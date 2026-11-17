"""完整動作表示（full-actions）的觀測：Loop 4 前向特徵 ＋ 三處補齊。

規格：內部規格文件 §第一段精確實作規格。

為什麼要 wrapper 而不是改 `rl/loop4_obs.py`：舊維度與舊權重要保持可重現，所以這裡
**只呼叫**既有的 `loop4_obs.observation`，再把三處缺的可見資訊接上去：

1. **global**（24 → 32）：自家五種作物的**種子存量**（`state.seeds`，舊 24 維完全沒有）
   ＋ 自家 shed 的**三種動物**（`state.shed[:, seat, 9:12]`；舊 product 特徵只取
   `shed[:, seat, :N_PRODUCTS]`＝9 類，動物槽看不到）。
2. **unit**（18 → 42，obs v2／T3 F5）：保留 loop4 的 6 維基本資訊，丟掉它 ``clip(inv/12)``
   的 12 維（12 與 100 件無法區分），改成每品項 ``log1p(n)/log1p(1000)``（不 clip）與
   ``clip(n/100)`` 兩通道，再接 12 維規範化插入順位（T3 F4）。

`tiles`（35）與 `products`（10）不動。

**`in_progress` 一律 0**：這條路徑是直接單位動作（verb 含四向 MOVE），沒有 Loop 4 的
topk／assign 執行器，「上一回合有人在做這格」沒有對應語意；teacher（IL）與 runtime（部署）
都傳零，不從 teacher label 構造假進度。

遮罩：新增的都是 private 量，**只放自己那一座**；
對手單位的 36 維持有量／順位填 0，死單位的這 36 維為 0。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import loop4_obs as O
from rl import rules
from rl.state import SimState, unit_alive

F32 = jnp.float32

#: 觀測契約版本（寫進 checkpoint meta）。
#: v2（T3 第二批 F5）：unit 30→42（6 基本＋12 log 量＋12 線性量＋12 規範化順位）
OBS_VERSION = "full-actions-obs-v2"

TILE_DIM = O.TILE_DIM  # 35，不動
PRODUCT_DIM = O.PRODUCT_DIM  # 10，不動
#: `loop4_obs.global_features` 的 market_trips／market_units 兩維（T3 F3 恆 0）
MARKET_COUNTER_IDX = (22, 23)
#: 觀測執行契約（T3 第一批：F3 歸零、F4 規範化順位；維度與 OBS_VERSION 不變）
EXEC_OBS_CONTRACT = "t3-batch1"
#: 24 ＋ 5 種子 ＋ 3 動物
GLOBAL_DIM = O.GLOBAL_DIM + rules.N_CROPS + rules.N_ANIMALS
#: unit layout（T3 F5）：``[0:6]`` 原 6 維基本資訊、``[6:18]`` log 數量、``[18:30]`` 線性數量、
#: ``[30:42]`` F4 規範化順位。舊的 ``inv/12`` 裁切（loop4 `[6:18]`）不再使用。
UNIT_BASE_DIM = 6
UNIT_DIM = UNIT_BASE_DIM + 3 * rules.N_ITEMS
#: log 通道 ``log1p(n)/log1p(1000)``：**不 clip 到 1**，n>1000 仍可區分尺度
QTY_LOG_SCALE = float(np.log1p(1000.0))
#: 線性通道 ``clip(n/100, 0, 1)``（shed 容量尺度）
QTY_LIN_SCALE = 100.0

#: 種子數量是非負 int32，**不在 200 截掉差異**（規格）；用 int32 正域當尺度。
SEED_SCALE = float(jnp.log1p(jnp.asarray(2**31 - 1, F32)))
#: shed 容量 100（與任務 30 的資產估值同一常數來源）。
SHED_CAP = 100.0


def seed_features(state: SimState, seat: int) -> jax.Array:
    """自家各作物種子存量 ``[B, 5]``，`log1p(x) / log1p(INT32_MAX)`。"""
    seeds = state.seeds[:, seat, : rules.N_CROPS].astype(F32)
    return jnp.clip(jnp.log1p(jnp.maximum(seeds, 0.0)) / SEED_SCALE, 0.0, 1.0)


def shed_animal_features(state: SimState, seat: int) -> jax.Array:
    """自家 shed 的三種動物 ``[B, 3]``，除以 shed 容量。"""
    animals = state.shed[:, seat, rules.ANIMAL_ITEM_BASE :].astype(F32)
    return jnp.clip(animals / SHED_CAP, 0.0, 1.0)


def inv_qty_features(state: SimState, caps: caps_mod.Caps) -> tuple[jax.Array, jax.Array]:
    """T3 F5：每單位 12 品項的持有量兩通道 ``[B, 2, U, 12]``（死單位為 0）。

    ``log1p(float32(n)) / log1p(1000)``（不 clip）與 ``clip(float32(n) / 100, 0, 1)``。
    取代 loop4 的 ``clip(inv/12)``：12 與 100 件在舊通道完全相同（審查者 F5）。
    """
    n = jnp.maximum(state.inv.astype(F32), 0.0)
    alive = unit_alive(state, caps)[..., None]
    log_q = jnp.where(alive, jnp.log1p(n) / QTY_LOG_SCALE, 0.0)
    lin_q = jnp.where(alive, jnp.clip(n / QTY_LIN_SCALE, 0.0, 1.0), 0.0)
    return log_q, lin_q


def inv_rank_features(state: SimState, caps: caps_mod.Caps) -> jax.Array:
    """每單位 12 品項的**規範化**插入順位 ``[B, 2, U, 12]``（T3 F4）。

    ``rank_i = 1 + #{j : present_j and seq_j < seq_i}``，``feature_i = rank_i / (N_present + 1)``；
    不存在、死單位為 0。只用 `inv_seq` 的**相對順序**，不用它的絕對值：kernel 的 `inv_seq`／
    `inv_ctr` 在「移除→再插入」後會持續累加（`rl/kernels.py::_update_inv_bookkeeping`），
    部署端則每步由官方 dict 重建成 1..N；舊式 ``seq / inv_ctr`` 兩邊數值不同（F4）。
    kernel 的 `inv_seq`／`inv_ctr` 本身**不改**（DROP／日終倒包仍依它排序）。
    存在品項 seq 非正或重複屬 bookkeeping 錯誤：jit 內不能丟例外，用
    :func:`inv_rank_violations` 另行計數（測試與 parity 驅動必須斷言為 0）。
    """
    seq = state.inv_seq.astype(jnp.int32)  # [B, 2, U, 12]
    present = (state.inv > 0) & unit_alive(state, caps)[..., None]
    earlier = present[..., None, :] & (seq[..., None, :] < seq[..., :, None])  # [.., i, j]
    rank = 1 + jnp.sum(earlier, axis=-1)
    n_present = jnp.sum(present, axis=-1, keepdims=True)
    return jnp.where(present, rank.astype(F32) / (n_present + 1).astype(F32), 0.0)


def inv_rank_violations(state: SimState, caps: caps_mod.Caps) -> jax.Array:
    """存在品項中 seq ≤ 0 或與同單位其他存在品項 seq 重複的個數 ``[B]``（應恆為 0）。"""
    seq = state.inv_seq.astype(jnp.int32)
    present = (state.inv > 0) & unit_alive(state, caps)[..., None]
    nonpos = present & (seq <= 0)
    same = present[..., None, :] & present[..., :, None] & (seq[..., None, :] == seq[..., :, None])
    dup = jnp.sum(same, axis=-1) > 1
    return jnp.sum(nonpos | (present & dup), axis=(1, 2, 3))


def observation(state: SimState, seat: int, caps: caps_mod.Caps) -> dict[str, jax.Array]:
    """一座的 token 觀測 ``tiles[B,2,100,35]／units[B,2,U,30]／products[B,9,10]／global[B,32]``。

    `seat` 是 Python int（單座）。座位輪換要逐 env 不同時，由呼叫端對兩座各算一次再
    **明確 gather／vmap**（規格：不以 array seat 套舊 scalar 索引）——見
    `observation_seat`。
    """
    base = O.observation(state, seat, caps, assignment=None, in_progress=None)
    other = 1 - seat
    order = jnp.asarray([seat, other])

    rank = inv_rank_features(state, caps)[:, order]  # [B, 2, U, 12]
    log_q, lin_q = inv_qty_features(state, caps)
    priv = jnp.concatenate([log_q[:, order], lin_q[:, order], rank], axis=-1)
    priv = priv.at[:, 1].set(0.0)  # 對手持有量與順位是 private
    units = jnp.concatenate([base["units"][..., :UNIT_BASE_DIM], priv], axis=-1)

    # T3 F3：global 22/23（market_trips／market_units）在 IL 資料與部署端恆為 0，
    # full-actions 觀測一律歸零；simulator 的原始欄位不動（工程診斷仍可讀）。
    base_glob = base["global"].at[:, MARKET_COUNTER_IDX].set(0.0)
    glob = jnp.concatenate(
        [base_glob, seed_features(state, seat), shed_animal_features(state, seat)], axis=-1
    )
    assert units.shape[-1] == UNIT_DIM, units.shape
    assert glob.shape[-1] == GLOBAL_DIM, glob.shape
    return {"tiles": base["tiles"], "units": units, "products": base["products"], "global": glob}


def observation_seat(state: SimState, seat_vec: jax.Array, caps: caps_mod.Caps) -> dict:
    """逐 env 座位：兩座各算一次，再依 `seat_vec` ``[B]`` **明確選**。

    不把 array 塞進 `seat`——舊的 scalar 索引（`state.money[:, seat]`、`[seat, 1 - seat]`
    重排）吃 Python int，傳 array 會靜默算錯。
    """
    o0 = observation(state, 0, caps)
    o1 = observation(state, 1, caps)
    sv = jnp.asarray(seat_vec, jnp.int32)
    return {
        k: jnp.where(sv.reshape((-1,) + (1,) * (o0[k].ndim - 1)) == 0, o0[k], o1[k]) for k in o0
    }


def dims() -> dict[str, int]:
    """寫進 checkpoint meta 的維度契約。"""
    return {
        "obs_version": OBS_VERSION,
        "tile_dim": TILE_DIM,
        "unit_dim": UNIT_DIM,
        "product_dim": PRODUCT_DIM,
        "global_dim": GLOBAL_DIM,
        "in_progress": "always_zero",
        "seed_scale": "log1p(x)/log1p(2**31-1)",
        "shed_animal_scale": f"x/{SHED_CAP:g}",
        "unit_layout": "[0:6] base, [6:18] log1p(n)/log1p(1000), [18:30] clip(n/100), [30:42] rank",
        "inv_rank": "rank/(n_present+1), rank=1+#(present & seq<seq_i), dead unit zeroed",
        "market_counters": "global[22:24] always zero",
        "execution_obs_contract": EXEC_OBS_CONTRACT,
    }


__all__ = [
    "GLOBAL_DIM",
    "OBS_VERSION",
    "PRODUCT_DIM",
    "TILE_DIM",
    "UNIT_DIM",
    "dims",
    "inv_qty_features",
    "inv_rank_features",
    "inv_rank_violations",
    "observation",
    "observation_seat",
    "seed_features",
    "shed_animal_features",
]
