"""Loop 4 的 PPO 迴圈骨架（任務 14 B3）。**本模組不啟動任何正式 run。**

配方照 內部規格文件 v3 §B 與 11th Supergiants：

| 項 | 值 | 依據 |
|---|---|---|
| rollout | T=719 整局、N=256 | B1 實測 3,239 transitions/s、VRAM 6.33 GiB |
| gamma／λ | 1.0／0.95 | Isaiah gamma 1；λ 全體 0.90–0.97 |
| clip／epochs | 0.2／1 | felix、boey |
| minibatch | 1,024（B1 量到 minibatch 256 時 update 佔 40.8 s／epoch，是唯一可優化的一段） | B1 |
| LR | 3e-5 | Ender |
| 起點 | IL 權重 | v5 §1 |
| **KL** | 對 IL policy 的**常數 KL 0.05**，**無 entropy** | 11th |
| advantage | **整 batch** normalization | 11th |
| reward | 終局現金差（telescoping 版留旗標對照） | 12／13 隊終局稀疏 |
| 對手 | tape 重播（`rl/loop4_tape.py`）＋ 自己的快照池（1−winrate）＋ IL 權重底線 | v4 §3 |

守衛沿 Loop 3 任務 09 的裁定：`old_logp` 重算的比較用 **median／越線 share／max>1.0** 三條，
不用 max——max 在飽和 logit 上會被單一 token 的浮點離群支配。

**還沒實作的**（B3 正式開工時要補）：rollout 的 scan、GAE、minibatch 更新、league 抽樣、
checkpoint／resume、history。本檔目前只有**配方常數與形狀契約**，讓 B3 的實作有一個
被測試釘住的起點，而不是邊寫邊定。
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from rl import caps as caps_mod
from rl import loop4_actions as A
from rl import loop4_model as M
from rl import loop4_obs as O
from rl import rules

# ---------------------------------------------------------------- 配方常數
ROLLOUT_T = rules.N_TRANSITIONS  # 719，整局
N_ENVS = 256
GAMMA = 1.0
LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 1
MINIBATCH = 1024
LR = 3e-5
#: 對 IL policy 的常數 KL（11th）。**無 entropy**——Ender 的教訓是 entropy 治不了數量頭退化。
KL_TO_IL = 0.05
ENTROPY = 0.0
#: 整 batch 的 advantage normalization（11th），不是 per-minibatch。
ADV_NORM_WHOLE_BATCH = True

#: logp 守衛（任務 09 裁定；不用 max 當統計量）。
STEPWISE_LOGP_WARN = 1e-2
STEPWISE_LOGP_MEDIAN_STOP = 1e-3
STEPWISE_LOGP_SHARE_STOP = 1e-4
STEPWISE_LOGP_ABS_STOP = 1.0

#: league 的三類對手。
OPP_TAPE = "tape"
OPP_SNAPSHOT = "snapshot"
OPP_IL = "il"


class HeadSpec(NamedTuple):
    """一個 head 的形狀契約：logits 的最後一維，以及它對應的決策欄位。"""

    name: str
    field: str
    n_classes: int
    per_entity: int  # 1 = 全域，其餘 = 實體數（格／產品／作物…）


def head_specs(caps: caps_mod.Caps) -> tuple[HeadSpec, ...]:
    """所有 head 的形狀，logp 聚合與 KL 都照這張表逐 head 算。"""
    del caps
    return (
        HeadSpec("task", "tile_task", A.N_TASKS, rules.N_TILES),
        HeadSpec("tile_qty", "tile_qty", A.N_QTY_BUCKETS, rules.N_TILES),
        HeadSpec("sell", "sell", A.N_SELL, rules.N_PRODUCTS),
        HeadSpec("buy_seed", "buy_seed", A.N_BUY_SEED, rules.N_CROPS),
        HeadSpec("buy_product", "buy_product", A.N_BUY_PRODUCT, len(A.BUYABLE_PRODUCTS)),
        HeadSpec("buy_animal", "buy_animal", A.N_BUY_ANIMAL, rules.N_ANIMALS),
        HeadSpec("hire", "hire", A.N_HIRE, 1),
        HeadSpec("buy_land", "buy_land", A.N_BUY_LAND, 1),
    )


def masked_logits(out: dict, name: str, logit_mask: dict | None):
    """套上**與抽樣時完全相同**的合法性遮罩（任務 25c，審查者 2026-09-20 裁定）。

    `sample_decision` 對 task 是從 `jnp.where(tile_mask, logits, -inf)` 抽的，但 logp 一度
    對**未遮罩**的 logits 做 log_softmax——那樣 `exp(old_logp)` 不是抽樣機率，ratio 會差
    一個 `Z_old/Z_new`（Z ＝合法類別的機率和，**它也隨參數變**）。rollout 與 update 現在
    走同一個函式、同一個 mask 來源。
    """
    logits = out[name]
    m = None if logit_mask is None else logit_mask.get(name)
    return logits if m is None else jnp.where(m, logits, -jnp.inf)


def logp_terms(
    out: dict,
    decision: A.Loop4Decision,
    active: dict,
    seat: int = 0,
    logit_mask: dict | None = None,
):
    """逐 head 的 ``(名字, picked [B, E], active [B, E])``——**所有 logp 分解的唯一來源**。

    ``E`` 是那個 head 的實體數（task／tile_qty 是 100 格、sell 是 9 個產品、hire 與 buy_land
    是 1）。被遮掉的項 picked 設 0、mask 設 False，語意與 Loop 3 的 `rl/ppo.py` 一致。

    ``logit_mask`` 是**合法性**遮罩（哪些類別抽得到），與 ``active``（哪些項計入 ratio）
    是兩件事：前者改分佈的分母，後者決定這一項算不算數。
    """
    for spec in head_specs(caps_mod.BOUNDED_TRAINING):
        logits = masked_logits(out, spec.name, logit_mask)
        choice = getattr(decision, spec.field)
        if spec.field in ("tile_task", "tile_qty"):
            choice = choice[:, seat]  # 只有自己那一座出決策
        elif choice.ndim >= 2 and choice.shape[1] == 2:
            choice = choice[:, seat]
        lp = jax.nn.log_softmax(logits, axis=-1)
        picked = jnp.take_along_axis(lp, choice[..., None], axis=-1)[..., 0]
        m = active.get(spec.name)
        if picked.ndim == 1:  # 全域 head：攤成 [B, 1]，讓所有 head 形狀一致
            picked = picked[:, None]
            m = None if m is None else m.reshape(picked.shape)
        mask = jnp.ones(picked.shape, bool) if m is None else m
        yield spec.name, jnp.where(mask, picked, 0.0), mask


def per_head_logp(
    out: dict,
    decision: A.Loop4Decision,
    active: dict,
    seat: int = 0,
    logit_mask: dict | None = None,
) -> dict[str, jax.Array]:
    """`joint_logp` 的逐 head 分解：每個 head 一份 ``[B]``，相加就是 joint。

    診斷用（任務 25b 第 2 點：joint logp 的變化主要來自哪個 head）。
    """
    return {
        n: jnp.sum(p, axis=-1) for n, p, _ in logp_terms(out, decision, active, seat, logit_mask)
    }


def component_logp(
    out: dict,
    decision: A.Loop4Decision,
    active: dict,
    seat: int = 0,
    logit_mask: dict | None = None,
) -> tuple[jax.Array, jax.Array]:
    """**展開到最細**的逐 component logp ``[B, C]`` 與 active 遮罩 ``[B, C]``（任務 25c）。

    C ＝ 100 格 task ＋ 100 格 qty ＋ 9 sell ＋ 各市場品項 ＋ hire ＋ buy_land。
    每一項是**一個決策**的 log 機率；PPO 的 clip 要作用在這一層，不是 108 項的和上。

    根因（smoke C 實測）：`joint_logp` 把這些全部加起來，每格平移 0.0045 nat 就讓 joint
    ratio 變 1.57、每格 0.15 就 e^15。一次 lr 3e-5 的 Adam step 就足以讓 100 格同向平移，
    joint 上的 clip 完全管不住每一格。
    """
    parts = list(logp_terms(out, decision, active, seat, logit_mask))
    return (
        jnp.concatenate([p for _, p, _ in parts], axis=-1),
        jnp.concatenate([m for _, _, m in parts], axis=-1),
    )


def component_slices() -> dict[str, slice]:
    """`component_logp` 的 ``[B, C]`` 裡每個 head 佔哪一段（診斷／報表用）。"""
    out, o = {}, 0
    for spec in head_specs(caps_mod.BOUNDED_TRAINING):
        n = spec.per_entity
        out[spec.name] = slice(o, o + n)
        o += n
    return out


def factored_surrogate(
    logp_h: jax.Array, old_logp_h: jax.Array, mask_h: jax.Array, adv: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """**逐 component** 的 clipped surrogate（任務 25c 裁定；審查者 2026-09-20 改為求和）。

    ``pg = −mean_rows( Σ_h active_h · min(r_h·A, clip(r_h)·A) )``：每個 component 一個
    ratio、**各自 clip**，一列把 active 項**相加**（不是平均、也不除常數）。

    為什麼是相加：ratio 全等於 1 時這個式子與 joint 版的梯度量級相同，``v`` 0.5 與
    ``KL`` 0.05 的係數本來就是對那個量級定的；每列 active 數不同造成的權重差與 joint 版
    一樣（joint 也是把 218 項加起來）。與 joint 版的差別只在**先 clip 再加**：
    joint 是 ``e^(ΣΔ_h)``（100 格各動 0.15 就 e^15），這裡是 ``Σ clip(e^Δ_h)``，
    每一項各自被 clip 咬住，所以隨項數線性成長、不是指數。
    """
    r = jnp.exp(logp_h - old_logp_h)
    a = adv[:, None]
    surr = jnp.minimum(r * a, jnp.clip(r, 1 - CLIP, 1 + CLIP) * a)
    m = mask_h.astype(surr.dtype)
    pg = -jnp.mean(jnp.sum(surr * m, axis=-1))
    rm = jnp.where(mask_h, r, jnp.nan)
    return pg, {
        "ratio_h_median": jnp.nanmedian(rm),
        "ratio_h_max": jnp.max(jnp.where(mask_h, r, -jnp.inf)),
        "ratio_h_min": jnp.min(jnp.where(mask_h, r, jnp.inf)),
        "clip_frac_h": jnp.sum(m * (jnp.abs(r - 1.0) > CLIP)) / jnp.maximum(jnp.sum(m), 1.0),
        "n_components_active": jnp.sum(m),
    }


def joint_logp(
    out: dict,
    decision: A.Loop4Decision,
    active: dict,
    seat: int = 0,
    logit_mask: dict | None = None,
) -> jax.Array:
    """每個 seat-turn 一個 joint ``log π``：所有 head 的 log 機率相加。

    ``active`` 給每個 head 一份 ``[B, ...]`` 的遮罩——被遮掉的項貢獻 0，語意與 Loop 3 的
    `rl/ppo.py` 相同（「inactive／被強制項貢獻 0」），所以 ratio 仍是正確的 importance weight。
    每個 seat-turn **一個 joint ratio**，不按 component 拆。
    """
    total = None
    for v in per_head_logp(out, decision, active, seat, logit_mask).values():
        total = v if total is None else total + v
    return total


def kl_to_reference(
    out: dict, ref_out: dict, active: dict, logit_mask: dict | None = None
) -> jax.Array:
    """對參考 policy（IL 權重）的 KL，逐 head 算再相加。常數係數 `KL_TO_IL`。

    ``logit_mask`` 時**兩邊都用遮罩後的分佈**、且只在合法類別上加總——非法類別的
    ``log_softmax`` 是 −inf，``exp(lq)·(lq−lp)`` 會是 ``0 × nan``，必須明確排除。
    """
    total = None
    for spec in head_specs(caps_mod.BOUNDED_TRAINING):
        lgm = None if logit_mask is None else logit_mask.get(spec.name)
        lp = jax.nn.log_softmax(masked_logits(out, spec.name, logit_mask), axis=-1)
        lq = jax.nn.log_softmax(masked_logits(ref_out, spec.name, logit_mask), axis=-1)
        term = jnp.exp(lq) * (lq - lp)
        kl = jnp.sum(term if lgm is None else jnp.where(lgm, term, 0.0), axis=-1)
        m = active.get(spec.name)
        if m is not None:
            kl = jnp.where(m, kl, 0.0)
        s = kl if kl.ndim == 1 else jnp.sum(kl, axis=-1)
        total = s if total is None else total + s
    return total


def normalize_advantages(adv: jax.Array, valid: jax.Array | None = None) -> jax.Array:
    """**整 batch** normalization（11th），不是 per-minibatch。

    給 `valid` 時 mean／std **只用有效列**算，無效列維持 0——否則局末後的 padding 列
    （adv 已被 `gae` 歸零）會把 mean 拉向 0、std 壓小，等於用無效資料改變有效列的尺度。
    """
    if valid is None:
        return (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
    m = valid.astype(adv.dtype)
    n = jnp.maximum(jnp.sum(m), 1.0)
    mean = jnp.sum(adv * m) / n
    var = jnp.sum(((adv - mean) * m) ** 2) / n
    return ((adv - mean) / (jnp.sqrt(var) + 1e-8)) * m


#: 終局現金差要除掉的尺度。教師的終局現金是 10^4–10^5 量級，critic 直接回歸原尺度時
#: ``v_loss`` 是 10^9–10^10，會把 policy gradient 完全淹掉（B3 smoke 實測 v=185,401、
#: pg=3.3e-7，兩者差 11 個數量級）。除以 10,000 讓 reward 落在個位數。
REWARD_SCALE = 1e-4


def cash_margin(money: jax.Array, seat, scale: float = REWARD_SCALE) -> jax.Array:
    """現金差（自己 − 對手）× `scale`。``money`` 是 ``[..., B, 2]``，回傳 ``[...,  B]``。

    終局版（`terminal_reward`）與逐步版（`dense_rewards`）共用同一個定義與同一個 seat 邏輯，
    這樣兩者的尺度才真的可比——A/B 只差在「什麼時候把這個差發給 agent」。
    """
    b = money.shape[-2]
    sv = seat_vec(seat, b)
    idx = sv.reshape((1,) * (money.ndim - 2) + (b, 1))
    idx = jnp.broadcast_to(idx, money.shape[:-1] + (1,))
    own = jnp.take_along_axis(money, idx, axis=-1)[..., 0]
    opp = jnp.take_along_axis(money, 1 - idx, axis=-1)[..., 0]
    return (own - opp).astype(jnp.float32) * scale


def terminal_reward(money: jax.Array, seat, scale: float = REWARD_SCALE) -> jax.Array:
    """終局現金差，按 內部規格文件 v3 第 3 點的預設（終局 only），再乘 `REWARD_SCALE`。

    `seat` 可以是 int，也可以是 `[B]`（座位輪換後每個 env 不同）。
    """
    sv = seat_vec(seat, money.shape[0])
    own = jnp.take_along_axis(money, sv[:, None], axis=1)[:, 0]
    opp = jnp.take_along_axis(money, (1 - sv)[:, None], axis=1)[:, 0]
    return (own - opp).astype(jnp.float32) * scale


def dense_rewards(traj, final, seat, valid: jax.Array | None = None) -> jax.Array:
    """逐步現金差（設計 v6 §4）：``r_t = margin(s_{t+1}) − margin(s_t)``，回傳 ``[T, B]``。

    `traj.state` 存的是**每一步行動前**的 state，所以 `traj.state.money[t]` ＝ margin(s_t)，
    最後一步的下一個 state 是 `final`。

    **telescoping 恆等式**：``Σ_t r_t = terminal_reward(final) − margin(初始 state)``。
    day 0 開局時雙方現金相同、初始 margin ＝ 0，兩者因此完全相等；但
    **教師狀態重開局（`--p-teacher`）的初始 margin 不是 0**——那是「接手時就已經存在的
    差距」，不是這個 policy 打出來的。這裡**不**把它塞進 `r_0`：逐步版發的是
    「這一步讓差距變動多少」，繼承來的差距不歸功也不歸咎於 agent。
    呼叫端要對回 `terminal_reward` 時，用 `initial_margin` 補回去。

    **局末之後（padding）**：rollout 已把 state 凍結，所以 post − pre ＝ 0；給了 `valid`
    就再明確乘一次遮罩，讓「無效列 reward 必為 0」不依賴凍結這個間接性質。
    """
    pre = cash_margin(traj.state.money, seat)  # [T, B]，每一步行動前
    post = jnp.concatenate([pre[1:], cash_margin(final.money, seat)[None]], axis=0)
    r = post - pre
    return r if valid is None else r * valid.astype(r.dtype)


# ------------------------------------------------ 任務 30：資產淨值勢函數（networth）
#: 價格常數**全部來自引擎／`rl/rules.py`**，不在這裡寫死：
#: * 土地＝已解鎖象限的累計購價（`rules.LAND_PRICES` 1,000／2,000／4,000 的前綴和）
#: * 動物＝`rules.ANIMAL_COST`（300／400／500）
#: * 種子與田裡作物＝`rules.CROP_SEED_COST`（10／20／50／100／80，保守估，不用預期售價）
#: * 產品＝市價表在**初始庫存 `rules.MARKET_I0`** 上的價（WHEAT 25 … MELON 250 … WOOL 200）
#: * 肥料、人手、coop／pasture 結構＝0
_ASSET_TABLES: dict[str, Any] = {}


def asset_tables() -> dict[str, Any]:
    """Φ 用到的價格向量（lazy，因為市價表要讀／建 9 × 80,001 的快取）。"""
    if not _ASSET_TABLES:
        import numpy as np  # noqa: PLC0415

        from rl import pricing  # noqa: PLC0415

        base = pricing.price_table()[:, rules.MARKET_I0 - pricing.INV_MIN].astype(np.float32)
        item = np.zeros(rules.N_ITEMS, np.float32)
        item[: rules.N_PRODUCTS] = base
        item[rules.I_FERTILIZER] = 0.0  # 肥料不計入資產
        item[rules.ANIMAL_ITEM_BASE :] = np.asarray(rules.ANIMAL_COST, np.float32)
        _ASSET_TABLES.update(
            item=jnp.asarray(item),
            seed=jnp.asarray(np.asarray(rules.CROP_SEED_COST, np.float32)),
            animal=jnp.asarray(np.asarray(rules.ANIMAL_COST, np.float32)),
            land=jnp.asarray(
                np.concatenate([[0.0], np.cumsum(rules.LAND_PRICES)]).astype(np.float32)
            ),
        )
    return _ASSET_TABLES


def asset_value(state, seat) -> jax.Array:
    """自己那一座的資產估值 Φ(s) ``[B]``（**未乘 `REWARD_SCALE`**，單位是引擎的錢）。

    現金 ＋ 土地（已解鎖象限的累計購價）＋ 田裡作物（種子價）＋ 畜舍裡的動物（購價）
    ＋ shed 與人手身上的產品（初始市價）與種子（種子價）。肥料、人手、結構算 0。

    為什麼要它（任務 30，作者 指出的問題）：`--reward dense` 每步發「現金差的變動」，
    買地 −1,000／買動物 −300～−500／買種子在**當下全是負 reward**，回收在幾十到幾百步後；
    λ=0.95 的 advantage 主要看接下來約 20 步，於是梯度推向「不投資」——正好跟教師
    day 0 就把錢花光的核心行為相反。把買進來的東西**按買價計入資產**，投資當下就是零，
    真正的損益留給「賣多少錢」與「東西爛掉」去表達。
    """
    t = asset_tables()
    b = state.money.shape[0]
    sv = seat_vec(seat, b)
    money = pick_seat(state.money, sv).astype(jnp.float32)
    land = t["land"][pick_seat(state.n_quads, sv).astype(jnp.int32)]
    tiles = pick_seat(state.tiles, sv)
    animal = tiles[..., rules.C_ANIMAL].astype(jnp.int32)
    animal_v = jnp.sum(
        jnp.where(animal >= 0, t["animal"][jnp.clip(animal, 0, rules.N_ANIMALS - 1)], 0.0), -1
    )
    planted = tiles[..., rules.C_KIND].astype(jnp.int32) == rules.TK_PLANT
    crop = tiles[..., rules.C_CROP].astype(jnp.int32)
    crop_v = jnp.sum(jnp.where(planted, t["seed"][jnp.clip(crop, 0, rules.N_CROPS - 1)], 0.0), -1)
    shed_v = jnp.sum(pick_seat(state.shed, sv).astype(jnp.float32) * t["item"], -1)
    inv_v = jnp.sum(pick_seat(state.inv, sv).astype(jnp.float32) * t["item"], (-2, -1))
    seeds_v = jnp.sum(pick_seat(state.seeds, sv).astype(jnp.float32) * t["seed"], -1)
    return money + land + animal_v + crop_v + shed_v + inv_v + seeds_v


def opponent_money(state, seat) -> jax.Array:
    """對手那一座的現金 ``[B]``（對手側不算資產，比賽只算現金）。"""
    sv = seat_vec(seat, state.money.shape[0])
    return pick_seat(state.money, 1 - sv).astype(jnp.float32)


def networth_rewards(
    traj, final, seat, scale: float = REWARD_SCALE, valid: jax.Array | None = None
) -> jax.Array:
    """勢函數 shaping 的逐步 reward ``[T, B]``（Ng et al. 1999：不改最優策略）。

    ``r_t = [Φ(s_{t+1}) − Φ(s_t)] − [opp_money(s_{t+1}) − opp_money(s_t)]``，乘 `REWARD_SCALE`。

    **終局把非現金資產歸零**：比賽只算現金，所以最後一步用 ``Φ_final = money_final``
    ——手上還沒變現的東西在那一步變成負 reward，這正是「該賣要賣」的訊號。

    加總恆等式：``Σ_t r_t = terminal_reward(final) − initial_asset_margin(s_0)``，
    其中初始資產差 ＝ ``(Φ(s_0) − opp_money(s_0)) × scale``。教師狀態重開局繼承的資產與
    現金差**不歸功也不歸咎**於這個 policy，與 `dense_rewards` 的處理一致。
    """
    phi_pre = jax.vmap(lambda st: asset_value(st, seat))(traj.state)  # [T, B]
    opp_pre = jax.vmap(lambda st: opponent_money(st, seat))(traj.state)
    sv = seat_vec(seat, final.money.shape[0])
    phi_fin = pick_seat(final.money, sv).astype(jnp.float32)  # 終局只認現金
    opp_fin = pick_seat(final.money, 1 - sv).astype(jnp.float32)
    phi_post = jnp.concatenate([phi_pre[1:], phi_fin[None]], axis=0)
    opp_post = jnp.concatenate([opp_pre[1:], opp_fin[None]], axis=0)
    if valid is not None:
        # **資產歸零要發生在各 env 自己的真局末那一步**，不是 scan 的最後一步：教師狀態
        # 重開局的 env 在 scan 末步早就是 padding，歸零會被 valid 遮掉，Σr 變成
        # 「現金差 ＋ 資產差」，「該賣要賣」的訊號整個消失。
        is_last = jnp.arange(phi_pre.shape[0])[:, None] == last_valid_step(valid)[None, :]
        phi_post = jnp.where(is_last, phi_fin[None, :], phi_post)
        opp_post = jnp.where(is_last, opp_fin[None, :], opp_post)
    r = ((phi_post - phi_pre) - (opp_post - opp_pre)) * scale
    return r if valid is None else r * valid.astype(r.dtype)


def initial_asset_margin(traj, seat, scale: float = REWARD_SCALE) -> jax.Array:
    """開局那一刻的「資產 − 對手現金」``[B]`` × scale（`networth_rewards` 的加總補正項）。"""
    s0 = jax.tree.map(lambda x: x[0], traj.state)
    return (asset_value(s0, seat) - opponent_money(s0, seat)) * scale


def initial_margin(traj, seat) -> jax.Array:
    """開局那一刻的 margin ``[B]``（教師狀態重開局時不是 0）。"""
    return cash_margin(traj.state.money, seat)[0]


def logp_guard_stats(delta: jax.Array) -> dict[str, jax.Array]:
    """守衛的三個統計量（任務 09 裁定）。呼叫端決定 STOP／WARN，本函式只算。"""
    a = jnp.abs(delta)
    return {
        "median": jnp.median(a),
        "n_over_warn": jnp.sum(a > STEPWISE_LOGP_WARN),
        "max": jnp.max(a),
        "n_rows": jnp.asarray(a.size),
    }


#: reward 模式（任務 28）。`dense` 是設計 v6 §4 的逐步現金差，telescoping 回同一個終局差。
REWARD_TERMINAL = "terminal_cash_margin"
REWARD_DENSE = "dense_cash_margin_telescoping"
REWARD_NETWORTH = "networth_potential_telescoping"
#: `--reward` 的旗標值 → 寫進 result.json 的名字。**旗標與名字只在這裡對一次**
#: （任務 30 的教訓：分散在 trainer 裡的三元式改漏了一處，result.json 標成 terminal，
#: 但實際跑的是 networth——這種靜默錯標會被後人當證據引用）。
REWARD_MODES = {
    "terminal": REWARD_TERMINAL,
    "dense": REWARD_DENSE,
    "networth": REWARD_NETWORTH,
}


def config(reward: str = REWARD_TERMINAL) -> dict:
    """寫進 result.json 的配方，一個地方定義，不讓它散在指令列裡。"""
    return {
        "rollout_T": ROLLOUT_T,
        "n_envs": N_ENVS,
        "gamma": GAMMA,
        "lambda": LAMBDA,
        "clip": CLIP,
        "epochs": EPOCHS,
        "minibatch": MINIBATCH,
        "lr": LR,
        "kl_to_il": KL_TO_IL,
        "entropy": ENTROPY,
        "adv_norm_whole_batch": ADV_NORM_WHOLE_BATCH,
        "reward": reward,
        "reward_scale": REWARD_SCALE,
        "opponents": [OPP_TAPE, OPP_SNAPSHOT, OPP_IL],
        "model": dict(M.Config.default(caps_mod.BOUNDED_TRAINING)),
        "guard": {
            "warn": STEPWISE_LOGP_WARN,
            "median_stop": STEPWISE_LOGP_MEDIAN_STOP,
            "share_stop": STEPWISE_LOGP_SHARE_STOP,
            "abs_stop": STEPWISE_LOGP_ABS_STOP,
        },
    }


# ================================================================ B3 骨架
# 以下是 rollout／GAE／更新的實作。**本模組仍不啟動任何正式 run**——這裡只有純函式，
# 由 `scripts/loop4_ppo_train.py`（尚未寫）負責 I/O、checkpoint 與 run_capped。


def seat_vec(seat, b: int) -> jax.Array:
    """把 `seat`（int 或 `[B]` 陣列）正規化成 `[B]` 的 int32。"""
    if isinstance(seat, int):
        return jnp.full((b,), seat, jnp.int32)
    return jnp.asarray(seat, jnp.int32)


def pick_seat(x: jax.Array, sv: jax.Array) -> jax.Array:
    """`[B, 2, ...]` → `[B, ...]`，每個 env 取自己那一座。"""
    idx = sv.reshape((-1,) + (1,) * (x.ndim - 1))
    return jnp.take_along_axis(x, idx, axis=1)[:, 0]


def place_seat(x: jax.Array, sv: jax.Array) -> jax.Array:
    """`[B, ...]` → `[B, 2, ...]`，放進自己那一座，另一座留 0。"""
    z = jnp.zeros_like(x)
    at0 = jnp.stack([x, z], axis=1)
    at1 = jnp.stack([z, x], axis=1)
    return jnp.where(sv.reshape((-1,) + (1,) * (at0.ndim - 1)) == 0, at0, at1)


def merge_by_seat(own, opp, sv: jax.Array):
    """兩份各自只填了一座的 `SimAction`，按每個 env 的座位合成一份。

    `rl/loop4_tape.py::merge_seat_actions` 只吃純量 seat；座位輪換後每個 env 不同，
    所以這裡用逐 env 的遮罩選，不是切片賦值。
    """

    def f(a, b):
        keep_own = jnp.arange(2).reshape((1, 2) + (1,) * (a.ndim - 2)) == sv.reshape(
            (-1, 1) + (1,) * (a.ndim - 2)
        )
        return jnp.where(keep_own, a, b)

    return jax.tree.map(f, own, opp)


def observation_seat(state, sv: jax.Array, caps, assignment=None):
    """逐 env 座位的觀測：兩座各算一次再逐 env 選。

    `rl/loop4_obs.py::observation` 的 `seat` 是 Python int（它用 `[seat, 1-seat]` 重排格與人手），
    座位輪換要的是逐 env 不同的視角。不改共用的 obs 程式（IL 路徑也在用），改在這裡選。
    """
    o0 = O.observation(state, 0, caps, assignment)
    o1 = O.observation(state, 1, caps, assignment)
    return {
        k: jnp.where(sv.reshape((-1,) + (1,) * (o0[k].ndim - 1)) == 0, o0[k], o1[k]) for k in o0
    }


def sample_decision(
    params, obs, mask, key, cfg, greedy: bool = False, seat=0, fast=None, with_components=False
):
    """從 model 輸出抽一份**自己這座**的決策，同時回傳 joint logp、value 與 active 遮罩。

    **與 IL 推論期的 top-k 解碼不同**：那裡用「按 1 − P(NOOP) 取前 k 格」是個決定論的排序
    啟發式，沒有對應的機率密度，ratio 算不出來。PPO 直接從每格（含 NOOP）的遮罩後 categorical
    抽，NOOP 由梯度自己學要不要選——這正是 IL 階段缺的訊號。

    回傳 ``(decision, logp, value, active)``；``decision`` 是兩座的 `Loop4Decision`，
    自己的欄位放在 ``seat``、對手那座全 0，由呼叫端用 tape／快照覆蓋。
    ``obs`` 必須是**同一個 seat 視角**的觀測，遮罩也從該座取——B2 的 `loop4_eval` 曾經
    觀測取第 0 座、state 卻把農場放在真實座位，半數的局在替對手決策。
    """
    out = M.forward(params, obs, cfg, fast=fast)
    b = obs["global"].shape[0]
    sv = seat_vec(seat, b)
    keys = jax.random.split(key, len(head_specs(caps_mod.BOUNDED_TRAINING)))

    def pick(logits, i, m=None):
        lg = logits if m is None else jnp.where(m, logits, -jnp.inf)
        if greedy:
            return jnp.argmax(lg, axis=-1).astype(jnp.int32)
        return jax.random.categorical(keys[i], lg, axis=-1).astype(jnp.int32)

    tile_mask = pick_seat(mask, sv)  # [B, 100, 17]
    task = pick(out["task"], 0, tile_mask)
    fields = {
        "tile_task": task,
        "tile_qty": pick(out["tile_qty"], 1),
        "sell": pick(out["sell"], 2),
        "buy_seed": pick(out["buy_seed"], 3),
        "buy_product": pick(out["buy_product"], 4),
        "buy_animal": pick(out["buy_animal"], 5),
        "hire": pick(out["hire"], 6),
        "buy_land": pick(out["buy_land"], 7),
    }

    decision = A.Loop4Decision(**{k: place_seat(v, sv) for k, v in fields.items()})
    active = active_masks(fields["tile_task"])
    # joint logp 只看自己選的那份，所以兩座都塞同一份，seat 指到哪一座都一樣
    own = A.Loop4Decision(**{k: jnp.stack([v, v], axis=1) for k, v in fields.items()})
    # **logp 用與抽樣完全相同的遮罩後分佈**（task 是唯一有合法性遮罩的 head）
    logit_mask = {"task": tile_mask}
    logp_h, mask_h = component_logp(out, own, active, seat=0, logit_mask=logit_mask)
    logp = jnp.sum(logp_h, axis=-1)
    if with_components:
        # 任務 25c：PPO 的 ratio 逐 component 算，所以 rollout 要存逐 component 的 old_logp
        return decision, logp, out["value"], active, logp_h, mask_h
    return decision, logp, out["value"], active


def active_masks(tile_task: jax.Array) -> dict[str, jax.Array]:
    """哪些 head 的選擇要計入 joint logp。

    ``tile_qty`` 只有在該格的任務**真的帶數量**（PICKUP／PLACE）時才是自由選擇，
    其餘情況解碼會忽略它，計進 ratio 只會加無關的噪音。其他 head 全部計入
    （「不賣／不買」本身就是那些 head 的第 0 類，是真的決策，不是 inactive）。
    """
    qty_used = (tile_task == A.T_PICKUP) | (tile_task == A.T_PLACE)
    return {"tile_qty": qty_used}


class Transition(NamedTuple):
    """rollout 每一步留下的東西。**不存 obs**——`[719,256,2,100,35]` 的每格特徵是 51 GiB，
    存不下也不該存；更新期用存下來的 state 重算 obs（前向 2.6 ms／batch，遠比搬 51 GiB 便宜）。
    """

    state: Any  # SimState，重算 obs 用
    assignment: jax.Array  # [B, 2, U]，obs 的 in-progress 維要它才能重現
    decision: A.Loop4Decision
    logp: jax.Array  # [B]，joint（對照用）
    logp_h: jax.Array  # [B, C]，逐 component 的 old_logp（任務 25c：ratio 在這一層算）
    value: jax.Array  # [B]
    done: jax.Array  # [B]，這一步之後是否已到真局末
    seat: jax.Array  # [B]，這一列是坐哪一座出的決策（座位輪換）
    valid: jax.Array  # [B] bool，pre-state 還在局內（False ＝ 局末後的 padding，不進訓練）


def rollout(
    params, cfg, caps, init_state, key, opp_action_fn, seat=0, t: int = ROLLOUT_T, fast=None
):
    """整局 rollout，``jax.lax.scan`` 跑 T 步。

    ``opp_action_fn(state, turn, key) -> SimAction`` 由 league 提供（tape 重播／自己的快照／
    IL 權重），**它自己負責逐 env 的對手混合與對手座位**；本函式只把兩座的動作合起來送進引擎。

    ``seat`` 可以是 int 或 ``[B]``：座位輪換時每個 env 坐不同的座位，觀測、遮罩、決策放置、
    動作合併與 reward 全部按各自的座位來。
    """
    from rl import env as env_mod

    b = init_state.step.shape[0]
    sv = seat_vec(seat, b)

    def step(carry, _):
        state, assign, k = carry
        k, k_pol, k_opp = jax.random.split(k, 3)
        # **逐 env 的有效列**（任務 P1）：pre-state 還沒到真局末的那些 env。教師狀態重開局帶著
        # bank 的原 step 進來，scan 仍固定跑 t 步，所以會越過真局末；越過之後凍結 state 與
        # assignment，後面的列不再改變任何東西（reward／GAE／loss 都以 valid 為準）。
        valid = state.step < rules.N_TRANSITIONS  # [B]
        obs = observation_seat(state, sv, caps, assign)
        mask = A.tile_task_mask(state, caps)
        decision, logp, value, _, logp_h, _ = sample_decision(
            params, obs, mask, k_pol, cfg, seat=sv, fast=fast, with_components=True
        )
        nxt, own_action, new_assign = A.loop4_step(state, decision, caps, None, assign)
        opp = opp_action_fn(state, state.step, k_opp)
        action = merge_by_seat(own_action, opp, sv)
        nxt = env_mod.step(state, action, caps)

        # 凍結：無效列的 next state／assignment 原封不動（`jnp.where` 在 carry 上做，
        # 不動 `rl/env.py::step`——那支 IL／eval／NumPy parity 三條線都在用）
        def keep(a_, b_):
            return jnp.where(valid.reshape((-1,) + (1,) * (a_.ndim - 1)), a_, b_)

        nxt = jax.tree.map(keep, nxt, state)
        new_assign = jnp.where(valid[:, None, None], new_assign, assign)
        tr = Transition(
            state, assign, decision, logp, logp_h, value, nxt.step >= rules.N_TRANSITIONS, sv, valid
        )
        return (nxt, new_assign, k), tr

    zero_assign = jnp.full((b, 2, caps.max_units), A.NO_ASSIGNMENT, jnp.int32)
    (final, _, _), traj = jax.lax.scan(step, (init_state, zero_assign, key), None, length=t)
    return final, traj


def last_valid_step(valid: jax.Array) -> jax.Array:
    """每個 env 最後一個有效步的時間索引 ``[B]``（`valid` 是 ``[T, B]``）。

    `valid` 沿 T 只會由 True 變 False（step 單調遞增、到局末後凍結），所以總數 −1 就是索引。
    """
    return jnp.maximum(jnp.sum(valid.astype(jnp.int32), axis=0) - 1, 0)


def scatter_to_last_valid(reward_b: jax.Array, valid: jax.Array) -> jax.Array:
    """``[B]`` 的終局 reward → ``[T, B]``，放在**各 env 自己的真局末那一步**。

    不能沿用「放在 index T−1」：教師狀態重開局的 env 在 T−1 已經是 padding，那一列會被
    valid 遮掉，reward 直接消失。
    """
    t = valid.shape[0]
    idx = last_valid_step(valid)  # [B]
    hit = jnp.arange(t)[:, None] == idx[None, :]
    return jnp.where(hit, reward_b[None, :], 0.0)


def gae(
    values: jax.Array,
    last_value: jax.Array,
    reward: jax.Array,
    valid: jax.Array | None = None,
    done: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """GAE。``values`` 是 ``[T, B]``；``reward`` 可以是

    * ``[B]``——終局稀疏（只在最後一步給）；有 `valid` 時放在**各 env 的真局末**那一步；
    * ``[T, B]``——逐步 reward（任務 28 的 `dense_rewards`）。

    ``valid`` ``[T, B]`` 給了就做**局末截斷**：無效（padding）列的 `v_next` 與 `adv_next`
    一律當 0，所以最後一個有效步不會被 padding 的 `delta = 0 − v_frozen` 以 ``λ^k``
    滲回來（教師 day 3 開局時 padding 只有 72 步，λ^72 ≈ 0.025，足以汙染）。
    無效列自己的 adv／ret 回 0。

    ``done`` ``[T, B]``（`Transition.done`：這一步之後已到真局末）給了就**顯式截斷
    bootstrap**：該步的 `v_next` 與 `adv_next` 都當 0。不能只靠「padding 列把 carry
    歸零」——那只在 padding 真的存在、且 `last_value` 為 0 時成立；fresh env 的真末步
    落在 index T−1，carry 直接是傳進來的 `last_value`，非 0 就會 bootstrap 過終局。

    gamma = 1（Isaiah）所以沒有折扣項；λ 只控制 bootstrap 的權重。
    """
    tt = values.shape[0]
    if reward.ndim == 1:  # [B] → 只有終局那一步有值的 [T, B]
        reward = (
            jnp.zeros_like(values).at[tt - 1].set(reward)
            if valid is None
            else scatter_to_last_valid(reward, valid)
        )
    vmask = jnp.ones_like(values) if valid is None else valid.astype(values.dtype)
    cont = jnp.ones_like(values) if done is None else 1.0 - done.astype(values.dtype)

    def back(carry, i):
        adv_next, v_next = carry
        m, c = vmask[i], cont[i]
        r = reward[i] * m
        # done 的那一步不 bootstrap；無效列不傳遞、也不留自己的 adv
        delta = r + GAMMA * v_next * c * m - values[i] * m
        adv = (delta + GAMMA * LAMBDA * adv_next * c * m) * m
        return (adv, values[i] * m), adv

    _, advs = jax.lax.scan(
        back, (jnp.zeros_like(last_value), last_value), jnp.arange(tt), reverse=True
    )
    return advs, (advs + values) * vmask


def ppo_loss(params, ref_params, cfg, caps, batch, seat=None, fast=None, remat: bool = True):
    """一個 minibatch 的 loss。``batch`` 是 rollout 攤平後抽出的列（含重算 obs 要的 state）。

    座位用 ``batch["seat"]``（逐列），沒有就退回 ``seat``（int，預設 0）——rollout 存的是
    每一列自己的座位，更新期必須用同一個，否則半數的列會用對手視角重算 obs。
    """
    sv = batch.get("seat")
    if sv is None:
        sv = seat_vec(0 if seat is None else seat, batch["old_logp"].shape[0])
    obs = observation_seat(batch["state"], sv, caps, batch["assignment"])
    out = M.forward(params, obs, cfg, remat=remat, fast=fast)
    own_task = pick_seat(batch["decision"].tile_task, sv)
    active = active_masks(own_task)
    # 與 rollout 抽樣時同一個 mask 來源（`A.tile_task_mask` 只看 state，與參數無關）
    logit_mask = {"task": pick_seat(A.tile_task_mask(batch["state"], caps), sv)}
    own_dec = jax.tree.map(lambda x: jnp.stack([pick_seat(x, sv)] * 2, axis=1), batch["decision"])
    logp_h, mask_h = component_logp(out, own_dec, active, seat=0, logit_mask=logit_mask)
    logp = jnp.sum(logp_h, axis=-1)
    adv = batch["adv"]
    # 任務 25c：surrogate 逐 component clip。joint ratio 仍算出來寫 history 對照，不進 loss。
    pg, hstats = factored_surrogate(logp_h, batch["old_logp_h"], mask_h, adv)
    ratio = jnp.exp(logp - batch["old_logp"])
    v_loss = jnp.mean((out["value"] - batch["ret"]) ** 2)
    # KL 錨的參考分佈：優先用 rollout 時就算好、存在 batch 裡的 logits。
    # 在 grad_fn 裡再跑一次 IL 前向等於多一份 attention 激活（任務 23 實測 grad 峰值 12.13 GiB，
    # 預算 13.53 GiB），而錨的權重是固定的，這些 logits 本來就不必每個 epoch 重算。
    ref_out = batch.get("ref_logits")
    if ref_out is None:
        ref_out = jax.lax.stop_gradient(M.forward(ref_params, obs, cfg, fast=fast))
    else:
        ref_out = jax.tree.map(jax.lax.stop_gradient, ref_out)
    kl_row = kl_to_reference(out, ref_out, active, logit_mask=logit_mask)  # [B]
    kl = jnp.mean(kl_row)
    total = pg + 0.5 * v_loss + KL_TO_IL * kl
    # 診斷欄位（任務 25b 第 2 點）：pg 在負 advantage 那側沒有上界，所以要看得到**哪些列**
    # 的 ratio 跑掉、它們的 advantage 是不是負的。全是純量，不影響 grad。
    gt2 = ratio > 2.0
    # 任務 34：依「這一列的 env 是不是從教師狀態開局」分組回報 KL 的**和與列數**
    # （history 端再加總相除；直接回平均會讓各 micro-batch 的組大小不同而加權錯）。
    ft = batch.get("from_teacher")
    if ft is None:
        ft = jnp.zeros_like(kl_row, bool)
    ftf = ft.astype(kl_row.dtype)
    return total, {
        "kl_sum_teacher": jnp.sum(kl_row * ftf),
        "kl_n_teacher": jnp.sum(ftf),
        "kl_sum_fresh": jnp.sum(kl_row * (1.0 - ftf)),
        "kl_n_fresh": jnp.sum(1.0 - ftf),
        "pg": pg,
        "v": v_loss,
        "kl": kl,
        "ratio_median": jnp.median(ratio),
        "clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > CLIP),
        **hstats,
        "ratio_min": jnp.min(ratio),
        "ratio_max": jnp.max(ratio),
        "n_ratio_gt_2": jnp.sum(gt2),
        "n_ratio_gt_2_adv_neg": jnp.sum(gt2 & (adv < 0)),
        "old_logp_min": jnp.min(batch["old_logp"]),
        "new_logp_min": jnp.min(logp),
        "dlogp_max_abs": jnp.max(jnp.abs(logp - batch["old_logp"])),
    }
