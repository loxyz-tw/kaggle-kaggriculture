"""Reward：純勝負終局 ＋ potential-based shaping（嚴格保最優）。

Phase 2 binding review 第 2 條的逐字實作，常數全部寫死、不可調：

* **終局**：win ``+1``／tie ``0``／loss ``−1``。**不用 bank margin 當終局獎勵**
  ——天梯只看勝負。
* **Shaping**：``F_t = γ·Φ(s_{t+1}) − Φ(s_t)``，
  ``Φ(s) = clip((W(me) − W(opp)) / 50_000, −2, 2)``，
  **終局狀態強制 ``Φ(s_T) = 0``**。
* ``γ = 0.999``（``0.999^719 ≈ 0.49``）、``GAE λ = 0.95``、terminal bootstrap 0。

S3″ 的唯一 reward 修改：``W`` 的非現金部分乘上**時間衰減 ``κ(t) = (T − t) / T``**
（``T = 719``），亦即 ``W_t(p) = money_p + κ(t)·A(p)``。

原因（S3′ 失效根因，同樣歸屬 審查者 的規格缺陷）：常數權重的 ``W`` 讓
「把現金換成種子抱著」變成**零成本、零風險**的最優解——``BUY_SEED`` 在 ``W`` 上零變化、
持有沒有任何損耗，而 ``PLANT`` 會因斷水變雜草、``SELL`` 幾乎必然把價格壓到基準價以下。
S3′ run 1 的最終策略因此以 **bank 中位數 0、W 中位數 4,455（其中未播種子佔 55%）** 收場。

``κ(t)`` 把「離季末還有多遠」放進估值：季初資產與現金等值（κ=1），季末資產幾乎不值錢
（``t = T−1`` 時 κ = 1/719）。於是——

* ``BUY_SEED`` 在 ``t = 0`` 仍是平價，但 ``t > 0`` 時 ``ΔW = −(1 − κ)·cost``：
  **越晚囤貨越虧**，抱著不用的資產會隨時間自然貶值。
* ``SELL`` 在成交價恰為 ``κ·V`` 時 ``ΔW = 0``；季末逼近時 ``κ·V → 0``，
  **任何價格賣出都是賺的**，變現終於有了正向誘因。

終局目標（銀行存款勝負）與 ``Φ`` 的底層量因此在時間上對齊：``κ(T) → 0`` 讓
``W_T ≈ money``，正是終局實際計分的量。

--- S3′ 的原始修改（保留說明）：``Φ`` 的底層量從 **bank-only 改成淨資產 ``W``**。

原因（S3 失效根因，歸屬 審查者 的規格缺陷，不是實作缺陷）：bank-only 的 ``Φ`` 把
「花錢投資」在局部一律看成負訊號——買種子、買動物、雇工都只讓 bank 下降，
要等好幾天後賣出才回本，而在完全對稱的 self-play 下「先動的人先落後」。
於是策略收斂到把開局 3000 元原封不動抱到季末的**不作為均衡**
（S3 run 2 最終 greedy 自我對弈 64 場全和、雙方 bank 皆 3000.0）。

``W`` 用引擎常數把「還沒變現的資產」計入，讓平價轉換在 ``Φ`` 上是**零變化**：

* ``BUY_SEED``：money ``−SEED_COST``、seeds ``+SEED_COST`` ⇒ ΔW = 0
* ``PLANT``：seeds ``−SEED_COST``、田地 ``+SEED_COST`` ⇒ ΔW = 0
* ``PLACE``：shed ``−ANIMAL_COST``、圈舍 ``+ANIMAL_COST`` ⇒ ΔW = 0
* ``HARVEST``：田地 ``−yield·V``、shed ``+yield·V`` ⇒ ΔW = 0

真正讓 ``W`` 變動的只有：**產出成長**（``yield_units`` 上升）、
**死亡／逃逸**（作物變雜草、動物逃走，整格估值歸零）、
**市場價差**（成交價偏離 I0 基準價）、以及城鎮消耗造成的價格移動。

``W`` 的估值常數全部取自引擎（``CROP_SEED_COST``／``ANIMAL_COST``／
``price_table`` 在 ``I0`` 的基準價），**沒有任何 replay-derived 參數**，
也沒有任何可調旋鈕。structure／land／hire／fertilizer flag／care bonus 一律估值 0
（保守：只認引擎定價得出的資產，不替沒有市價的東西編價）。

為什麼這樣不會改變最優政策（Ng et al. 1999）：shaping 項在一條軌跡上 telescoping，

``Σ_t γ^t (γ·Φ(s_{t+1}) − Φ(s_t)) = γ^T·Φ(s_T) − Φ(s_0)``

強制 ``Φ(s_T) = 0``，而雙方同起始局面使 ``Φ(s_0) = 0``，因此**折扣回報恆等於未加
shaping 的純勝負回報**。這個恆等式**與 Φ 的定義無關**——換成 ``W`` 之後依然成立，
`tests/test_rl_s1_core.py` 以隨機軌跡的 property test 守住它，它是本模組唯一的
正確性定義。

Φ 只用得到公開欄位與自己的私有欄位：對手 money／tiles 是公開的
（``obs["farms"][p]``），對手的 shed／seeds／inv 不可見。訓練時走的是模擬器的
full state，兩邊都算得出來；**部署時不需要 Φ**（只有訓練用得到 shaping）。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rl import pricing, rules
from rl.state import SimState

# --- 寫死的常數（binding review 第 2 條）；不得在未經 審查者 核准下調整 ---
GAMMA = 0.999
GAE_LAMBDA = 0.95
POTENTIAL_SCALE = 50_000.0
POTENTIAL_CLIP = 2.0
WIN_REWARD = 1.0
TIE_REWARD = 0.0
LOSS_REWARD = -1.0

F32 = jnp.float32


# --------------------------------------------------------------------------
# 估值常數：全部由引擎導出，無旋鈕、無 replay 參數
# --------------------------------------------------------------------------


def _engine_base_prices() -> tuple[float, ...]:
    """9 種產品在 ``I0``（庫存 10,000）的引擎基準價。"""
    table = pricing.price_table()
    col = rules.MARKET_I0 - pricing.INV_MIN
    return tuple(float(table[i, col]) for i in range(rules.N_PRODUCTS))


#: ``V[i]``，12 項：產品用 I0 基準價、動物用 ``ANIMAL_COST``。
ITEM_VALUE = _engine_base_prices() + tuple(float(c) for c in rules.ANIMAL_COST)

_ITEM_VALUE = jnp.asarray(ITEM_VALUE, F32)  # [12]
_SEED_VALUE = jnp.asarray(rules.CROP_SEED_COST, F32)  # [5]
_ANIMAL_VALUE = jnp.asarray(rules.ANIMAL_COST, F32)  # [3]
#: 每種作物的產出品估值——作物的產出就是自己（``CROPS == ITEMS[:5]``）。
_CROP_PRODUCT_VALUE = jnp.asarray([ITEM_VALUE[i] for i in range(rules.N_CROPS)], F32)
#: 每種動物的產出品估值（EGG／MILK／WOOL）。
_ANIMAL_PRODUCT_VALUE = jnp.asarray([ITEM_VALUE[i] for i in rules.ANIMAL_PRODUCT_ITEM], F32)

#: :func:`wealth_components` 的欄位順序，供報表與測試引用。
WEALTH_COMPONENTS = (
    "money",  # 現金
    "seeds",  # 未播的種子
    "plant_stock",  # 田裡作物的種子成本（死亡即歸零）
    "plant_yield",  # 田裡待收的產出
    "animal_stock",  # 圈舍裡的動物本身（逃逸即歸零）
    "animal_yield",  # 動物身上待收的產出
    "held",  # shed ＋ 各單位身上的物品
)


def wealth_components(state: SimState) -> dict[str, jax.Array]:
    """``W`` 的七個分項，各為 ``[B, 2]``，**皆未折現**。

    ``money`` 以外的六項即 ``A(p)``；:func:`wealth` 會把它們乘上 ``κ(t)`` 再加上
    ``money``，所以**這裡的七項直接相加不等於 ``W_t``**（等於 κ=1 的情況）。

    分項存在的目的是讓 Φ 的變化能被歸因（平價轉換／生長／死亡／市場），
    而不是只看到一個合計數字在動。
    """
    tiles = state.tiles  # [B, 2, 100, C]
    kind = tiles[..., rules.C_KIND]
    yld = tiles[..., rules.C_YIELD].astype(F32)

    # --- 作物 ---
    is_plant = kind == rules.TK_PLANT
    crop = jnp.clip(tiles[..., rules.C_CROP], 0, rules.N_CROPS - 1)
    plant_stock = jnp.sum(jnp.where(is_plant, _SEED_VALUE[crop], 0.0), axis=-1)
    plant_yield = jnp.sum(jnp.where(is_plant, yld * _CROP_PRODUCT_VALUE[crop], 0.0), axis=-1)

    # --- 動物（C_ANIMAL 為 -1 表示圈舍是空的；空圈舍估值 0）---
    animal = tiles[..., rules.C_ANIMAL]
    has_animal = animal >= 0
    species = jnp.clip(animal, 0, rules.N_ANIMALS - 1)
    animal_stock = jnp.sum(jnp.where(has_animal, _ANIMAL_VALUE[species], 0.0), axis=-1)
    animal_yield = jnp.sum(
        jnp.where(has_animal, yld * _ANIMAL_PRODUCT_VALUE[species], 0.0), axis=-1
    )

    # --- 手上與倉庫的物品 ---
    held_units = state.shed.astype(F32) + jnp.sum(state.inv.astype(F32), axis=2)  # [B, 2, 12]
    held = jnp.sum(held_units * _ITEM_VALUE, axis=-1)

    return {
        "money": state.money.astype(F32),
        "seeds": jnp.sum(state.seeds.astype(F32) * _SEED_VALUE, axis=-1),
        "plant_stock": plant_stock,
        "plant_yield": plant_yield,
        "animal_stock": animal_stock,
        "animal_yield": animal_yield,
        "held": held,
    }


#: ``A(p)`` 的欄位——``W`` 裡會被 ``κ(t)`` 折現的非現金部分。
ASSET_COMPONENTS = tuple(n for n in WEALTH_COMPONENTS if n != "money")


def kappa(step: jax.Array) -> jax.Array:
    """``κ(t) = (T − t) / T``，``[B]``；``T = 719``（一局的 agent transition 數）。

    ``t = 0`` 時 κ = 1（資產與現金等值）、``t = T−1`` 時 κ = 1/719
    （資產幾乎不值錢，只有現金算數）。這條線性衰減是 S3″ 的唯一 reward 變更。
    """
    t = step.astype(F32)
    return (F32(rules.N_TRANSITIONS) - t) / F32(rules.N_TRANSITIONS)


def assets(state: SimState) -> jax.Array:
    """``A(p)``，``[B, 2]``：``W`` 的非現金分項合計（未折現）。"""
    parts = wealth_components(state)
    return sum(parts[name] for name in ASSET_COMPONENTS)


def wealth(state: SimState) -> jax.Array:
    """``W_t(p) = money_p + κ(t)·A(p)``，``[B, 2]``。

    平價轉換（買種子／播種／放置動物／收成）在 ``t = 0`` 是零變化；``t > 0`` 時
    「現金 → 資產」會虧掉 ``(1 − κ)`` 的比例，「資產 → 現金」則對應地有賺頭。
    這正是 S3″ 要修掉的東西：常數權重的 ``W`` 讓囤貨零成本
    （S3′ 終局 bank 中位數 0、未播種子佔淨資產 55%）。
    """
    parts = wealth_components(state)
    a = sum(parts[name] for name in ASSET_COMPONENTS)
    return parts["money"] + kappa(state.step)[:, None] * a


#: ``Φ_abs`` 的固定參照點：引擎常數 ``startingMoney``。**不是可調旋鈕。**
W_REF = float(rules.DEFAULT_CONFIG["startingMoney"])  # 3_000


def absolute_potential(value: jax.Array, done: jax.Array) -> jax.Array:
    """``Φ_abs(s, p) = clip((W_t(p) − W_REF) / 50_000, −2, 2)``，``[B, 2]``。

    S3A 的唯一變更：**參照點**從對手座位換成引擎常數 3,000。
    ``W``／``κ``／``POTENTIAL_SCALE``／``POTENTIAL_CLIP`` 全部不動。

    為什麼要換（S3″ 量測到的根因）：相對式 ``Φ`` 的 shaping 是
    ``Δ(W_me) − Δ(W_opp)``——自身進步訊號疊加**等量級的對手雜訊**，而純 self-play
    共用參數使該差值期望為 0。S3″ 實測末段 W 差只有 110／−59（W 量級 2,400–3,600），
    ``|Φ|`` 平均 **1.23e-03**，只有 clip 量程 2.0 的 **0.06%**。
    換成固定參照後 shaping 變成**純粹的 ``ΔW_me``**，對手雜訊項整個消失，
    ``|Φ| ≈ |W − 3000| / 50000 ≈ 0.01–0.14``（10–100×），且隨成功放大。

    **逐座位獨立**：本函式只吃自己座位的 ``W_t``，公式裡沒有 ``−W_opp`` 項，
    因此「被另一座位對稱抵消」在結構上不可能發生——抵消需要那一項存在。
    兩座位同時變好時各自的 ``F`` 各自為正，不會互相歸零。

    ``done`` 為真的狀態強制回 0：這是 telescoping 的前提，也擋掉
    ``γ^T·(W_T − 3000)/50k`` 這個會違反「終局只認勝負」的 margin 項。
    """
    w = value.astype(F32)
    phi = jnp.clip((w - W_REF) / POTENTIAL_SCALE, -POTENTIAL_CLIP, POTENTIAL_CLIP)
    return jnp.where(done[:, None], 0.0, phi)


def state_absolute_potential(state: SimState, done: jax.Array | None = None) -> jax.Array:
    """便利包裝：直接從 :class:`~rl.state.SimState` 取 ``Φ_abs``。"""
    if done is None:
        from rl.env import is_done

        done = is_done(state)
    return absolute_potential(wealth(state), done)


def potential(value: jax.Array, done: jax.Array) -> jax.Array:
    """``Φ(s)``，``[B, 2]``。``done`` 為真的狀態強制回 0（telescoping 的前提）。

    ``value`` 是 ``[B, 2]`` 的雙方淨資產 ``W``（見 :func:`wealth`）；``done`` 是 ``[B]``。
    函式本身只認「我方 vs 對手」的對稱結構，換底層量不影響 telescoping。

    .. deprecated::
       這是 S3／S3′／S3″ 用的**相對式** Φ，三次都失敗（見
       `內部規格文件` §7）。S3A 起訓練走
       :func:`absolute_potential`；本函式保留供既有 property test 與歷史對照，
       **不得**在未經 審查者 裁定下重新接回訓練迴圈。
    """
    mine = value.astype(F32)
    opp = mine[:, ::-1]
    phi = jnp.clip((mine - opp) / POTENTIAL_SCALE, -POTENTIAL_CLIP, POTENTIAL_CLIP)
    return jnp.where(done[:, None], 0.0, phi)


def terminal_reward(money: jax.Array) -> jax.Array:
    """``[B, 2]`` 純勝負：多者 +1、少者 −1、相等 0。"""
    mine = money.astype(F32)
    opp = mine[:, ::-1]
    return jnp.where(mine > opp, WIN_REWARD, jnp.where(mine < opp, LOSS_REWARD, TIE_REWARD))


def shaped_reward(
    phi_t: jax.Array,
    phi_tp1: jax.Array,
    money_tp1: jax.Array,
    done_tp1: jax.Array,
    gamma: float = GAMMA,
) -> jax.Array:
    """``r'_t = R_t + γ·Φ(s_{t+1}) − Φ(s_t)``，``[B, 2]``。

    ``R_t`` 只有在 ``s_{t+1}`` 是終局時才非零（純勝負）。``phi_tp1`` 應該已經由
    :func:`potential` 在 done 時歸零；這裡再乘一次 ``~done`` 是為了讓函式在被單獨
    呼叫時也不會破壞 telescoping。
    """
    base = jnp.where(done_tp1[:, None], terminal_reward(money_tp1), 0.0)
    phi_next = jnp.where(done_tp1[:, None], 0.0, phi_tp1)
    return base + gamma * phi_next - phi_t


def state_potential(state: SimState, done: jax.Array | None = None) -> jax.Array:
    """便利包裝：直接從 :class:`~rl.state.SimState` 取 ``Φ``。"""
    if done is None:
        from rl.env import is_done

        done = is_done(state)
    return potential(wealth(state), done)


def discounted_return(rewards: jax.Array, gamma: float = GAMMA) -> jax.Array:
    """``Σ_t γ^t r_t``，對前導軸（時間）折扣。供 property test 與診斷使用。"""
    t = jnp.arange(rewards.shape[0], dtype=rewards.dtype)
    disc = gamma ** t.reshape((-1,) + (1,) * (rewards.ndim - 1))
    return jnp.sum(disc * rewards, axis=0)
