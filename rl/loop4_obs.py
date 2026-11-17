"""Loop 4 觀測 v2：規則直接算出的前向預演特徵（內部規格文件 §2）。

設計依 內部量測紀錄 §1。四類 token，每一維都有界：

* **每格 35 維**（雙方各 100 格）：one-hot 14（kind 6／crop 5／animal 3）＋
  既有純量 4（watered／unwatered／yield／fert_until−day）＋ 動物狀態 5（fed／unfed／
  cared／fert_avail／care_bonus）＋ **前向預演 11**（距首產、距最大產量、剩餘壽命與
  未排定旗標、ongoing 剩餘產次、非 ongoing 剩餘澆水日、今天不澆會枯、按當前價的預期
  產值、動物剩餘產次、動物距下次產出、本回合已被單位佔用）＋ **進行中 1**
  （上一回合有人手正在做這格；沒有它 policy 每回合重挑目標會掉頭）。
  內部量測紀錄 §1a 寫 33，是把動物狀態那 5 維同時算進「既有」與
  「新」的算術錯誤；實際是 34，已在該檔更正。
* **每單位 18 維**、**每產品 10 維**、**全域 24 維**。

前向預演的三個閉式外推，契約逐字對 `rl/kernels.py`：

1. 最後一次 ``end_of_day`` 在 step 695（``day=28``、``nd=29``），
   所以產出事件的 ``nd`` 範圍是 **1–29**。
2. 非 ongoing 作物（WHEAT／CARROT／MELON）在 ``_daily_refresh`` 裡**不產出**；
   它們的 ``yield_units`` 只由 WATER 在 ``age ∈ [WINDOW_START, MAX_YIELD_DAY]`` 內增加。
3. ongoing 作物的產出日是 ``nd = planted_day + FIRST_YIELD + k·interval``
   （``k = 0..MAX_YIELD-1``）；動物是 ``nd = placed_day + FIRST_YIELD + k·interval``
   （``k`` 無上限，上限在持有量）。

三個閉式都以 ``kernels._daily_refresh`` 的逐日遞推為 oracle 釘在 ``tests/test_loop4_obs.py``。

**不動 Loop 3 的檔**：本模組只讀 `rl.rules`／`rl.kernels`／`rl.state`，不 import `rl.obs`。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import kernels, pricing, rules
from rl.state import SimState, unit_alive

I32 = jnp.int32
F32 = jnp.float32

OBS_SPEC_VERSION = "loop4-v2"

TILE_DIM = 35
UNIT_DIM = 18
PRODUCT_DIM = 10
GLOBAL_DIM = 24

#: 最後一個執行 ``end_of_day`` 的 step；``(695 + 1) % 24 == 0`` 且 step 719 沒有 transition。
LAST_EOD_STEP = 695
#: 因此產出事件的 ``nd``（``_daily_refresh`` 的「隔天」）最大值。
LAST_PRODUCTION_ND = LAST_EOD_STEP // rules.TURNS_PER_DAY + 1  # 29

_CROP_FIRST_YIELD = jnp.asarray(rules.CROP_FIRST_YIELD_DAY, I32)
_CROP_MAX_YIELD_DAY = jnp.asarray(rules.CROP_MAX_YIELD_DAY, I32)
_CROP_INTERVAL = jnp.asarray(rules.CROP_INTERVAL, I32)
_CROP_MAX_YIELD = jnp.asarray(rules.CROP_MAX_YIELD, I32)
_CROP_ONGOING = jnp.asarray(rules.CROP_ONGOING, I32)
_CROP_WINDOW_START = jnp.asarray(rules.CROP_WATER_WINDOW_START, I32)
_ANIMAL_FIRST_YIELD = jnp.asarray(rules.ANIMAL_FIRST_YIELD_DAY, I32)
_ANIMAL_INTERVAL = jnp.asarray(rules.ANIMAL_INTERVAL, I32)
_ANIMAL_MAX_HELD = jnp.asarray(rules.ANIMAL_MAX_HELD, I32)
_ANIMAL_PRODUCT = jnp.asarray(rules.ANIMAL_PRODUCT_ITEM, I32)
_LAND_PRICES = jnp.asarray(rules.LAND_PRICES, I32)
_SHED_ACCESS_FLAT = jnp.asarray(rules.SHED_ACCESS_FLAT, I32)


#: 一次最多賣幾單位——倉庫上限就是上限（`rl/rules.py` ``SHED_CAPACITY``）。
MAX_SALE_UNITS = rules.SHED_CAPACITY


def sale_revenue(product: jax.Array, inventory: jax.Array, k: jax.Array) -> jax.Array:
    """賣 ``k`` 單位 ``product`` 的總收入（庫存從 ``inventory`` 起算）。

    **為什麼不用前綴和表**：價格表 ``[9, 80001]`` 的整體前綴和會到 4.2e10，float32 的尾數
    只有 24 bit，解析不出兩個相鄰位置間 100 量級的差（實測 ``cum[hi] - cum[lo]`` 回 0.0）。
    賣量本來就被 ``SHED_CAPACITY = 100`` 夾住，所以直接 gather 100 個價格再遮罩求和，
    整數精確、成本固定。

    地板規則：價格到 1 之後庫存不再上升（`rl/kernels.py` 的 ``mkt_delta``），而價格表在更高
    庫存處也是 1，所以「庫存逐筆上升」與「停在地板」兩種算法的總額相同。
    """
    js = jnp.arange(MAX_SALE_UNITS, dtype=I32)
    shape = jnp.broadcast_shapes(product.shape, inventory.shape, k.shape)
    prod = jnp.broadcast_to(product, shape)[..., None]
    inv = jnp.broadcast_to(inventory, shape)[..., None]
    kk = jnp.broadcast_to(k, shape)[..., None]
    price, _ = kernels.lookup_price(
        jnp.broadcast_to(prod, shape + (MAX_SALE_UNITS,)),
        jnp.clip(inv + js, pricing.INV_MIN, pricing.INV_MAX),
    )
    return jnp.sum(jnp.where(js < kk, price, 0), axis=-1).astype(F32)


# ------------------------------------------------------------- 閉式外推
def crop_productions_left(crop: jax.Array, planted_day: jax.Array, day: jax.Array) -> jax.Array:
    """ongoing 作物在終局前還會產出幾次（假設不枯死）。非 ongoing 恆 0。

    產出日 ``nd = planted_day + FIRST_YIELD + k·interval``，``k = 0..MAX_YIELD-1``，
    只算 ``nd ∈ [day + 1, LAST_PRODUCTION_ND]``。
    """
    interval = jnp.maximum(_CROP_INTERVAL[crop], 1)
    base = planted_day + _CROP_FIRST_YIELD[crop]
    # 第一個 nd >= day + 1 的 k
    need = day + 1 - base
    k_lo = jnp.maximum(0, -(-need // interval))  # ceil division
    # 最後一個 nd <= LAST_PRODUCTION_ND 的 k
    k_hi = jnp.minimum(_CROP_MAX_YIELD[crop] - 1, (LAST_PRODUCTION_ND - base) // interval)
    n = jnp.maximum(0, k_hi - k_lo + 1)
    return jnp.where(_CROP_ONGOING[crop] == 1, n, 0)


def crop_water_days_left(crop: jax.Array, planted_day: jax.Array, day: jax.Array) -> jax.Array:
    """非 ongoing 作物還剩幾天可以靠 WATER 加產量（窗內、且在終局前）。ongoing 恆 0。

    WATER 的加成條件是 ``age = day - planted_day ∈ [WINDOW_START, MAX_YIELD_DAY]``
    （`rl/kernels.py:290-295`），澆水發生在當天的單位相位，所以從 ``day`` 本身起算。
    """
    a_lo = jnp.maximum(_CROP_WINDOW_START[crop], day - planted_day)
    a_hi = jnp.minimum(_CROP_MAX_YIELD_DAY[crop], rules.N_DAYS - 1 - planted_day)
    n = jnp.maximum(0, a_hi - a_lo + 1)
    return jnp.where(_CROP_ONGOING[crop] == 0, n, 0)


def animal_productions_left(animal: jax.Array, placed_day: jax.Array, day: jax.Array) -> jax.Array:
    """動物在終局前還會產出幾次（假設不逃跑）。產出次數無上限，上限在持有量。"""
    interval = _ANIMAL_INTERVAL[animal]
    base = placed_day + _ANIMAL_FIRST_YIELD[animal]
    need = day + 1 - base
    k_lo = jnp.maximum(0, -(-need // interval))
    k_hi = (LAST_PRODUCTION_ND - base) // interval
    return jnp.maximum(0, k_hi - k_lo + 1)


def animal_days_to_next(animal: jax.Array, placed_day: jax.Array, day: jax.Array) -> jax.Array:
    """距離下一次產出還有幾天；終局前不會再產出時回傳 ``N_DAYS``。"""
    interval = _ANIMAL_INTERVAL[animal]
    base = placed_day + _ANIMAL_FIRST_YIELD[animal]
    need = day + 1 - base
    k_lo = jnp.maximum(0, -(-need // interval))
    nd = base + k_lo * interval
    gap = nd - day
    return jnp.where(nd <= LAST_PRODUCTION_ND, gap, rules.N_DAYS)


# ------------------------------------------------------------- 每格特徵
def assignment_to_mask(assignment: jax.Array, n_tiles: int = rules.N_TILES) -> jax.Array:
    """``[B, 2, U]`` 的「每個人手在做哪一格」→ ``[B, 2, 100]`` 的「這格有人在做」。"""
    return (
        jnp.sum(
            (jnp.arange(n_tiles, dtype=I32)[None, None, None, :] == assignment[..., None]).astype(
                I32
            ),
            axis=2,
        )
        > 0
    )


def tile_features(
    state: SimState,
    caps: caps_mod.Caps,
    assignment: jax.Array | None = None,
    in_progress: jax.Array | None = None,
) -> jax.Array:
    """``[B, 2, 100, 35]``，每一維都有界。"""
    tiles = state.tiles
    b = tiles.shape[0]
    day = (state.step // rules.TURNS_PER_DAY)[:, None, None]
    kind = tiles[..., rules.C_KIND]
    crop_raw = tiles[..., rules.C_CROP]
    crop = jnp.clip(crop_raw, 0, rules.N_CROPS - 1)
    animal_raw = tiles[..., rules.C_ANIMAL]
    animal = jnp.clip(animal_raw, 0, rules.N_ANIMALS - 1)
    is_plant = kind == rules.TK_PLANT
    has_animal = animal_raw >= 0
    planted = tiles[..., rules.C_PLANTED_DAY]
    placed = tiles[..., rules.C_PLACED_DAY]
    yld = tiles[..., rules.C_YIELD]

    kind_oh = jax.nn.one_hot(kind, len(rules.TILE_KIND_NAMES), dtype=F32)
    crop_oh = jax.nn.one_hot(crop_raw, rules.N_CROPS, dtype=F32)
    animal_oh = jax.nn.one_hot(animal_raw, rules.N_ANIMALS, dtype=F32)

    age = day - planted
    days_to_first = jnp.where(is_plant, _CROP_FIRST_YIELD[crop] - age, rules.N_DAYS)
    days_to_max = jnp.where(is_plant, _CROP_MAX_YIELD_DAY[crop] - age, rules.N_DAYS)
    mls = tiles[..., rules.C_MLS]
    life_steps = jnp.where(mls >= 0, mls - state.step[:, None, None], rules.EPISODE_STEPS)
    life_unset = (mls < 0).astype(F32)
    prod_left = jnp.where(is_plant, crop_productions_left(crop, planted, day), 0)
    water_left = jnp.where(is_plant, crop_water_days_left(crop, planted, day), 0)
    dies = is_plant & (tiles[..., rules.C_WATERED] == 0) & (tiles[..., rules.C_UNWATERED] >= 1)

    prices = kernels.prices_of(state.market_inv)  # [B, 9]
    crop_price = jnp.take_along_axis(prices[:, None, :], crop[..., :1] * 0 + crop, axis=2)
    animal_price = jnp.take_along_axis(prices[:, None, :], _ANIMAL_PRODUCT[animal], axis=2)
    unit_price = jnp.where(has_animal, animal_price, crop_price)
    expected_value = yld.astype(F32) * unit_price.astype(F32)

    a_left = jnp.where(has_animal, animal_productions_left(animal, placed, day), 0)
    a_next = jnp.where(has_animal, animal_days_to_next(animal, placed, day), rules.N_DAYS)

    flat = state.unit_y * rules.BOARD_SIZE + state.unit_x  # [B, 2, U]
    alive = unit_alive(state, caps)
    occupied = (
        jnp.sum(
            (jnp.arange(rules.N_TILES, dtype=I32)[None, None, None, :] == flat[..., None])
            & alive[..., None],
            axis=2,
        )
        > 0
    )

    # 上一回合還在做這格嗎——policy 沒有記憶，這一維讓它看得到「正在進行中的工作」，
    # 否則每回合重新挑目標會讓人手掉頭（實測無記憶時 35.3% 的連續移動是直接掉頭，
    # 內部量測紀錄）。
    if in_progress is None:
        in_progress = (
            jnp.zeros(kind.shape, bool) if assignment is None else assignment_to_mask(assignment)
        )

    days = float(rules.N_DAYS)
    scalars = jnp.stack(
        [
            tiles[..., rules.C_WATERED].astype(F32),
            jnp.clip(tiles[..., rules.C_UNWATERED], 0, 3).astype(F32) / 3.0,
            yld.astype(F32) / float(max(rules.CROP_MAX_YIELD + rules.ANIMAL_MAX_HELD)),
            jnp.clip(tiles[..., rules.C_FERT_UNTIL] - day, -1, 3).astype(F32) / 3.0,
            jnp.clip(days_to_first, -days, days).astype(F32) / days,
            jnp.clip(days_to_max, -days, days).astype(F32) / days,
            jnp.clip(life_steps, 0, rules.EPISODE_STEPS).astype(F32) / rules.EPISODE_STEPS,
            life_unset,
            prod_left.astype(F32) / days,
            water_left.astype(F32) / days,
            dies.astype(F32),
            jnp.clip(jnp.log1p(jnp.maximum(expected_value, 0.0)) / np.log1p(3000.0), 0.0, 1.0),
            a_left.astype(F32) / days,
            jnp.clip(a_next, 0, days).astype(F32) / days,
            tiles[..., rules.C_FED].astype(F32),
            jnp.clip(tiles[..., rules.C_UNFED], 0, 3).astype(F32) / 3.0,
            tiles[..., rules.C_CARED].astype(F32),
            tiles[..., rules.C_FERT_AVAIL].astype(F32),
            jnp.clip(tiles[..., rules.C_CARE_BONUS], 0, 6).astype(F32) / 6.0,
            occupied.astype(F32),
            in_progress.astype(F32),
        ],
        axis=-1,
    )
    out = jnp.concatenate([kind_oh, crop_oh, animal_oh, scalars], axis=-1)
    assert out.shape == (b, 2, rules.N_TILES, TILE_DIM), out.shape
    return out


# ----------------------------------------------------------- 每單位特徵
def unit_features(state: SimState, caps: caps_mod.Caps) -> jax.Array:
    """``[B, 2, U, 18]``。"""
    u = caps.max_units
    alive = unit_alive(state, caps).astype(F32)
    flat = state.unit_y * rules.BOARD_SIZE + state.unit_x
    shed_adj = jnp.any(flat[..., None] == _SHED_ACCESS_FLAT, axis=-1).astype(F32)
    is_farmer = (jnp.arange(u, dtype=I32)[None, None, :] == 0).astype(F32)
    carried = state.inv.astype(F32) / 12.0
    out = jnp.concatenate(
        [
            (state.unit_x.astype(F32) / (rules.BOARD_SIZE - 1))[..., None],
            (state.unit_y.astype(F32) / (rules.BOARD_SIZE - 1))[..., None],
            alive[..., None],
            jnp.broadcast_to(is_farmer[..., None], alive.shape + (1,)),
            shed_adj[..., None],
            (flat.astype(F32) / (rules.N_TILES - 1))[..., None],
            jnp.clip(carried, 0.0, 1.0),
        ],
        axis=-1,
    )
    assert out.shape[-1] == UNIT_DIM, out.shape
    return out


# ---------------------------------------------------------- 每產品特徵
def product_features(state: SimState, seat: int) -> jax.Array:
    """``[B, 9, 10]``：當前價、賣光後的價與總收入、賣 1/一半/1/4 的邊際價、需求、庫存、持有。"""
    b = state.step.shape[0]
    inv = state.market_inv  # [B, 9]
    prices = kernels.prices_of(inv).astype(F32)
    held = state.shed[:, seat, : rules.N_PRODUCTS]
    prod = jnp.broadcast_to(jnp.arange(rules.N_PRODUCTS, dtype=I32)[None, :], inv.shape)

    def price_after(k):
        p, _ = kernels.lookup_price(prod, jnp.clip(inv + k, pricing.INV_MIN, pricing.INV_MAX))
        return p.astype(F32)

    rev_all = sale_revenue(prod, inv, held)
    shop = _shop_demand(state).astype(F32)
    town = (jnp.arange(rules.N_PRODUCTS, dtype=I32) < len(rules.TOWN_CENTER_ITEMS)).astype(F32)
    # 價格在極低庫存時會爆到百萬（價格表實測最大 4,266,252，TOMATO），
    # 所以用 log1p 除以表上限，不用線性除以一個猜的上限。
    price_scale = np.log1p(float(np.max(pricing.price_table())))
    norm = lambda v: jnp.clip(jnp.log1p(jnp.maximum(v, 0.0)) / price_scale, 0.0, 1.0)  # noqa: E731
    out = jnp.stack(
        [
            norm(prices),
            norm(price_after(held)),
            norm(price_after(1)),
            norm(price_after(jnp.maximum(held // 2, 0))),
            norm(price_after(jnp.maximum(held // 4, 0))),
            jnp.clip(jnp.log1p(jnp.maximum(rev_all, 0.0)) / np.log1p(50_000.0), 0.0, 1.0),
            held.astype(F32) / rules.SHED_CAPACITY,
            jnp.clip((inv - rules.MARKET_I0).astype(F32), -2000.0, 2000.0) / 2000.0,
            shop / 16.0,
            jnp.broadcast_to(town[None, :], (b, rules.N_PRODUCTS)),
        ],
        axis=-1,
    )
    assert out.shape == (b, rules.N_PRODUCTS, PRODUCT_DIM), out.shape
    return out


def _shop_demand(state: SimState) -> jax.Array:
    ok = state.shop_seq >= 0
    idx = jnp.clip(state.shop_seq, 0, rules.N_SHOPS - 1)
    counts = jnp.sum(
        jnp.where(ok[..., None], jax.nn.one_hot(idx, rules.N_SHOPS, dtype=I32), 0), axis=1
    )
    return counts @ jnp.asarray(rules.SHOP_DEMAND, I32)


# ------------------------------------------------------------- 全域特徵
def _max_hires_affordable(money: jax.Array, hires_today: jax.Array) -> jax.Array:
    """從目前的 ``hires_today`` 起算，這些錢還雇得起幾個人（fib 累積）。"""
    n = kernels._FIB_LEN
    ks = jnp.arange(n, dtype=I32)
    fib = kernels._FIB_TABLE
    cum = jnp.cumsum(fib)  # cum[i] = fib[0..i] 的和
    start = jnp.clip(hires_today, 0, n - 1)
    before = jnp.where(start > 0, cum[jnp.maximum(start - 1, 0)], 0)
    cost_of_k_more = cum[jnp.clip(start[..., None] + ks, 0, n - 1)] - before[..., None]
    return jnp.sum((cost_of_k_more <= money[..., None]).astype(I32), axis=-1) - 1


def global_features(state: SimState, seat: int) -> jax.Array:
    """``[B, 24]``，全部有界；對手側只放公開量。"""
    other = 1 - seat
    day = state.step // rules.TURNS_PER_DAY
    hour = state.step % rules.TURNS_PER_DAY
    kind = state.tiles[..., rules.C_KIND]
    empty = jnp.sum(kind == rules.TK_EMPTY, axis=2).astype(F32) / rules.N_TILES
    planted = jnp.sum(kind == rules.TK_PLANT, axis=2).astype(F32) / rules.N_TILES
    money = state.money.astype(F32)
    nq = state.n_quads
    land_cost = _LAND_PRICES[jnp.clip(nq, 0, len(rules.LAND_PRICES) - 1)].astype(F32)
    hire_cost = kernels._FIB_TABLE[jnp.clip(state.hires_today, 0, kernels._FIB_LEN - 1)].astype(F32)
    more = _max_hires_affordable(state.money, state.hires_today).astype(F32)
    shop = _shop_demand(state).astype(F32)
    money_scale = np.log1p(200_000.0)
    out = jnp.concatenate(
        [
            ((rules.N_DAYS - 1 - day) / (rules.N_DAYS - 1)).astype(F32)[:, None],
            (hour / rules.TURNS_PER_DAY).astype(F32)[:, None],
            jnp.clip(jnp.log1p(jnp.maximum(money[:, [seat, other]], 0.0)) / money_scale, 0.0, 1.0),
            (nq[:, [seat, other]].astype(F32) / len(rules.LAND_PRICES)),
            jnp.clip(jnp.log1p(land_cost[:, [seat, other]]) / np.log1p(4000.0), 0.0, 1.0),
            (nq[:, [seat, other]] < len(rules.LAND_PRICES)).astype(F32),
            empty[:, [seat, other]],
            planted[:, [seat, other]],
            jnp.clip(state.hires_today[:, [seat, other]].astype(F32) / 28.0, 0.0, 1.0),
            jnp.clip(jnp.log1p(hire_cost[:, [seat, other]]) / np.log1p(2000.0), 0.0, 1.0),
            jnp.clip(more[:, [seat, other]], 0.0, 20.0) / 20.0,
            jnp.sum(shop, axis=-1, keepdims=True) / 64.0,
            state.n_shops.astype(F32)[:, None] / rules.MAX_SHOP_INSTANCES,
            jnp.clip(state.market_trips.astype(F32)[:, None], 0.0, 200.0) / 200.0,
            jnp.clip(state.market_units.astype(F32)[:, None], 0.0, 200.0) / 200.0,
        ],
        axis=-1,
    )
    assert out.shape[-1] == GLOBAL_DIM, out.shape
    return out


def observation(
    state: SimState,
    seat: int,
    caps: caps_mod.Caps,
    assignment: jax.Array | None = None,
    in_progress: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """一座的 token 觀測。自己的格在前、對手的格在後（對手只有公開量）。

    ``assignment`` ``[B, 2, U]`` 是上一回合每個人手在做的格
    （`rl/loop4_actions.py::loop4_step` 的第三個回傳值）。
    IL 訓練沒有「人手→格」的對應，只有「上一回合哪些格被做了」，所以另外收
    ``in_progress`` ``[B, 2, 100]`` 的布林遮罩；兩個都沒有就當作沒有進行中的工作。
    """
    order = jnp.asarray([seat, 1 - seat])
    tiles = tile_features(state, caps, assignment, in_progress)[:, order]
    units = unit_features(state, caps)[:, order]
    # 對手的私有量（攜帶物）遮掉
    units = units.at[:, 1, :, 6:].set(0.0)
    return {
        "tiles": tiles,
        "units": units,
        "products": product_features(state, seat),
        "global": global_features(state, seat),
    }
