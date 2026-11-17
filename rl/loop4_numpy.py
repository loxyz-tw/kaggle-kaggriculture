"""Loop 4 部署路徑的**純 NumPy** 版（任務 20）：只 import ``numpy``、標準庫與 jax-free 的
``rl.rules``／``rl.caps``，不碰 jax／jaxlib。

為什麼要有這一份
----------------
任務 19 量到：延遲不是問題（單核 p99 8.5 ms），**包大小才是**——jax＋jaxlib 的 CPU 路徑
安裝後 371 MiB，提交包上限 100 MiB。Loop 3 走的是同一招（`rl/numpy_agent.py`），
但那一份對的是 v2 的觀測／head，Loop 4 的觀測、模型、解碼、走路執行器、市場桶全是新的。

對照的是哪一條
--------------
逐行對 `scripts/loop4_eval.py::make_policy_step` 的 **greedy** 路徑，也就是任務 19 量的那條::

    官方 obs dict → scripts/loop4_parity.py::_state_from_obs（canonical=True）
      → rl/loop4_obs.py::observation
      → rl/loop4_model.py::forward
      → top-k 解碼（1 − P(NOOP) 取前 k 格）
      → rl/loop4_actions.py::loop4_step 的單位相位 ＋ 走路執行器 ＋ 市場桶整數
      → scripts/loop4_parity.py::action_to_engine_json

紀律（沿 `rl/numpy_agent.py`）
-----------------------------
* 常數一律 **import** 自 `rl.rules`／`rl.caps`（這兩個模組是 jax-free 的），不手抄——
  手抄會漂移，`rl/numpy_agent.py` 是因為來源檔 import jax 才不得不抄。
* **市價用引擎的閉式公式在 float64 重算**，不帶 80,001×9 的價格表進包。
  公式逐字抄自 `docs/official/kaggriculture-1.32.7/kaggriculture.py.txt:192-206`；
  `scripts/loop4_numpy_parity.py` 會拿它對 `rl/pricing.py` 的整張表逐格核對。
* 整數運算（桶、執行器、單位相位）與 JAX 版**逐位相同**；浮點只出現在觀測與 forward，
  判準沿 `rl/export_numpy.py` 的 ``LOGIT_ATOL``／``LOGIT_RTOL``／``TIE_EPS``。
* 排序一律 ``kind="stable"``：JAX 的 ``argsort`` 預設穩定，NumPy 預設不是；
  top-k 選格與 DROP 的丟棄順序都靠這個平手規則。

單座、無 batch
--------------
部署只有自己這一座、batch 1。狀態把**自己放索引 0**（`_state_from_obs(canonical=True)`
的做法），所以決策一律讀索引 0；對手那一側只保留公開量，供觀測的「對手的格」用。
觀測輸出保留 ``B=1`` 的前導軸，形狀與 JAX 版一致，parity 才好逐張比。
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from rl import caps as caps_mod
from rl import rules

I32 = np.int32
F32 = np.float32

# ------------------------------------------------------------------ 觀測維度
# 與 rl/loop4_obs.py 同值；那個模組 import jax，所以這裡重列並由 parity 測試核對。
TILE_DIM = 35
UNIT_DIM = 18
PRODUCT_DIM = 10
GLOBAL_DIM = 24
LAST_EOD_STEP = 695
LAST_PRODUCTION_ND = LAST_EOD_STEP // rules.TURNS_PER_DAY + 1  # 29
MAX_SALE_UNITS = rules.SHED_CAPACITY

# ------------------------------------------------------------------ 動作空間
# 與 rl/loop4_actions.py 同值。
T_NOOP = 0
T_PLANT0 = 1
T_WATER = 6
T_HARVEST = 7
T_FERTILIZE = 8
T_FEED = 9
T_CARE = 10
T_COLLECT = 11
T_DIG = 12
T_BUILD_COOP = 13
T_BUILD_PASTURE = 14
T_PICKUP = 15
T_PLACE = 16
N_TASKS = 17

TASK_TO_OP = np.asarray(
    [
        rules.OP_PASS,
        *([rules.OP_PLANT] * rules.N_CROPS),
        rules.OP_WATER,
        rules.OP_HARVEST,
        rules.OP_FERTILIZE,
        rules.OP_FEED,
        rules.OP_CARE,
        rules.OP_COLLECT_FERTILIZER,
        rules.OP_DIG,
        rules.OP_BUILD_COOP,
        rules.OP_BUILD_PASTURE,
        rules.OP_PICKUP,
        rules.OP_PLACE,
    ],
    I32,
)

QTY_BUCKETS = (1, 2, 4, -1)
N_QTY_BUCKETS = 4
N_SELL = 10
N_BUY_SEED = 8
N_BUY_PRODUCT = 9
N_BUY_ANIMAL = 4
N_HIRE = 11
N_BUY_LAND = 2
BUYABLE_PRODUCTS = (rules.I_WHEAT, rules.I_FERTILIZER)
SWITCH_COST = 4
NO_ASSIGNMENT = -1

_CROP_FIRST_YIELD = np.asarray(rules.CROP_FIRST_YIELD_DAY, I32)
_CROP_MAX_YIELD_DAY = np.asarray(rules.CROP_MAX_YIELD_DAY, I32)
_CROP_INTERVAL = np.asarray(rules.CROP_INTERVAL, I32)
_CROP_MAX_YIELD = np.asarray(rules.CROP_MAX_YIELD, I32)
_CROP_ONGOING = np.asarray(rules.CROP_ONGOING, I32)
_CROP_WINDOW_START = np.asarray(rules.CROP_WATER_WINDOW_START, I32)
_ANIMAL_FIRST_YIELD = np.asarray(rules.ANIMAL_FIRST_YIELD_DAY, I32)
_ANIMAL_INTERVAL = np.asarray(rules.ANIMAL_INTERVAL, I32)
_ANIMAL_PRODUCT = np.asarray(rules.ANIMAL_PRODUCT_ITEM, I32)
_ANIMAL_STRUCT = np.asarray(rules.ANIMAL_STRUCTURE_KIND, I32)
_LAND_PRICES = np.asarray(rules.LAND_PRICES, I32)
_SHED_ACCESS_FLAT = np.asarray(rules.SHED_ACCESS_FLAT, I32)
_SHOP_DEMAND = np.asarray(rules.SHOP_DEMAND, I32)
_MOVE_DX = np.asarray(rules.MOVE_DX, I32)
_MOVE_DY = np.asarray(rules.MOVE_DY, I32)
_FIB_LEN = 46
_FIB_TABLE = np.asarray([min(rules.fib(n), (1 << 31) - 1) for n in range(_FIB_LEN)], I32)

_EMPTY_ROW = np.zeros(rules.N_TILE_CHANNELS, I32)
_EMPTY_ROW[rules.C_CROP] = -1
_EMPTY_ROW[rules.C_ANIMAL] = -1


def _kind_row(kind: int) -> np.ndarray:
    row = _EMPTY_ROW.copy()
    row[rules.C_KIND] = kind
    return row


_WEED_ROW = _kind_row(rules.TK_WEED)
_COOP_ROW = _kind_row(rules.TK_COOP)
_PASTURE_ROW = _kind_row(rules.TK_PASTURE)

# ------------------------------------------------------------------- 市價
# 逐字抄自 docs/official/kaggriculture-1.32.7/kaggriculture.py.txt:39-74, 192-206。
PRICE_FLOOR = 1
HINGE_GAIN = 8.0
INV_MIN = -20_000
INV_MAX = 60_000
MARKET_PARAMS = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}


def _shape(func: str, x: np.ndarray, t: float) -> np.ndarray:
    """``kaggriculture._shape`` 的向量化版；全程 float64。"""
    x = np.maximum(0.0, np.asarray(x, np.float64))
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return np.sqrt(x)
    if func == "log":
        return np.log(1.0 + x)
    if func == "log10":
        return np.log10(1.0 + x)
    if func == "hinge":
        if not t or t <= 0:
            return x
        u = x / t
        return u + HINGE_GAIN * np.maximum(0.0, u - 1.0) ** 2
    return x


def _price_one(pi: int, inventory: np.ndarray) -> np.ndarray:
    base, t, below_f, below_tgt, above_f, above_tgt = MARKET_PARAMS[rules.PRODUCTS[pi]]
    inv = np.asarray(inventory, np.float64)
    i0 = float(rules.MARKET_I0)
    below_amp = below_tgt * base / _shape(below_f, np.float64(t), t)
    above_amp = above_tgt * base / _shape(above_f, np.float64(t), t)
    price = np.where(
        inv < i0,
        base + below_amp * _shape(below_f, i0 - inv, t),
        base - above_amp * _shape(above_f, inv - i0, t),
    )
    # `int(round(x))`：Python 與 np.rint 都是 half-to-even，語意相同。
    return np.maximum(PRICE_FLOOR, np.rint(price).astype(np.int64)).astype(I32)


def price_at(product: np.ndarray, inventory: np.ndarray) -> np.ndarray:
    """``kernels.lookup_price`` 的等價：窗外先夾再算，與表的定義域一致。"""
    product = np.asarray(product, I32)
    inv = np.clip(np.asarray(inventory, np.int64), INV_MIN, INV_MAX)
    out = np.zeros(np.broadcast(product, inv).shape, I32)
    prod_b, inv_b = np.broadcast_arrays(product, inv)
    for pi in range(rules.N_PRODUCTS):
        sel = prod_b == pi
        if np.any(sel):
            out[sel] = _price_one(pi, inv_b[sel])
    return out


def prices_of(market_inv: np.ndarray) -> np.ndarray:
    """``[9]`` 目前報價。"""
    return price_at(np.arange(rules.N_PRODUCTS, dtype=I32), market_inv)


def sale_revenue(product: np.ndarray, inventory: np.ndarray, k: np.ndarray) -> np.ndarray:
    """賣 ``k`` 單位的總收入；與 `rl/loop4_obs.py::sale_revenue` 同一個算法。"""
    js = np.arange(MAX_SALE_UNITS, dtype=I32)
    prod, inv, kk = np.broadcast_arrays(
        np.asarray(product)[..., None], np.asarray(inventory)[..., None], np.asarray(k)[..., None]
    )
    price = price_at(
        np.broadcast_to(prod, prod.shape[:-1] + (MAX_SALE_UNITS,)),
        np.clip(inv + js, INV_MIN, INV_MAX),
    )
    return np.sum(np.where(js < kk, price, 0), axis=-1).astype(F32)


# ------------------------------------------------------------------- 狀態
class NpState:
    """`rl/state.py::SimState` 的單 batch NumPy 版；欄位名與型別逐項對齊。

    自己一律在索引 0（`_state_from_obs(canonical=True)` 的做法）。對手那一側只有公開量
    （格、單位座標、錢、hires、解鎖象限），私有欄位留 0——我們讀不到，也不讀。
    """

    __slots__ = (
        "step",
        "money",
        "tiles",
        "unit_x",
        "unit_y",
        "hires_today",
        "n_quads",
        "shed",
        "seeds",
        "inv",
        "inv_seq",
        "inv_ctr",
        "market_inv",
        "shop_seq",
        "n_shops",
    )

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def copy(self) -> NpState:
        return NpState(
            **{
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in ((k, getattr(self, k)) for k in self.__slots__)
            }
        )


def encode_tile(tile: Any) -> np.ndarray:
    """`rl/state.py::encode_tile` 的複本（那個模組 import jax）。"""
    row = _EMPTY_ROW.copy()
    if tile is None:
        return row
    if tile == "LOCKED":
        row[rules.C_KIND] = rules.TK_LOCKED
        return row
    kind = tile.get("kind")
    if kind == "WEED":
        row[rules.C_KIND] = rules.TK_WEED
        return row
    if kind == "PLANT":
        row[rules.C_KIND] = rules.TK_PLANT
        row[rules.C_CROP] = rules.ITEM_INDEX[tile["crop"]]
        row[rules.C_PLANTED_DAY] = tile["planted_day"]
        row[rules.C_WATERED] = int(bool(tile["watered_today"]))
        row[rules.C_UNWATERED] = tile["consecutive_unwatered"]
        row[rules.C_YIELD] = tile["yield_units"]
        row[rules.C_MLS] = tile["max_lifespan_step"]
        row[rules.C_FERT_UNTIL] = tile.get("fertilized_until_day", -1)
        return row
    row[rules.C_KIND] = rules.TK_COOP if kind == "COOP" else rules.TK_PASTURE
    if "animal" in tile:
        row[rules.C_ANIMAL] = rules.ITEM_INDEX[tile["animal"]] - rules.ANIMAL_ITEM_BASE
        row[rules.C_PLACED_DAY] = tile["placed_day"]
        row[rules.C_YIELD] = tile["yield_units"]
        row[rules.C_UNFED] = tile["consecutive_unfed"]
        row[rules.C_FED] = int(bool(tile["fed_today"]))
        row[rules.C_CARED] = int(bool(tile["cared_today"]))
        row[rules.C_FERT_AVAIL] = int(bool(tile["fertilizer_available"]))
        row[rules.C_CARE_BONUS] = tile.get("care_bonus", 0)
    return row


def state_from_obs(obs: dict, seat: int, caps: caps_mod.Caps) -> NpState:
    """`scripts/loop4_parity.py::_state_from_obs(canonical=True)` 的 NumPy 等價。"""
    farms = obs["farms"]
    u = caps.max_units
    tiles = np.zeros((2, rules.N_TILES, rules.N_TILE_CHANNELS), I32)
    unit_x = np.full((2, u), rules.DEFAULT_SPAWN[0], I32)
    unit_y = np.full((2, u), rules.DEFAULT_SPAWN[1], I32)
    money = np.zeros(2, I32)
    hires = np.zeros(2, I32)
    n_quads = np.zeros(2, I32)
    shed = np.zeros((2, rules.N_ITEMS), I32)
    seeds = np.zeros((2, rules.N_CROPS), I32)
    inv = np.zeros((2, u, rules.N_ITEMS), I32)
    inv_seq = np.zeros_like(inv)
    inv_ctr = np.ones((2, u), I32)
    for p, src in enumerate((seat, 1 - seat)):
        farm = farms[src]
        for flat in range(rules.N_TILES):
            y, x = divmod(flat, rules.BOARD_SIZE)
            tiles[p, flat] = encode_tile(farm["tiles"][y][x])
        money[p] = int(farm["money"])
        hires[p] = farm["hires_today"]
        n_quads[p] = len(farm["unlocked_quadrants"]) - 1
        unit_x[p, 0], unit_y[p, 0] = farm["farmer"][0], farm["farmer"][1]
        for h, pos in enumerate(farm["hands"]):
            if h + 1 < u:
                unit_x[p, h + 1], unit_y[p, h + 1] = pos[0], pos[1]
    private = obs["private"]
    for item, n in private["shed"].items():
        shed[0, rules.ITEM_INDEX[item]] = n
    for crop, n in private["seeds"].items():
        seeds[0, rules.ITEM_INDEX[crop]] = n
    for ui, unit_inv in enumerate(private["inventories"]):
        if ui >= u:
            break
        for rank, (item, n) in enumerate(unit_inv.items(), start=1):
            if n > 0:
                inv[0, ui, rules.ITEM_INDEX[item]] = n
                inv_seq[0, ui, rules.ITEM_INDEX[item]] = rank
        inv_ctr[0, ui] = len([1 for n in unit_inv.values() if n > 0]) + 1
    shops = obs.get("town", {}).get("shops", []) or []
    shop_seq = np.full(rules.MAX_SHOP_INSTANCES, -1, I32)
    for i, name in enumerate(shops[: rules.MAX_SHOP_INSTANCES]):
        shop_seq[i] = rules.SHOP_NAMES.index(name)
    return NpState(
        step=int(obs["step"]) if "step" in obs else 0,
        money=money,
        tiles=tiles,
        unit_x=unit_x,
        unit_y=unit_y,
        hires_today=hires,
        n_quads=n_quads,
        shed=shed,
        seeds=seeds,
        inv=inv,
        inv_seq=inv_seq,
        inv_ctr=inv_ctr,
        market_inv=np.asarray([obs["market"]["inventory"][p] for p in rules.PRODUCTS], I32),
        shop_seq=shop_seq,
        n_shops=len(shops),
    )


def unit_alive(state: NpState, caps: caps_mod.Caps) -> np.ndarray:
    idx = np.arange(caps.max_units, dtype=I32)
    live = np.minimum(state.hires_today, caps.max_units - 1)
    return idx[None, :] <= live[:, None]


# ------------------------------------------------------------- 閉式外推
def crop_productions_left(crop, planted_day, day):
    interval = np.maximum(_CROP_INTERVAL[crop], 1)
    base = planted_day + _CROP_FIRST_YIELD[crop]
    need = day + 1 - base
    k_lo = np.maximum(0, -(-need // interval))
    k_hi = np.minimum(_CROP_MAX_YIELD[crop] - 1, (LAST_PRODUCTION_ND - base) // interval)
    n = np.maximum(0, k_hi - k_lo + 1)
    return np.where(_CROP_ONGOING[crop] == 1, n, 0)


def crop_water_days_left(crop, planted_day, day):
    a_lo = np.maximum(_CROP_WINDOW_START[crop], day - planted_day)
    a_hi = np.minimum(_CROP_MAX_YIELD_DAY[crop], rules.N_DAYS - 1 - planted_day)
    n = np.maximum(0, a_hi - a_lo + 1)
    return np.where(_CROP_ONGOING[crop] == 0, n, 0)


def animal_productions_left(animal, placed_day, day):
    interval = _ANIMAL_INTERVAL[animal]
    base = placed_day + _ANIMAL_FIRST_YIELD[animal]
    need = day + 1 - base
    k_lo = np.maximum(0, -(-need // interval))
    k_hi = (LAST_PRODUCTION_ND - base) // interval
    return np.maximum(0, k_hi - k_lo + 1)


def animal_days_to_next(animal, placed_day, day):
    interval = _ANIMAL_INTERVAL[animal]
    base = placed_day + _ANIMAL_FIRST_YIELD[animal]
    need = day + 1 - base
    k_lo = np.maximum(0, -(-need // interval))
    nd = base + k_lo * interval
    return np.where(nd <= LAST_PRODUCTION_ND, nd - day, rules.N_DAYS)


def _one_hot(idx: np.ndarray, n: int, dtype=F32) -> np.ndarray:
    """`jax.nn.one_hot` 的等價：負索引整列為 0（JAX 用 ``x == axis_index`` 做的）。"""
    return (np.asarray(idx)[..., None] == np.arange(n, dtype=I32)).astype(dtype)


# --------------------------------------------------------------- 觀測
#: `rl/loop4_obs.py` 用 ``np.max(pricing.price_table())`` 當價格的正規化上限。
#: 價格對庫存單調遞減，所以整張表的最大值就是每個產品在 ``INV_MIN`` 的報價取最大；
#: `scripts/loop4_numpy_parity.py` 會拿它對真表核對，不靠這句註解。
PRICE_TABLE_MAX = int(max(int(_price_one(p, np.asarray(INV_MIN))) for p in range(rules.N_PRODUCTS)))
_PRICE_SCALE = float(np.log1p(float(PRICE_TABLE_MAX)))


def _shop_demand(state: NpState) -> np.ndarray:
    ok = state.shop_seq >= 0
    idx = np.clip(state.shop_seq, 0, rules.N_SHOPS - 1)
    counts = np.sum(np.where(ok[..., None], _one_hot(idx, rules.N_SHOPS, I32), 0), axis=0)
    return counts @ _SHOP_DEMAND


def tile_features(
    state: NpState, caps: caps_mod.Caps, in_progress: np.ndarray | None = None
) -> np.ndarray:
    """``[2, 100, 35]``，逐項對 `rl/loop4_obs.py::tile_features`。"""
    tiles = state.tiles
    day = state.step // rules.TURNS_PER_DAY
    kind = tiles[..., rules.C_KIND]
    crop_raw = tiles[..., rules.C_CROP]
    crop = np.clip(crop_raw, 0, rules.N_CROPS - 1)
    animal_raw = tiles[..., rules.C_ANIMAL]
    animal = np.clip(animal_raw, 0, rules.N_ANIMALS - 1)
    is_plant = kind == rules.TK_PLANT
    has_animal = animal_raw >= 0
    planted = tiles[..., rules.C_PLANTED_DAY]
    placed = tiles[..., rules.C_PLACED_DAY]
    yld = tiles[..., rules.C_YIELD]

    kind_oh = _one_hot(kind, len(rules.TILE_KIND_NAMES))
    crop_oh = _one_hot(crop_raw, rules.N_CROPS)
    animal_oh = _one_hot(animal_raw, rules.N_ANIMALS)

    age = day - planted
    days_to_first = np.where(is_plant, _CROP_FIRST_YIELD[crop] - age, rules.N_DAYS)
    days_to_max = np.where(is_plant, _CROP_MAX_YIELD_DAY[crop] - age, rules.N_DAYS)
    mls = tiles[..., rules.C_MLS]
    life_steps = np.where(mls >= 0, mls - state.step, rules.EPISODE_STEPS)
    life_unset = (mls < 0).astype(F32)
    prod_left = np.where(is_plant, crop_productions_left(crop, planted, day), 0)
    water_left = np.where(is_plant, crop_water_days_left(crop, planted, day), 0)
    dies = is_plant & (tiles[..., rules.C_WATERED] == 0) & (tiles[..., rules.C_UNWATERED] >= 1)

    prices = prices_of(state.market_inv)  # [9]
    unit_price = np.where(has_animal, prices[_ANIMAL_PRODUCT[animal]], prices[crop])
    expected_value = yld.astype(F32) * unit_price.astype(F32)

    a_left = np.where(has_animal, animal_productions_left(animal, placed, day), 0)
    a_next = np.where(has_animal, animal_days_to_next(animal, placed, day), rules.N_DAYS)

    flat = state.unit_y * rules.BOARD_SIZE + state.unit_x  # [2, U]
    alive = unit_alive(state, caps)
    occupied = (
        np.sum(
            (np.arange(rules.N_TILES, dtype=I32)[None, None, :] == flat[..., None])
            & alive[..., None],
            axis=1,
        )
        > 0
    )
    if in_progress is None:
        in_progress = np.zeros(kind.shape, bool)

    days = float(rules.N_DAYS)
    yield_scale = float(max(rules.CROP_MAX_YIELD + rules.ANIMAL_MAX_HELD))
    scalars = np.stack(
        [
            tiles[..., rules.C_WATERED].astype(F32),
            np.clip(tiles[..., rules.C_UNWATERED], 0, 3).astype(F32) / 3.0,
            yld.astype(F32) / yield_scale,
            np.clip(tiles[..., rules.C_FERT_UNTIL] - day, -1, 3).astype(F32) / 3.0,
            np.clip(days_to_first, -days, days).astype(F32) / days,
            np.clip(days_to_max, -days, days).astype(F32) / days,
            np.clip(life_steps, 0, rules.EPISODE_STEPS).astype(F32) / float(rules.EPISODE_STEPS),
            life_unset,
            prod_left.astype(F32) / days,
            water_left.astype(F32) / days,
            dies.astype(F32),
            np.clip(
                np.log1p(np.maximum(expected_value, F32(0.0))) / F32(np.log1p(3000.0)), 0.0, 1.0
            ),
            a_left.astype(F32) / days,
            np.clip(a_next, 0, days).astype(F32) / days,
            tiles[..., rules.C_FED].astype(F32),
            np.clip(tiles[..., rules.C_UNFED], 0, 3).astype(F32) / 3.0,
            tiles[..., rules.C_CARED].astype(F32),
            tiles[..., rules.C_FERT_AVAIL].astype(F32),
            np.clip(tiles[..., rules.C_CARE_BONUS], 0, 6).astype(F32) / 6.0,
            occupied.astype(F32),
            in_progress.astype(F32),
        ],
        axis=-1,
    ).astype(F32)
    return np.concatenate([kind_oh, crop_oh, animal_oh, scalars], axis=-1).astype(F32)


def unit_features(state: NpState, caps: caps_mod.Caps) -> np.ndarray:
    """``[2, U, 18]``。"""
    u = caps.max_units
    alive = unit_alive(state, caps).astype(F32)
    flat = state.unit_y * rules.BOARD_SIZE + state.unit_x
    shed_adj = np.any(flat[..., None] == _SHED_ACCESS_FLAT, axis=-1).astype(F32)
    is_farmer = (np.arange(u, dtype=I32)[None, :] == 0).astype(F32)
    carried = state.inv.astype(F32) / F32(12.0)
    return np.concatenate(
        [
            (state.unit_x.astype(F32) / F32(rules.BOARD_SIZE - 1))[..., None],
            (state.unit_y.astype(F32) / F32(rules.BOARD_SIZE - 1))[..., None],
            alive[..., None],
            np.broadcast_to(is_farmer[..., None], alive.shape + (1,)),
            shed_adj[..., None],
            (flat.astype(F32) / F32(rules.N_TILES - 1))[..., None],
            np.clip(carried, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(F32)


def product_features(state: NpState, seat: int) -> np.ndarray:
    """``[9, 10]``。"""
    inv = state.market_inv
    prices = prices_of(inv).astype(F32)
    held = state.shed[seat, : rules.N_PRODUCTS]
    prod = np.arange(rules.N_PRODUCTS, dtype=I32)

    def price_after(k):
        return price_at(prod, np.clip(inv + k, INV_MIN, INV_MAX)).astype(F32)

    rev_all = sale_revenue(prod, inv, held)
    shop = _shop_demand(state).astype(F32)
    town = (np.arange(rules.N_PRODUCTS, dtype=I32) < len(rules.TOWN_CENTER_ITEMS)).astype(F32)
    scale = F32(_PRICE_SCALE)

    def norm(v):
        return np.clip(np.log1p(np.maximum(v, F32(0.0))) / scale, 0.0, 1.0)

    return np.stack(
        [
            norm(prices),
            norm(price_after(held)),
            norm(price_after(1)),
            norm(price_after(np.maximum(held // 2, 0))),
            norm(price_after(np.maximum(held // 4, 0))),
            np.clip(np.log1p(np.maximum(rev_all, F32(0.0))) / F32(np.log1p(50_000.0)), 0.0, 1.0),
            held.astype(F32) / F32(rules.SHED_CAPACITY),
            np.clip((inv - rules.MARKET_I0).astype(F32), -2000.0, 2000.0) / F32(2000.0),
            shop / F32(16.0),
            town,
        ],
        axis=-1,
    ).astype(F32)


def _max_hires_affordable(money: np.ndarray, hires_today: np.ndarray) -> np.ndarray:
    n = _FIB_LEN
    ks = np.arange(n, dtype=I32)
    cum = np.cumsum(_FIB_TABLE, dtype=I32)
    start = np.clip(hires_today, 0, n - 1)
    before = np.where(start > 0, cum[np.maximum(start - 1, 0)], 0)
    cost_of_k_more = cum[np.clip(start[..., None] + ks, 0, n - 1)] - before[..., None]
    return np.sum((cost_of_k_more <= money[..., None]).astype(I32), axis=-1) - 1


def global_features(state: NpState, seat: int) -> np.ndarray:
    """``[24]``。"""
    other = 1 - seat
    pair = [seat, other]
    day = state.step // rules.TURNS_PER_DAY
    hour = state.step % rules.TURNS_PER_DAY
    kind = state.tiles[..., rules.C_KIND]
    empty = np.sum(kind == rules.TK_EMPTY, axis=1).astype(F32) / F32(rules.N_TILES)
    planted = np.sum(kind == rules.TK_PLANT, axis=1).astype(F32) / F32(rules.N_TILES)
    money = state.money.astype(F32)
    nq = state.n_quads
    land_cost = _LAND_PRICES[np.clip(nq, 0, len(rules.LAND_PRICES) - 1)].astype(F32)
    hire_cost = _FIB_TABLE[np.clip(state.hires_today, 0, _FIB_LEN - 1)].astype(F32)
    more = _max_hires_affordable(state.money, state.hires_today).astype(F32)
    shop = _shop_demand(state).astype(F32)
    money_scale = F32(np.log1p(200_000.0))
    return np.concatenate(
        [
            np.asarray([(rules.N_DAYS - 1 - day) / (rules.N_DAYS - 1)], F32),
            np.asarray([hour / rules.TURNS_PER_DAY], F32),
            np.clip(np.log1p(np.maximum(money[pair], F32(0.0))) / money_scale, 0.0, 1.0),
            nq[pair].astype(F32) / F32(len(rules.LAND_PRICES)),
            np.clip(np.log1p(land_cost[pair]) / F32(np.log1p(4000.0)), 0.0, 1.0),
            (nq[pair] < len(rules.LAND_PRICES)).astype(F32),
            empty[pair],
            planted[pair],
            np.clip(state.hires_today[pair].astype(F32) / F32(28.0), 0.0, 1.0),
            np.clip(np.log1p(hire_cost[pair]) / F32(np.log1p(2000.0)), 0.0, 1.0),
            np.clip(more[pair], 0.0, 20.0) / F32(20.0),
            np.asarray([np.sum(shop) / 64.0], F32),
            np.asarray([state.n_shops / rules.MAX_SHOP_INSTANCES], F32),
            np.zeros(1, F32),  # market_trips：obs 重建的 state 一律 0（同 _state_from_obs）
            np.zeros(1, F32),  # market_units：同上
        ]
    ).astype(F32)


def observation(
    state: NpState, caps: caps_mod.Caps, in_progress: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """一座的 token 觀測，形狀含 ``B=1`` 的前導軸，與 `rl/loop4_obs.py::observation` 一致。

    狀態已經是 canonical（自己在 0），所以 `observation` 裡的 ``order`` 重排是恆等。
    """
    tiles = tile_features(state, caps, in_progress)[None]
    units = unit_features(state, caps)[None].copy()
    units[:, 1, :, 6:] = 0.0  # 對手的攜帶物遮掉
    return {
        "tiles": tiles,
        "units": units,
        "products": product_features(state, 0)[None],
        "global": global_features(state, 0)[None],
    }


# --------------------------------------------------------------- 模型
def _ln(x, g, b, eps=1e-5):
    mu = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    var = np.mean(np.square(x - mu), axis=-1, keepdims=True, dtype=np.float32)
    return ((x - mu) / np.sqrt(var + F32(eps)) * g + b).astype(F32)


def _gelu(x):
    """``jax.nn.gelu`` 的預設是 **approximate=True**（tanh 近似），不是 erf 版。"""
    c = F32(math.sqrt(2.0 / math.pi))
    # x*x*x 而非 x**3：float32 整數次方走一般 pow 慢路徑（部署 profile：占每步 44%）；兩者差 ≤1 ulp
    return (x * F32(0.5) * (F32(1.0) + np.tanh(c * (x + F32(0.044715) * (x * x * x))))).astype(F32)


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(F32)


def _log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return (shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))).astype(F32)


def build_tokens(params, obs, cfg):
    """`rl/loop4_model.py::build_tokens` 的等價。"""
    tiles = obs["tiles"]
    b = tiles.shape[0]
    d = cfg["d_model"]
    n_tile_tokens = 2 * rules.N_TILES
    tile_tok = tiles @ params["in_tile_w"] + params["in_tile_b"]
    tile_tok = tile_tok + params["tile_pos_emb"][None, None, :, :]
    tile_tok = tile_tok + np.stack([params["seg_emb"][0], params["seg_emb"][1]])[None, :, None, :]
    tile_tok = tile_tok.reshape(b, n_tile_tokens, d)

    units = obs["units"][:, 0]
    unit_tok = units @ params["in_unit_w"] + params["in_unit_b"] + params["seg_emb"][2]
    prod_tok = obs["products"] @ params["in_product_w"] + params["in_product_b"]
    prod_tok = prod_tok + params["seg_emb"][3]
    glob_tok = obs["global"] @ params["in_global_w"] + params["in_global_b"]
    glob_tok = (glob_tok + params["seg_emb"][4])[:, None, :]

    x = np.concatenate([tile_tok, unit_tok, prod_tok, glob_tok], axis=1).astype(F32)
    alive = obs["units"][:, 0, :, 2] > 0.5
    mask = np.concatenate(
        [
            np.ones((b, n_tile_tokens), bool),
            alive,
            np.ones((b, rules.N_PRODUCTS + 1), bool),
        ],
        axis=1,
    )
    return x, mask


def _block(x, p, i, n_heads, mask):
    d = x.shape[-1]
    h = _ln(x, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
    qkv = h @ p[f"l{i}_qkv_w"]
    q, k, v = np.split(qkv, 3, axis=-1)
    b, t, _ = q.shape
    dh = d // n_heads
    shape = (b, t, n_heads, dh)
    q = q.reshape(shape).transpose(0, 2, 1, 3)
    k = k.reshape(shape).transpose(0, 2, 1, 3)
    v = v.reshape(shape).transpose(0, 2, 1, 3)
    # np.matmul 而非 np.einsum：兩運算元 einsum 不走 BLAS（部署 profile：占每步 31%）；只改累加順序
    att = np.matmul(q, k.transpose(0, 1, 3, 2)) / F32(np.sqrt(dh))
    att = np.where(mask[:, None, None, :], att, F32(-1e30))
    att = _softmax(att, axis=-1)
    out = np.matmul(att, v).transpose(0, 2, 1, 3).reshape(b, t, d)
    x = x + out @ p[f"l{i}_proj_w"]
    h = _ln(x, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
    h = _gelu(h @ p[f"l{i}_ff1_w"] + p[f"l{i}_ff1_b"])
    return (x + h @ p[f"l{i}_ff2_w"] + p[f"l{i}_ff2_b"]).astype(F32)


def forward(params, obs, cfg) -> dict[str, np.ndarray]:
    """`rl/loop4_model.py::forward` 的等價；層數／寬度／head 數一律讀 ``cfg``。"""
    x, mask = build_tokens(params, obs, cfg)
    for i in range(cfg["n_layers"]):
        x = _block(x, params, i, cfg["n_heads"], mask)
    x = _ln(x, params["ln_f_g"], params["ln_f_b"])
    n_tile_tokens = 2 * rules.N_TILES
    own = x[:, : rules.N_TILES]
    mu = cfg["max_units"]
    prod = x[:, n_tile_tokens + mu : n_tile_tokens + mu + 9]
    glob = x[:, -1]
    farm = glob @ params["farm_w"] + params["farm_b"]
    o = 0

    def take(n):
        nonlocal o
        out = farm[:, o : o + n]
        o += n
        return out

    return {
        "task": own @ params["task_w"] + params["task_b"],
        "tile_qty": own @ params["tqty_w"] + params["tqty_b"],
        "sell": prod @ params["sell_w"] + params["sell_b"],
        "buy_seed": take(rules.N_CROPS * N_BUY_SEED).reshape(-1, rules.N_CROPS, N_BUY_SEED),
        "buy_product": take(len(BUYABLE_PRODUCTS) * N_BUY_PRODUCT).reshape(
            -1, len(BUYABLE_PRODUCTS), N_BUY_PRODUCT
        ),
        "buy_animal": take(rules.N_ANIMALS * N_BUY_ANIMAL).reshape(
            -1, rules.N_ANIMALS, N_BUY_ANIMAL
        ),
        "hire": take(N_HIRE),
        "buy_land": take(N_BUY_LAND),
        "value": (glob @ params["value_w"] + params["value_b"])[:, 0],
    }


# ------------------------------------------------------------ 每格遮罩
def tile_task_mask(state: NpState, caps: caps_mod.Caps, seat: int = 0) -> np.ndarray:
    """``[100, 17]``，逐項對 `rl/loop4_actions.py::tile_task_mask`（只做一座）。"""
    del caps
    tiles = state.tiles[seat]
    kind = tiles[:, rules.C_KIND]
    crop = np.clip(tiles[:, rules.C_CROP], 0, rules.N_CROPS - 1)
    animal = tiles[:, rules.C_ANIMAL]
    has_animal = animal >= 0
    yld = tiles[:, rules.C_YIELD]
    day = state.step // rules.TURNS_PER_DAY
    empty = kind == rules.TK_EMPTY
    is_plant = kind == rules.TK_PLANT
    mature = (day - tiles[:, rules.C_PLANTED_DAY]) >= _CROP_FIRST_YIELD[crop]

    seeds = state.seeds[seat][None, :]
    shed = state.shed[seat][None, :]
    inv_total = np.sum(state.inv[seat], axis=0)[None, :]
    held = shed + inv_total
    shed_total = int(np.sum(state.shed[seat]))
    flat = np.arange(rules.N_TILES, dtype=I32)
    shed_adj = np.any(flat[:, None] == _SHED_ACCESS_FLAT, axis=-1)

    struct_match = kind[:, None] == _ANIMAL_STRUCT[None, :]
    animal_held = inv_total[:, rules.ANIMAL_ITEM_BASE :] > 0
    place_animal = np.any(struct_match & (~has_animal)[:, None] & animal_held, axis=-1)
    place_shed = shed_adj & (np.sum(inv_total) > 0) & (shed_total < rules.SHED_CAPACITY)

    m = np.zeros((rules.N_TILES, N_TASKS), bool)
    m[:, T_NOOP] = True
    m[:, T_PLANT0 : T_PLANT0 + rules.N_CROPS] = empty[:, None] & (seeds > 0)
    m[:, T_WATER] = is_plant & (tiles[:, rules.C_WATERED] == 0)
    m[:, T_HARVEST] = (yld > 0) & ((is_plant & mature) | has_animal)
    m[:, T_FERTILIZE] = is_plant & (held[:, rules.I_FERTILIZER] > 0)
    m[:, T_FEED] = has_animal & (tiles[:, rules.C_FED] == 0) & (held[:, rules.I_WHEAT] > 0)
    m[:, T_CARE] = has_animal & (tiles[:, rules.C_CARED] == 0)
    m[:, T_COLLECT] = has_animal & (tiles[:, rules.C_FERT_AVAIL] == 1)
    m[:, T_DIG] = (kind != rules.TK_EMPTY) & (kind != rules.TK_LOCKED) & (~has_animal)
    m[:, T_BUILD_COOP] = empty
    m[:, T_BUILD_PASTURE] = empty
    m[:, T_PICKUP] = shed_adj & (shed_total > 0)
    m[:, T_PLACE] = place_animal | place_shed
    return m


# ------------------------------------------------------------- 解碼
class Decision:
    """`rl/loop4_actions.py::Loop4Decision` 的單座 NumPy 版。"""

    __slots__ = (
        "tile_task",
        "tile_qty",
        "sell",
        "buy_seed",
        "buy_product",
        "buy_animal",
        "hire",
        "buy_land",
    )

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw[k])


def _categorical(rng, logits: np.ndarray) -> np.ndarray:
    """從 logits 抽一個類別（逆 CDF），形狀 ``[..., K]`` → ``[...]``。

    對應 JAX 的 ``jax.random.categorical``，但 **RNG 流不可能一致**：JAX 用 threefry
    counter-based PRNG、我們用 numpy 的 PCG64。取樣模式下兩個 backend 不會逐局相同，
    只能比分佈（任務 22 的口徑）。
    """
    p = _softmax(np.asarray(logits, F32), axis=-1)
    u = rng.random(p.shape[:-1])[..., None]
    idx = np.sum(np.cumsum(p, axis=-1) < u, axis=-1)
    return np.minimum(idx, p.shape[-1] - 1).astype(I32)


def decode(out: dict, mask: np.ndarray, n_alive: int, rng=None) -> Decision:
    """`scripts/loop4_eval.py::make_policy_step` 的 **greedy ＋ top-k** 路徑。

    top-k：取「最不像 NOOP」的前 k 格（k = 活著的人手數），其餘保持 NOOP；被選中的格取
    「非 NOOP 類裡最大的那一個」。

    排序鍵是 **logp_NOOP 升冪**（這裡取負號後降冪，同一件事），與 `loop4_eval.py` 同步改。
    舊鍵 ``log1p(-exp(logp_NOOP))`` 在 P(NOOP)→1 時會把 logits 上 1e-7 的 float32 差放大約
    1e5 倍，任務 20 實測因此翻轉平手的格（`reports/opus_task20/PROGRESS.md`）。

    **排序用 stable**——JAX 的 ``argsort`` 預設穩定，NumPy 預設是 quicksort，
    不指定的話平手時選到的格會不一樣。

    ``rng`` 給了就走**取樣**（其餘 head 從 categorical 抽，對應 `loop4_eval.py` 的
    ``greedy=False``）；``None`` 就是 greedy（argmax）。**每格任務一律走 top-k**，
    與 JAX 端 ``topk=True`` 時一致——那條路本來就不抽樣。
    抽樣順序固定為 tile_qty → sell → buy_seed → buy_product → buy_animal → hire → buy_land，
    對上 JAX 那邊 ``k[1]``…``k[7]`` 的順序。
    """
    task_logits = out["task"][0]  # [100, 17]
    task_lg = np.where(mask, task_logits, F32(-np.inf))
    logp = _log_softmax(task_lg, axis=-1)
    want = -logp[:, T_NOOP]
    want = np.where(np.any(mask[:, 1:], axis=-1), want, -np.inf)
    order = np.argsort(-want, axis=-1, kind="stable")
    rank = np.argsort(order, axis=-1, kind="stable")
    chosen = rank < n_alive
    non_noop = np.where(mask[:, 1:], task_lg[:, 1:], F32(-np.inf))
    best_non_noop = np.argmax(non_noop, axis=-1).astype(I32) + 1
    tile_task = np.where(chosen, best_non_noop, T_NOOP).astype(I32)

    def pick(name):
        lg = out[name][0]
        if rng is None:
            return np.argmax(lg, axis=-1).astype(I32)
        return _categorical(rng, lg)

    return Decision(
        tile_task=tile_task,
        tile_qty=pick("tile_qty"),
        sell=pick("sell"),
        buy_seed=pick("buy_seed"),
        buy_product=pick("buy_product"),
        buy_animal=pick("buy_animal"),
        hire=int(pick("hire")),
        buy_land=int(pick("buy_land")),
    )


# --------------------------------------------------------- 走路執行器
def _carry_item(state: NpState, ui: int, task: int, qbucket: int, seat: int = 0):
    """`rl/loop4_actions.py::_carry_item` 的等價。"""
    inv_k = state.inv[seat, ui]
    seq_k = state.inv_seq[seat, ui]
    if task == T_PICKUP:
        item = int(np.argmax(state.shed[seat]))
        have = int(state.shed[seat, item])
    else:
        item = int(np.argmin(np.where(inv_k > 0, seq_k, 1 << 29)))
        have = int(inv_k[item])
    raw = QTY_BUCKETS[int(np.clip(qbucket, 0, N_QTY_BUCKETS - 1))]
    n = have if raw < 0 else min(raw, max(have, 1))
    return item, max(n, 1)


def assign_and_walk(
    state: NpState,
    tile_task: np.ndarray,
    tile_qty: np.ndarray,
    caps: caps_mod.Caps,
    prev_assignment: np.ndarray | None = None,
    switch_cost: int = SWITCH_COST,
    seat: int = 0,
):
    """`rl/loop4_actions.py::assign_and_walk` 的等價（只做一座）。"""
    u = caps.max_units
    if prev_assignment is None:
        prev_assignment = np.full(u, NO_ASSIGNMENT, I32)
    alive = unit_alive(state, caps)[seat]
    want = tile_task != T_NOOP
    flats = np.arange(rules.N_TILES, dtype=I32)
    ys, xs = np.divmod(flats, rules.BOARD_SIZE)
    big = np.int32(1 << 29)
    units = np.arange(u, dtype=I32)

    ux_all = state.unit_x[seat]
    uy_all = state.unit_y[seat]
    dist = np.abs(xs[None, :] - ux_all[:, None]) + np.abs(ys[None, :] - uy_all[:, None])
    stay = flats[None, :] == prev_assignment[:, None]
    eff = dist + np.where(stay, 0, switch_cost)
    order_key = (
        eff * (u * rules.N_TILES) + units[:, None] * rules.N_TILES + flats[None, :]
    ).astype(I32)

    free_unit = alive.copy()
    free_tile = np.ones(rules.N_TILES, bool)
    op = np.zeros(u, I32)
    arg = np.full(u, -1, I32)
    qty = np.ones(u, I32)
    assign = np.full(u, NO_ASSIGNMENT, I32)

    for _ in range(u):
        ok = free_unit[:, None] & (free_tile & want)[None, :]
        key = np.where(ok, order_key, big).reshape(-1)
        best = int(np.argmin(key))
        if key[best] >= big:
            break
        ui, ti = best // rules.N_TILES, best % rules.N_TILES
        free_unit[ui] = False
        free_tile[ti] = False
        ux, uy = int(ux_all[ui]), int(uy_all[ui])
        tgt_x, tgt_y = ti % rules.BOARD_SIZE, ti // rules.BOARD_SIZE
        task = int(tile_task[ti])
        qbucket = int(tile_qty[ti])
        on_tile = ux == tgt_x and uy == tgt_y
        if tgt_x > ux:
            move_op = rules.OP_EAST
        elif tgt_x < ux:
            move_op = rules.OP_WEST
        elif tgt_y > uy:
            move_op = rules.OP_SOUTH
        else:
            move_op = rules.OP_NORTH
        op[ui] = TASK_TO_OP[task] if on_tile else move_op
        item, n = _carry_item(state, ui, task, qbucket, seat)
        if T_PLANT0 <= task < T_PLANT0 + rules.N_CROPS:
            a = task - T_PLANT0
        elif task >= T_PICKUP:
            a = item
        else:
            a = -1
        arg[ui] = a if on_tile else -1
        qty[ui] = n if (on_tile and task >= T_PICKUP) else 1
        assign[ui] = ti
    return op, arg, qty, assign


# ------------------------------------------------------- 單位相位（引擎）
def apply_unit_actions(
    state: NpState,
    unit_op: np.ndarray,
    unit_arg: np.ndarray,
    unit_qty: np.ndarray,
    caps: caps_mod.Caps,
    seat: int = 0,
) -> NpState:
    """`rl/kernels.py::apply_unit_actions` 的等價（只做一座）。

    為什麼部署路徑需要它：SELL 的「全部／一半／1/4」桶要用**單位相位之後**的 shed 算
    （`rl/loop4_actions.py::market_orders` 的契約），所以動作要送出去之前，得先在本地把
    這一座的單位相位跑一遍。另一座這回合全是 PASS（policy 只對自己出任務），不受影響。
    """
    st = state.copy()
    u = caps.max_units
    day = st.step // rules.TURNS_PER_DAY
    live = int(min(st.hires_today[seat], u - 1))

    op = unit_op.astype(I32).copy()
    arg = unit_arg.astype(I32)
    qty = unit_qty.astype(I32)

    # PLANT 的原子驗證：某作物的 PLANT 請求總數超過種子就全部變 PASS
    crop_ok = (arg >= 0) & (arg < rules.N_CROPS)
    is_plant_op = op == rules.OP_PLANT
    crop_idx = np.clip(arg, 0, rules.N_CROPS - 1)
    demand = np.sum(_one_hot(crop_idx, rules.N_CROPS, I32) * (is_plant_op & crop_ok)[:, None], 0)
    blocked = demand > st.seeds[seat]
    op = np.where(is_plant_op & crop_ok & blocked[crop_idx], rules.OP_PASS, op)

    for k in range(live + 1):
        opk, argk, qtyk = int(op[k]), int(arg[k]), int(qty[k])
        x, y = int(st.unit_x[seat, k]), int(st.unit_y[seat, k])
        inv_k = st.inv[seat, k].copy()
        seq_k = st.inv_seq[seat, k].copy()
        ctr_k = int(st.inv_ctr[seat, k])

        flat = y * rules.BOARD_SIZE + x
        t = st.tiles[seat, flat].copy()
        kind = int(t[rules.C_KIND])
        crop = int(np.clip(t[rules.C_CROP], 0, rules.N_CROPS - 1))
        animal = int(t[rules.C_ANIMAL])
        anim_i = int(np.clip(animal, 0, rules.N_ANIMALS - 1))
        yield_u = int(t[rules.C_YIELD])

        # ---- 移動（可以走上 LOCKED；出界是 no-op）
        if rules.OP_NORTH <= opk <= rules.OP_WEST:
            nx, ny = x + int(_MOVE_DX[opk]), y + int(_MOVE_DY[opk])
            if 0 <= nx < rules.BOARD_SIZE and 0 <= ny < rules.BOARD_SIZE:
                st.unit_x[seat, k], st.unit_y[seat, k] = nx, ny

        shed_adj = bool(np.any(flat == _SHED_ACCESS_FLAT))
        item_ok = 0 <= argk < rules.N_ITEMS
        item_i = int(np.clip(argk, 0, rules.N_ITEMS - 1))
        inv_at = int(inv_k[item_i])
        shed_at = int(st.shed[seat, item_i])
        shed_total = int(np.sum(st.shed[seat]))
        room = max(0, rules.SHED_CAPACITY - shed_total)

        d_inv = np.zeros_like(inv_k)
        d_shed = np.zeros_like(st.shed[seat])

        # ---- PICKUP（在 LOCKED 守衛之前結算，照抄引擎）
        n_pick = min(qtyk, shed_at)
        if opk == rules.OP_PICKUP and shed_adj and item_ok and qtyk > 0 and n_pick > 0:
            d_inv[item_i] += n_pick
            d_shed[item_i] -= n_pick

        # ---- PLACE：動物放進空結構，否則丟進倉庫
        is_anim_item = rules.ANIMAL_ITEM_BASE <= argk < rules.N_ITEMS
        a_local = int(np.clip(argk - rules.ANIMAL_ITEM_BASE, 0, rules.N_ANIMALS - 1))
        c_place_a = (
            opk == rules.OP_PLACE
            and is_anim_item
            and kind == int(_ANIMAL_STRUCT[a_local])
            and animal < 0
        )
        c_place_a_ok = c_place_a and inv_at >= 1
        if c_place_a_ok:
            d_inv[item_i] -= 1
        n_drop1 = min(qtyk, inv_at, room)
        if (
            opk == rules.OP_PLACE
            and not c_place_a
            and shed_adj
            and item_ok
            and qtyk > 0
            and n_drop1 > 0
        ):
            d_inv[item_i] -= n_drop1
            d_shed[item_i] += n_drop1

        # ---- DROP：整身家倒進倉庫，依 dict 插入序
        if opk == rules.OP_DROP and shed_adj:
            sort_key = np.where(inv_k > 0, seq_k, 1 << 30)
            order = np.argsort(sort_key, kind="stable")
            n_sorted = np.where(sort_key[order] >= (1 << 30), 0, inv_k[order])
            excl = np.cumsum(n_sorted) - n_sorted
            take_sorted = np.clip(
                np.minimum(n_sorted, rules.SHED_CAPACITY - shed_total - excl), 0, None
            )
            take = np.empty_like(take_sorted)
            take[order] = take_sorted
            d_shed += take
            d_inv -= inv_k

        # ---- 格上的動作（都要求非 LOCKED）
        not_locked = kind != rules.TK_LOCKED
        empty = kind == rules.TK_EMPTY
        is_plant_tile = kind == rules.TK_PLANT
        has_animal = animal >= 0

        crop_idx_k = int(np.clip(argk, 0, rules.N_CROPS - 1))
        c_plant = (
            opk == rules.OP_PLANT
            and empty
            and 0 <= argk < rules.N_CROPS
            and int(st.seeds[seat, crop_idx_k]) > 0
        )
        c_water = opk == rules.OP_WATER and is_plant_tile and int(t[rules.C_WATERED]) == 0
        age = day - int(t[rules.C_PLANTED_DAY])
        in_window = int(_CROP_WINDOW_START[crop]) <= age <= int(_CROP_MAX_YIELD_DAY[crop])
        bonus = 2 if int(t[rules.C_FERT_UNTIL]) >= day else 1
        watered_yield = min(int(_CROP_MAX_YIELD[crop]), yield_u + bonus)
        c_water_bonus = c_water and int(_CROP_ONGOING[crop]) == 0 and in_window

        c_harv = opk == rules.OP_HARVEST and kind >= rules.TK_WEED and yield_u > 0
        mature = (day - int(t[rules.C_PLANTED_DAY])) >= int(_CROP_FIRST_YIELD[crop])
        c_harv_p = c_harv and is_plant_tile and mature
        c_harv_a = c_harv and has_animal
        if c_harv_p:
            d_inv[crop] += yield_u
        if c_harv_a:
            d_inv[int(_ANIMAL_PRODUCT[anim_i])] += yield_u
        c_harv_clear = c_harv_p and int(_CROP_ONGOING[crop]) == 0

        c_fert = opk == rules.OP_FERTILIZE and is_plant_tile and int(inv_k[rules.I_FERTILIZER]) >= 1
        if c_fert:
            d_inv[rules.I_FERTILIZER] -= 1

        c_dig = opk == rules.OP_DIG and not_locked and not empty and not has_animal
        c_coop = opk == rules.OP_BUILD_COOP and empty
        c_past = opk == rules.OP_BUILD_PASTURE and empty

        c_feed = (
            opk == rules.OP_FEED
            and has_animal
            and int(t[rules.C_FED]) == 0
            and int(inv_k[rules.I_WHEAT]) >= 1
        )
        if c_feed:
            d_inv[rules.I_WHEAT] -= 1
        c_coll = (
            opk == rules.OP_COLLECT_FERTILIZER and has_animal and int(t[rules.C_FERT_AVAIL]) == 1
        )
        if c_coll:
            d_inv[rules.I_FERTILIZER] += 1
        c_care = opk == rules.OP_CARE and has_animal and int(t[rules.C_CARED]) == 0

        # ---- 組新的格
        nt = t
        if c_water:
            nt[rules.C_WATERED] = 1
        if c_water_bonus:
            nt[rules.C_YIELD] = watered_yield
        if c_harv_p or c_harv_a:
            nt[rules.C_YIELD] = 0
        if c_fert:
            nt[rules.C_FERT_UNTIL] = max(int(nt[rules.C_FERT_UNTIL]), day + 2)
        if c_feed:
            nt[rules.C_FED] = 1
        if c_care:
            nt[rules.C_CARED] = 1
        if c_coll:
            nt[rules.C_FERT_AVAIL] = 0
        if c_place_a_ok:
            nt[rules.C_ANIMAL] = a_local
            nt[rules.C_PLACED_DAY] = day
            for ch in (
                rules.C_YIELD,
                rules.C_UNFED,
                rules.C_FED,
                rules.C_CARED,
                rules.C_FERT_AVAIL,
                rules.C_CARE_BONUS,
            ):
                nt[ch] = 0
        if c_plant:
            ongoing_new = int(_CROP_ONGOING[crop_idx_k])
            nt = _EMPTY_ROW.copy()
            nt[rules.C_KIND] = rules.TK_PLANT
            nt[rules.C_CROP] = crop_idx_k
            nt[rules.C_PLANTED_DAY] = day
            nt[rules.C_UNWATERED] = 1
            nt[rules.C_YIELD] = 0 if ongoing_new == 1 else 1
            nt[rules.C_MLS] = (
                -1
                if ongoing_new == 1
                else (day + int(_CROP_MAX_YIELD_DAY[crop_idx_k]) + 1) * rules.TURNS_PER_DAY
            )
            nt[rules.C_FERT_UNTIL] = -1
            st.seeds[seat, crop_idx_k] -= 1
        if c_dig or c_harv_clear:
            nt = _EMPTY_ROW.copy()
        if c_coop:
            nt = _COOP_ROW.copy()
        if c_past:
            nt = _PASTURE_ROW.copy()
        st.tiles[seat, flat] = nt

        new_inv_k = inv_k + d_inv
        newly = (inv_k == 0) & (new_inv_k > 0)
        gone = (inv_k > 0) & (new_inv_k == 0)
        seq_k = np.where(newly, ctr_k, np.where(gone, 0, seq_k))
        st.inv[seat, k] = new_inv_k
        st.inv_seq[seat, k] = seq_k
        st.inv_ctr[seat, k] = ctr_k + int(np.sum(newly))
        st.shed[seat] = st.shed[seat] + d_shed
    return st


# --------------------------------------------------------------- 市場桶
def _sell_quantity(bucket: int, held: int) -> int:
    table_abs = (0, 0, 0, 0, 1, 2, 3, 4, 6, 8)
    if bucket == 0:
        return 0
    if bucket == 1:
        n = held
    elif bucket == 2:
        n = held // 2
    elif bucket == 3:
        n = held // 4
    else:
        n = table_abs[min(max(bucket, 0), 9)]
    return min(max(n, 0), held)


def _buy_seed_quantity(bucket: int, n_empty: int) -> int:
    table = (0, 1, 2, 3, 4, 5, 10, 0)
    if bucket == 0:
        return 0
    n = n_empty if bucket == 7 else table[min(max(bucket, 0), 7)]
    return max(n, 0)


def _buy_product_quantity(bucket: int, feed_all: int) -> int:
    table = (0, 1, 2, 3, 4, 5, 10, 15, 0)
    if bucket == 0:
        return 0
    n = feed_all if bucket == 8 else table[min(max(bucket, 0), 8)]
    return max(n, 0)


def _buy_animal_quantity(bucket: int, fill: int) -> int:
    table = (0, 1, 2, 0)
    if bucket == 0:
        return 0
    n = fill if bucket == 3 else table[min(max(bucket, 0), 3)]
    return max(n, 0)


def market_counts(state: NpState, seat: int = 0) -> dict[str, Any]:
    kind = state.tiles[seat, :, rules.C_KIND]
    animal = state.tiles[seat, :, rules.C_ANIMAL]
    n_empty = int(np.sum(kind == rules.TK_EMPTY))
    feed_all = int(np.sum((animal >= 0) & (state.tiles[seat, :, rules.C_FED] == 0)))
    free_coop = int(np.sum((kind == rules.TK_COOP) & (animal < 0)))
    free_past = int(np.sum((kind == rules.TK_PASTURE) & (animal < 0)))
    fill = [free_coop if k == rules.TK_COOP else free_past for k in rules.ANIMAL_STRUCTURE_KIND]
    return {"n_empty": n_empty, "feed_all": feed_all, "fill": fill}


def market_orders(
    state: NpState, decision: Decision, caps: caps_mod.Caps, seat: int = 0
) -> list[tuple[int, int, int]]:
    """桶 → ``(op, item, qty)`` 的清單，已按 ``EMIT_ORDER`` 的優先序截斷到 ``max_orders``。

    ``state`` 必須是**單位相位之後**的狀態（同 `rl/loop4_actions.py::market_orders` 的契約）。
    """
    cnt = market_counts(state, seat)
    cand: list[tuple[int, int, int]] = []
    for p in range(rules.N_PRODUCTS):
        cand.append(
            (rules.MO_SELL, p, _sell_quantity(int(decision.sell[p]), int(state.shed[seat, p])))
        )
    cand.append((rules.MO_BUY_LAND, -1, 1 if decision.buy_land > 0 else 0))
    for i in range(N_HIRE - 1):
        cand.append((rules.MO_HIRE, -1, 1 if decision.hire > i else 0))
    for c in range(rules.N_CROPS):
        cand.append(
            (rules.MO_BUY_SEED, c, _buy_seed_quantity(int(decision.buy_seed[c]), cnt["n_empty"]))
        )
    for j, item in enumerate(BUYABLE_PRODUCTS):
        cand.append(
            (
                rules.MO_BUY_PRODUCT,
                item,
                _buy_product_quantity(int(decision.buy_product[j]), cnt["feed_all"]),
            )
        )
    for a in range(rules.N_ANIMALS):
        cand.append(
            (
                rules.MO_BUY_ANIMAL,
                rules.ANIMAL_ITEM_BASE + a,
                _buy_animal_quantity(int(decision.buy_animal[a]), cnt["fill"][a]),
            )
        )
    out = []
    for op, item, qty in cand:
        if qty <= 0:
            continue
        out.append((op, item if qty > 0 else -1, qty))
        if len(out) == caps.max_orders:
            break
    return out


# ------------------------------------------------------------ 引擎 JSON
def action_to_engine_json(
    unit_op: np.ndarray,
    unit_arg: np.ndarray,
    unit_qty: np.ndarray,
    orders: list[tuple[int, int, int]],
    n_alive: int,
) -> dict:
    """`scripts/loop4_parity.py::action_to_engine_json` 的等價。"""
    units = []
    for k in range(n_alive):
        o = int(unit_op[k])
        name = rules.UNIT_OP_NAMES[o]
        if o == rules.OP_PLANT:
            units.append([name, rules.CROPS[int(unit_arg[k])]])
        elif o in (rules.OP_PICKUP, rules.OP_PLACE):
            units.append([name, rules.ITEMS[int(unit_arg[k])], int(unit_qty[k])])
        else:
            units.append([name])
    market = []
    for op, item, qty in orders:
        if op == rules.MO_HIRE:
            market.append(["HIRE"])
        elif op == rules.MO_BUY_LAND:
            market.append(["BUY_LAND"])
        else:
            market.append([rules.MARKET_OP_NAMES[op], rules.ITEMS[int(item)], int(qty)])
    return {"farmer": units[0], "hands": units[1:], "market": market}


# --------------------------------------------------------------- agent
def load_params(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: np.asarray(z[k], F32) for k in z.files}


def config_from_params(params: dict[str, np.ndarray], caps: caps_mod.Caps) -> dict:
    """從權重的形狀反推 cfg，不寫死 4 層 d192。"""
    d = int(params["in_tile_w"].shape[1])
    # `ln_f_g` 也以 l 開頭，所以要認 `l<digits>_` 這個形狀，不能只看首字母。
    layers = [int(m.group(1)) for m in (re.match(r"l(\d+)_", k) for k in params) if m]
    n_layers = 1 + max(layers)
    n_heads = 6 if d % 6 == 0 else 4
    return {
        "d_model": d,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": int(params["l0_ff1_w"].shape[1]),
        "max_units": caps.max_units,
        "n_tokens": 2 * rules.N_TILES + caps.max_units + rules.N_PRODUCTS + 1,
    }


class Loop4NumpyAgent:
    """部署用的 agent：``agent(obs)`` → 引擎動作 dict。全程 NumPy。"""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        seat: int,
        caps: caps_mod.Caps | None = None,
        cfg: dict | None = None,
        rng=None,
    ) -> None:
        self.caps = caps or caps_mod.EXACT_REFERENCE
        self.params = params
        self.cfg = cfg or config_from_params(params, self.caps)
        self.seat = seat
        #: 給了就走取樣，``None`` 是 greedy。部署預設 greedy。
        self.rng = rng
        self.assignment = np.full(self.caps.max_units, NO_ASSIGNMENT, I32)
        self.last: dict[str, Any] = {}

    def __call__(self, obs: dict, _config: Any = None) -> dict:
        state = state_from_obs(obs, self.seat, self.caps)
        in_progress = np.zeros((2, rules.N_TILES), bool)
        hit = self.assignment[self.assignment >= 0]
        in_progress[0, hit] = True
        ob = observation(state, self.caps, in_progress)
        out = forward(self.params, ob, self.cfg)
        mask = tile_task_mask(state, self.caps)
        n_alive_decode = int(state.hires_today[0]) + 1
        decision = decode(out, mask, n_alive_decode, self.rng)
        unit_op, unit_arg, unit_qty, assignment = assign_and_walk(
            state, decision.tile_task, decision.tile_qty, self.caps, self.assignment
        )
        post = apply_unit_actions(state, unit_op, unit_arg, unit_qty, self.caps)
        orders = market_orders(post, decision, self.caps)
        n_alive = min(1 + int(obs["farms"][self.seat]["hires_today"]), self.caps.max_units)
        js = action_to_engine_json(unit_op, unit_arg, unit_qty, orders, n_alive)
        # 日末人手收回 spawn、hires_today 歸零，上一天的配對對新的一天沒有意義
        if (state.step + 1) % rules.TURNS_PER_DAY == 0:
            assignment = np.full(self.caps.max_units, NO_ASSIGNMENT, I32)
        self.assignment = assignment
        self.last = {"obs": ob, "logits": out, "mask": mask, "decision": decision, "action": js}
        return js
