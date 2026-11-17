"""Rule constants transcribed from ``kaggle_environments.envs.kaggriculture``.

Every table here is verified against the installed engine by
``rl.rules.verify_against_engine()`` (exercised by ``tests/test_rl_rules.py``),
so an engine upgrade fails loudly instead of silently drifting.

Index spaces
------------
``ITEMS`` is the single 12-wide item index space used by shed / farmer
inventories / market orders::

    0 WHEAT  1 CARROT  2 TOMATO  3 STRAWBERRY  4 MELON
    5 EGG    6 MILK    7 WOOL    8 FERTILIZER
    9 GOOSE 10 COW    11 SHEEP

``PRODUCTS`` are items 0..8 (market-tradable), ``CROPS`` are items 0..4 (seed
index == item index) and ``ANIMALS`` are items 9..11 (animal index == item - 9).
"""

from __future__ import annotations

ENGINE_VERSION = "1.32.7"

# ---------------------------------------------------------------- item space
ITEMS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
)
ITEM_INDEX = {name: i for i, name in enumerate(ITEMS)}
N_ITEMS = len(ITEMS)

PRODUCTS = ITEMS[:9]
N_PRODUCTS = 9
CROPS = ITEMS[:5]
N_CROPS = 5
ANIMALS = ITEMS[9:]
N_ANIMALS = 3
ANIMAL_ITEM_BASE = 9

I_WHEAT = 0
I_FERTILIZER = 8

# Products the town center pulls each tick (everything except FERTILIZER).
TOWN_CENTER_ITEMS = tuple(range(8))

# ---------------------------------------------------------------- crop table
# columns match CROPS order.
CROP_SEED_COST = (10, 20, 50, 100, 80)
CROP_FIRST_YIELD_DAY = (2, 2, 8, 10, 10)
CROP_MAX_YIELD_DAY = (4, 3, 8, 10, 12)
CROP_INTERVAL = (0, 0, 1, 2, 0)
CROP_MAX_YIELD = (6, 4, 4, 4, 6)
CROP_ONGOING = (0, 0, 1, 1, 0)
# ``(max_yield_day + 1) // 2`` -- start of the WATER yield window.
CROP_WATER_WINDOW_START = tuple((m + 1) // 2 for m in CROP_MAX_YIELD_DAY)

# -------------------------------------------------------------- animal table
ANIMAL_COST = (300, 400, 500)
ANIMAL_FIRST_YIELD_DAY = (4, 8, 6)
ANIMAL_INTERVAL = (1, 2, 3)
ANIMAL_MAX_HELD = (4, 6, 6)
ANIMAL_PRODUCT_ITEM = (5, 6, 7)  # EGG, MILK, WOOL

# ------------------------------------------------------------- tile encoding
TK_EMPTY = 0
TK_LOCKED = 1
TK_WEED = 2
TK_COOP = 3
TK_PASTURE = 4
TK_PLANT = 5
TILE_KIND_NAMES = ("EMPTY", "LOCKED", "WEED", "COOP", "PASTURE", "PLANT")

ANIMAL_STRUCTURE_KIND = (TK_COOP, TK_PASTURE, TK_PASTURE)

C_KIND = 0
C_CROP = 1
C_PLANTED_DAY = 2
C_WATERED = 3
C_UNWATERED = 4
C_YIELD = 5
C_MLS = 6  # max_lifespan_step
C_FERT_UNTIL = 7
C_ANIMAL = 8
C_PLACED_DAY = 9
C_UNFED = 10
C_FED = 11
C_CARED = 12
C_FERT_AVAIL = 13
C_CARE_BONUS = 14
N_TILE_CHANNELS = 15
TILE_CHANNEL_NAMES = (
    "kind",
    "crop",
    "planted_day",
    "watered_today",
    "consecutive_unwatered",
    "yield_units",
    "max_lifespan_step",
    "fertilized_until_day",
    "animal",
    "placed_day",
    "consecutive_unfed",
    "fed_today",
    "cared_today",
    "fertilizer_available",
    "pending_care_bonus",
)

# ------------------------------------------------------------------ unit ops
OP_PASS = 0
OP_NORTH = 1
OP_SOUTH = 2
OP_EAST = 3
OP_WEST = 4
OP_PICKUP = 5
OP_PLANT = 6
OP_WATER = 7
OP_HARVEST = 8
OP_FERTILIZE = 9
OP_BUILD_COOP = 10
OP_BUILD_PASTURE = 11
OP_DIG = 12
OP_PLACE = 13
OP_FEED = 14
OP_COLLECT_FERTILIZER = 15
OP_CARE = 16
OP_DROP = 17
N_UNIT_OPS = 18
UNIT_OP_NAMES = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "PLACE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
    "DROP",
)
UNIT_OP_INDEX = {name: i for i, name in enumerate(UNIT_OP_NAMES)}
# (dx, dy) per op index; non-move ops are (0, 0). y grows downward.
MOVE_DX = (0, 0, 0, 1, -1) + (0,) * 13
MOVE_DY = (0, -1, 1, 0, 0) + (0,) * 13

# ---------------------------------------------------------------- market ops
MO_NONE = 0
MO_BUY_SEED = 1
MO_BUY_PRODUCT = 2
MO_BUY_ANIMAL = 3
MO_SELL = 4
MO_HIRE = 5
MO_BUY_LAND = 6
N_MARKET_OPS = 7
MARKET_OP_NAMES = (
    "NONE",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "HIRE",
    "BUY_LAND",
)
MARKET_OP_INDEX = {name: i for i, name in enumerate(MARKET_OP_NAMES)}

# ------------------------------------------------------------------ economy
LAND_PRICES = (1000, 2000, 4000)
LAND_ORDER = ("NE", "SW", "SE")
FARM_HAND_COST_MULT = 1
MARKET_I0 = 10000
PRICE_FLOOR = 1
MAX_SHOP_INSTANCES = 8

SHOP_NAMES = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)
N_SHOPS = len(SHOP_NAMES)
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "YARN_STORE": ("WOOL",),
}


def _shop_demand_matrix() -> tuple[tuple[int, ...], ...]:
    rows = []
    for shop in SHOP_NAMES:
        products = _SHOP_PRODUCTS[shop]
        mult = 2 if len(products) == 1 else 1
        row = [0] * N_PRODUCTS
        for item in products:
            row[ITEM_INDEX[item]] += mult
        rows.append(tuple(row))
    return tuple(rows)


# [shop, product] units pulled per shop tick by one instance.
SHOP_DEMAND = _shop_demand_matrix()

# ------------------------------------------------------------- configuration
DEFAULT_CONFIG = {
    "episodeSteps": 720,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
}

BOARD_SIZE = 10
N_TILES = BOARD_SIZE * BOARD_SIZE
TURNS_PER_DAY = 24
N_DAYS = 30
EPISODE_STEPS = 720
# The framework records 720 observations, so only 719 interpreter transitions
# ever carry an agent action (steps 0..718).
N_TRANSITIONS = EPISODE_STEPS - 1
SHED_CAPACITY = 100

HALF = BOARD_SIZE // 2
SHED_ACCESS_TILES = ((HALF - 1, HALF - 1), (HALF, HALF - 1), (HALF - 1, HALF), (HALF, HALF))
SHED_ACCESS_FLAT = tuple(y * BOARD_SIZE + x for (x, y) in SHED_ACCESS_TILES)
DEFAULT_SPAWN = SHED_ACCESS_TILES[0]  # (4, 4) -- first shed-access tile in NW


def quadrant_of(x: int, y: int) -> str:
    return ("N" if y < HALF else "S") + ("W" if x < HALF else "E")


QUAD_ORDER = ("NW",) + LAND_ORDER
# quadrant rank per flat tile index: 0 = NW (always unlocked), 1..3 = LAND_ORDER.
TILE_QUAD_RANK = tuple(
    QUAD_ORDER.index(quadrant_of(i % BOARD_SIZE, i // BOARD_SIZE)) for i in range(N_TILES)
)


def fib(n: int) -> int:
    """``fib(0) == 1``, ``fib(1) == 1``, ``fib(2) == 2`` -- engine indexing."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(n_already_today: int, mult: int = FARM_HAND_COST_MULT) -> int:
    return mult * fib(n_already_today)


def cumulative_hire_cost(n: int, mult: int = FARM_HAND_COST_MULT) -> int:
    return sum(hire_cost(i, mult) for i in range(n))


def max_hires_for_money(money: float, mult: int = FARM_HAND_COST_MULT) -> int:
    """Largest ``n`` whose cumulative Fibonacci hire cost fits in ``money``."""
    n, spent = 0, 0
    while True:
        spent += hire_cost(n, mult)
        if spent > money:
            return n
        n += 1


def verify_against_engine() -> None:
    """Raise ``AssertionError`` if the installed engine no longer matches."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as eng

    assert tuple(eng.PRODUCTS) == PRODUCTS, "PRODUCTS order drifted"
    assert tuple(eng.CROPS) == CROPS, "CROPS order drifted"
    assert tuple(eng.ANIMALS) == ANIMALS, "ANIMALS order drifted"
    assert tuple(sorted(eng.SHOPS)) == SHOP_NAMES, "SHOPS drifted"
    assert tuple(eng.TOWN_CENTER_PRODUCTS) == PRODUCTS[:8], "town centre basket drifted"
    for i, crop in enumerate(CROPS):
        cd = eng.CROPS[crop]
        assert cd["seed"] == CROP_SEED_COST[i]
        assert cd["first_yield_day"] == CROP_FIRST_YIELD_DAY[i]
        assert cd["max_yield_day"] == CROP_MAX_YIELD_DAY[i]
        assert cd["interval"] == CROP_INTERVAL[i]
        assert cd["max_yield"] == CROP_MAX_YIELD[i]
        assert int(cd["ongoing"]) == CROP_ONGOING[i]
    for i, animal in enumerate(ANIMALS):
        ad = eng.ANIMALS[animal]
        assert ad["cost"] == ANIMAL_COST[i]
        assert ad["first_yield_day"] == ANIMAL_FIRST_YIELD_DAY[i]
        assert ad["interval"] == ANIMAL_INTERVAL[i]
        assert ad["max_held"] == ANIMAL_MAX_HELD[i]
        assert ITEM_INDEX[ad["product"]] == ANIMAL_PRODUCT_ITEM[i]
        kind = TK_COOP if ad["structure"] == "COOP" else TK_PASTURE
        assert kind == ANIMAL_STRUCTURE_KIND[i]
    for si, shop in enumerate(SHOP_NAMES):
        assert tuple(eng.SHOPS[shop]) == _SHOP_PRODUCTS[shop]
        assert SHOP_DEMAND[si] == _shop_demand_matrix()[si]
    assert tuple(eng.LAND_PRICES) == LAND_PRICES
    assert tuple(eng.LAND_ORDER) == LAND_ORDER
    assert eng.MARKET_I0 == MARKET_I0
    assert eng.PRICE_FLOOR == PRICE_FLOOR
    assert eng.MAX_SHOP_INSTANCES == MAX_SHOP_INSTANCES
    assert eng.FARM_HAND_COST_MULT == FARM_HAND_COST_MULT
    assert tuple(eng._shed_access_tiles(BOARD_SIZE)) == SHED_ACCESS_TILES
    assert tuple(eng._default_spawn(BOARD_SIZE)) == DEFAULT_SPAWN
    for n in range(12):
        assert eng._fib(n) == fib(n)
    spec = eng.specification
    for key, value in DEFAULT_CONFIG.items():
        got = spec["configuration"][key]
        got = got if not isinstance(got, dict) else got["default"]
        assert got == value, f"config default {key}: {got} != {value}"
