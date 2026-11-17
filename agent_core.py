"""Kaggriculture Day 0 共用核心（observation-driven planner）。

設計原則：

* **完全無狀態。** `plan()` 只讀 observation，不保留任何跨回合或跨對局的資料，
  因此同一個 process 內連續評估多場對局不會互相污染。
* **存活優先。** 會枯死的作物與會逃走的動物永遠排在最前面（tier 0）。
* **能力受限才擴張。** 播種、買地、買動物都要先通過現金與當日 action capacity 檢查。
* **市場有價格底線。** 賣出量由「賣完之後的價格仍在底線之上」決定，期末才無條件清倉。

引擎常數鏡像自 kaggle_environments/envs/kaggriculture/kaggriculture.py（1.32.7）。
"""

import math

# --------------------------------------------------------------------------- #
# 引擎常數鏡像
# --------------------------------------------------------------------------- #

TURNS_PER_DAY = 24
FINAL_DAY = 29  # 720 steps -> 最後一個可下動作的回合是 day 29 hour 22
LAST_HOUR = 22
# 批准計畫規定：obs["day"] >= 26 起停止一切「新投資」（買種／買動物／買地／播種／建造）。
# 注意邊界方向：day 26 當天就已經停止，不是 day 27 才停。
HARD_INVEST_STOP_DAY = 26
# 最後一個「餵食仍有意義」的日子：day 29 之後沒有跨日結算，餵了也不會產出。
LAST_FEED_DAY = FINAL_DAY - 1
SHED_CAPACITY = 100
MAX_MARKET_ORDERS = 10
PRICE_FLOOR = 1
MARKET_I0 = 10000
HINGE_GAIN = 8.0

CROPS = {
    "WHEAT": {
        "seed": 10,
        "first_yield_day": 2,
        "max_yield_day": 4,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
    "CARROT": {
        "seed": 20,
        "first_yield_day": 2,
        "max_yield_day": 3,
        "interval": 0,
        "max_yield": 4,
        "ongoing": False,
    },
    "TOMATO": {
        "seed": 50,
        "first_yield_day": 8,
        "max_yield_day": 8,
        "interval": 1,
        "max_yield": 4,
        "ongoing": True,
    },
    "STRAWBERRY": {
        "seed": 100,
        "first_yield_day": 10,
        "max_yield_day": 10,
        "interval": 2,
        "max_yield": 4,
        "ongoing": True,
    },
    "MELON": {
        "seed": 80,
        "first_yield_day": 10,
        "max_yield_day": 12,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
}

ANIMALS = {
    "GOOSE": {
        "cost": 300,
        "structure": "COOP",
        "first_yield_day": 4,
        "interval": 1,
        "max_held": 4,
        "product": "EGG",
    },
    "COW": {
        "cost": 400,
        "structure": "PASTURE",
        "first_yield_day": 8,
        "interval": 2,
        "max_held": 6,
        "product": "MILK",
    },
    "SHEEP": {
        "cost": 500,
        "structure": "PASTURE",
        "first_yield_day": 6,
        "interval": 3,
        "max_held": 6,
        "product": "WOOL",
    },
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

MARKET_PARAMS = {
    "WHEAT": {
        "base": 25,
        "T": 400,
        "below_func": "sqrt",
        "below_target": 0.80,
        "above_func": "log",
        "above_target": 0.20,
    },
    "CARROT": {
        "base": 35,
        "T": 450,
        "below_func": "hinge",
        "below_target": 1.00,
        "above_func": "sqrt",
        "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60,
        "T": 200,
        "below_func": "hinge",
        "below_target": 0.40,
        "above_func": "sqrt",
        "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120,
        "T": 100,
        "below_func": "sqrt",
        "below_target": 0.70,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "MELON": {
        "base": 250,
        "T": 300,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.60,
    },
    "EGG": {
        "base": 50,
        "T": 332,
        "below_func": "hinge",
        "below_target": 0.40,
        "above_func": "log",
        "above_target": 0.20,
    },
    "MILK": {
        "base": 160,
        "T": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "T": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "linear",
        "above_target": 0.40,
    },
}

LAND_PRICES = (1000, 2000, 4000)

# 每日工時模型：一格作物約需 1 次澆水 + 移動 + 攤提的播種/收成；
# 一隻動物需 FEED + CARE + 攤提的 HARVEST/COLLECT_FERTILIZER + 移動。
WORK_PER_TILE = 3.0
WORK_PER_ANIMAL = 4.5
WORK_PER_UNIT_DAY = 22.0

# 一次性作物在「完美澆水」下實際可拿到的收量與到期天數。
# （max_yield_day 之後就開始衰減，所以 harvest_age 就是實際採收日。）
HARVEST_AGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10, "STRAWBERRY": 10, "TOMATO": 8}


def _shape(func, x, T):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory):
    """鏡像引擎的定價函式；用來預估「再賣 n 單位之後」的價格。"""
    p = MARKET_PARAMS[item]
    base = p["base"]
    T = p["T"]
    if inventory < MARKET_I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, MARKET_I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inventory - MARKET_I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def hire_cost(n_already_today):
    a, b = 1, 1
    for _ in range(n_already_today):
        a, b = b, a + b
    return a


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #


class Profile:
    """兩個候選之間唯一的差異來源。

    Opening Tempo subsystem 的參數全部放在這裡（而不是 module-global），有兩個
    理由：一是 Gate 0 禁止新增 module-global tuning；二是 `ANCHOR` 必須逐位元
    維持原行為，把開關與數值都放進 profile，anchor 只要不設定就完全走舊路徑。
    預設值一律是「關閉」的中性值，真正的數值只出現在 `CHALLENGER`。
    """

    __slots__ = (
        "name",
        "crops",
        "use_animals",
        "animal_targets",
        "last_animal_day",
        "max_hands",
        "land_plan",
        "wheat_reserve_per_animal",
        "wheat_carry",
        "use_fertilizer",
        "sell_floor_ratio",
        "opening_tempo",
        "opening_free_structs",
        "opening_last_day",
        "opening_seed_reserve",
        "land_backlog_tiles",
        "land_cash_buffer",
        "early_hands_floor",
        "opening_wheat_share",
        "opening_crop_last_day",
    )

    def __init__(
        self,
        name,
        crops,
        use_animals=False,
        animal_targets=(),
        last_animal_day=18,
        max_hands=11,
        land_plan=((0, 3, 20), (1, 4, 16)),
        wheat_reserve_per_animal=3,
        wheat_carry=6,
        use_fertilizer=False,
        sell_floor_ratio=((20, 0.55), (26, 0.40), (29, 0.15)),
        opening_tempo=False,
        opening_free_structs=0,
        opening_last_day=-1,
        opening_seed_reserve=0.0,
        land_backlog_tiles=0,
        land_cash_buffer=0.0,
        early_hands_floor=(),
        opening_wheat_share=0.0,
        opening_crop_last_day=None,
    ):
        self.name = name
        self.crops = tuple(crops)
        self.use_animals = use_animals
        # ((animal, count), ...) 依序建置
        self.animal_targets = tuple(animal_targets)
        self.last_animal_day = last_animal_day
        self.max_hands = max_hands
        # ((land_index, cash_multiple_required, last_day), ...)
        self.land_plan = tuple(land_plan)
        self.wheat_reserve_per_animal = wheat_reserve_per_animal
        self.wheat_carry = wheat_carry
        self.use_fertilizer = use_fertilizer
        self.sell_floor_ratio = tuple(sell_floor_ratio)
        # Opening Tempo subsystem 的總開關與參數。預設全部關閉／中性，`ANCHOR`
        # 因此完全不受影響（main.py 的行為必須逐位元維持不變）。
        self.opening_tempo = bool(opening_tempo)
        self.opening_free_structs = int(opening_free_structs)
        self.opening_last_day = int(opening_last_day)
        self.opening_seed_reserve = float(opening_seed_reserve)
        self.land_backlog_tiles = int(land_backlog_tiles)
        self.land_cash_buffer = float(land_cash_buffer)
        # ((day_limit, hands), ...)：day < day_limit 時的 hands 下限，語義與
        # sell_floor_ratio 相同；空 tuple 代表沒有 ramp。
        self.early_hands_floor = tuple(early_hands_floor)
        self.opening_wheat_share = float(opening_wheat_share)
        # 「開局配比窗口」與「小額 seed reserve 窗口」本來是兩個政策，卻意外共用
        # 同一個 `opening_last_day`。把它們分開：未指定時回落到 `opening_last_day`，
        # 兩個窗口合一，行為與分離之前逐位元相同（由等價回歸測試強制證明）。
        # `_seed_reserve` 一律只讀 `opening_last_day`，永遠不看這個欄位。
        self.opening_crop_last_day = int(
            self.opening_last_day if opening_crop_last_day is None else opening_crop_last_day
        )

    def floor_ratio(self, day):
        for limit, ratio in self.sell_floor_ratio:
            if day < limit:
                return ratio
        return 0.0

    def early_hands(self, day):
        """有 expansion backlog 時的 hands 下限；沒設定就回 0（等於沒有 ramp）。"""
        for limit, hands in self.early_hands_floor:
            if day < limit:
                return hands
        return self.early_hands_floor[-1][1] if self.early_hands_floor else 0


ANCHOR = Profile(
    name="day0_anchor",
    crops=("MELON", "WHEAT", "CARROT", "STRAWBERRY"),
    use_animals=False,
    max_hands=11,
    land_plan=((0, 2.2, 22), (1, 2.5, 18)),
    use_fertilizer=False,
)

CHALLENGER = Profile(
    name="day0_challenger",
    crops=("MELON", "STRAWBERRY", "WHEAT"),
    use_animals=True,
    animal_targets=(("COW", 9), ("SHEEP", 3)),
    last_animal_day=18,
    max_hands=12,
    land_plan=((0, 2.0, 22), (1, 2.5, 18)),
    wheat_reserve_per_animal=3,
    wheat_carry=6,
    use_fertilizer=True,
    # --- Opening Tempo subsystem（依 內部規格文件）---
    # top-10 每個家族都在 day 0 就把現金換成資產（收盤現金中位 43、19 作物、
    # 4 動物、**0 個空結構**），day 4–6 買 Q2、day 8–11 買 Q3，開局 hands 5–8。
    # 舊版依「最終 12 隻動物」預留結構與格子，三場 replay 固定停在 d0 的
    # 13 作物 + 2 動物 + 10 空 pasture、d7 的 1 象限 / 3 hands / 4 動物 / 13 作物。
    opening_tempo=True,
    opening_free_structs=2,  # 允許同時存在的空結構上限（top-10 觀測範圍 0–2）
    opening_last_day=2,  # 小額 seed reserve 只用在 d0–d2
    opening_seed_reserve=60.0,  # 開局固定保留金；真正的下限是隔日餵食成本
    land_backlog_tiles=12,  # 判定「格子不夠」時前瞻的格數（約半塊象限）
    land_cash_buffer=150.0,  # 買地後仍必須留下的營運現金
    early_hands_floor=((6, 6), (FINAL_DAY + 1, 8)),  # d0–d5 至少 6；d6 起至少 8
    opening_wheat_share=0.40,  # 配比窗口內新播種中 WHEAT 的最低比例
    # v2.1：配比／缺貨 fallback 的窗口延長到 d8（唯一的策略變量）。
    # `opening_last_day` 維持 2——小額 seed reserve 的窗口不變，d3 起回到
    # 標準的 `150 + 60·n_animals`，飼料現金緩衝不被削弱。
    opening_crop_last_day=8,
)


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #


class Task:
    __slots__ = ("tier", "value", "targets", "action", "key", "exclusive_tile", "need", "only_unit")

    def __init__(
        self,
        tier,
        value,
        targets,
        action,
        key=None,
        exclusive_tile=None,
        need=None,
        only_unit=None,
    ):
        self.tier = tier
        self.value = value
        self.targets = targets  # ((x, y), ...) 任一可執行位置
        self.action = action  # 送出的動作 list
        self.key = key  # 保留鍵，None 表示可重複指派
        self.exclusive_tile = exclusive_tile  # (x, y)：整格獨占
        self.need = need  # (item, count) 單位 inventory 前置條件
        self.only_unit = only_unit  # 綁定特定單位（卸貨這類任務只有持貨的人能做）


# tier 權重：tier 之間相差夠大，確保存活任務永遠壓過一切；
# tier 內部則由 value / (1 + 2*dist) 決定，讓單位優先做手邊的工作。
_TIER_W = (1.0e7, 1.0e5, 1.0e4, 1.0e3, 60.0, 8.0, 1.0)

# 單回合最多考慮的任務數（延遲上限保護）。
MAX_TASKS = 160


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(pos, target):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if dx > 0:
        return ["EAST"]
    if dx < 0:
        return ["WEST"]
    if dy > 0:
        return ["SOUTH"]
    if dy < 0:
        return ["NORTH"]
    return ["PASS"]


# --------------------------------------------------------------------------- #
# 每回合規劃器
# --------------------------------------------------------------------------- #


class _Turn:
    def __init__(self, obs, profile):
        self.p = profile
        self.player = int(obs["player"])
        self.farm = obs["farms"][self.player]
        self.priv = obs["private"]
        self.day = int(obs["day"])
        self.hour = int(obs["hour"])
        self.money = float(self.farm["money"])
        self.tiles = self.farm["tiles"]
        self.board = len(self.tiles)
        half = self.board // 2
        self.shed_tiles = ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
        self.shed_center = (half - 1, half - 1)

        market = obs["market"]
        self.m_inv = dict(market["inventory"])
        self.m_price = dict(market["prices"])

        self.shed = dict(self.priv.get("shed", {}) or {})
        self.seeds = dict(self.priv.get("seeds", {}) or {})
        raw_inv = self.priv.get("inventories", []) or []

        positions = [tuple(self.farm["farmer"])]
        for h in self.farm["hands"]:
            positions.append((int(h[0]), int(h[1])))
        self.pos = positions
        self.n_units = len(positions)
        self.inv = []
        for i in range(self.n_units):
            self.inv.append(dict(raw_inv[i]) if i < len(raw_inv) and raw_inv[i] else {})

        self.hires_today = int(self.farm.get("hires_today", 0))
        self.n_quadrants = len(self.farm.get("unlocked_quadrants", ["NW"]))

        self.carried_total = sum(sum(d.values()) for d in self.inv)
        self.shed_total = sum(self.shed.values())

        self.plants = []  # (x, y, tile)
        self.animals = []  # (x, y, tile)
        self.free_structs = []  # (x, y, tile)
        self.weeds = []  # (x, y)
        self.empty = []  # (x, y)
        self._scan()

        self.n_animals = len(self.animals)
        # 飼料保留量 = 每隻動物 × 剩餘仍需餵食的天數（上限為 profile 設定值）。
        # day 26 起禁止 BUY_PRODUCT，所以這份存量必須在 day 25 之前就備妥；
        # 之後逐日遞減，day 29 歸零讓終局清算可以把小麥全部賣掉。
        if self.p.use_animals:
            days_left = max(0, min(self.p.wheat_reserve_per_animal, LAST_FEED_DAY - self.day + 1))
            self.wheat_reserve = days_left * self.n_animals
        else:
            self.wheat_reserve = 0

        # 作物 pipeline（田面尚未收成的預估產量）用來算邊際價格。
        self.pipeline = {}
        for _x, _y, t in self.plants:
            crop = t["crop"]
            cd = CROPS[crop]
            add = cd["max_yield"] if cd["ongoing"] else HARVEST_UNITS[crop]
            self.pipeline[crop] = self.pipeline.get(crop, 0) + add
        for _x, _y, t in self.animals:
            prod = ANIMALS[t["animal"]]["product"]
            self.pipeline[prod] = self.pipeline.get(prod, 0) + 6

        # shed 壓力：倉庫 + 所有單位身上的貨。日終會全部倒回倉庫，超過 100 直接消失，
        # 所以壓力越高就要越積極出貨，並且停止把更多貨收進身上。
        self.projected_shed = self.shed_total + self.carried_total
        self.pressure = self.projected_shed / float(SHED_CAPACITY)

        # 新投資閘門的唯一判準。所有「買種／買動物／買地／播種／建造／清雜草」
        # 都必須經過它，避免邊界條件散落在各處而漂移。
        self.investing = self.day < HARD_INVEST_STOP_DAY

        self.load = WORK_PER_TILE * len(self.plants) + WORK_PER_ANIMAL * self.n_animals
        self.capacity = (1 + self.p.max_hands) * WORK_PER_UNIT_DAY
        self.desired_hands = self._desired_hands()

        self.orders = []
        self.planned_plants = {}  # crop -> 本回合已排的播種數
        self.spend = 0.0  # 本回合已排定的市場支出

    # ---------------- 掃描 ---------------- #

    def _scan(self):
        tiles = self.tiles
        for y in range(self.board):
            row = tiles[y]
            for x in range(self.board):
                t = row[x]
                if t is None:
                    self.empty.append((x, y))
                    continue
                if not isinstance(t, dict):
                    continue  # "LOCKED"
                kind = t.get("kind")
                if kind == "PLANT":
                    self.plants.append((x, y, t))
                elif "animal" in t:
                    self.animals.append((x, y, t))
                elif kind == "WEED":
                    self.weeds.append((x, y))
                elif kind in ("COOP", "PASTURE"):
                    self.free_structs.append((x, y, t))

    # ---------------- 價格 ---------------- #

    def price_of(self, item):
        return float(self.m_price.get(item, MARKET_PARAMS[item]["base"]))

    def projected_price(self, item):
        """把自己田面 pipeline 與倉庫存量灌進市場後的邊際價格。"""
        inv = self.m_inv.get(item, MARKET_I0)
        extra = self.pipeline.get(item, 0) + self.shed.get(item, 0)
        return market_price(item, inv + extra)

    # ---------------- 雇工 ---------------- #

    def _desired_hands(self):
        """由「今天實際要做的工時」反推需要幾個工人。

        雇工採 Fibonacci 計價（12 人／天約 $376），相對於一格甜瓜的產值極便宜，
        所以瓶頸應該是土地而不是勞力；但仍以現金比例設上限避免早期把錢燒光。
        """
        if self.day > FINAL_DAY:
            return 0
        workable = len(self.plants)
        if self.investing:
            workable += len(self.empty) + len(self.weeds)
        work = WORK_PER_TILE * workable + WORK_PER_ANIMAL * self.n_animals
        if self.p.use_animals and self.investing:
            # 只有還在投資期時才為「即將買進的動物」預留人力；day 26 起已禁止
            # BUY_ANIMAL，再為未達 target 的缺口增聘就等於為不會存在的資產雇工。
            work += WORK_PER_ANIMAL * min(self._animals_pending(), 6)
        target = int(math.ceil(work / WORK_PER_UNIT_DAY)) - 1
        target = max(0, min(self.p.max_hands, target))
        if self.day == 0:
            target = max(target, 5)
        # Opening Tempo：工時模型只看「當下已經存在的資產」，開局資產少就雇不到人，
        # 人少又吃不下更多格子——這是 replay 裡 day 7 卡在 3 hands 的直接原因。
        # 只要還有可執行的開發 backlog，就先把人力拉到 top-10 的開局水位；
        # max_hands 上限與下面的現金可負擔迴圈都照舊生效。
        if self.p.opening_tempo and self._expansion_backlog() > 0:
            target = max(target, min(self.p.max_hands, self.p.early_hands(self.day)))
        budget = self._hire_budget()
        total = 0
        allowed = 0
        for i in range(target):
            total += hire_cost(i)
            if total > budget:
                break
            allowed = i + 1
        return allowed

    def wheat_target(self):
        """倉庫希望持有的小麥量——buy 與 sell 的**唯一**依據。

        舊版賣的門檻用 wheat_reserve、買的門檻用 wheat_reserve + n_animals，
        兩者中間的區間會同時觸發買與賣。兩邊都讀這個函式就不可能對沖。
        """
        return self.wheat_reserve

    def _hire_budget(self):
        """當日雇工的現金上限（_desired_hands 與 _market 共用同一條規則）。"""
        return max(80.0, self.money * 0.5)

    def _animals_pending(self):
        if not self.p.use_animals:
            return 0
        target = sum(n for _a, n in self.p.animal_targets)
        return max(0, target - self.n_animals)

    # ---------------- Opening Tempo subsystem ---------------- #

    def _held_animals(self):
        """已經買到、正在等結構的動物數（倉庫 + 單位身上）。

        這些是**現在就需要**一個空結構的動物，和「最終 target 還缺幾隻」不同——
        後者是未來的事，拿它來提前蓋結構會把開局的格子全部佔死。
        """
        if not self.p.use_animals:
            return 0
        n = sum(self.shed.get(a, 0) for a in ANIMALS)
        for d in self.inv:
            n += sum(d.get(a, 0) for a in ANIMALS)
        return int(n)

    def _hire_plan(self):
        """這一回合的 HIRE `(張數, 花費)`——`_market` 與現金估算的唯一來源。"""
        if not (self.hour <= 1 and self.day <= FINAL_DAY):
            return 0, 0.0
        want = max(0, self.desired_hands - (self.n_units - 1))
        budget = self._hire_budget()
        spent = 0.0
        n = 0
        for k in range(want):
            c = hire_cost(self.hires_today + k)
            if spent + c > budget:
                break
            spent += c
            n += 1
            if n >= 9:
                break
        return n, spent

    def _feed_order(self, cash):
        """飼料小麥訂單 `(order, 花費)`，沒有就回 None——買與估都走這一份。"""
        if not (self.p.use_animals and self.n_animals > 0):
            return None
        have = self.shed.get("WHEAT", 0) + sum(d.get("WHEAT", 0) for d in self.inv)
        deficit = self.wheat_target() - have
        price = self.price_of("WHEAT")
        room = SHED_CAPACITY - self.shed_total - 4
        if deficit > 0 and price <= 60 and room > 0:
            n = int(min(deficit, room, cash // max(1.0, price * 1.2)))
            if n > 0:
                return ["BUY_PRODUCT", "WHEAT", n], n * price
        return None

    def _cash_for_animals(self):
        """輪到買動物時實際還剩多少現金。

        `_tasks()`（JIT 建結構）跑在 `_market()` 之前，那時 `self.spend` 還是 0；
        如果 JIT 用 `self.money` 估、`_buy_orders` 用扣掉 HIRE 與飼料之後的餘額估，
        就會出現「結構蓋了、動物卻買不成」的錯配，白白吃掉一格與一個 build 動作。
        因此兩邊都改用這個純函數，數字由建構方式保證相同。
        """
        cash = self.money - self._hire_plan()[1]
        feed = self._feed_order(cash)
        if feed is not None:
            cash -= feed[1]
        return cash

    def _animal_counts(self):
        """每種動物「已經算數」的數量：田面 + 倉庫 + 單位身上。"""
        placed = {}
        for _x, _y, t in self.animals:
            placed[t["animal"]] = placed.get(t["animal"], 0) + 1
        for a in ANIMALS:
            carried = sum(d.get(a, 0) for d in self.inv)
            placed[a] = placed.get(a, 0) + self.shed.get(a, 0) + carried
        return placed

    def _animal_orders(self, cash):
        """這一回合要送出的 `BUY_ANIMAL` 批次清單 `[(animal, n), ...]`。

        `_buy_orders` 與 JIT 建結構共用同一份判斷，兩邊才不會漂移：結構只為
        「已持有」或「這一回合就會下單買到」的動物而蓋。

        **day 0 允許兩批（先 COW 後 SHEEP），且整個 day 0 的物種排程固定為
        2 COW + 2 SHEEP**——湊滿就不再下單，第五隻要等 day 1 之後才由現行規則
        接手。其餘日子維持現行規則（一回合只推進一種動物、最多一批、牛先於羊）。

        引擎按 market list 的順序逐單結算、現金不足即停，而 `_buy_orders` 本來
        就把動物排在種子之前——所以「排序」本身就是資金分配機制，不需要任何
        funding floor 或預算欄位。
        """
        if not self.p.use_animals or self.day > self.p.last_animal_day:
            return []
        placed = self._animal_counts()
        two_batches = self.p.opening_tempo and self.day == 0
        out = []
        load = self.load
        for a, target in self.p.animal_targets:
            if two_batches:
                # d0 的排程上限就是每種 2 隻；總 target（9c+3s）不變，只是延後。
                target = min(target, 2)
            missing = target - placed.get(a, 0)
            if missing <= 0:
                continue
            cost = ANIMALS[a]["cost"]
            reserve = 300.0 + 40.0 * self.n_animals
            affordable = int(max(0.0, cash - reserve) // cost)
            room = SHED_CAPACITY - self.shed_total - 4 - sum(k for _b, k in out)
            n = min(missing, affordable, max(0, room), 2)
            if n > 0 and load + WORK_PER_ANIMAL * n <= self.capacity:
                out.append((a, n))
                cash -= n * cost
                load += WORK_PER_ANIMAL * n
            if not two_batches or len(out) >= 2:
                break
        return out

    def _jit_animal_demand(self):
        """現在**真的**需要結構的動物數＝已持有待放置 ＋ 這一回合會買的那一批。

        和 `_animals_pending()`（最終 target 的缺口，開局是 10）不同——那是未來的
        事，拿來預留格子就會把第一象限鎖死。
        """
        if not self.p.use_animals or self.day > self.p.last_animal_day:
            return 0
        incoming = sum(n for _a, n in self._animal_orders(self._cash_for_animals()))
        want = self._held_animals() + incoming
        target = sum(n for _a, n in self.p.animal_targets)
        room_to_target = max(0, target - (self.n_animals + len(self.free_structs)))
        return max(0, min(want, room_to_target))

    def _jit_tile_reserve(self):
        """要留給 JIT 結構、不可以拿去播種的空格數。

        完全不保留是不行的：實測作物會把 25 格佔滿，`_build_and_plant_tasks` 因
        `not self.empty` 直接返回，倉庫裡已經買好的動物永遠等不到 pasture。
        保留量只跟「已持有 ＋ 一個近期批次」有關，不是最終 target 的缺口。
        """
        if not self.p.opening_tempo:
            return 0
        return max(0, self._jit_animal_demand() - len(self.free_structs))

    def _structs_needed(self):
        """這一回合該蓋幾個 pasture（JIT）。

        舊版依最終 target 預建（`target - (animals + free_structs)`，每回合 3 個），
        day 0 就蓋滿 12 個、其中 10 個空置，把第一象限鎖死。改成只為
        「已持有待放置」＋「這一回合買得起的一批」而蓋，且同時存在的空結構數不得
        超過 profile 的 `opening_free_structs`。
        """
        if not self.p.use_animals or self.day > self.p.last_animal_day:
            return 0
        if self.load + 4.0 > self.capacity:
            return 0
        free = len(self.free_structs)
        if not self.p.opening_tempo:
            target = sum(n for _a, n in self.p.animal_targets)
            return max(0, min(target - (self.n_animals + free), 3))
        cap = self.p.opening_free_structs
        return max(0, min(self._jit_animal_demand() - free, cap - free, 3))

    def _expansion_backlog(self):
        """「多一雙手就有事做」的開發待辦，以格數計。

        用於 early hire ramp：只要還有可開發的空地／雜草，或還有動物等著放置，
        就代表雇工不會閒置。day 26 起停止投資，backlog 一律視為 0。
        """
        if not self.investing:
            return 0
        return len(self.empty) + len(self.weeds) + self._held_animals()

    def _land_backlog(self):
        """買地觸發：可用格子已經少於「今天還吸收得了的量」。

        舊規則是「`empty + weeds > 8` 就不買地」，但開局只有 Q1 的 25 格時這條
        永遠成立，Q2 因此被拖到 day 9–10（top-10 是 day 4–6）。
        """
        free_tiles = len(self.empty) + len(self.weeds)
        room = max(0, int((self.capacity - self.load) // WORK_PER_TILE))
        return max(0, min(room, self.p.land_backlog_tiles) - free_tiles)

    def _opening_crop(self, crop, planting=False):
        """開局配比：窗口內的新播種中，WHEAT 不得低於 `opening_wheat_share`。

        窗口由 **`opening_crop_last_day`** 決定（與 `_seed_reserve` 的
        `opening_last_day` 是兩個獨立政策，不可共用）。只在窗口內生效，而且只
        決定「這一批要種／要買哪一種」，`_choose_crop()` 的一般評分完全不動。

        為什麼必須混種：`_choose_crop()` 在 pipeline 還是 0 時一定選 MELON，而
        MELON 要 day 10 才有第一筆收入，d2–d5 完全沒有現金買第三批動物；小麥
        兩天就能收，而且同時是飼料。live top10 每個家族 day 0 都是 WHEAT+MELON
        混種。

        `planting=True` 是播種端：配比挑到的種子如果手上沒有，必須改種手上有的，
        不能整個回合放棄——實測 d0 買了 21 顆小麥種子、種滿 5 格後配比切到 MELON，
        而甜瓜種子已用完，`available <= 0` 讓剩下 16 顆種子與空地整天沒動。
        （比例是下限，多種小麥不會違反配比。）
        """
        window = self.p.opening_crop_last_day
        if not (self.p.opening_tempo and self.day <= window):
            return crop
        if "WHEAT" not in self.p.crops or self.p.opening_wheat_share <= 0:
            return crop
        total = 0
        wheat = 0
        for _x, _y, t in self.plants:
            if int(t.get("planted_day", 0)) > window:
                continue
            total += 1
            if t.get("crop") == "WHEAT":
                wheat += 1
        want = "WHEAT" if wheat < self.p.opening_wheat_share * (total + 1) else crop
        if planting and self.seeds.get(want, 0) <= 0:
            for alt in ("WHEAT", *self.p.crops):
                if self.seeds.get(alt, 0) > 0:
                    return alt
        return want

    def _seed_reserve(self):
        """買種子時要留下的現金。

        舊版 `150 + 60·n_animals` 在低現金開局會直接停掉播種。d0–d2 改成小額固定
        保留，下限是「隔天餵食小麥的採購成本」——動物不能斷糧，但除此之外開局
        應該把現金全部換成資產（top-10 的 d0 收盤現金中位數是 43）。
        """
        feed = 0.0
        if self.p.use_animals:
            feed = self.n_animals * self.price_of("WHEAT")
        if self.p.opening_tempo and self.day <= self.p.opening_last_day:
            return max(self.p.opening_seed_reserve, feed)
        return 150.0 + (60.0 * self.n_animals if self.p.use_animals else 0.0)

    # ---------------- 作物選擇 ---------------- #

    def _crop_units_by_end(self, crop):
        cd = CROPS[crop]
        if not cd["ongoing"]:
            age = HARVEST_AGE[crop]
            if self.day + age > FINAL_DAY:
                return 0, 0
            return HARVEST_UNITS[crop], age + 1
        units = 0
        last = 0
        for k in range(cd["max_yield"]):
            d = cd["first_yield_day"] + k * cd["interval"]
            if self.day + d <= FINAL_DAY:
                units += 1
                last = d
        return units, last + 1

    def _choose_crop(self):
        best = None
        best_score = 0.0
        for crop in self.p.crops:
            if not self.investing:
                continue
            units, tile_days = self._crop_units_by_end(crop)
            if units <= 0 or tile_days <= 0:
                continue
            price = self.projected_price(crop)
            floor = MARKET_PARAMS[crop]["base"] * self.p.floor_ratio(self.day)
            if price < floor:
                continue
            revenue = units * price
            score = (revenue - CROPS[crop]["seed"]) / float(tile_days)
            if self.p.use_animals and crop == "WHEAT":
                have = self.shed.get("WHEAT", 0) + sum(d.get("WHEAT", 0) for d in self.inv)
                if have < self.wheat_reserve:
                    score *= 1.6
            if score > best_score:
                best_score = score
                best = crop
        return best

    # ---------------- 任務 ---------------- #

    def _tasks(self):
        out = []
        day = self.day
        hour = self.hour
        final = day >= FINAL_DAY

        # --- 作物 ---
        for x, y, t in self.plants:
            crop = t["crop"]
            cd = CROPS[crop]
            age = day - int(t["planted_day"])
            price = self.price_of(crop)
            yu = int(t.get("yield_units", 0))

            if not t.get("watered_today", False):
                critical = int(t.get("consecutive_unwatered", 0)) >= 1
                gain = self._water_gain(t, cd, age, day)
                if final:
                    # day 29 之後沒有跨日結算，作物不會枯死，所以只有「今天還能
                    # 增加收量」的澆水才有意義；存活性澆水在最後一天是純浪費。
                    if gain > 0:
                        out.append(Task(2, gain * price, ((x, y),), ["WATER"], key=(x, y, "W")))
                elif critical:
                    remain = self._plant_future_value(crop, cd, age, yu, price)
                    out.append(Task(0, remain, ((x, y),), ["WATER"], key=(x, y, "W")))
                elif gain > 0:
                    out.append(Task(2, gain * price, ((x, y),), ["WATER"], key=(x, y, "W")))
                else:
                    out.append(Task(4, 1.0, ((x, y),), ["WATER"], key=(x, y, "W")))

            if yu > 0 and age >= cd["first_yield_day"]:
                ready = (
                    cd["ongoing"] or yu >= cd["max_yield"] or age >= cd["max_yield_day"] or final
                )
                if ready:
                    tier = 1 if final else (6 if self.projected_shed >= SHED_CAPACITY - 4 else 2)
                    out.append(
                        Task(
                            tier,
                            yu * price,
                            ((x, y),),
                            ["HARVEST"],
                            key=(x, y, "H"),
                            exclusive_tile=None if cd["ongoing"] else (x, y),
                        )
                    )

        if self.p.use_fertilizer:
            out.extend(self._fertilize_tasks())

        # --- 動物 ---
        if self.p.use_animals:
            out.extend(self._animal_tasks())

        # --- 雜草 ---
        want_tiles = self.investing  # 清雜草只為了騰出可播種的格子
        if want_tiles:
            for x, y in self.weeds:
                out.append(
                    Task(5, 20.0, ((x, y),), ["DIG"], key=(x, y, "D"), exclusive_tile=(x, y))
                )

        # --- 播種 / 建牧場 ---
        out.extend(self._build_and_plant_tasks())

        # --- 倉庫物流 ---
        out.extend(self._logistics_tasks())

        _ = hour
        return out

    def _water_gain(self, t, cd, age, day):
        fert = int(t.get("fertilized_until_day", -1)) >= day
        if cd["ongoing"]:
            if not fert:
                return 0
            ds = day + 1 - int(t["planted_day"]) - cd["first_yield_day"]
            if ds >= 0 and cd["interval"] > 0 and ds % cd["interval"] == 0:
                if ds // cd["interval"] + 1 <= cd["max_yield"]:
                    return 1
            return 0
        ws = (cd["max_yield_day"] + 1) // 2
        if ws <= age <= cd["max_yield_day"] and int(t.get("yield_units", 0)) < cd["max_yield"]:
            return 2 if fert else 1
        return 0

    def _plant_future_value(self, crop, cd, age, yu, price):
        if cd["ongoing"]:
            return max(1.0, cd["max_yield"] * price * 0.5 + yu * price)
        remaining_age = max(0, HARVEST_AGE[crop] - age)
        if self.day + remaining_age > FINAL_DAY:
            return yu * price
        return max(1.0, HARVEST_UNITS[crop] * price)

    def _fertilize_tasks(self):
        out = []
        fert_price = self.price_of("FERTILIZER")
        carried = sum(d.get("FERTILIZER", 0) for d in self.inv)
        if carried <= 0:
            return out
        day = self.day
        for x, y, t in self.plants:
            if int(t.get("fertilized_until_day", -1)) >= day:
                continue
            crop = t["crop"]
            cd = CROPS[crop]
            age = day - int(t["planted_day"])
            price = self.price_of(crop)
            if cd["ongoing"]:
                # 覆蓋接下來 3 天內的產出日
                gain = 0
                for d in range(day, day + 3):
                    ds = d + 1 - int(t["planted_day"]) - cd["first_yield_day"]
                    if ds >= 0 and cd["interval"] > 0 and ds % cd["interval"] == 0:
                        if ds // cd["interval"] + 1 <= cd["max_yield"] and d + 1 <= FINAL_DAY:
                            gain += 1
            else:
                ws = (cd["max_yield_day"] + 1) // 2
                head = max(0, int(t.get("yield_units", 0)))
                room = cd["max_yield"] - head
                planted = int(t["planted_day"])
                days = [
                    d for d in range(day, day + 3) if ws <= (d - planted) <= cd["max_yield_day"]
                ]
                gain = min(room, len(days))
                if self.day + max(0, HARVEST_AGE[crop] - age) > FINAL_DAY:
                    gain = 0
            value = gain * price
            if value > fert_price * 1.15 and gain > 0:
                out.append(
                    Task(
                        3,
                        value,
                        ((x, y),),
                        ["FERTILIZE"],
                        key=(x, y, "F"),
                        need=("FERTILIZER", 1),
                    )
                )
        return out

    def _animal_tasks(self):
        out = []
        day = self.day
        final = day >= FINAL_DAY
        unfed_critical = 0
        unfed_total = 0
        for x, y, t in self.animals:
            a = ANIMALS[t["animal"]]
            price = self.price_of(a["product"])
            yu = int(t.get("yield_units", 0))
            fed = bool(t.get("fed_today", False))
            if not fed and not final:
                # 最後一天沒有跨日結算：既不會產出也不會逃走，餵食純屬浪費。
                unfed_total += 1
                if int(t.get("consecutive_unfed", 0)) >= 1:
                    unfed_critical += 1
                    out.append(
                        Task(0, 6 * price, ((x, y),), ["FEED"], key=(x, y, "FD"), need=("WHEAT", 1))
                    )
                else:
                    out.append(
                        Task(1, 2 * price, ((x, y),), ["FEED"], key=(x, y, "FD"), need=("WHEAT", 1))
                    )
            if yu > 0:
                tier = 1 if (final or yu >= a["max_held"] - 1) else 3
                if not final and self.projected_shed >= SHED_CAPACITY - 4:
                    tier = 6
                out.append(Task(tier, yu * price, ((x, y),), ["HARVEST"], key=(x, y, "H")))
            if not final and not t.get("cared_today", False):
                out.append(Task(3, price * 0.9, ((x, y),), ["CARE"], key=(x, y, "C")))
            if t.get("fertilizer_available", False):
                out.append(
                    Task(
                        4,
                        self.price_of("FERTILIZER"),
                        ((x, y),),
                        ["COLLECT_FERTILIZER"],
                        key=(x, y, "CF"),
                    )
                )

        # 放置動物
        shed_animals = [a for a in ANIMALS if self.shed.get(a, 0) > 0]
        carried_animals = {}
        for d in self.inv:
            for a in ANIMALS:
                if d.get(a, 0) > 0:
                    carried_animals[a] = carried_animals.get(a, 0) + d[a]
        structs_by_kind = {}
        for x, y, t in self.free_structs:
            structs_by_kind.setdefault(t.get("kind"), []).append((x, y))
        if self.free_structs and not final:
            for a, n in carried_animals.items():
                spots = structs_by_kind.get(ANIMALS[a]["structure"], [])
                for j in range(min(n, len(spots))):
                    x, y = spots[j]
                    out.append(
                        Task(
                            2,
                            ANIMALS[a]["cost"],
                            ((x, y),),
                            ["PLACE", a],
                            key=(x, y, "P"),
                            exclusive_tile=(x, y),
                            need=(a, 1),
                        )
                    )
            # 從倉庫領動物：每個空結構最多對應一次領取
            for a in shed_animals:
                kind = ANIMALS[a]["structure"]
                spots = len(structs_by_kind.get(kind, []))
                have = carried_animals.get(a, 0)
                want = min(self.shed.get(a, 0), max(0, spots - have))
                for _k in range(want):
                    out.append(
                        Task(
                            2,
                            ANIMALS[a]["cost"] * 0.9,
                            self.shed_tiles,
                            ["PICKUP", a, 1],
                            key=None,
                        )
                    )

        # 領小麥去餵
        if unfed_total > 0 and not final:
            carried_wheat = sum(d.get("WHEAT", 0) for d in self.inv)
            deficit = unfed_total - carried_wheat
            if deficit > 0 and self.shed.get("WHEAT", 0) > 0:
                carry = max(1, self.p.wheat_carry)
                n = int(math.ceil(deficit / float(carry)))
                n = min(n, self.n_units, max(1, self.shed.get("WHEAT", 0) // carry + 1))
                tier = 0 if unfed_critical > carried_wheat else 1
                take = min(carry, self.shed.get("WHEAT", 0))
                for _k in range(n):
                    out.append(
                        Task(tier, 500.0, self.shed_tiles, ["PICKUP", "WHEAT", take], key=None)
                    )
        return out

    def _build_and_plant_tasks(self):
        out = []
        if not self.investing or self.hour > 20:
            return out
        if not self.empty:
            return out

        empties = sorted(self.empty, key=lambda p: _dist(p, self.shed_center))

        # 動物結構優先占用離倉庫最近的格子。`_structs_needed()` 在 opening_tempo
        # 之下是 JIT 的（只為已持有／這回合買得起的動物），否則維持舊的預建行為。
        n_struct_needed = self._structs_needed()
        # 同一回合的建造格與播種格互斥：先把最近的格子切給結構，剩下的才播種
        # （另有 exclusive_tile 在 _assign 層再擋一次）。
        struct_sites = empties[:n_struct_needed]
        for x, y in struct_sites:
            out.append(
                Task(4, 300.0, ((x, y),), ["BUILD_PASTURE"], key=(x, y, "B"), exclusive_tile=(x, y))
            )

        # 播種格從「保留給 JIT 結構的格子」之後開始。保留量 >= 本回合要蓋的數量，
        # 因為已買到但還沒蓋得完的動物也需要格子。
        reserved = max(n_struct_needed, self._jit_tile_reserve())
        plant_sites = empties[reserved:]
        crop = self._choose_crop()
        if crop is None:
            return out
        crop = self._opening_crop(crop, planting=True)  # d0–d2 開局配比（一般評分不動）
        available = self.seeds.get(crop, 0)
        if available <= 0:
            return out
        if self.load + WORK_PER_TILE > self.capacity:
            return out
        # 每格新作物當天必須澆水；播種任務的下一回合會以 tier 0 出現在同一格
        # （consecutive_unwatered = 1），因此只要留下至少一個可用回合即可保證。
        room = max(0, int((self.capacity - self.load) // WORK_PER_TILE))
        # 播種是 tier 3、建造是 tier 4，同回合出現時單位一定先被播種搶走。JIT 結構
        # 因此必須連「單位額度」一起保留，否則結構永遠排不到人，動物管線會卡死
        # （實測：不保留時 day 7 仍只有 2 隻動物）。這裡只限制本回合送出的播種
        # 任務數，沒有動到任何 tier 或 _assign。
        units_for_plants = self.n_units
        if n_struct_needed > 0:
            units_for_plants = max(1, self.n_units - n_struct_needed)
        n = min(available, len(plant_sites), room, units_for_plants)
        price = self.projected_price(crop)
        for k in range(n):
            x, y = plant_sites[k]
            out.append(
                Task(
                    3,
                    price * 2.0,
                    ((x, y),),
                    ["PLANT", crop],
                    key=(x, y, "PL"),
                    exclusive_tile=(x, y),
                )
            )
        return out

    def _logistics_tasks(self):
        out = []
        final = self.day >= FINAL_DAY
        overflow = self.pressure > 0.70
        for i in range(self.n_units):
            d = self.inv[i]
            if not d:
                continue
            total = sum(d.values())
            keep_wheat = self.p.use_animals and not final
            sellable = {
                k: v
                for k, v in d.items()
                if v > 0 and k in PRODUCTS and not (keep_wheat and k == "WHEAT")
            }
            if not sellable:
                continue
            dist_shed = min(_dist(self.pos[i], s) for s in self.shed_tiles)
            must_return = final and (self.hour + dist_shed >= LAST_HOUR - 1)
            if must_return:
                tier = 0
            elif overflow:
                tier = 1
            elif total >= 12:
                tier = 3
            else:
                continue
            if keep_wheat:
                item = max(sellable, key=lambda k: sellable[k] * self.price_of(k))
                act = ["PLACE", item, sellable[item]]
            else:
                act = ["DROP"]
            value = sum(v * self.price_of(k) for k, v in sellable.items())
            out.append(Task(tier, value, self.shed_tiles, act, key=("unit", i), only_unit=i))
        return out

    # ---------------- 指派 ---------------- #

    def _assign(self, tasks):
        """距離折扣的全域貪婪配對。

        先前版本依 (tier, value) 排序後把最近的單位配給最高價值任務，導致單位
        每回合被不同任務拉走，實測 65-70% 的動作都是 MOVE。改成對每組
        (unit, task) 計分 `tier_weight * value / (1 + 2*dist)`，tier 仍然主導，
        但同 tier 內距離會決定歸屬，因此單位會自然把附近的工作做完再移動。
        """
        n = self.n_units
        actions = [None] * n
        free = list(range(n))
        used_keys = set()
        locked_tiles = set()
        plant_budget = dict(self.seeds)

        tasks.sort(key=lambda t: (t.tier, -t.value))
        del tasks[MAX_TASKS:]

        while free and tasks:
            best_score = 0.0
            best = None
            for ti, task in enumerate(tasks):
                if task.key is not None and task.key in used_keys:
                    continue
                if task.exclusive_tile is not None and task.exclusive_tile in locked_tiles:
                    continue
                if task.action[0] == "PLANT" and plant_budget.get(task.action[1], 0) <= 0:
                    continue
                w = _TIER_W[task.tier] * (task.value + 10.0)
                need = task.need
                only = task.only_unit
                for u in free:
                    if only is not None and u != only:
                        continue
                    if need is not None and self.inv[u].get(need[0], 0) < need[1]:
                        continue
                    pu = self.pos[u]
                    d = task.targets[0]
                    dd = _dist(pu, d)
                    for tg in task.targets[1:]:
                        v = _dist(pu, tg)
                        if v < dd:
                            dd = v
                            d = tg
                    if self.hour + dd > LAST_HOUR:
                        continue
                    score = w / (1.0 + 2.0 * dd)
                    if score > best_score:
                        best_score = score
                        best = (ti, u, dd, d)
            if best is None:
                break
            ti, u, dd, tgt = best
            task = tasks[ti]
            if dd == 0:
                actions[u] = list(task.action)
                if task.action[0] == "PLANT":
                    plant_budget[task.action[1]] = plant_budget.get(task.action[1], 0) - 1
            else:
                actions[u] = _step_toward(self.pos[u], tgt)
            free.remove(u)
            if task.key is not None:
                used_keys.add(task.key)
            if task.exclusive_tile is not None:
                locked_tiles.add(task.exclusive_tile)
            tasks.pop(ti)

        for u in range(n):
            if actions[u] is None:
                actions[u] = self._idle_action(u)
        return actions

    def _idle_action(self, u):
        """沒有任務時直接 PASS。

        日終引擎會自動把所有 inventory 倒回 shed，所以「為了卸貨而走回倉庫」
        是純浪費；真正需要提前入庫的情況（溢出、最後一天）由 _logistics_tasks
        以任務形式產生，會走正常的指派流程。
        """
        return ["PASS"]

    # ---------------- 市場 ---------------- #

    def _market(self):
        orders = []
        day = self.day
        hour = self.hour

        # 1) 雇工（越早越好，hands 於下一回合才會出現）
        #
        # HIRE 不受 day 26 投資閘門限制：工人當日雇、當日消失、不形成任何資產，
        # 屬於計畫中「終局只保留能在 day 29 結束前兌現的餵食／維護」的營運成本。
        # day 26 起 _desired_hands 已不再把空地與雜草計入 workable，因此人力
        # 只會覆蓋既有作物／動物的維護與最後一天的搬運清倉，無法用來擴產。
        n_hire, hire_spend = self._hire_plan()
        for _k in range(n_hire):
            orders.append(["HIRE"])
        self.spend += hire_spend
        _ = (day, hour)

        # 2) 賣出
        orders.extend(self._sell_orders(MAX_MARKET_ORDERS - len(orders)))

        # 3) 買種子 / 飼料 / 動物 / 土地
        if len(orders) < MAX_MARKET_ORDERS and self.investing:
            orders.extend(self._buy_orders(MAX_MARKET_ORDERS - len(orders)))

        return orders[:MAX_MARKET_ORDERS]

    def _sell_orders(self, slots):
        if slots <= 0:
            return []
        out = []
        day = self.day
        urgent = day >= FINAL_DAY or self.pressure >= 0.88
        if urgent:
            ratio = 0.0
        else:
            ratio = self.p.floor_ratio(day)
            # 倉容是硬約束：壓力從 0.55 升到 0.88 時，價格底線線性收斂到 0，
            # 讓「賣不掉就爛在倉庫」永遠不會發生。
            if self.pressure > 0.55:
                ratio *= max(0.0, (0.88 - self.pressure) / 0.33)
        items = sorted(
            (k for k in PRODUCTS if self.shed.get(k, 0) > 0),
            key=lambda k: -self.shed[k] * self.price_of(k),
        )
        for item in items:
            if len(out) >= slots:
                break
            have = self.shed.get(item, 0)
            if item == "WHEAT" and self.p.use_animals and day <= LAST_FEED_DAY:
                # 賣與買共用 wheat_target()：賣 = have - target、買 = target - have，
                # 因此同一回合結構上不可能同時出現 SELL WHEAT 與 BUY_PRODUCT WHEAT。
                # 線上實測舊版有 117 個回合同時買賣小麥（對手 1 個），白白吃掉
                # 25% 的市場訂單 slot。這裡刻意沒有倉壓例外——加了例外對沖就復發。
                have -= self.wheat_target()
            if have <= 0:
                continue
            floor = max(PRICE_FLOOR, MARKET_PARAMS[item]["base"] * ratio)
            inv = self.m_inv.get(item, MARKET_I0)
            n = 0
            while n < have:
                if market_price(item, inv + n) < floor:
                    break
                n += 1
            if n <= 0:
                continue
            out.append(["SELL", item, n])
        return out

    def _buy_orders(self, slots):
        out = []
        if slots <= 0:
            return []
        cash = self.money - self.spend

        # 飼料小麥（challenger）
        if len(out) < slots:
            feed = self._feed_order(cash)
            if feed is not None:
                out.append(feed[0])
                cash -= feed[1]

        # 動物。現金基準用 `_cash_for_animals()`（= money − HIRE − 飼料）：JIT 建結構
        # 在 `_tasks()` 階段用的是同一個純函數，所以「蓋了結構就一定買得起這隻動物」
        # 由建構方式保證。動物一律排在 BUY_SEED 之前——引擎按順序逐單結算，排序
        # 本身就是資金分配。
        for a, n in self._animal_orders(self._cash_for_animals()):
            if len(out) >= slots:
                break
            out.append(["BUY_ANIMAL", a, n])
            cash -= n * ANIMALS[a]["cost"]

        # 種子
        if len(out) < slots:
            crop = self._choose_crop()
            if crop is not None:
                crop = self._opening_crop(crop)  # d0–d2 開局配比（一般評分不動）
                have = self.seeds.get(crop, 0)
                want_sites = len(self.empty)
                if self.p.use_animals:
                    if self.p.opening_tempo:
                        # 只扣掉「已持有待放置 ＋ 一個近期批次」真正需要的格子。
                        # 舊版扣掉整個 `_animals_pending()`（最終 target 的缺口），
                        # 等於把未來十隻動物的格子永久保留成空地，開局播種因此
                        # 被封在 13 格。
                        want_sites = max(0, want_sites - self._jit_tile_reserve())
                    else:
                        want_sites = max(0, want_sites - self._animals_pending())
                room = max(0, int((self.capacity - self.load) // WORK_PER_TILE))
                want = min(want_sites, room, 25) - have
                cost = CROPS[crop]["seed"]
                n = int(min(want, max(0.0, cash - self._seed_reserve()) // cost))
                if n > 0:
                    out.append(["BUY_SEED", crop, n])
                    cash -= n * cost

        # 土地
        if len(out) < slots:
            idx = self.n_quadrants - 1
            for want_idx, mult, last_day in self.p.land_plan:
                if want_idx != idx:
                    continue
                if self.day > last_day:
                    break
                price = LAND_PRICES[idx]
                if self.load + WORK_PER_TILE * 12 > self.capacity:
                    break
                if self.p.opening_tempo:
                    # backlog + 可負擔。舊版的「`empty + weeds > 8` 就不買」在只有
                    # Q1 的開局永遠成立，加上現金倍數閘（2.0/2.5），Q2 被拖到 d9–d10。
                    # 這裡不寫任何「第幾天無條件買」，購買時點由 backlog 與現金決定；
                    # land_plan 的 last_day 上限保留。
                    if self._land_backlog() <= 0:
                        break
                    if cash >= price + self.p.land_cash_buffer:
                        out.append(["BUY_LAND"])
                        cash -= price
                    break
                # 只有在現有土地快用完、且還有工時可吸收新格子時才擴張
                if len(self.empty) + len(self.weeds) > 8:
                    break
                if cash >= price * mult:
                    out.append(["BUY_LAND"])
                    cash -= price
                break

        return out[:slots]

    # ---------------- 主流程 ---------------- #

    def run(self):
        tasks = self._tasks()
        actions = self._assign(tasks)
        market = self._market()
        return {"farmer": actions[0], "hands": actions[1:], "market": market}


# 一次性作物在完美澆水下的實際收量（yield_units 從 1 起算，窗內每天 +1）。
HARVEST_UNITS = {}
for _c, _d in CROPS.items():
    if _d["ongoing"]:
        HARVEST_UNITS[_c] = _d["max_yield"]
    else:
        _ws = (_d["max_yield_day"] + 1) // 2
        _days = _d["max_yield_day"] - _ws + 1
        HARVEST_UNITS[_c] = min(_d["max_yield"], 1 + _days)


_SAFE = {"farmer": ["PASS"], "hands": [], "market": []}


def plan(obs, profile):
    """回傳一個合法的動作 dict；任何例外都退回安全的 PASS。"""
    try:
        return _Turn(obs, profile).run()
    except Exception:
        try:
            n = len(obs["farms"][obs["player"]]["hands"])
        except Exception:
            n = 0
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n, "market": []}
