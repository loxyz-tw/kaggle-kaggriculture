"""Day 0 回歸測試。

分四類：
1. 引擎常數 / 定價鏡像是否與 kaggle_environments 一致（規則漂移偵測）。
2. planner 的合約：無狀態、動作合法、市場訂單不超量、例外有安全退路。
3. 存活不變式：新播種當天必澆、動物不因缺糧逃走、shed 不溢出。
4. 打包契約：單檔產物最後一個 callable 是 agent、tar 根目錄只有 main.py。
"""

import subprocess
import sys
import tarfile
import tempfile
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.kaggriculture import kaggriculture as ENG  # noqa: E402

import agent_core as AC  # noqa: E402

VALID_UNIT_OPS = {
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
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
}
VALID_MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}


def _raw_turn(profile, errors):
    """直接呼叫 _Turn.run()（不經過 plan() 的 try/except），把例外記下來。"""

    def raw(obs):
        try:
            return AC._Turn(obs, profile).run()
        except Exception:  # noqa: BLE001
            if not errors:
                errors.append(traceback.format_exc())
            return {"farmer": ["PASS"], "hands": [], "market": []}

    return raw


def make_probe(fn, inspect):
    """包一層探針，逐回合檢查 agent 的輸出。

    **必須用閉包，不能用預設參數**：`Agent.act` 會依 `__code__.co_argcount`
    把 [observation, configuration] 截斷後傳入，所以
    `def probe(obs, _fn=fn)` 的 `_fn` 會被 configuration 蓋掉，probe 直接拋例外、
    整場 0 次成功呼叫，而 status 仍是 DONE——測試看起來過了，其實什麼都沒驗。
    （本專案真的踩過：好幾個 full-episode 測試曾經是空轉的。）
    """
    calls = []

    def probe(obs):
        act = fn(obs)
        inspect(obs, act)
        calls.append(1)
        return act

    probe.calls = calls
    return probe


# --------------------------------------------------------------------------- #


class TestEngineMirror(unittest.TestCase):
    """agent_core 內鏡像的引擎常數必須與實際引擎一致。"""

    def test_crops(self):
        self.assertEqual(AC.CROPS, ENG.CROPS)

    def test_animals(self):
        self.assertEqual(AC.ANIMALS, ENG.ANIMALS)

    def test_products(self):
        self.assertEqual(AC.PRODUCTS, ENG.PRODUCTS)

    def test_land_prices(self):
        self.assertEqual(list(AC.LAND_PRICES), ENG.LAND_PRICES)

    def test_scalars(self):
        self.assertEqual(AC.MARKET_I0, ENG.MARKET_I0)
        self.assertEqual(AC.PRICE_FLOOR, ENG.PRICE_FLOOR)
        self.assertEqual(AC.HINGE_GAIN, ENG.HINGE_GAIN)

    def test_market_params_subset(self):
        for item, p in AC.MARKET_PARAMS.items():
            for k, v in p.items():
                self.assertEqual(v, ENG.MARKET_PARAMS[item][k], f"{item}.{k}")

    def test_market_price_matches_engine(self):
        for item in AC.PRODUCTS:
            for inv in (0, 5000, 9500, 9900, 10000, 10100, 10400, 11000, 20000):
                self.assertEqual(
                    AC.market_price(item, inv),
                    ENG.market_price(item, inv),
                    f"{item}@{inv}",
                )

    def test_hire_cost_matches_engine(self):
        for n in range(15):
            self.assertEqual(AC.hire_cost(n), ENG._hire_cost(n), f"hire {n}")

    def test_harvest_units_match_simulated_watering(self):
        """HARVEST_UNITS 必須等於「每天澆水直到 max_yield_day」實際拿到的收量。"""
        for crop, cd in AC.CROPS.items():
            if cd["ongoing"]:
                continue
            tile = ENG._new_plant(crop, 0, 24)
            for day in range(AC.HARVEST_AGE[crop] + 1):
                ws = (cd["max_yield_day"] + 1) // 2
                if ws <= day <= cd["max_yield_day"]:
                    tile["yield_units"] = min(cd["max_yield"], tile["yield_units"] + 1)
            self.assertEqual(tile["yield_units"], AC.HARVEST_UNITS[crop], crop)

    def test_final_day_matches_episode_length(self):
        """FINAL_DAY / LAST_HOUR 必須對得上 720 步真正的最後一個可下動作回合。"""
        seen = []

        def probe(obs):
            seen.append((obs["day"], obs["hour"]))
            return {"farmer": ["PASS"], "hands": [], "market": []}

        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1}, debug=False)
        env.run([probe, "pass"])
        self.assertEqual(seen[-1], (AC.FINAL_DAY, AC.LAST_HOUR))


# --------------------------------------------------------------------------- #


def _first_obs(seed=0):
    """取得一個真實的初始 observation（Struct 形態，與線上一致）。"""
    grabbed = {}

    def probe(obs):
        if "obs" not in grabbed:
            grabbed["obs"] = obs
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": seed}, debug=False)
    env.run([probe, "pass"])
    return grabbed["obs"]


class TestPlannerContract(unittest.TestCase):
    def test_stateless_same_obs_same_action(self):
        obs = _first_obs()
        for profile in (AC.ANCHOR, AC.CHALLENGER):
            a = AC.plan(obs, profile)
            b = AC.plan(obs, profile)
            self.assertEqual(a, b, profile.name)

    def test_profiles_differ(self):
        """兩個 profile 必須是不同的策略，不能只是換個名字。"""
        a, c = AC.ANCHOR, AC.CHALLENGER
        self.assertNotEqual(a.name, c.name)
        self.assertFalse(a.use_animals)
        self.assertTrue(c.use_animals)
        self.assertFalse(a.use_fertilizer)
        self.assertTrue(c.use_fertilizer)
        self.assertNotEqual(set(a.crops), set(c.crops))
        self.assertEqual(a.animal_targets, ())
        self.assertTrue(sum(n for _x, n in c.animal_targets) >= 10)

    def test_malformed_obs_falls_back_to_pass(self):
        for bad in ({}, {"player": 0}, {"player": 5, "farms": []}, None):
            out = AC.plan(bad, AC.ANCHOR)
            self.assertEqual(out["farmer"], ["PASS"])
            self.assertEqual(out["market"], [])

    def test_actions_are_well_formed_over_full_episode(self):
        """跑滿一場，逐回合檢查送出的每個動作與訂單都合法且數量在上限內。"""
        for path, label in ((ROOT / "main.py", "anchor"), (ROOT / "challenger.py", "challenger")):
            with self.subTest(label):
                fn = _load(path)
                problems = []

                def inspect(obs, act, problems=problems):
                    n_hands = len(obs["farms"][obs["player"]]["hands"])
                    if not isinstance(act, dict):
                        problems.append(("not a dict", act))
                        return
                    if set(act) != {"farmer", "hands", "market"}:
                        problems.append(("bad keys", sorted(act)))
                    if len(act["hands"]) != n_hands:
                        problems.append(("hands length", len(act["hands"]), n_hands))
                    for a in [act["farmer"]] + list(act["hands"]):
                        if not isinstance(a, list) or not a or a[0] not in VALID_UNIT_OPS:
                            problems.append(("bad unit action", a))
                    if len(act["market"]) > AC.MAX_MARKET_ORDERS:
                        problems.append(("too many orders", len(act["market"])))
                    for o in act["market"]:
                        if not isinstance(o, list) or not o or o[0] not in VALID_MARKET_OPS:
                            problems.append(("bad order", o))
                        elif o[0] in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
                            if len(o) != 3 or not isinstance(o[2], int) or o[2] <= 0:
                                problems.append(("bad order qty", o))

                probe = make_probe(fn, inspect)
                cfg = {"episodeSteps": 720, "seed": 3}
                env = make("kaggriculture", configuration=cfg, debug=False)
                env.run([probe, "starter"])
                self.assertEqual(len(probe.calls), 719, "探針沒有被實際呼叫滿全場")
                self.assertEqual(problems[:5], [], f"{label}: {problems[:5]}")
                self.assertEqual([s.status for s in env.steps[-1]], ["DONE", "DONE"])

    def test_never_requests_more_plants_than_seeds(self):
        """引擎對超額 PLANT 會整批丟棄同作物的所有請求，必須永不觸發。"""
        for path, label in ((ROOT / "main.py", "anchor"), (ROOT / "challenger.py", "challenger")):
            with self.subTest(label):
                fn = _load(path)
                bad = []

                def inspect(obs, act, bad=bad):
                    seeds = obs["private"]["seeds"]
                    want = {}
                    for a in [act["farmer"]] + list(act["hands"]):
                        if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
                            want[a[1]] = want.get(a[1], 0) + 1
                    for crop, n in want.items():
                        if n > seeds.get(crop, 0):
                            bad.append((obs["day"], obs["hour"], crop, n, seeds.get(crop, 0)))

                probe = make_probe(fn, inspect)
                cfg = {"episodeSteps": 720, "seed": 4}
                env = make("kaggriculture", configuration=cfg, debug=False)
                env.run([probe, "starter"])
                self.assertEqual(len(probe.calls), 719, "探針沒有被實際呼叫滿全場")
                self.assertEqual(bad[:5], [], label)


# --------------------------------------------------------------------------- #


def _fake_obs(day, hour=0, n_empty=25, money=20000.0, player=0, n_animals=0, shed=None):
    """組一個最小但合法的 observation，用來單獨測邊界條件。

    n_empty 以外的格子維持 LOCKED（而非雜草），因為 _desired_hands 把雜草也算成
    可整理的工時；用 LOCKED 才能乾淨地只變動「空地數」這一個變因。
    n_animals 會先佔用格子放上已餵過的乳牛（不產生任何待辦任務）。
    """
    board = 10
    tiles = [["LOCKED"] * board for _ in range(board)]
    cells = [(x, y) for y in range(5) for x in range(5)]
    for x, y in cells[:n_animals]:
        tiles[y][x] = {
            "kind": "PASTURE",
            "animal": "COW",
            "placed_day": 0,
            "yield_units": 0,
            "consecutive_unfed": 0,
            "fed_today": True,
            "cared_today": True,
            "fertilizer_available": False,
            "pending_care_bonus": 0,
        }
    for x, y in cells[n_animals : n_animals + n_empty]:
        tiles[y][x] = None
    return {
        "player": player,
        "day": day,
        "hour": hour,
        "step": day * 24 + hour,
        "farms": [
            {
                "money": money,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            }
            for _ in range(2)
        ],
        "private": {
            "shed": dict(shed or {}),
            "seeds": {c: 20 for c in AC.CROPS},
            "inventories": [{}],
        },
        "market": {
            "inventory": {p: AC.MARKET_I0 for p in AC.PRODUCTS},
            "prices": {p: AC.MARKET_PARAMS[p]["base"] for p in AC.PRODUCTS},
        },
        "town": {"unlocked_shops": []},
    }


INVEST_ORDERS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND"}
INVEST_ACTIONS = {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}


class TestHardInvestStopDay(unittest.TestCase):
    """批准計畫：obs["day"] >= 26 起停止一切新投資（day 26 當天就要停）。"""

    def test_boundary_is_inclusive_of_day_26(self):
        self.assertEqual(AC.HARD_INVEST_STOP_DAY, 26)
        for profile in (AC.ANCHOR, AC.CHALLENGER):
            for day in (24, 25):
                act = AC.plan(_fake_obs(day), profile)
                self.assertTrue(
                    AC._Turn(_fake_obs(day), profile).investing,
                    f"{profile.name} day={day} 應仍在投資期",
                )
                self.assertIsInstance(act, dict)
            for day in (26, 27, 28, 29):
                t = AC._Turn(_fake_obs(day), profile)
                self.assertFalse(t.investing, f"{profile.name} day={day} 應已停止投資")

    def test_no_investment_on_and_after_day_26_synthetic(self):
        """直接以合成狀態逼出投資機會：有錢、有種子、25 格全空。"""
        for profile in (AC.ANCHOR, AC.CHALLENGER):
            for day in (26, 27, 28, 29):
                for hour in (0, 1, 5, 12):
                    obs = _fake_obs(day, hour)
                    act = AC.plan(obs, profile)
                    bad_o = [o for o in act["market"] if o[0] in INVEST_ORDERS]
                    bad_a = [
                        a for a in [act["farmer"]] + list(act["hands"]) if a[0] in INVEST_ACTIONS
                    ]
                    self.assertEqual(bad_o, [], f"{profile.name} day={day} h={hour}")
                    self.assertEqual(bad_a, [], f"{profile.name} day={day} h={hour}")

    def test_day_25_still_invests(self):
        """反向驗證：邊界不是把投資整個關掉，day 25 仍必須會買種／播種。"""
        obs = _fake_obs(25)
        act = AC.plan(obs, AC.ANCHOR)
        got = [o[0] for o in act["market"] if o[0] in INVEST_ORDERS]
        self.assertIn("BUY_SEED", got, f"day 25 應仍會買種，實際 market={act['market']}")

    def test_no_investment_on_and_after_day_26_full_episodes(self):
        """整場對局逐回合檢查，涵蓋真實狀態分布。"""
        for path, label in ((ROOT / "main.py", "anchor"), (ROOT / "challenger.py", "challenger")):
            for seed in (0, 6):
                with self.subTest(label=label, seed=seed):
                    fn = _load(path)
                    bad = []

                    def inspect(obs, act, bad=bad):
                        if obs["day"] >= AC.HARD_INVEST_STOP_DAY:
                            for o in act["market"]:
                                if o[0] in INVEST_ORDERS:
                                    bad.append((obs["day"], obs["hour"], o))
                            for a in [act["farmer"]] + list(act["hands"]):
                                if a[0] in INVEST_ACTIONS:
                                    bad.append((obs["day"], obs["hour"], a))

                    probe = make_probe(fn, inspect)
                    cfg = {"episodeSteps": 720, "seed": seed}
                    env = make("kaggriculture", configuration=cfg, debug=False)
                    env.run([probe, "starter"])
                    self.assertEqual(len(probe.calls), 719, "探針沒有被實際呼叫滿全場")
                    self.assertEqual(bad[:5], [], f"{label} seed={seed}")

    def test_hire_after_day_26_is_maintenance_only(self):
        """day 26 起仍可雇工（清算與維護的必要營運成本），但人力不得隨空地擴張。

        判準：把空地數從 0 提高到 25，day 25 需要的工人數必須上升（要播種），
        day 26 之後則必須完全不受空地影響——這就是「只為清算、不得擴產」。
        """
        for profile in (AC.ANCHOR, AC.CHALLENGER):
            d25_few = AC._Turn(_fake_obs(25, n_empty=0), profile).desired_hands
            d25_many = AC._Turn(_fake_obs(25, n_empty=25), profile).desired_hands
            self.assertGreater(d25_many, d25_few, f"{profile.name}: day 25 應因空地而增聘")

            d26_few = AC._Turn(_fake_obs(26, n_empty=0), profile).desired_hands
            d26_many = AC._Turn(_fake_obs(26, n_empty=25), profile).desired_hands
            self.assertEqual(d26_many, d26_few, f"{profile.name}: day 26 的人力不得受空地影響")

    def test_feed_reserve_covers_days_26_to_28(self):
        """day 26 起不能再買飼料，保留量必須正好覆蓋剩餘餵食天數並在 day 29 歸零。

        直接讀合成 observation 上真正算出來的 `t.wheat_reserve`，而不是在測試裡
        重算同一條公式（那樣等於什麼都沒驗到）。
        """
        n = 10
        expected_per_animal = {20: 3, 25: 3, 26: 3, 27: 2, 28: 1, 29: 0}
        for day, per_animal in expected_per_animal.items():
            obs = _fake_obs(day, n_animals=n, n_empty=0, shed={"WHEAT": 60})
            t = AC._Turn(obs, AC.CHALLENGER)
            self.assertEqual(t.n_animals, n, f"day {day}: 合成盤面應有 {n} 隻動物")
            self.assertEqual(t.wheat_reserve, per_animal * n, f"day {day} 的飼料保留量不符")

        # anchor 不養動物，任何一天都不該保留飼料
        for day in (20, 26, 29):
            t = AC._Turn(_fake_obs(day, n_animals=0), AC.ANCHOR)
            self.assertEqual(t.wheat_reserve, 0, f"anchor day {day}")

    def test_feed_reserve_is_not_sold_off_before_last_feed_day(self):
        """day 26-28 之間不得為了紓解倉壓把飼料賣掉（賣了就補不回來）。"""
        n = 10
        for day in (26, 27, 28):
            obs = _fake_obs(day, n_animals=n, n_empty=0, shed={"WHEAT": 95})
            t = AC._Turn(obs, AC.CHALLENGER)
            sold = 0
            for o in t._sell_orders(10):
                if o[1] == "WHEAT":
                    sold = o[2]
            kept = 95 - sold
            self.assertGreaterEqual(
                kept, t.wheat_reserve, f"day {day}: 賣掉後剩 {kept}，低於保留量 {t.wheat_reserve}"
            )

    def test_pending_animal_labour_stops_at_day_26(self):
        """challenger 未達動物 target 時，day 25 會為待買動物增聘，day 26 不得再增聘。"""
        p = AC.CHALLENGER
        target = sum(k for _a, k in p.animal_targets)
        placed = 2  # 遠低於 target，_animals_pending() 必然 > 0
        self.assertGreater(target - placed, 0)

        def hands(day):
            t = AC._Turn(_fake_obs(day, n_animals=placed, n_empty=0), p)
            self.assertGreater(t._animals_pending(), 0, f"day {day} 應仍有動物缺口")
            return t.desired_hands, t

        h25, t25 = hands(25)
        h26, t26 = hands(26)
        self.assertTrue(t25.investing)
        self.assertFalse(t26.investing)
        self.assertGreater(h25, h26, "day 25 應為待買動物預留人力，day 26 起不得再為未來動物增聘")

        # 對照組：把動物補滿到 target，day 25 就不該再有「待買動物」的加成
        full = AC._Turn(_fake_obs(25, n_animals=target, n_empty=0), p)
        self.assertEqual(full._animals_pending(), 0)


# --------------------------------------------------------------------------- #


class TestNoSilentFallback(unittest.TestCase):
    """plan() 的 try/except 不得掩蓋真正的錯誤。

    實際踩過：一次縮排錯誤讓 _animal_tasks 拋 UnboundLocalError，plan() 靜靜地
    退回 PASS，整季照跑完、status 全 DONE，只有銀行從 87k 掉到 4k 才看得出來。
    這個測試直接走 _Turn.run()（沒有 except），確保正常盤面不會有例外。
    """

    def test_turn_run_never_raises_over_full_episode(self):
        for profile in (AC.ANCHOR, AC.CHALLENGER):
            with self.subTest(profile.name):
                errors = []
                raw = _raw_turn(profile, errors)
                probe = make_probe(raw, lambda o, a: None)
                cfg = {"episodeSteps": 720, "seed": 11}
                env = make("kaggriculture", configuration=cfg, debug=False)
                env.run([probe, "starter"])
                self.assertEqual(len(probe.calls), 719, "探針沒有被實際呼叫滿全場")
                self.assertEqual(errors, [], errors[0] if errors else "")

    def test_agent_beats_trivial_floor(self):
        """靜默退回 PASS 的另一道防線：分數不能掉到 starter 等級。"""
        for path in (ROOT / "main.py", ROOT / "challenger.py"):
            with self.subTest(path.name):
                fn = _load(path)
                cfg = {"episodeSteps": 720, "seed": 11}
                env = make("kaggriculture", configuration=cfg, debug=False)
                env.run([fn, "starter"])
                bank = env.steps[-1][0].reward
                self.assertGreater(bank, 20000, f"{path.name} 只有 {bank}，疑似退回 PASS")


class TestDeterminism(unittest.TestCase):
    """指派必須完全由 observation 重建（無跨回合狀態）。"""

    def test_same_obs_gives_bit_identical_action(self):
        for day in (0, 5, 12, 26, 29):
            obs = _fake_obs(day, n_empty=15, n_animals=3)
            for profile in (AC.ANCHOR, AC.CHALLENGER):
                a = AC.plan(obs, profile)
                b = AC.plan(obs, profile)
                self.assertEqual(a, b, f"{profile.name} day={day}")


class TestWheatRoundTrip(unittest.TestCase):
    """R2-5：同品項 buy+sell 對沖必須結構上不可能發生。"""

    def test_sell_and_buy_never_coexist_for_wheat(self):
        for path in (ROOT / "main.py", ROOT / "challenger.py"):
            with self.subTest(path.name):
                fn = _load(path)
                hits = []

                def checked(obs, _fn=fn, _h=hits):
                    act = _fn(obs)
                    sells = {o[1] for o in act["market"] if o[0] == "SELL" and len(o) > 1}
                    buys = {o[1] for o in act["market"] if o[0] == "BUY_PRODUCT" and len(o) > 1}
                    if sells & buys:
                        _h.append((obs["day"], obs["hour"], sorted(sells & buys)))
                    return act

                cfg = {"episodeSteps": 720, "seed": 7}
                env = make("kaggriculture", configuration=cfg, debug=False)
                env.run([checked, "starter"])
                self.assertEqual(hits[:5], [], f"{path.name} 出現同品項對沖")

    def test_wheat_target_drives_both_sides(self):
        obs = _fake_obs(10, n_animals=10, n_empty=0, shed={"WHEAT": 50})
        t = AC._Turn(obs, AC.CHALLENGER)
        target = t.wheat_target()
        self.assertEqual(target, 30)
        sell = [o for o in t._sell_orders(10) if o[1] == "WHEAT"]
        self.assertTrue(sell and sell[0][2] == 50 - target)
        buys = [o for o in t._buy_orders(10) if o[0] == "BUY_PRODUCT"]
        self.assertEqual(buys, [], "倉庫高於 target 時不得買入")


class TestSurvivalInvariants(unittest.TestCase):
    """完整對局後，由 evaluate.analyse 還原的存活事件必須全為零。"""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import evaluate  # noqa: E402

        cls.E = evaluate

    def _check(self, path, seed):
        r = self.E.play(("x", str(path)), ("starter", "starter"), seed, 0)
        r.pop("latency", None)
        ev = r["events"]
        self.assertEqual(r["status"], ["DONE", "DONE"])
        for key in (
            "death_thirst",
            "death_new_plant_unwatered",
            "new_plant_dry_at_eod",
            "animal_escape",
            "shed_overflow_risk",
        ):
            self.assertEqual(ev.get(key, 0), 0, f"{path.name} seed={seed}: {key} -> {ev}")
        # 所有送出的工作動作都必須真的生效（引擎對非法動作是靜默 no-op）
        noops = {k: v for k, v in ev.items() if k.startswith("noop_") or k == "act_no_such_unit"}
        self.assertEqual(noops, {}, f"{path.name} seed={seed}: {noops}")
        return r

    def test_anchor_survival(self):
        self._check(ROOT / "main.py", 5)

    def test_challenger_survival(self):
        r = self._check(ROOT / "challenger.py", 5)
        # challenger 必須真的養出動物，否則等於退化成 anchor
        self.assertGreater(r["events"].get("act_work", 0), 0)


# --------------------------------------------------------------------------- #


def _load(path):
    src = Path(path).read_text(encoding="utf-8")
    env = {}
    sys.path.insert(0, str(Path(path).parent))
    try:
        exec(compile(src, str(path), "exec"), env)
    finally:
        sys.path.pop(0)
    return [v for v in env.values() if callable(v)][-1]


class TestPackagingContract(unittest.TestCase):
    def test_repo_entrypoints_last_callable_is_agent(self):
        for path in (ROOT / "main.py", ROOT / "challenger.py"):
            src = path.read_text(encoding="utf-8")
            env = {}
            sys.path.insert(0, str(ROOT))
            try:
                exec(compile(src, str(path), "exec"), env)
            finally:
                sys.path.pop(0)
            names = [k for k, v in env.items() if callable(v)]
            self.assertEqual(names[-1], "agent", f"{path.name}: {names}")

    def test_entrypoints_handle_kaggle_path(self):
        """進入點必須同時支援 __file__ 目錄與 /kaggle_simulations/agent/。"""
        for path in (ROOT / "main.py", ROOT / "challenger.py"):
            src = path.read_text(encoding="utf-8")
            self.assertIn("/kaggle_simulations/agent", src, path.name)
            self.assertIn('globals().get("__file__"', src, path.name)

    def test_built_artifacts(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import build_submissions as B  # noqa: E402

        manifest = B.build()
        self.assertEqual(set(manifest["artifacts"]), {"day0_anchor", "day0_challenger"})
        for name, art in manifest["artifacts"].items():
            self.assertEqual(art["members"], ["main.py"], name)
            self.assertLess(art["tar_bytes"], 100 * 1024 * 1024, name)
            tar = ROOT / art["tar"]
            with tarfile.open(tar, "r:gz") as tf:
                self.assertEqual(tf.getnames(), ["main.py"])
        # 兩個產物必須真的不同
        a = manifest["artifacts"]["day0_anchor"]["main_py_sha256"]
        c = manifest["artifacts"]["day0_challenger"]["main_py_sha256"]
        self.assertNotEqual(a, c)

    def test_artifact_runs_from_temp_dir_without_repo_on_path(self):
        """解壓到臨時目錄、切換 cwd、把倉庫從 sys.path 移除後仍能跑短局。"""
        sys.path.insert(0, str(ROOT / "scripts"))
        import build_submissions as B  # noqa: E402

        manifest = B.build()
        for name, art in manifest["artifacts"].items():
            with self.subTest(name):
                d = tempfile.mkdtemp(prefix="kagg-test-")
                w = tempfile.mkdtemp(prefix="kagg-testcwd-")
                with tarfile.open(ROOT / art["tar"], "r:gz") as tf:
                    tf.extractall(d)
                code = (
                    "import os,sys\n"
                    f"sys.path=[p for p in sys.path if os.path.abspath(p or '.')!={str(ROOT)!r}]\n"
                    f"src=open({str(Path(d) / 'main.py')!r},encoding='utf-8').read()\n"
                    "env={}\n"
                    f"exec(compile(src,{str(Path(d) / 'main.py')!r},'exec'),env)\n"
                    "cs=[k for k,v in env.items() if callable(v)]\n"
                    "assert cs[-1]=='agent', cs\n"
                    "from kaggle_environments import make\n"
                    "e=make('kaggriculture',configuration={'episodeSteps':120,'seed':2},debug=False)\n"
                    "e.run([env['agent'],'starter'])\n"
                    "print('OK',[s.status for s in e.steps[-1]])\n"
                )
                p = subprocess.run(
                    [sys.executable, "-c", code], cwd=w, capture_output=True, text=True
                )
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
                self.assertIn("OK", p.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
