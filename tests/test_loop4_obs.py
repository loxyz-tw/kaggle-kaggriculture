"""Loop 4 觀測 v2 的反例測試（任務 12 A1）。

核心是三個**閉式外推**必須與 `rl/kernels.py` 的逐日遞推逐格相等：作物剩餘產次、
非 ongoing 作物剩餘澆水日、動物剩餘產次。oracle 是真的把 `kernels._daily_refresh`
一天一天跑到終局並數事件，不是另一份公式。
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rl import caps as caps_mod
from rl import kernels, loop4_obs, rules
from rl import state as rl_state

CAPS = caps_mod.BOUNDED_TRAINING


def _blank_state(batch: int) -> rl_state.SimState:
    return rl_state.reset(batch, CAPS, seed=0)


def _set_step(state, step):
    return state._replace(step=jnp.full_like(state.step, step))


def _plant_tile(crop: int, planted_day: int, yield_units: int = 1) -> np.ndarray:
    row = rl_state.CANONICAL_EMPTY.copy()
    row[rules.C_KIND] = rules.TK_PLANT
    row[rules.C_CROP] = crop
    row[rules.C_PLANTED_DAY] = planted_day
    row[rules.C_YIELD] = yield_units
    row[rules.C_MLS] = (
        -1
        if rules.CROP_ONGOING[crop]
        else (planted_day + rules.CROP_MAX_YIELD_DAY[crop] + 1) * rules.TURNS_PER_DAY
    )
    row[rules.C_FERT_UNTIL] = -1
    row[rules.C_WATERED] = 1
    return row


def _animal_tile(animal: int, placed_day: int) -> np.ndarray:
    row = rl_state.CANONICAL_EMPTY.copy()
    row[rules.C_KIND] = rules.ANIMAL_STRUCTURE_KIND[animal]
    row[rules.C_ANIMAL] = animal
    row[rules.C_PLACED_DAY] = placed_day
    row[rules.C_FED] = 1
    return row


def _oracle_productions(row: np.ndarray, day: int, kind: str) -> int:
    """把 ``_daily_refresh`` 一天一天跑到終局，數 yield 實際增加的次數。

    每天都假設有澆水／餵食（把 ``C_WATERED``／``C_FED`` 設回 1），對上閉式「不枯死／不逃跑」
    的前提；並且**每天把 ``C_YIELD`` 歸零**，模擬「有去收成」。歸零是必要的：動物的
    ``yield_units`` 會被 ``ANIMAL_MAX_HELD`` 夾住（GOOSE 上限 4），不歸零的話 26 次產出只數得到
    4 次。作物側歸零不影響結果——``produces`` 只看 ``dsf``，不看目前的 yield。
    作物不算 WATER 的加成——那是單位相位做的，不在 ``_daily_refresh`` 裡。
    """
    tiles = np.tile(rl_state.CANONICAL_EMPTY, (1, 2, rules.N_TILES, 1)).astype(np.int32)
    tiles[0, 0, 0] = row
    state = _blank_state(1)._replace(tiles=jnp.asarray(tiles, jnp.int32))
    events = 0
    for d in range(day, rules.N_DAYS):
        step = d * rules.TURNS_PER_DAY + rules.TURNS_PER_DAY - 1
        if step > loop4_obs.LAST_EOD_STEP:
            break
        state = _set_step(state, step)
        t = np.array(state.tiles)
        if kind == "crop":
            t[0, 0, 0, rules.C_WATERED] = 1
        else:
            t[0, 0, 0, rules.C_FED] = 1
        t[0, 0, 0, rules.C_YIELD] = 0
        state = state._replace(tiles=jnp.asarray(t))
        state = kernels._daily_refresh(state, jnp.asarray([d], jnp.int32))
        if int(np.asarray(state.tiles)[0, 0, 0, rules.C_YIELD]) > 0:
            events += 1
    return events


@pytest.mark.parametrize("crop", range(rules.N_CROPS))
def test_crop_productions_left_matches_daily_recurrence(crop):
    """ongoing 作物的剩餘產次；非 ongoing 的閉式必須回 0 且 oracle 也真的 0 次。"""
    for planted_day in range(0, rules.N_DAYS):
        for day in range(planted_day, rules.N_DAYS):
            got = int(
                loop4_obs.crop_productions_left(
                    jnp.asarray(crop), jnp.asarray(planted_day), jnp.asarray(day)
                )
            )
            want = _oracle_productions(_plant_tile(crop, planted_day, 0), day, "crop")
            assert got == want, (
                f"crop={rules.CROPS[crop]} planted={planted_day} day={day}: {got} != {want}"
            )


@pytest.mark.parametrize("animal", range(rules.N_ANIMALS))
def test_animal_productions_left_matches_daily_recurrence(animal):
    """動物剩餘產次。yield 會被 ANIMAL_MAX_HELD 夾住，所以 oracle 從 0 起算才數得到每一次。"""
    for placed_day in range(0, rules.N_DAYS):
        for day in range(placed_day, rules.N_DAYS):
            got = int(
                loop4_obs.animal_productions_left(
                    jnp.asarray(animal), jnp.asarray(placed_day), jnp.asarray(day)
                )
            )
            want = _oracle_productions(_animal_tile(animal, placed_day), day, "animal")
            assert got == want, (
                f"animal={rules.ANIMALS[animal]} placed={placed_day} day={day}: {got} != {want}"
            )


def test_animal_days_to_next_agrees_with_productions_left():
    """還有產次時 days_to_next 必須落在 1..剩餘天數；沒有產次時必須是哨兵 N_DAYS。"""
    for animal in range(rules.N_ANIMALS):
        for placed_day in range(rules.N_DAYS):
            for day in range(placed_day, rules.N_DAYS):
                left = int(
                    loop4_obs.animal_productions_left(
                        jnp.asarray(animal), jnp.asarray(placed_day), jnp.asarray(day)
                    )
                )
                nxt = int(
                    loop4_obs.animal_days_to_next(
                        jnp.asarray(animal), jnp.asarray(placed_day), jnp.asarray(day)
                    )
                )
                if left > 0:
                    assert 1 <= nxt <= loop4_obs.LAST_PRODUCTION_ND - day
                else:
                    assert nxt == rules.N_DAYS


def _oracle_water_days(crop: int, planted_day: int, day: int) -> int:
    """逐日數：從 ``day`` 起到 day 29 為止，有幾天澆水會真的加產量。

    條件逐字取自 ``kernels.apply_unit_actions`` 的 ``c_water_bonus``：
    ``ongoing == 0`` 且 ``age ∈ [WINDOW_START, MAX_YIELD_DAY]``。
    """
    if rules.CROP_ONGOING[crop]:
        return 0
    n = 0
    for d in range(day, rules.N_DAYS):
        age = d - planted_day
        if rules.CROP_WATER_WINDOW_START[crop] <= age <= rules.CROP_MAX_YIELD_DAY[crop]:
            n += 1
    return n


@pytest.mark.parametrize("crop", range(rules.N_CROPS))
def test_crop_water_days_left_matches_engine_window(crop):
    for planted_day in range(rules.N_DAYS):
        for day in range(planted_day, rules.N_DAYS):
            got = int(
                loop4_obs.crop_water_days_left(
                    jnp.asarray(crop), jnp.asarray(planted_day), jnp.asarray(day)
                )
            )
            want = _oracle_water_days(crop, planted_day, day)
            assert got == want, f"crop={rules.CROPS[crop]} p={planted_day} d={day}: {got}!={want}"


def test_last_eod_step_is_695():
    """最後一次 end_of_day 的 step，以及它推出的 nd 上界。"""
    eods = [s for s in range(rules.N_TRANSITIONS) if (s + 1) % rules.TURNS_PER_DAY == 0]
    assert eods[-1] == loop4_obs.LAST_EOD_STEP == 695
    assert eods[-1] // rules.TURNS_PER_DAY + 1 == loop4_obs.LAST_PRODUCTION_ND == 29


def test_sale_revenue_matches_engine_lockstep():
    """賣 k 單位的前綴和收入，必須等於引擎逐筆報價的總和。"""
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = int(rng.integers(0, rules.N_PRODUCTS))
        inv = int(rng.integers(rules.MARKET_I0 - 500, rules.MARKET_I0 + 500))
        k = int(rng.integers(0, 60))
        want = 0
        cur = inv
        for _ in range(k):
            price, _ = kernels.lookup_price(jnp.asarray(p), jnp.asarray(cur))
            want += int(price)
            if int(price) > 1:
                cur += 1
        got = float(loop4_obs.sale_revenue(jnp.asarray(p), jnp.asarray(inv), jnp.asarray(k)))
        assert got == pytest.approx(want), f"p={p} inv={inv} k={k}: {got} != {want}"


def test_observation_shapes_and_bounds_on_random_states():
    """1,000 個隨機 state：形狀固定、全部有限、且落在 [-1, 1]。"""
    rng = np.random.default_rng(1)
    batch = 50
    for it in range(20):
        state = _blank_state(batch)
        tiles = np.asarray(state.tiles).copy()
        for b in range(batch):
            for seat in range(2):
                for flat in range(rules.N_TILES):
                    roll = rng.random()
                    if roll < 0.35:
                        tiles[b, seat, flat] = _plant_tile(
                            int(rng.integers(0, rules.N_CROPS)),
                            int(rng.integers(0, rules.N_DAYS)),
                            int(rng.integers(0, 7)),
                        )
                    elif roll < 0.5:
                        tiles[b, seat, flat] = _animal_tile(
                            int(rng.integers(0, rules.N_ANIMALS)),
                            int(rng.integers(0, rules.N_DAYS)),
                        )
        state = state._replace(
            tiles=jnp.asarray(tiles, jnp.int32),
            step=jnp.asarray(rng.integers(0, rules.N_TRANSITIONS, batch), jnp.int32),
            money=jnp.asarray(rng.integers(0, 300_000, (batch, 2)), jnp.int32),
            hires_today=jnp.asarray(rng.integers(0, 20, (batch, 2)), jnp.int32),
            n_quads=jnp.asarray(rng.integers(0, 4, (batch, 2)), jnp.int32),
            shed=jnp.asarray(rng.integers(0, 40, (batch, 2, rules.N_ITEMS)), jnp.int32),
            market_inv=jnp.asarray(
                rng.integers(rules.MARKET_I0 - 3000, rules.MARKET_I0 + 3000, (batch, 9)),
                jnp.int32,
            ),
        )
        obs = loop4_obs.observation(state, it % 2, CAPS)
        assert obs["tiles"].shape == (batch, 2, rules.N_TILES, loop4_obs.TILE_DIM)
        assert obs["units"].shape == (batch, 2, CAPS.max_units, loop4_obs.UNIT_DIM)
        assert obs["products"].shape == (batch, rules.N_PRODUCTS, loop4_obs.PRODUCT_DIM)
        assert obs["global"].shape == (batch, loop4_obs.GLOBAL_DIM)
        for name, arr in obs.items():
            a = np.asarray(arr)
            assert np.all(np.isfinite(a)), name
            assert a.min() >= -1.0 and a.max() <= 1.0, (name, a.min(), a.max())


def test_opponent_private_inventory_is_masked():
    state = _blank_state(3)
    inv = np.asarray(state.inv).copy()
    inv[:, :, :, :] = 5
    state = state._replace(inv=jnp.asarray(inv, jnp.int32))
    obs = loop4_obs.observation(state, 0, CAPS)
    own = np.asarray(obs["units"])[:, 0, :, 6:]
    opp = np.asarray(obs["units"])[:, 1, :, 6:]
    assert own.max() > 0.0
    assert opp.max() == 0.0


def test_no_loop3_imports():
    """不得依賴 Loop 3 的模組（任務 12 規矩）。比對 AST 的 import，不是字串。"""
    import ast

    tree = ast.parse(open("rl/loop4_obs.py").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.add(base)
            imported.update(f"{base}.{a.name}" for a in node.names)
    banned = {"rl.obs", "rl.train", "rl.ordered_v3_ppo", "rl.ordered_v3_codec"}
    assert not (imported & banned), imported & banned
    assert not any("ordered_v3" in name for name in imported), imported
