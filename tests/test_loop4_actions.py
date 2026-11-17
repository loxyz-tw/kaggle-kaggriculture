"""Loop 4 動作表示的測試（任務 12 A2）。

三件必須釘住的事：

1. **遮罩是「以格為主體」的**，而且對每一種任務，遮罩為真時引擎至少有可能做成——
   用 `rl/kernels.py` 實跑一步來驗，不是另寫一份判準。
2. **走路執行器是確定性的**，而且永遠不會產生非法欄位（PLANT 一定帶作物、PICKUP／PLACE
   一定帶品項與正數量、其他動作不帶 arg）。
3. **市場桶 → 整數 → 截斷**：不超過 10 筆、不產生零數量訂單、優先序正確、
   SELL 的整數用的是**單位相位之後**的 shed。
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rl import caps as caps_mod
from rl import kernels, rules
from rl import loop4_actions as A
from rl import state as rl_state

CAPS = caps_mod.BOUNDED_TRAINING


def _decision(batch, **kw):
    d = dict(
        tile_task=jnp.zeros((batch, 2, rules.N_TILES), jnp.int32),
        tile_qty=jnp.zeros((batch, 2, rules.N_TILES), jnp.int32),
        sell=jnp.zeros((batch, 2, rules.N_PRODUCTS), jnp.int32),
        buy_seed=jnp.zeros((batch, 2, rules.N_CROPS), jnp.int32),
        buy_product=jnp.zeros((batch, 2, len(A.BUYABLE_PRODUCTS)), jnp.int32),
        buy_animal=jnp.zeros((batch, 2, rules.N_ANIMALS), jnp.int32),
        hire=jnp.zeros((batch, 2), jnp.int32),
        buy_land=jnp.zeros((batch, 2), jnp.int32),
    )
    d.update(kw)
    return A.Loop4Decision(**d)


def _random_state(rng, batch=8):
    state = rl_state.reset(batch, CAPS, seed=0)
    tiles = np.array(state.tiles)
    for b in range(batch):
        for seat in range(2):
            for flat in range(rules.N_TILES):
                roll = rng.random()
                row = rl_state.CANONICAL_EMPTY.copy()
                if roll < 0.3:
                    crop = int(rng.integers(0, rules.N_CROPS))
                    row[rules.C_KIND] = rules.TK_PLANT
                    row[rules.C_CROP] = crop
                    row[rules.C_PLANTED_DAY] = int(rng.integers(0, 20))
                    row[rules.C_YIELD] = int(rng.integers(0, 5))
                    row[rules.C_MLS] = -1 if rules.CROP_ONGOING[crop] else 700
                    row[rules.C_WATERED] = int(rng.integers(0, 2))
                elif roll < 0.45:
                    a = int(rng.integers(0, rules.N_ANIMALS))
                    row[rules.C_KIND] = rules.ANIMAL_STRUCTURE_KIND[a]
                    row[rules.C_ANIMAL] = a
                    row[rules.C_PLACED_DAY] = int(rng.integers(0, 20))
                    row[rules.C_YIELD] = int(rng.integers(0, 4))
                    row[rules.C_FERT_AVAIL] = int(rng.integers(0, 2))
                elif roll < 0.55:
                    row[rules.C_KIND] = rules.TK_WEED
                elif roll < 0.6:
                    row[rules.C_KIND] = rules.TK_LOCKED
                tiles[b, seat, flat] = row
    hires = rng.integers(0, 6, (batch, 2))
    ux = rng.integers(0, rules.BOARD_SIZE, (batch, 2, CAPS.max_units))
    uy = rng.integers(0, rules.BOARD_SIZE, (batch, 2, CAPS.max_units))
    inv = np.zeros((batch, 2, CAPS.max_units, rules.N_ITEMS), np.int32)
    inv[:, :, :, rules.I_WHEAT] = rng.integers(0, 4, inv.shape[:3])
    inv[:, :, :, rules.I_FERTILIZER] = rng.integers(0, 4, inv.shape[:3])
    inv[:, :, :, rules.ANIMAL_ITEM_BASE] = rng.integers(0, 2, inv.shape[:3])
    return state._replace(
        tiles=jnp.asarray(tiles, jnp.int32),
        step=jnp.asarray(rng.integers(0, rules.N_TRANSITIONS, batch), jnp.int32),
        money=jnp.asarray(rng.integers(0, 50_000, (batch, 2)), jnp.int32),
        hires_today=jnp.asarray(hires, jnp.int32),
        unit_x=jnp.asarray(ux, jnp.int32),
        unit_y=jnp.asarray(uy, jnp.int32),
        inv=jnp.asarray(inv, jnp.int32),
        inv_seq=jnp.asarray((inv > 0).astype(np.int32), jnp.int32),
        shed=jnp.asarray(rng.integers(0, 12, (batch, 2, rules.N_ITEMS)), jnp.int32),
        seeds=jnp.asarray(rng.integers(0, 6, (batch, 2, rules.N_CROPS)), jnp.int32),
        n_quads=jnp.asarray(rng.integers(0, 4, (batch, 2)), jnp.int32),
    )


def test_task_table_is_consistent():
    assert A.TASK_TO_OP.shape[0] == A.N_TASKS == len(A.TASK_NAMES)
    assert int(A.TASK_TO_OP[A.T_NOOP]) == rules.OP_PASS
    for c in range(rules.N_CROPS):
        assert int(A.TASK_TO_OP[A.T_PLANT0 + c]) == rules.OP_PLANT
    assert int(A.TASK_TO_OP[A.T_PICKUP]) == rules.OP_PICKUP
    assert int(A.TASK_TO_OP[A.T_PLACE]) == rules.OP_PLACE


def test_mask_is_tile_centric_not_unit_centric():
    """同一個盤面把所有單位移到別處，遮罩必須一個 bit 都不變。"""
    rng = np.random.default_rng(0)
    state = _random_state(rng)
    m1 = np.asarray(A.tile_task_mask(state, CAPS))
    moved = state._replace(unit_x=jnp.zeros_like(state.unit_x), unit_y=jnp.zeros_like(state.unit_y))
    m2 = np.asarray(A.tile_task_mask(moved, CAPS))
    assert np.array_equal(m1, m2)


def test_locked_tiles_allow_only_noop_pickup_and_place():
    """LOCKED 格的合法集合必須與引擎一致。

    引擎的 **PICKUP 在 LOCKED 守衛之前結算**（`rl/kernels.py` 的 "PICKUP (resolves before
    the LOCKED guard)"），而 PLACE-to-shed 的 ``c_place_b`` 也沒有 ``not_locked`` 條件。
    所以 LOCKED 格上這兩件事是真的做得成的，其餘一律不行。實際盤面上 shed 相鄰格永遠在
    NW（一開始就解鎖），所以這條只有在壓力測試的隨機盤面才碰得到。
    """
    rng = np.random.default_rng(1)
    state = _random_state(rng)
    m = np.asarray(A.tile_task_mask(state, CAPS))
    assert m[..., A.T_NOOP].all()
    kind = np.asarray(state.tiles)[..., rules.C_KIND]
    locked = kind == rules.TK_LOCKED
    allowed = {A.T_NOOP, A.T_PICKUP, A.T_PLACE}
    for task in range(A.N_TASKS):
        if task in allowed:
            continue
        assert not m[..., task][locked].any(), f"LOCKED 格不該允許 {A.TASK_NAMES[task]}"
    shed_flat = set(rules.SHED_ACCESS_FLAT)
    non_shed = np.zeros(rules.N_TILES, bool)
    non_shed[[f for f in range(rules.N_TILES) if f not in shed_flat]] = True
    locked_non_shed = locked & non_shed[None, None, :]
    for task in (A.T_PICKUP, A.T_PLACE):
        assert not m[..., task][locked_non_shed].any(), (
            f"不在 shed 相鄰格的 LOCKED 格不該允許 {A.TASK_NAMES[task]}"
        )


def test_walk_emits_only_wellformed_unit_fields():
    """執行器產生的 ``unit_op``／``unit_arg``／``unit_qty`` 必須永遠是合語法的。"""
    rng = np.random.default_rng(2)
    state = _random_state(rng, batch=6)
    mask = np.asarray(A.tile_task_mask(state, CAPS))
    task = np.zeros((6, 2, rules.N_TILES), np.int32)
    for b in range(6):
        for seat in range(2):
            for flat in range(rules.N_TILES):
                legal = np.nonzero(mask[b, seat, flat])[0]
                task[b, seat, flat] = int(rng.choice(legal))
    qty = rng.integers(0, A.N_QTY_BUCKETS, (6, 2, rules.N_TILES)).astype(np.int32)
    op, arg, n, _ = A.assign_and_walk(state, jnp.asarray(task), jnp.asarray(qty), CAPS)
    op, arg, n = np.asarray(op), np.asarray(arg), np.asarray(n)
    assert op.min() >= 0 and op.max() < rules.N_UNIT_OPS
    plant = op == rules.OP_PLANT
    assert (arg[plant] >= 0).all() and (arg[plant] < rules.N_CROPS).all()
    carry = (op == rules.OP_PICKUP) | (op == rules.OP_PLACE)
    assert (arg[carry] >= 0).all() and (arg[carry] < rules.N_ITEMS).all()
    assert (n[carry] >= 1).all()
    plain = ~(plant | carry)
    assert (arg[plain] == -1).all()
    assert (n >= 1).all()


def test_walk_is_deterministic():
    rng = np.random.default_rng(3)
    state = _random_state(rng, batch=4)
    task = jnp.asarray(rng.integers(0, A.N_TASKS, (4, 2, rules.N_TILES)).astype(np.int32))
    qty = jnp.asarray(rng.integers(0, A.N_QTY_BUCKETS, (4, 2, rules.N_TILES)).astype(np.int32))
    a = [np.asarray(x) for x in A.assign_and_walk(state, task, qty, CAPS)]
    b = [np.asarray(x) for x in A.assign_and_walk(state, task, qty, CAPS)]
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_walk_moves_toward_target_or_acts_on_it():
    """指派一格給唯一存活的農夫：要嘛站上去做事，要嘛曼哈頓距離必須減 1。"""
    for target in (0, 37, 55, 99):
        state = rl_state.reset(1, CAPS, seed=0)
        tiles = np.array(state.tiles)
        tiles[0, :, target] = rl_state.CANONICAL_EMPTY  # 一定是 EMPTY，BUILD_COOP 合法
        state = state._replace(tiles=jnp.asarray(tiles, jnp.int32))
        task = np.zeros((1, 2, rules.N_TILES), np.int32)
        task[0, :, target] = A.T_BUILD_COOP
        op, _, _, _ = A.assign_and_walk(
            state, jnp.asarray(task), jnp.zeros_like(jnp.asarray(task)), CAPS
        )
        ux, uy = int(state.unit_x[0, 0, 0]), int(state.unit_y[0, 0, 0])
        tx, ty = target % rules.BOARD_SIZE, target // rules.BOARD_SIZE
        d0 = abs(tx - ux) + abs(ty - uy)
        o = int(op[0, 0, 0])
        if d0 == 0:
            assert o == rules.OP_BUILD_COOP
        else:
            nx = ux + rules.MOVE_DX[o]
            ny = uy + rules.MOVE_DY[o]
            assert abs(tx - nx) + abs(ty - ny) == d0 - 1, (target, o, ux, uy)


def test_market_orders_never_exceed_cap_or_emit_zero_quantities():
    rng = np.random.default_rng(4)
    state = _random_state(rng, batch=8)
    d = _decision(
        8,
        sell=jnp.asarray(rng.integers(0, A.N_SELL, (8, 2, rules.N_PRODUCTS)).astype(np.int32)),
        buy_seed=jnp.asarray(rng.integers(0, A.N_BUY_SEED, (8, 2, rules.N_CROPS)).astype(np.int32)),
        buy_product=jnp.asarray(rng.integers(0, A.N_BUY_PRODUCT, (8, 2, 2)).astype(np.int32)),
        buy_animal=jnp.asarray(
            rng.integers(0, A.N_BUY_ANIMAL, (8, 2, rules.N_ANIMALS)).astype(np.int32)
        ),
        hire=jnp.asarray(rng.integers(0, A.N_HIRE, (8, 2)).astype(np.int32)),
        buy_land=jnp.asarray(rng.integers(0, 2, (8, 2)).astype(np.int32)),
    )
    op, item, qty = A.market_orders(state, d, CAPS)
    op, item, qty = np.asarray(op), np.asarray(item), np.asarray(qty)
    assert op.shape[-1] == CAPS.max_orders
    live = op != rules.MO_NONE
    assert live.sum(-1).max() <= CAPS.max_orders
    # 需要數量的訂單一定是正數；HIRE／BUY_LAND 本來就沒有數量欄位
    needs_qty = live & (op != rules.MO_HIRE) & (op != rules.MO_BUY_LAND)
    assert (qty[needs_qty] > 0).all()
    # 有效訂單必須連續排在前面（引擎按 list 順序結算）
    first_dead = np.argmax(~live, axis=-1)
    for idx in np.ndindex(live.shape[:2]):
        n = int(live[idx].sum())
        assert live[idx][:n].all() and not live[idx][n:].any(), (idx, first_dead[idx])
    # 品項欄位合法
    sell = live & (op == rules.MO_SELL)
    assert (item[sell] >= 0).all() and (item[sell] < rules.N_PRODUCTS).all()
    buyp = live & (op == rules.MO_BUY_PRODUCT)
    assert np.isin(item[buyp], list(A.BUYABLE_PRODUCTS)).all()


def test_emit_priority_puts_sell_first_and_buy_animal_last():
    """同時有六種訂單時的實際順序。"""
    state = rl_state.reset(1, CAPS, seed=0)
    shed = np.array(state.shed)
    shed[0, :, :] = 5
    seeds = np.array(state.seeds)
    state = state._replace(
        shed=jnp.asarray(shed, jnp.int32),
        seeds=jnp.asarray(seeds, jnp.int32),
        money=jnp.full_like(state.money, 100_000),
    )
    d = _decision(
        1,
        sell=jnp.zeros((1, 2, rules.N_PRODUCTS), jnp.int32).at[:, :, 0].set(1),
        buy_seed=jnp.zeros((1, 2, rules.N_CROPS), jnp.int32).at[:, :, 0].set(1),
        buy_product=jnp.zeros((1, 2, 2), jnp.int32).at[:, :, 0].set(1),
        buy_animal=jnp.zeros((1, 2, rules.N_ANIMALS), jnp.int32).at[:, :, 0].set(1),
        hire=jnp.ones((1, 2), jnp.int32),
        buy_land=jnp.ones((1, 2), jnp.int32),
    )
    op, _, _ = A.market_orders(state, d, CAPS)
    seq = [int(x) for x in np.asarray(op)[0, 0] if int(x) != rules.MO_NONE]
    assert seq == [
        rules.MO_SELL,
        rules.MO_BUY_LAND,
        rules.MO_HIRE,
        rules.MO_BUY_SEED,
        rules.MO_BUY_PRODUCT,
        rules.MO_BUY_ANIMAL,
    ], seq


def test_sell_all_uses_post_unit_phase_shed():
    """回合開始時倉庫是空的、但這回合 HARVEST→DROP 會補進來，「賣全部」必須賣得到。

    這正是任務 11 量到的形狀（教師 48.2% 的 SELL 數量超過回合開始時的庫存）。
    """
    state = rl_state.reset(1, CAPS, seed=0)
    tiles = np.array(state.tiles)
    # 農夫站的那一格放一株成熟的 WHEAT，yield 3
    spawn = rules.DEFAULT_SPAWN[1] * rules.BOARD_SIZE + rules.DEFAULT_SPAWN[0]
    row = rl_state.CANONICAL_EMPTY.copy()
    row[rules.C_KIND] = rules.TK_PLANT
    row[rules.C_CROP] = rules.ITEM_INDEX["WHEAT"]
    row[rules.C_PLANTED_DAY] = 0
    row[rules.C_YIELD] = 3
    row[rules.C_MLS] = 700
    tiles[0, 0, spawn] = row
    state = state._replace(
        tiles=jnp.asarray(tiles, jnp.int32),
        step=jnp.full_like(state.step, 5 * rules.TURNS_PER_DAY),
    )
    assert int(state.shed[0, 0, rules.I_WHEAT]) == 0

    task = np.zeros((1, 2, rules.N_TILES), np.int32)
    task[0, 0, spawn] = A.T_HARVEST
    d = _decision(
        1,
        tile_task=jnp.asarray(task),
        sell=jnp.zeros((1, 2, rules.N_PRODUCTS), jnp.int32),
    )
    # 先不賣：確認 HARVEST 讓小麥進了單位身上（還沒進 shed）
    nxt, act, _ = A.loop4_step(state, d, CAPS)
    assert int(nxt.inv[0, 0, 0, rules.I_WHEAT]) == 3

    # 農夫站在 shed 相鄰格，所以同一回合 DROP 就會進倉庫；改成「收成後先 DROP」兩回合驗賣全部
    dropped = nxt
    task2 = np.zeros((1, 2, rules.N_TILES), np.int32)
    task2[0, 0, spawn] = A.T_PLACE  # PLACE 到 shed（農夫在 shed 相鄰格）
    d2 = _decision(1, tile_task=jnp.asarray(task2), tile_qty=jnp.full((1, 2, 100), 3, jnp.int32))
    after_drop, _, _ = A.loop4_step(dropped, d2, CAPS)
    held = int(after_drop.shed[0, 0, rules.I_WHEAT])
    assert held > 0, "PLACE 應該把小麥放進倉庫"

    d3 = _decision(1, sell=jnp.zeros((1, 2, rules.N_PRODUCTS), jnp.int32).at[:, :, 0].set(1))
    _, act3, _ = A.loop4_step(after_drop, d3, CAPS)
    op3 = np.asarray(act3.order_op)[0, 0]
    qty3 = np.asarray(act3.order_qty)[0, 0]
    sold = int(qty3[op3 == rules.MO_SELL].sum())
    assert sold == held, (sold, held)


def test_loop4_step_phase_order_matches_env_step():
    """同一份 SimAction 餵給 `rl/env.py::step`，結果必須與 loop4_step 逐欄位相同。"""
    from rl import env as rl_env

    rng = np.random.default_rng(5)
    state = _random_state(rng, batch=4)
    task = jnp.asarray(rng.integers(0, A.N_TASKS, (4, 2, rules.N_TILES)).astype(np.int32))
    qty = jnp.asarray(rng.integers(0, A.N_QTY_BUCKETS, (4, 2, rules.N_TILES)).astype(np.int32))
    d = _decision(
        4,
        tile_task=task,
        tile_qty=qty,
        sell=jnp.asarray(rng.integers(0, A.N_SELL, (4, 2, rules.N_PRODUCTS)).astype(np.int32)),
        hire=jnp.asarray(rng.integers(0, A.N_HIRE, (4, 2)).astype(np.int32)),
    )
    got, action, _ = A.loop4_step(state, d, CAPS)
    want = rl_env.step(state, action, CAPS)
    for name in state._fields:
        if name == "key":
            continue
        assert np.array_equal(np.asarray(getattr(got, name)), np.asarray(getattr(want, name))), name


@pytest.mark.parametrize("task", [A.T_WATER, A.T_HARVEST, A.T_CARE, A.T_COLLECT, A.T_DIG])
def test_mask_true_means_engine_can_succeed(task):
    """遮罩為真的格，把單位放上去執行該任務，引擎必須真的改變狀態（至少一格會變）。"""
    rng = np.random.default_rng(6 + task)
    state = _random_state(rng, batch=4)
    mask = np.asarray(A.tile_task_mask(state, CAPS))
    hits = np.argwhere(mask[:, 0, :, task])
    assert len(hits) > 0, f"隨機盤面沒有任何 {A.TASK_NAMES[task]} 合法格"
    b, flat = hits[0]
    ux = np.array(state.unit_x)
    uy = np.array(state.unit_y)
    ux[b, 0, 0] = flat % rules.BOARD_SIZE
    uy[b, 0, 0] = flat // rules.BOARD_SIZE
    st = state._replace(unit_x=jnp.asarray(ux, jnp.int32), unit_y=jnp.asarray(uy, jnp.int32))
    op = np.zeros((4, 2, CAPS.max_units), np.int32)
    op[b, 0, 0] = int(A.TASK_TO_OP[task])
    from rl.actions import SimAction

    act = SimAction(
        unit_op=jnp.asarray(op),
        unit_arg=jnp.full((4, 2, CAPS.max_units), -1, jnp.int32),
        unit_qty=jnp.ones((4, 2, CAPS.max_units), jnp.int32),
        order_op=jnp.zeros((4, 2, CAPS.max_orders), jnp.int32),
        order_item=jnp.full((4, 2, CAPS.max_orders), -1, jnp.int32),
        order_qty=jnp.zeros((4, 2, CAPS.max_orders), jnp.int32),
    )
    after = kernels.apply_unit_actions(st, act, CAPS)
    changed = not np.array_equal(
        np.asarray(st.tiles)[b, 0, flat], np.asarray(after.tiles)[b, 0, flat]
    ) or not np.array_equal(np.asarray(st.inv)[b, 0, 0], np.asarray(after.inv)[b, 0, 0])
    assert changed, f"{A.TASK_NAMES[task]} 遮罩為真但引擎沒有任何改變"
