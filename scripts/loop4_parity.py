"""任務 12 A4：Loop 4 的動作表示與官方引擎逐步等價。

沿 `scripts/rl_parity.py` 的結構——**官方引擎是 oracle**，移植該 seed 的 RNG（`rl/rng.py`），
用引擎實際收到的動作重播模擬器，每一回合比對 `rl/compare.py` 的 14 個欄位，零容忍。
新增的是一個 Loop 4 的 agent：它在引擎裡用 `rl/loop4_obs.py` 的遮罩選任務、用
`rl/loop4_actions.py` 的桶與走路執行器產生引擎 JSON。

兩種決策來源（掃描 §4 要的兩條）：

* ``random``——在合法遮罩上均勻亂選（每格任務、每個市場 head 的桶），固定 seed。
* ``teacher``——把教師語料的動作經 `scripts/loop4_il_labels.py` 映射成標籤、再解碼回動作。
  這條驗的是「映射再解碼」不會產生引擎不接受的東西。

四層檢查（掃描 §4）：

1. 逐回合全狀態比對（引擎 oracle）。
2. 解碼器往返：`loop4_step` 在**模擬器自己的狀態**上重算出來的 SimAction，必須與送進引擎的
   那一份逐欄位相同。
3. 溢位計數器全零。
4. 動作合法性：無非法訂單（品項／op 不合引擎語法）、無零數量訂單、每回合不超過 10 筆。
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

from rl import actions as rl_actions  # noqa: E402
from rl import caps as caps_mod  # noqa: E402
from rl import env as rl_env  # noqa: E402
from rl import kernels, rng, rules  # noqa: E402
from rl import loop4_actions as A  # noqa: E402
from rl import state as rl_state  # noqa: E402
from rl.compare import compare  # noqa: E402

CAPS = caps_mod.EXACT_REFERENCE


def _state_from_obs(obs: dict, seat: int, canonical: bool = False) -> rl_state.SimState:
    """從**一座自己的** observation 造 SimState；對手的私有欄位留 0（我們不讀它）。

    ``canonical=True`` 時把**自己放到索引 0**、對手放到索引 1。給 policy 用：
    `rl/loop4_model.py` 的每格 head 讀的是 token 的前 100 格，也就是觀測裡的「自己」，
    而 `loop4_eval` 的解碼固定取 seat 0——不做這個正規化的話，坐 seat 1 的那些局
    policy 會對著**對手的農場**下決定。parity 那條路徑用真實索引且前後一致，不要開這個旗標。
    """
    farms = obs["farms"]
    u = CAPS.max_units
    tiles = np.zeros((1, 2, rules.N_TILES, rules.N_TILE_CHANNELS), np.int32)
    unit_x = np.full((1, 2, u), rules.DEFAULT_SPAWN[0], np.int32)
    unit_y = np.full((1, 2, u), rules.DEFAULT_SPAWN[1], np.int32)
    money = np.zeros((1, 2), np.int32)
    hires = np.zeros((1, 2), np.int32)
    n_quads = np.zeros((1, 2), np.int32)
    shed = np.zeros((1, 2, rules.N_ITEMS), np.int32)
    seeds = np.zeros((1, 2, rules.N_CROPS), np.int32)
    inv = np.zeros((1, 2, u, rules.N_ITEMS), np.int32)
    inv_seq = np.zeros_like(inv)
    inv_ctr = np.ones((1, 2, u), np.int32)
    order = [seat, 1 - seat] if canonical else [0, 1]
    own = 0 if canonical else seat
    for p, src in enumerate(order):
        farm = farms[src]
        for flat in range(rules.N_TILES):
            y, x = divmod(flat, rules.BOARD_SIZE)
            tiles[0, p, flat] = rl_state.encode_tile(farm["tiles"][y][x])
        money[0, p] = int(farm["money"])
        hires[0, p] = farm["hires_today"]
        n_quads[0, p] = len(farm["unlocked_quadrants"]) - 1
        unit_x[0, p, 0], unit_y[0, p, 0] = farm["farmer"][0], farm["farmer"][1]
        for h, pos in enumerate(farm["hands"]):
            if h + 1 < u:
                unit_x[0, p, h + 1], unit_y[0, p, h + 1] = pos[0], pos[1]
    private = obs["private"]
    for item, n in private["shed"].items():
        shed[0, own, rules.ITEM_INDEX[item]] = n
    for crop, n in private["seeds"].items():
        seeds[0, own, rules.ITEM_INDEX[crop]] = n
    for ui, unit_inv in enumerate(private["inventories"]):
        if ui >= u:
            break
        for rank, (item, n) in enumerate(unit_inv.items(), start=1):
            if n > 0:
                inv[0, own, ui, rules.ITEM_INDEX[item]] = n
                inv_seq[0, own, ui, rules.ITEM_INDEX[item]] = rank
        inv_ctr[0, own, ui] = len([1 for n in unit_inv.values() if n > 0]) + 1
    market_inv = np.asarray([obs["market"]["inventory"][p] for p in rules.PRODUCTS], np.int32)[None]
    shop_seq = np.full((1, rules.MAX_SHOP_INSTANCES), -1, np.int32)
    shops = obs.get("town", {}).get("shops", []) or []
    for i, name in enumerate(shops[: rules.MAX_SHOP_INSTANCES]):
        shop_seq[0, i] = rules.SHOP_NAMES.index(name)
    return rl_state.SimState(
        step=jnp.asarray([int(obs["step"])] if "step" in obs else [0], jnp.int32),
        money=jnp.asarray(money),
        tiles=jnp.asarray(tiles),
        unit_x=jnp.asarray(unit_x),
        unit_y=jnp.asarray(unit_y),
        hires_today=jnp.asarray(hires),
        n_quads=jnp.asarray(n_quads),
        shed=jnp.asarray(shed),
        seeds=jnp.asarray(seeds),
        inv=jnp.asarray(inv),
        inv_seq=jnp.asarray(inv_seq),
        inv_ctr=jnp.asarray(inv_ctr),
        market_inv=jnp.asarray(market_inv),
        shop_seq=jnp.asarray(shop_seq),
        n_shops=jnp.asarray([len(shops)], jnp.int32),
        key=jnp.zeros((1, 2), jnp.uint32),
        overflow=jnp.zeros((1, caps_mod.N_OVERFLOW), jnp.int32),
        market_trips=jnp.zeros((1,), jnp.int32),
        market_units=jnp.zeros((1,), jnp.int32),
        market_closed=jnp.zeros((1,), jnp.int32),
    )


def random_decision(state: rl_state.SimState, key: jax.Array) -> A.Loop4Decision:
    """在合法遮罩上均勻亂選。桶側只遮「引擎語法上不可能」的（品項），不遮付不起——
    付不起由引擎自己跳過，正是我們要驗的東西。"""
    mask = A.tile_task_mask(state, CAPS)  # [1, 2, 100, 17]
    k = jax.random.split(key, 7)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    task = jax.random.categorical(k[0], logits, axis=-1).astype(jnp.int32)
    b = state.step.shape[0]
    return A.Loop4Decision(
        tile_task=task,
        tile_qty=jax.random.randint(k[1], (b, 2, rules.N_TILES), 0, A.N_QTY_BUCKETS),
        sell=jax.random.randint(k[2], (b, 2, rules.N_PRODUCTS), 0, A.N_SELL),
        buy_seed=jax.random.randint(k[3], (b, 2, rules.N_CROPS), 0, A.N_BUY_SEED),
        buy_product=jax.random.randint(k[4], (b, 2, len(A.BUYABLE_PRODUCTS)), 0, A.N_BUY_PRODUCT),
        buy_animal=jax.random.randint(k[5], (b, 2, rules.N_ANIMALS), 0, A.N_BUY_ANIMAL),
        hire=jax.random.randint(k[6], (b, 2), 0, A.N_HIRE),
        buy_land=jax.random.randint(jax.random.fold_in(key, 9), (b, 2), 0, 2),
    )


@functools.partial(jax.jit, static_argnums=(2,))
def _decide_and_step(state: rl_state.SimState, key: jax.Array, caps: caps_mod.Caps):
    """決策＋一步，整段 jit——沒有 jit 的話 eager dispatch 讓一局要好幾分鐘。

    只回傳 SimAction（引擎才是推進狀態的人），下一步的 state 由引擎的 observation 重建。
    """
    decision = random_decision(state, key)
    _, action, _ = A.loop4_step(state, decision, caps)
    return action


@functools.partial(jax.jit, static_argnums=(2,))
def _step_only(state: rl_state.SimState, decision, caps: caps_mod.Caps):
    """教師路徑的一步，整段 jit。

    沒有 jit 的話 batch-1 的 `loop4_step` 在 CPU 上 eager dispatch 要 **1.1 s／回合**，
    而引擎的 ``actTimeout`` 是 **1 s**——實測會在第 99–123 回合把兩座判 TIMEOUT。
    """
    _, action, _ = A.loop4_step(state, decision, caps)
    return action


def action_to_engine_json(action, seat: int, n_alive: int) -> dict:
    """SimAction 的一座 → 引擎 JSON。"""
    op = np.asarray(action.unit_op)[0, seat]
    arg = np.asarray(action.unit_arg)[0, seat]
    qty = np.asarray(action.unit_qty)[0, seat]
    units = []
    for k in range(n_alive):
        name = rules.UNIT_OP_NAMES[int(op[k])]
        if int(op[k]) == rules.OP_PLANT:
            units.append([name, rules.CROPS[int(arg[k])]])
        elif int(op[k]) in (rules.OP_PICKUP, rules.OP_PLACE):
            units.append([name, rules.ITEMS[int(arg[k])], int(qty[k])])
        else:
            units.append([name])
    oop = np.asarray(action.order_op)[0, seat]
    oitem = np.asarray(action.order_item)[0, seat]
    oqty = np.asarray(action.order_qty)[0, seat]
    market = []
    for s in range(len(oop)):
        o = int(oop[s])
        if o == rules.MO_NONE:
            continue
        if o == rules.MO_HIRE:
            market.append(["HIRE"])
        elif o == rules.MO_BUY_LAND:
            market.append(["BUY_LAND"])
        else:
            market.append([rules.MARKET_OP_NAMES[o], rules.ITEMS[int(oitem[s])], int(oqty[s])])
    return {"farmer": units[0], "hands": units[1:], "market": market}


def teacher_decision_source(path: Path, capture: int) -> dict:
    """A4 的第二個來源：教師語料的動作 → `scripts/loop4_il_labels.py` 的標籤 → 解碼。

    **整局 719 個回合的標籤一次算完**，不在 agent 裡逐回合算——桶的「索引」只需要教師自己的
    狀態（那是離線的），桶的「整數」才需要現場的 shed，而那一步由 `loop4_step` 在引擎裡重算。
    第一版在 agent 裡逐回合呼叫 `market_labels`，引擎在第 65 回合就把兩座判 TIMEOUT。
    """
    import zipfile

    from scripts.loop4_il_labels import (
        BIG_FIELDS,
        CALLS,
        STATE_FIELDS,
        _read,
        _read_slice,
        _state_from_parts,
        market_labels,
        qty_from_bytes,
        unit_labels,
    )

    lo, hi = capture * CALLS, (capture + 1) * CALLS
    with zipfile.ZipFile(path) as zf:
        marked = _read(zf, "marked.npy")[lo:hi]
        seat = np.argmax(marked, axis=1).astype(np.int64)
        verb = _read(zf, "d/unit_verb.npy")[lo:hi]
        crop = _read(zf, "d/unit_crop.npy")[lo:hi]
        uqty = qty_from_bytes(_read(zf, "d/unit_qty_bytes.npy")[lo:hi])
        ukeep = _read(zf, "k/unit_keep.npy")[lo:hi]
        mop = _read(zf, "d/market_op.npy")[lo:hi]
        mitem = _read(zf, "d/market_item.npy")[lo:hi]
        mqty = qty_from_bytes(_read(zf, "d/market_qty_bytes.npy")[lo:hi])
        mkeep = _read(zf, "k/market_keep.npy")[lo:hi]
        small = {k: _read(zf, f"s/{k}.npy") for k in STATE_FIELDS if k not in BIG_FIELDS}
        big = {k: _read_slice(zf, f"s/{k}.npy", lo, hi) for k in BIG_FIELDS}
    st = _state_from_parts(small, big, lo, hi)
    task, qb, _ = unit_labels(
        verb, crop, uqty, ukeep, np.asarray(st.unit_x), np.asarray(st.unit_y), seat
    )
    noop = rl_actions.SimAction(
        unit_op=jnp.asarray(verb)[:, None, :].repeat(2, axis=1),
        unit_arg=jnp.asarray(np.where(verb == rules.OP_PLANT, crop, -1))[:, None, :].repeat(
            2, axis=1
        ),
        unit_qty=jnp.ones_like(jnp.asarray(verb))[:, None, :].repeat(2, axis=1),
        order_op=jnp.zeros((CALLS, 2, CAPS.max_orders), jnp.int32),
        order_item=jnp.full((CALLS, 2, CAPS.max_orders), -1, jnp.int32),
        order_qty=jnp.zeros((CALLS, 2, CAPS.max_orders), jnp.int32),
    )
    post = kernels.apply_unit_actions(st, noop, CAPS)
    lab, _ = market_labels(mop, mitem, mqty, mkeep, post, seat)
    lb = lab["labels"]
    return {"task": task, "qbucket": qb, "market": lb}


def _teacher_action(state, teacher, t: int, seat: int):
    """用**預先算好的**教師標籤組一個 Loop4Decision 再解碼（桶的整數由 loop4_step 現場算）。"""
    row = min(t, teacher["task"].shape[0] - 1)
    b = 1
    task = np.zeros((b, 2, rules.N_TILES), np.int32)
    qb = np.zeros((b, 2, rules.N_TILES), np.int32)
    task[0, seat] = teacher["task"][row]
    qb[0, seat] = teacher["qbucket"][row]
    lb = teacher["market"]

    def two(x):
        out = np.zeros((b, 2) + x.shape[1:], np.int32)
        out[0, seat] = x[row]
        return jnp.asarray(out)

    decision = A.Loop4Decision(
        tile_task=jnp.asarray(task),
        tile_qty=jnp.asarray(qb),
        sell=two(lb["sell"]),
        buy_seed=two(lb["buy_seed"]),
        buy_product=two(lb["buy_product"]),
        buy_animal=two(lb["buy_animal"]),
        hire=jnp.full((b, 2), int(lb["hire"][row]), jnp.int32),
        buy_land=jnp.full((b, 2), int(lb["buy_land"][row]), jnp.int32),
    )
    return _step_only(state, decision, CAPS)


def make_loop4_agent(seat: int, seed: int, log: list, teacher: dict | None = None):
    """回傳一個 kaggle-environments 的 agent；同時把每回合的決策與動作記下來。"""
    counter = {"t": 0}

    def agent(obs, _config=None):
        state = _state_from_obs(obs, seat)
        t = counter["t"]
        if teacher is None:
            key = jax.random.fold_in(jax.random.PRNGKey(seed), t * 2 + seat)
            action = _decide_and_step(state, key, CAPS)
        else:
            action = _teacher_action(state, teacher, t, seat)
        n_alive = min(1 + int(obs["farms"][seat]["hires_today"]), CAPS.max_units)
        js = action_to_engine_json(action, seat, n_alive)
        log.append({"turn": counter["t"], "seat": seat, "json": js})
        counter["t"] += 1
        return js

    return agent


ILLEGAL = Counter()


def check_action_json(js: dict) -> list[str]:
    """第 4 層：無非法訂單、無零數量、不超過 10 筆。"""
    bad = []
    market = js.get("market") or []
    if len(market) > rules.DEFAULT_CONFIG["maxMarketOrdersPerTurn"]:
        bad.append(f"orders {len(market)} > 10")
    for e in market:
        if e[0] in ("HIRE", "BUY_LAND"):
            if len(e) != 1:
                bad.append(f"{e[0]} 帶了多餘欄位 {e}")
            continue
        if len(e) != 3:
            bad.append(f"訂單欄位數不是 3：{e}")
            continue
        op, item, n = e
        if op not in rules.MARKET_OP_NAMES:
            bad.append(f"未知 op {op}")
        if item not in rules.ITEMS:
            bad.append(f"未知品項 {item}")
        if not isinstance(n, int) or n <= 0:
            bad.append(f"非正數量 {e}")
        if op == "BUY_SEED" and item not in rules.CROPS:
            bad.append(f"BUY_SEED 品項不是作物：{item}")
        if op == "BUY_ANIMAL" and item not in rules.ANIMALS:
            bad.append(f"BUY_ANIMAL 品項不是動物：{item}")
        if op == "BUY_PRODUCT" and item not in ("WHEAT", "FERTILIZER"):
            bad.append(f"BUY_PRODUCT 品項引擎不收：{item}")
        if op == "SELL" and item not in rules.PRODUCTS:
            bad.append(f"SELL 品項不是產品：{item}")
    return bad


def one_game(seed: int, max_turns: int, teacher: dict | None = None) -> dict:
    t0 = time.perf_counter()
    log: list = []
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run(
        [
            make_loop4_agent(0, seed, log, teacher),
            make_loop4_agent(1, seed, log, teacher),
        ]
    )
    resolved = int(env.info.get("seed", seed))
    steps = env.steps
    tables = rng.record_tables([resolved])

    syntax_bad: list = []
    order_hist = Counter()
    for entry in log:
        b = check_action_json(entry["json"])
        if b:
            syntax_bad.append({"turn": entry["turn"], "seat": entry["seat"], "bad": b})
        order_hist[len(entry["json"].get("market") or [])] += 1

    state = rl_state.reset(1, CAPS, seed=resolved)
    ref0 = rl_state.encode_observation([s.observation for s in steps[0]], CAPS)
    bad0 = compare(rl_state.state_slice_np(state, 0, CAPS), ref0, CAPS)
    res = {
        "seed": seed,
        "resolved_seed": resolved,
        "engine_steps": len(steps),
        "turns_compared": 0,
        "clean": False,
        "first_mismatch": {"turn": 0, "detail": bad0} if bad0 else None,
        "syntax_violations": syntax_bad,
        "orders_per_turn": dict(sorted(order_hist.items())),
        "statuses": [s.status for s in steps[-1]],
    }
    if bad0:
        return res
    n = min(max_turns, len(steps) - 1)
    for i in range(1, n + 1):
        raw = [s.action for s in steps[i]]
        arrays, _ = rl_actions.encode_actions_np(raw, CAPS)
        action = rl_actions.stack_actions([arrays])
        state = rl_env.jit_step(state, action, CAPS, tables)
        sim = rl_state.state_slice_np(state, 0, CAPS)
        ref = rl_state.encode_observation([s.observation for s in steps[i]], CAPS)
        bad = compare(sim, ref, CAPS)
        sim_prices = np.asarray(kernels.prices_of(state.market_inv))[0].astype(np.int64)
        if not np.array_equal(sim_prices, ref.prices):
            bad.append(f"prices {sim_prices.tolist()} != {ref.prices.tolist()}")
        res["turns_compared"] = i
        if bad:
            res["first_mismatch"] = {"turn": i, "detail": bad}
            break
    else:
        res["clean"] = True
    res["overflow"] = dict(
        zip(caps_mod.OVERFLOW_NAMES, np.asarray(state.overflow)[0].tolist(), strict=True)
    )
    res["final_money"] = np.asarray(state.money)[0].tolist()
    res["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--max-turns", type=int, default=rules.N_TRANSITIONS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=("random", "teacher"), default="random")
    ap.add_argument("--shard", default=None, help="--source teacher 時必填：schema-3 shard npz")
    a = ap.parse_args()
    if a.source == "teacher" and not a.shard:
        ap.error("--source teacher 需要 --shard")
    games = []
    for i in range(a.games):
        teacher = teacher_decision_source(ROOT / a.shard, i) if a.source == "teacher" else None
        games.append(one_game(a.seed0 + i, a.max_turns, teacher))
    summary = {
        "games": len(games),
        "clean": sum(1 for g in games if g["clean"]),
        "syntax_violations": sum(len(g["syntax_violations"]) for g in games),
        "overflow_nonzero": sum(
            1 for g in games if any(v for v in (g.get("overflow") or {}).values())
        ),
        "max_orders_per_turn": max(
            (max(map(int, g["orders_per_turn"])) if g["orders_per_turn"] else 0) for g in games
        ),
        "turns_compared_total": sum(g["turns_compared"] for g in games),
        "first_mismatches": [g["first_mismatch"] for g in games if g["first_mismatch"]],
    }
    out = {"summary": summary, "games": games}
    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if summary["clean"] == len(games) and not summary["syntax_violations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
