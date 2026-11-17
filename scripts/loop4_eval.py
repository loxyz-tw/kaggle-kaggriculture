#!/usr/bin/env python3
"""Loop 4 權重的對局驗收（任務 14 B2 的驗收線）。

**在官方引擎裡跑**，不是在模擬器裡。理由：驗收線是「對 starter」，而 starter 是引擎內建的
Python agent（`kaggriculture.starter_agent`），模擬器沒有它的 JAX 版；
`rl/jax_bots.py::scripted_actions` 的檔頭明寫「不是策略、絕不可進 league 或提交」，不能拿來冒充。
引擎本來就是 oracle，A4 已經證明我們的動作在引擎裡逐步等價，所以在引擎裡驗收沒有損失。

agent 的每回合工作全部 jit（`_policy_action`）——引擎的 ``actTimeout`` 是 1 秒，
未 jit 的 batch-1 前向在 CPU 上要 1.1 秒（A4 踩過）。
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

from rl import loop4_actions as A  # noqa: E402
from rl import loop4_model as M  # noqa: E402
from rl import loop4_obs as O  # noqa: E402
from rl import rules  # noqa: E402
from scripts.loop4_parity import (  # noqa: E402
    CAPS,
    _state_from_obs,
    action_to_engine_json,
    check_action_json,
)


def load_params(path: Path) -> dict:
    z = np.load(path)
    return {k: jnp.asarray(z[k]) for k in z.files}


def make_policy_step(cfg, greedy: bool, topk: bool = True):
    @functools.partial(jax.jit, static_argnums=(4,))
    def policy_action(params, state, assignment, key, caps):
        obs = O.observation(state, 0, caps, assignment)
        out = M.forward(params, obs, cfg)
        mask = A.tile_task_mask(state, caps)[:, 0]
        k = jax.random.split(key, 8)

        def pick(logits, i, m=None):
            lg = logits if m is None else jnp.where(m, logits, -jnp.inf)
            if greedy:
                return jnp.argmax(lg, axis=-1).astype(jnp.int32)
            return jax.random.categorical(k[i], lg, axis=-1).astype(jnp.int32)

        b = state.step.shape[0]
        zero = jnp.zeros((b, rules.N_TILES), jnp.int32)

        def two(x, z):
            return jnp.stack([x, z], axis=1)

        # 每格任務：取「最不像 NOOP」的前 k 格（k = 活著的人手數），其餘保持 NOOP。
        # 每格獨立 argmax 會讓 100 格全部被指派，policy 就失去「要做哪幾格」的控制權。
        #
        # 排序鍵是 **logp_NOOP 升冪**，不是 `log1p(-exp(logp_NOOP))` 降冪。兩者在精確算術下
        # 同序（`log(1-P)` 對 `P` 單調遞減），但後者在 P(NOOP)→1 時是在取一個 1e-5 量級小量的
        # 對數，**把 logits 上 1e-7 的 float32 差放大約 1e5 倍**：任務 20 實測 JAX 端完全平手
        # （同一個 float32 值）的兩格，在 NumPy 端相差 6.3e-3 而翻轉，自走 parity 只剩 97.5%，
        # 四局都在第 6 回合分歧（內部量測紀錄）。
        # 改排序鍵不改決策定義，只是不再把浮點雜訊放大。
        task_lg = jnp.where(mask, out["task"], -jnp.inf)
        raw_task = pick(out["task"], 0, mask)
        if topk:
            logp = jax.nn.log_softmax(task_lg, axis=-1)
            want = -logp[..., A.T_NOOP]  # 越大＝越不像 NOOP
            want = jnp.where(jnp.any(mask[..., 1:], axis=-1), want, -jnp.inf)
            n_alive = jnp.sum(state.hires_today[:, 0]) + 1
            order = jnp.argsort(-want, axis=-1)
            rank = jnp.argsort(order, axis=-1)
            chosen = rank < n_alive
            # 被選中的格取「非 NOOP 類裡最大的那一個」
            non_noop = jnp.where(mask[..., 1:], task_lg[..., 1:], -jnp.inf)
            best_non_noop = jnp.argmax(non_noop, axis=-1).astype(jnp.int32) + 1
            tile_task = jnp.where(chosen, best_non_noop, A.T_NOOP)
        else:
            tile_task = raw_task

        decision = A.Loop4Decision(
            tile_task=two(tile_task, zero),
            tile_qty=two(pick(out["tile_qty"], 1), zero),
            sell=two(pick(out["sell"], 2), jnp.zeros((b, rules.N_PRODUCTS), jnp.int32)),
            buy_seed=two(pick(out["buy_seed"], 3), jnp.zeros((b, rules.N_CROPS), jnp.int32)),
            buy_product=two(pick(out["buy_product"], 4), jnp.zeros((b, 2), jnp.int32)),
            buy_animal=two(pick(out["buy_animal"], 5), jnp.zeros((b, rules.N_ANIMALS), jnp.int32)),
            hire=jnp.stack([pick(out["hire"], 6), jnp.zeros((b,), jnp.int32)], axis=1),
            buy_land=jnp.stack([pick(out["buy_land"], 7), jnp.zeros((b,), jnp.int32)], axis=1),
        )
        _, action, new_assign = A.loop4_step(state, decision, caps, None, assignment)
        return action, new_assign

    return policy_action


#: 九種產品裡 WHEAT／FERTILIZER 是唯二買得到的，所以只有其餘七種的倉庫減量
#: 可以無歧義地當成「結算賣出的單位數」（口徑同 `scripts/sim_daily_diff.py`）。
BUYABLE = (rules.I_WHEAT, rules.I_FERTILIZER)
SELL_ONLY = tuple(i for i in range(rules.N_PRODUCTS) if i not in BUYABLE)


def _daily_probe(obs: dict, seat: int) -> dict:
    """從一座自己的 observation 取逐日拆解要的量（全部是狀態，不是 action intent）。"""
    farm = obs["farms"][seat]
    tiles = [c for row in farm["tiles"] for c in row]
    planted = sum(1 for c in tiles if isinstance(c, dict) and c.get("kind") == "PLANT")
    mature = sum(
        1
        for c in tiles
        if isinstance(c, dict) and c.get("kind") == "PLANT" and c.get("yield_units", 0) > 0
    )
    watered = sum(
        1
        for c in tiles
        if isinstance(c, dict) and c.get("kind") == "PLANT" and c.get("watered_today")
    )
    animals = sum(1 for c in tiles if isinstance(c, dict) and "animal" in c)
    empty = sum(1 for c in tiles if c is None)
    weeds = sum(1 for c in tiles if isinstance(c, dict) and c.get("kind") == "WEED")
    pens = sum(1 for c in tiles if isinstance(c, dict) and c.get("kind") in ("COOP", "PASTURE"))
    locked = sum(1 for c in tiles if c == "LOCKED")
    yield_total = sum(
        int(c.get("yield_units", 0)) for c in tiles if isinstance(c, dict) and "kind" in c
    )
    shed = obs["private"]["shed"]
    held = np.asarray([int(shed.get(p, 0)) for p in rules.PRODUCTS], np.int64)
    prices = np.asarray([int(obs["market"]["prices"][p]) for p in rules.PRODUCTS], np.int64)
    return {
        "day": int(obs["step"]) // rules.TURNS_PER_DAY if "step" in obs else 0,
        "planted": planted,
        "mature": mature,
        "watered": watered,
        "animals": animals,
        "empty": empty,
        "weeds": weeds,
        "pens": pens,
        "locked": locked,
        "yield_total": yield_total,
        "held": held,
        "held_value": int((held * prices).sum()),
        "money": int(farm["money"]),
        "hands": len(farm["hands"]),
        "quads": len(farm["unlocked_quadrants"]),
    }


def fold_daily(probes: list[dict]) -> list[dict]:
    """逐回合 probe → 逐日：存量取當日最後一回合，流量取整日差分合計。"""
    out = []
    for d in range(rules.N_DAYS):
        lo = d * rules.TURNS_PER_DAY
        hi = min(lo + rules.TURNS_PER_DAY, len(probes))
        if lo >= len(probes):
            break
        window = probes[lo:hi]
        last = window[-1]
        sold = 0
        for a_, b_ in zip(window, window[1:], strict=False):
            drop = np.maximum(0, a_["held"] - b_["held"])
            sold += int(drop[list(SELL_ONLY)].sum())
        out.append(
            {
                "day": d,
                "planted_h23": last["planted"],
                "mature_h23": last["mature"],
                "watered_h23": last["watered"],
                "animals_h23": last["animals"],
                "empty_h23": last["empty"],
                "weeds_h23": last["weeds"],
                "pens_h23": last["pens"],
                "locked_h23": last["locked"],
                "yield_on_farm_h23": last["yield_total"],
                "shed_units_h23": int(last["held"].sum()),
                "shed_value_h23": last["held_value"],
                "money_h23": last["money"],
                "hands_h23": last["hands"],
                "quads_h23": last["quads"],
                "sold_units_day": sold,
            }
        )
    return out


def make_agent(
    params,
    cfg,
    seat: int,
    seed: int,
    greedy: bool,
    bad: list,
    topk: bool = True,
    probes: list | None = None,
):
    step = make_policy_step(cfg, greedy, topk)
    box = {"t": 0, "assign": None}

    def agent(obs, _config=None):
        # canonical=True：自己一律放索引 0，模型的每格 head 讀的就是自己的 100 格
        state = _state_from_obs(obs, seat, canonical=True)
        if box["assign"] is None:
            box["assign"] = jnp.full((1, 2, CAPS.max_units), A.NO_ASSIGNMENT, jnp.int32)
        key = jax.random.fold_in(jax.random.PRNGKey(seed), box["t"])
        action, box["assign"] = step(params, state, box["assign"], key, CAPS)
        n_alive = min(1 + int(obs["farms"][seat]["hires_today"]), CAPS.max_units)
        if probes is not None:
            probes.append(_daily_probe(obs, seat))
        js = action_to_engine_json(action, 0, n_alive)
        v = check_action_json(js)
        if v:
            bad.append({"turn": box["t"], "bad": v})
        box["t"] += 1
        return js

    return agent


def make_numpy_agent(
    params,
    cfg,
    seat: int,
    seed: int,
    greedy: bool,
    bad: list,
    probes: list | None = None,
    timing: list | None = None,
):
    """`--backend numpy`：走 `rl/loop4_numpy.py` 的**可部署**路徑，其餘全部同一套。

    為什麼要有這一條（任務 22）：任務 20／21 量到耦合 parity PASS，但自走時在精確平手處會翻格
    （IL 權重 P(NOOP)≈1 的固有性質），而 B3 的 PPO policy 是逐格 categorical 抽樣，
    JAX 與 NumPy 的 RNG 流不可能一致。所以**強度要直接在 NumPy agent 上量**，
    不能拿 JAX 面板代表提交包。

    取樣用 ``numpy.random.default_rng(seed)``；與 JAX 的取樣**不會**逐局相同，只比分佈。
    ``timing`` 給了就逐回合記秒數（`time.perf_counter`），給任務 22 的延遲用。
    """
    from rl import loop4_numpy as NP  # noqa: PLC0415  # 只有這條路徑需要它

    rng = None if greedy else np.random.default_rng(seed)
    agent_np = NP.Loop4NumpyAgent(params, seat, CAPS, cfg, rng)
    box = {"t": 0}

    def agent(obs, _config=None):
        if probes is not None:
            probes.append(_daily_probe(obs, seat))
        t0 = time.perf_counter()
        js = agent_np(obs)
        if timing is not None:
            timing.append(time.perf_counter() - t0)
        v = check_action_json(js)
        if v:
            bad.append({"turn": box["t"], "bad": v})
        box["t"] += 1
        return js

    return agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument(
        "--seat0",
        type=int,
        default=0,
        choices=(0, 1),
        help="第一局坐哪一座（之後逐局交替）。預設 0＝原行為；"
        "把 64 局切成多條 lane 平行跑時，用它把 lane 的座位奇偶對回整體序號",
    )
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-topk", action="store_true", help="每格獨立 argmax（對照用）")
    ap.add_argument("--anatomy", action="store_true", help="逐日拆解：種植格／收成／賣出／現金")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument(
        "--backend",
        choices=("jax", "numpy"),
        default="jax",
        help="jax＝既有路徑（預設，行為不變）；numpy＝rl/loop4_numpy.py 的可部署路徑",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.backend == "numpy":
        from rl import loop4_numpy as NP  # noqa: PLC0415

        if a.no_topk:
            raise SystemExit("--no-topk 只有 jax backend 有；numpy 部署路徑一律 top-k")
        params = NP.load_params(str(ROOT / a.params))
        cfg = NP.config_from_params(params, CAPS)
    else:
        params = load_params(ROOT / a.params)
        cfg = M.Config.default(CAPS, d_model=a.d_model)
    timing: list = []
    games = []
    t0 = time.perf_counter()
    for i in range(a.games):
        seed = a.seed0 + i
        seat = (a.seat0 + i) % 2  # 兩座各半，避免先手優勢混進勝率
        bad: list = []
        env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        probes: list | None = [] if a.anatomy else None
        if a.backend == "numpy":
            ours = make_numpy_agent(params, cfg, seat, seed, a.greedy, bad, probes, timing)
        else:
            ours = make_agent(params, cfg, seat, seed, a.greedy, bad, not a.no_topk, probes)
        agents = [ours, a.opponent] if seat == 0 else [a.opponent, ours]
        env.run(agents)
        rewards = env.steps[-1]
        money = [float(s.reward) for s in rewards]
        games.append(
            {
                "seed": seed,
                "seat": seat,
                "ours": money[seat],
                "opp": money[1 - seat],
                "win": money[seat] > money[1 - seat],
                "statuses": [s.status for s in rewards],
                "syntax_violations": len(bad),
                "daily": fold_daily(probes) if probes else None,
            }
        )
    ours = [g["ours"] for g in games]
    summary = {
        "params": a.params,
        "backend": a.backend,
        "opponent": a.opponent,
        "seed0": a.seed0,
        "seat0": a.seat0,
        "greedy": a.greedy,
        "topk": not a.no_topk,
        "games": len(games),
        "wins": sum(g["win"] for g in games),
        "own_cash_median": statistics.median(ours),
        "own_cash_min": min(ours),
        "own_cash_max": max(ours),
        "opp_cash_median": statistics.median(g["opp"] for g in games),
        "syntax_violations": sum(g["syntax_violations"] for g in games),
        "non_done": [g["seed"] for g in games if set(g["statuses"]) != {"DONE"}],
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    if timing:
        # 首回合另列：NumPy 沒有 JIT，但第一回合仍含 numpy／BLAS 的暖身
        srt = sorted(timing[1:])

        def _q(p):
            return srt[min(len(srt) - 1, max(0, int(round(p * (len(srt) - 1)))))]

        summary["turn_latency_s"] = {
            "n": len(srt),
            "first_turn": timing[0],
            "p50": _q(0.50),
            "p90": _q(0.90),
            "p99": _q(0.99),
            "max": srt[-1],
            "mean": statistics.fmean(srt),
            "over_act_timeout_1s": sum(1 for x in srt if x > 1.0),
        }
    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"summary": summary, "games": games}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
