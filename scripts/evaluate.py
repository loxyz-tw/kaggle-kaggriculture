"""Kaggriculture 候選評估器。

功能：

* 雙座位、多 seed 對局矩陣（對 starter / 對 self / 候選互打）。
* 每回合延遲分布（p50 / p95 / p99 / max）與引擎 overage 消耗。
* 存活事件偵測：渴死作物、當日未澆的新播種、逃走動物、shed 溢出。
* 無效動作（送出後引擎必然 no-op）稽核。
* 輸出機器可讀 JSON 與人類可讀 Markdown。

用法：
    uv run python scripts/evaluate.py --seeds 10 --out reports/day0-evaluation
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

import agent_core  # noqa: E402

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
SHED_CAP = 100
TURNS = 24


# --------------------------------------------------------------------------- #
# agent 載入（刻意複製引擎的 get_last_callable 語意，順便驗證「最後一個 callable」契約）
# --------------------------------------------------------------------------- #


def load_agent(path):
    path = str(path)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    env = {}
    d = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, d)
    try:
        exec(compile(src, path, "exec"), env)
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass
    cands = [v for v in env.values() if callable(v)]
    if not cands:
        raise RuntimeError(f"{path}: 找不到任何 callable")
    return cands[-1]


def timed(fn, bucket):
    def wrapped(obs):
        t0 = time.perf_counter()
        try:
            out = fn(obs)
        finally:
            bucket.append(time.perf_counter() - t0)
        return out

    return wrapped


# --------------------------------------------------------------------------- #
# 事件偵測
# --------------------------------------------------------------------------- #


def _tile_kind(t):
    if t is None:
        return "EMPTY"
    if not isinstance(t, dict):
        return "LOCKED"
    if t.get("kind") == "PLANT":
        return "PLANT"
    if "animal" in t:
        return "ANIMAL"
    return t.get("kind", "?")


def _acts_at(step_state, player):
    a = step_state[player].action
    if not isinstance(a, dict):
        return []
    out = []
    f = a.get("farmer")
    if isinstance(f, list) and f:
        out.append(f)
    for h in a.get("hands") or []:
        if isinstance(h, list) and h:
            out.append(h)
    return out


def _unit_positions(obs, player):
    farm = obs["farms"][player]
    pos = [tuple(farm["farmer"])]
    for h in farm["hands"]:
        pos.append((int(h[0]), int(h[1])))
    return pos


def analyse(env, player):
    """從 env.steps 還原我方的存活事件與無效動作。"""
    ev = Counter()
    steps = env.steps
    # env.steps[i].action 是「作用在 steps[i-1].observation 上、產生 steps[i].observation」
    # 的動作，因此觀測與動作要錯開一格配對：obs = steps[i]，act = steps[i+1]。
    n = len(steps) - 1

    planted_on = {}  # (x, y) -> day，記錄本日新播種的格子

    for i in range(n):
        obs = steps[i][0].observation
        priv = steps[i][player].observation["private"]
        farm = obs["farms"][player]
        tiles = farm["tiles"]
        day = int(obs["day"])
        hour = int(obs["hour"])
        pos = _unit_positions(obs, player)
        acts = _acts_at(steps[i + 1], player)
        inv = priv.get("inventories", []) or []
        seeds = dict(priv.get("seeds", {}) or {})

        # ---- 無效動作稽核 ----
        for u, a in enumerate(acts):
            if u >= len(pos):
                ev["act_no_such_unit"] += 1
                continue
            op = a[0]
            x, y = pos[u]
            tile = tiles[y][x]
            kind = _tile_kind(tile)
            uinv = dict(inv[u]) if u < len(inv) and inv[u] else {}
            if op in MOVES or op == "PASS":
                if op == "PASS":
                    ev["act_pass"] += 1
                else:
                    ev["act_move"] += 1
                continue
            ev["act_work"] += 1
            if op == "WATER":
                if kind != "PLANT" or tile.get("watered_today"):
                    ev["noop_water"] += 1
            elif op == "HARVEST":
                if kind not in ("PLANT", "ANIMAL") or int(tile.get("yield_units", 0)) <= 0:
                    ev["noop_harvest"] += 1
            elif op == "PLANT":
                if kind != "EMPTY" or seeds.get(a[1], 0) <= 0:
                    ev["noop_plant"] += 1
                else:
                    seeds[a[1]] -= 1
                    planted_on[(x, y)] = day
            elif op == "FEED":
                if kind != "ANIMAL" or tile.get("fed_today") or uinv.get("WHEAT", 0) <= 0:
                    ev["noop_feed"] += 1
            elif op == "CARE":
                if kind != "ANIMAL" or tile.get("cared_today"):
                    ev["noop_care"] += 1
            elif op == "COLLECT_FERTILIZER":
                if kind != "ANIMAL" or not tile.get("fertilizer_available"):
                    ev["noop_collect"] += 1
            elif op == "FERTILIZE":
                if kind != "PLANT" or uinv.get("FERTILIZER", 0) <= 0:
                    ev["noop_fertilize"] += 1
            elif op in ("BUILD_COOP", "BUILD_PASTURE"):
                if kind != "EMPTY":
                    ev["noop_build"] += 1
            elif op == "DIG":
                if kind in ("EMPTY", "LOCKED", "ANIMAL"):
                    ev["noop_dig"] += 1
            elif op == "PLACE":
                item = a[1] if len(a) > 1 else None
                ok_animal = (
                    item in agent_core.ANIMALS and kind == agent_core.ANIMALS[item]["structure"]
                )
                ok_shed = (x, y) in _shed_tiles(len(tiles))
                held = uinv.get(item, 0) > 0
                if not (ok_animal and held) and not (ok_shed and held):
                    ev["noop_place"] += 1
            elif op == "PICKUP":
                item = a[1] if len(a) > 1 else None
                if (x, y) not in _shed_tiles(len(tiles)) or priv["shed"].get(item, 0) <= 0:
                    ev["noop_pickup"] += 1
            elif op == "DROP":
                if (x, y) not in _shed_tiles(len(tiles)) or not uinv:
                    ev["noop_drop"] += 1

        # ---- 跨日轉換：檢查渴死 / 逃走 / 溢出 ----
        if hour != TURNS - 1 or i + 1 >= len(steps):
            continue
        nxt = steps[i + 1][0].observation
        ntiles = nxt["farms"][player]["tiles"]
        watered_now = {pos[u] for u, a in enumerate(acts) if a[0] == "WATER" and u < len(pos)}
        fed_now = {pos[u] for u, a in enumerate(acts) if a[0] == "FEED" and u < len(pos)}

        for y in range(len(tiles)):
            for x in range(len(tiles)):
                before, after = tiles[y][x], ntiles[y][x]
                kb, ka = _tile_kind(before), _tile_kind(after)
                if kb == "PLANT" and ka == "WEED":
                    thirsty = int(before.get("consecutive_unwatered", 0)) >= 1 and not before.get(
                        "watered_today"
                    )
                    if thirsty and (x, y) not in watered_now:
                        ev["death_thirst"] += 1
                        if planted_on.get((x, y)) == day:
                            ev["death_new_plant_unwatered"] += 1
                    else:
                        ev["death_lifespan"] += 1
                if kb == "ANIMAL" and ka != "ANIMAL":
                    starving = int(before.get("consecutive_unfed", 0)) >= 1 and not before.get(
                        "fed_today"
                    )
                    if starving and (x, y) not in fed_now:
                        ev["animal_escape"] += 1
                    else:
                        ev["animal_lost_other"] += 1
                # 播種當天沒澆到水（尚未變雜草，但明天必死）
                if kb == "PLANT" and planted_on.get((x, y)) == day:
                    if not before.get("watered_today") and (x, y) not in watered_now:
                        ev["new_plant_dry_at_eod"] += 1

        shed_before = sum(priv["shed"].values())
        carried = sum(sum(d.values()) for d in inv if d)
        if shed_before + carried > SHED_CAP:
            ev["shed_overflow_risk"] += 1
            ev["shed_overflow_units"] += shed_before + carried - SHED_CAP
        planted_on = {k: v for k, v in planted_on.items() if v >= day}

    return dict(ev)


def _shed_tiles(board):
    half = board // 2
    return {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}


# --------------------------------------------------------------------------- #
# 對局
# --------------------------------------------------------------------------- #


def play(spec_a, spec_b, seed, my_seat, steps=720):
    """spec_* 是 (name, path_or_builtin)。回傳我方（my_seat）視角的結果。"""
    bucket = []
    agents = []
    for seat, spec in enumerate((spec_a, spec_b)):
        name, ref = spec
        if isinstance(ref, str) and ref in ("pass", "random", "starter"):
            agents.append(ref)
        else:
            fn = load_agent(ref)
            agents.append(timed(fn, bucket) if seat == my_seat else fn)

    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    t0 = time.perf_counter()
    env.run(agents)
    wall = time.perf_counter() - t0

    last = env.steps[-1]
    rewards = [s.reward for s in last]
    statuses = [s.status for s in last]
    overage = []
    for p in range(2):
        try:
            overage.append(float(last[p].observation["remainingOverageTime"]))
        except Exception:
            overage.append(None)

    mine, theirs = rewards[my_seat], rewards[1 - my_seat]
    a, b = mine or 0, theirs or 0
    result = "win" if a > b else ("loss" if a < b else "draw")

    return {
        "seed": seed,
        "seat": my_seat,
        "opponent": spec_b[0] if my_seat == 0 else spec_a[0],
        "my_bank": mine,
        "opp_bank": theirs,
        "result": result,
        "status": statuses,
        "wall_s": round(wall, 2),
        "remaining_overage": overage[my_seat],
        "latency": bucket,
        "events": analyse(env, my_seat),
    }


def latency_stats(samples):
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)

    def q(p):
        return s[min(n - 1, int(round(p * (n - 1))))]

    return {
        "n": n,
        "p50_ms": round(q(0.50) * 1000, 3),
        "p95_ms": round(q(0.95) * 1000, 3),
        "p99_ms": round(q(0.99) * 1000, 3),
        "max_ms": round(s[-1] * 1000, 3),
        "mean_ms": round(statistics.fmean(s) * 1000, 3),
    }


def merge_events(rows):
    total = Counter()
    for r in rows:
        total.update(r["events"])
    return dict(total)


def summarise(rows):
    wins = sum(1 for r in rows if r["result"] == "win")
    losses = sum(1 for r in rows if r["result"] == "loss")
    draws = sum(1 for r in rows if r["result"] == "draw")
    banks = [r["my_bank"] for r in rows if r["my_bank"] is not None]
    by_seat = {}
    for seat in (0, 1):
        sub = [r for r in rows if r["seat"] == seat]
        if sub:
            by_seat[str(seat)] = {
                "n": len(sub),
                "wins": sum(1 for r in sub if r["result"] == "win"),
                "median_bank": round(statistics.median([r["my_bank"] for r in sub]), 1),
            }
    all_done = all(all(s == "DONE" for s in r["status"]) for r in rows)
    return {
        "n": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "all_done": all_done,
        "bank_min": min(banks) if banks else None,
        "bank_median": round(statistics.median(banks), 1) if banks else None,
        "bank_max": max(banks) if banks else None,
        "by_seat": by_seat,
        "events": merge_events(rows),
    }


# --------------------------------------------------------------------------- #
# 跨對局污染檢查
# --------------------------------------------------------------------------- #


def contamination_check(ref, opponent="starter", seed_a=0, seed_b=7):
    """同一個 process 內以 A, B, A 順序重跑；A 的兩次結果必須完全相同。"""
    fn = load_agent(ref)

    def run(seed):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
        env.run([fn, opponent])
        return [s.reward for s in env.steps[-1]], [s.status for s in env.steps[-1]]

    r1 = run(seed_a)
    _ = run(seed_b)
    r3 = run(seed_a)
    return {"seed": seed_a, "first": r1, "repeat": r3, "identical": r1 == r3}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", default="reports/day0-evaluation")
    ap.add_argument("--anchor", default=str(ROOT / "main.py"))
    ap.add_argument("--challenger", default=str(ROOT / "challenger.py"))
    ap.add_argument("--selfplay", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="只跑 2 seed，用於開發回圈")
    args = ap.parse_args()

    if args.quick:
        args.seeds = 2
        args.selfplay = 1

    seeds = list(range(args.seeds))
    cands = {"day0_anchor": args.anchor, "day0_challenger": args.challenger}

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": _engine_version(),
        "episode_steps": args.steps,
        "seeds": seeds,
        "candidates": {},
        "head_to_head": {},
    }

    for name, path in cands.items():
        print(f"\n=== {name} vs starter（{len(seeds)} seed × 2 座位）===", flush=True)
        rows = []
        lat = []
        for seed in seeds:
            for seat in (0, 1):
                a = (name, path) if seat == 0 else ("starter", "starter")
                b = ("starter", "starter") if seat == 0 else (name, path)
                r = play(a, b, seed, seat, args.steps)
                lat.extend(r.pop("latency"))
                rows.append(r)
                print(
                    f"  seed={seed} seat={seat} {r['result']:>4} "
                    f"me={r['my_bank']:>9.0f} opp={r['opp_bank']:>8.0f} "
                    f"status={r['status']} {r['wall_s']}s",
                    flush=True,
                )

        print(f"--- {name} self-play ({args.selfplay} 場) ---", flush=True)
        sp = []
        for k in range(args.selfplay):
            r = play((name, path), (name + "#2", path), 100 + k, 0, args.steps)
            lat.extend(r.pop("latency"))
            sp.append(r)
            print(
                f"  mirror seed={100 + k} {r['result']} {r['my_bank']:.0f} "
                f"vs {r['opp_bank']:.0f} {r['status']}",
                flush=True,
            )

        print(f"--- {name} 跨對局污染檢查 ---", flush=True)
        cc = contamination_check(path)
        print(f"  identical={cc['identical']} {cc['first'][0]} vs {cc['repeat'][0]}", flush=True)

        report["candidates"][name] = {
            "entry": os.path.relpath(path, ROOT),
            "vs_starter": summarise(rows),
            "vs_starter_matches": rows,
            "self_play": summarise(sp),
            "self_play_matches": sp,
            "latency": latency_stats(lat),
            "min_remaining_overage": min(
                [r["remaining_overage"] for r in rows + sp if r["remaining_overage"] is not None]
                or [None]
            ),
            "contamination_check": cc,
        }

    # ---- 候選互打 ----
    print("\n=== anchor vs challenger（paired head-to-head）===", flush=True)
    h2h = []
    for seed in seeds:
        for seat in (0, 1):
            anc = ("day0_anchor", cands["day0_anchor"])
            chl = ("day0_challenger", cands["day0_challenger"])
            a, b = (anc, chl) if seat == 0 else (chl, anc)
            r = play(a, b, seed, seat, args.steps)
            r.pop("latency")
            r["anchor_seat"] = seat
            h2h.append(r)
            print(
                f"  seed={seed} anchor_seat={seat} result={r['result']} "
                f"anchor={r['my_bank']:.0f} challenger={r['opp_bank']:.0f} status={r['status']}",
                flush=True,
            )
    report["head_to_head"] = {"anchor_view": summarise(h2h), "matches": h2h}
    report["failure_correlation"] = _failure_correlation(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(str(out) + ".md", "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    print(f"\n寫出 {out}.json / {out}.md")
    return report


def _engine_version():
    try:
        import kaggle_environments

        return str(getattr(kaggle_environments, "__version__", "unknown"))
    except Exception:
        return "unknown"


def render_markdown(rep):
    L = []
    L.append("# Day 0 評估報告\n")
    L.append(f"- 產生時間：{rep['generated_at']}")
    L.append(f"- 引擎：kaggle-environments {rep['engine']}")
    L.append(f"- episodeSteps：{rep['episode_steps']}；seeds：{rep['seeds']}\n")

    for name, c in rep["candidates"].items():
        s = c["vs_starter"]
        sp = c["self_play"]
        lat = c["latency"]
        L.append(f"## {name}（`{c['entry']}`）\n")
        L.append("### 對 starter（雙座位）\n")
        L.append(f"- 場次 {s['n']}：**{s['wins']} 勝 / {s['losses']} 敗 / {s['draws']} 和**")
        L.append(f"- 全場 DONE：{s['all_done']}")
        L.append(
            f"- 季末銀行 min/median/max：{s['bank_min']:.0f} / "
            f"{s['bank_median']:.0f} / {s['bank_max']:.0f}"
        )
        for seat, v in sorted(s["by_seat"].items()):
            L.append(f"- 座位 {seat}：{v['wins']}/{v['n']} 勝，中位銀行 {v['median_bank']:.0f}")
        L.append("")
        L.append("| seed | seat | 結果 | 我方 | 對手 | status |")
        L.append("|---:|---:|:--|---:|---:|:--|")
        for r in c["vs_starter_matches"]:
            L.append(
                f"| {r['seed']} | {r['seat']} | {r['result']} | {r['my_bank']:.0f} | "
                f"{r['opp_bank']:.0f} | {'/'.join(r['status'])} |"
            )
        L.append("")
        L.append("### Self-play（mirror）\n")
        L.append(f"- 場次 {sp['n']}，全場 DONE：{sp['all_done']}")
        for r in c["self_play_matches"]:
            L.append(
                f"  - seed {r['seed']}: {r['my_bank']:.0f} vs "
                f"{r['opp_bank']:.0f}，status {'/'.join(r['status'])}"
            )
        cc = c["contamination_check"]
        L.append(f"- 跨對局污染檢查（A→B→A 同 process）：identical = **{cc['identical']}**")
        L.append("")
        L.append("### 延遲\n")
        if lat:
            L.append(
                f"- n={lat['n']}，p50 {lat['p50_ms']} ms，p95 {lat['p95_ms']} ms，"
                f"p99 {lat['p99_ms']} ms，max {lat['max_ms']} ms"
            )
        L.append(f"- 引擎剩餘 overage（60 s 起算）最低值：{c['min_remaining_overage']}")
        L.append("")
        L.append("### 事件稽核（全部對局加總）\n")
        L.append("```")
        L.append(json.dumps(_event_view(s["events"], sp["events"]), indent=2, ensure_ascii=False))
        L.append("```")
        L.append("")

    h = rep["head_to_head"]["anchor_view"]
    L.append("## Anchor vs Challenger（paired，anchor 視角）\n")
    L.append(f"- 場次 {h['n']}：anchor **{h['wins']} 勝 / {h['losses']} 敗 / {h['draws']} 和**")
    L.append(f"- 全場 DONE：{h['all_done']}")
    L.append(
        f"- anchor 銀行 min/median/max：{h['bank_min']:.0f} / "
        f"{h['bank_median']:.0f} / {h['bank_max']:.0f}"
    )
    for seat, v in sorted(h["by_seat"].items()):
        L.append(
            f"- anchor 位於座位 {seat}：{v['wins']}/{v['n']} 勝，中位銀行 {v['median_bank']:.0f}"
        )
    L.append("")
    L.append("### Paired 失敗相關性\n")
    L.append("兩個候選在同一 seed／座位下的存活事件（渴死、逃走、shed 溢出）加總：\n")
    L.append("```")
    L.append(json.dumps(_event_view(h["events"], {}), indent=2, ensure_ascii=False))
    L.append("```")
    corr = _failure_correlation(rep)
    L.append(f"- 兩候選同時出現失敗事件的 seed 數：**{corr['both']}**")
    L.append(
        f"- 只有 anchor 出現：{corr['anchor_only']}；"
        f"只有 challenger 出現：{corr['challenger_only']}"
    )
    L.append(f"- 兩者都沒有失敗事件的 seed 數：{corr['neither']}")
    L.append("")
    L.append("| seed | anchor 座位 | anchor | challenger | 結果 | status |")
    L.append("|---:|---:|---:|---:|:--|:--|")
    for r in rep["head_to_head"]["matches"]:
        L.append(
            f"| {r['seed']} | {r['anchor_seat']} | {r['my_bank']:.0f} | {r['opp_bank']:.0f} | "
            f"{r['result']} | {'/'.join(r['status'])} |"
        )
    L.append("")
    return "\n".join(L)


FAILURE_KEYS = (
    "death_thirst",
    "death_new_plant_unwatered",
    "new_plant_dry_at_eod",
    "animal_escape",
    "animal_lost_other",
    "shed_overflow_risk",
)


def _failure_correlation(rep):
    """逐 seed 比對兩個候選是否同時出現存活失敗事件（失敗相關性）。"""

    def by_seed(name):
        out = {}
        for r in rep["candidates"][name]["vs_starter_matches"]:
            bad = sum(r["events"].get(k, 0) for k in FAILURE_KEYS)
            out[r["seed"]] = out.get(r["seed"], 0) + bad
        return out

    a = by_seed("day0_anchor")
    c = by_seed("day0_challenger")
    both = anchor_only = challenger_only = neither = 0
    for seed in sorted(set(a) | set(c)):
        fa, fc = a.get(seed, 0) > 0, c.get(seed, 0) > 0
        if fa and fc:
            both += 1
        elif fa:
            anchor_only += 1
        elif fc:
            challenger_only += 1
        else:
            neither += 1
    return {
        "both": both,
        "anchor_only": anchor_only,
        "challenger_only": challenger_only,
        "neither": neither,
    }


def _event_view(a, b):
    t = Counter(a)
    t.update(b)
    return dict(sorted(t.items()))


if __name__ == "__main__":
    main()
