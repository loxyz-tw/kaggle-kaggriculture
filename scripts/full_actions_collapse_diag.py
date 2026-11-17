#!/usr/bin/env python
"""短程 PPO「鏈啟動塌縮」診斷（計畫 內部規格文件）。

**不訓練、不改任何既有模組**：rollout／init／targets 全部呼叫訓練當時的同一支函式
（`scripts/full_actions_ppo_train.Ctx`、`PB.rollout`、`U.targets`），只在 host 上多算：

* M1 漂移：每個 checkpoint 的 intent 計數、observation 差分認定的效果、兩座絕對終局現金。
* M2 advantage 歸因：依「該步自己這座做了哪類動作（效果）」分組的 `adv_raw`，env-cluster bootstrap。
* M3 reward 分解：r ＝ (a) 自己現金Δ ＋ (b) 自己非現金資產Δ ＋ (c) −對手現金Δ；
  乾淨 SELL 步上「成交現金 − Φ 移除的產品估值」的缺口。

判讀規則照計畫 §2 預先登記的寫死在 `verdict()`，不看結果再挑。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl import full_actions_numpy as FN  # noqa: E402
from rl import full_actions_ppo as PB  # noqa: E402
from rl import full_actions_updater as U  # noqa: E402
from rl import loop4_ppo as P1  # noqa: E402
from rl import rules  # noqa: E402
from scripts.full_actions_il_train import _git_porcelain  # noqa: E402
from scripts.full_actions_ppo_cost import (  # noqa: E402
    Checks,
    leaves_sha256,
    require_gpu_highest,
    rss_gib,
    sha256_file,
)
from scripts.full_actions_ppo_train import Ctx  # noqa: E402

CAPS = PB.CAPS
#: 被診斷的 PPO leg 目錄（內含 result.json）；以環境變數 `KAGGRI_COLLAPSE_LEG_DIR` 指定。
LEG_DIR = Path(os.environ["KAGGRI_COLLAPSE_LEG_DIR"]) if os.environ.get("KAGGRI_COLLAPSE_LEG_DIR") else None
BOOT_R = 10_000
NOOP_MIN_ROWS = 30  # 審查者 BF2：無動作組少於此列數就改用全體 valid 列平均當對照

#: 乾淨 SELL 步允許的 unit verb：不改變自己的產品持有（除了賣掉的那些）。
CLEAN_SELL_VERBS = (
    rules.OP_PASS,
    rules.OP_NORTH,
    rules.OP_SOUTH,
    rules.OP_EAST,
    rules.OP_WEST,
    rules.OP_WATER,
    rules.OP_PLACE,
    rules.OP_CARE,
)
GROUPS = (
    "sell_exec",
    "buy_seed",
    "plant",
    "buy_animal",
    "buy_land",
    "harvest",
    "water_intent",
    "noop",
)
START_GROUPS = ("buy_seed", "plant", "buy_animal")  # 規則 2 的受檢組


# ------------------------------------------------------------------ checkpoint 身分
def checkpoint_table(leg_result: dict[str, Any]) -> dict[str, dict[str, str]]:
    """`il` ＋ `iterN` → {path, sha}。SHA 取自短程 leg 的 result.json（不是自己重算後當期望值）。"""
    out = {
        "il": {
            "path": leg_result["weights"]["il_start"],
            "sha256": leg_result["weights"]["sha256"],
        }
    }
    for name, st in leg_result["stages"].items():
        if name.endswith("/checkpoint") and "actor_sha256" in st:
            out[f"iter{int(st['iteration'])}"] = {
                "path": str(ROOT / st["actor"]),
                "sha256": st["actor_sha256"],
            }
    return out


def training_init_seeds(leg_result: dict[str, Any]) -> set[int]:
    seeds = set()
    for name, st in leg_result["stages"].items():
        if name.endswith("/rollout"):
            for seat in (0, 1):
                s = st.get(f"seat{seat}", {}).get("init_seed")
                if s is not None:
                    seeds.add(int(s))
    return seeds


def diag_keys(base_seed: int) -> dict[int, dict[str, Any]]:
    """每座固定一組 (init_seed, sample_key)，**所有 checkpoint 共用**，才能逐 env 配對。"""
    master = jax.random.PRNGKey(base_seed)
    out = {}
    for seat in (0, 1):
        k_init, k_sample = jax.random.split(jax.random.fold_in(master, seat))
        out[seat] = {
            "init_seed": int(jax.random.randint(k_init, (), 0, 2**31 - 1)),
            "sample_key": k_sample,
        }
    return out


# ------------------------------------------------------------------ reward 分解
def post_of(pre: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """`[T, B]` 的 pre 值 → post 值（最後一列接局末值）。"""
    return np.concatenate([pre[1:], fin[None]], axis=0)


def decompose(traj, final, *, seat: int, scale: float) -> dict[str, np.ndarray]:
    """把 `networth_rewards` 拆成三項，逐項照 P1 的真局末處理（非現金在真局末歸零）。"""
    phi = np.asarray(jax.vmap(lambda st: P1.asset_value(st, seat))(traj.state), np.float64)
    m_pre = np.asarray(traj.state.money[:, :, seat], np.float64)
    o_pre = np.asarray(traj.state.money[:, :, 1 - seat], np.float64)
    m_fin = np.asarray(final.money[:, seat], np.float64)
    o_fin = np.asarray(final.money[:, 1 - seat], np.float64)
    nc_pre = phi - m_pre
    valid = np.asarray(traj.valid)
    last = np.asarray(P1.last_valid_step(traj.valid))
    is_last = np.arange(valid.shape[0])[:, None] == last[None, :]
    m_post = np.where(is_last, m_fin[None], post_of(m_pre, m_fin))
    o_post = np.where(is_last, o_fin[None], post_of(o_pre, o_fin))
    nc_post = np.where(is_last, 0.0, post_of(nc_pre, np.zeros_like(m_fin)))
    v = valid.astype(np.float64)
    return {
        "a_own_cash": (m_post - m_pre) * scale * v,
        "b_own_noncash": (nc_post - nc_pre) * scale * v,
        "c_neg_opp_cash": -(o_post - o_pre) * scale * v,
        "dmoney": (m_post - m_pre) * v,
        "dnoncash": (nc_post - nc_pre) * v,
        "is_last": is_last & valid,
    }


# ------------------------------------------------------------------ 動作分組（效果）
def own_holdings(state, seat: int) -> dict[str, np.ndarray]:
    """`[T, B, …]` state → 自己這座的持有量（全部 `[T, B]`）。"""
    tiles = np.asarray(state.tiles[:, :, seat])
    return {
        "money": np.asarray(state.money[:, :, seat], np.int64),
        "products": np.asarray(state.shed[:, :, seat, : rules.N_PRODUCTS], np.int64).sum(-1)
        + np.asarray(state.inv[:, :, seat, :, : rules.N_PRODUCTS], np.int64).sum((-2, -1)),
        "seeds": np.asarray(state.seeds[:, :, seat], np.int64).sum(-1),
        "planted": (tiles[..., rules.C_KIND] == rules.TK_PLANT).sum(-1),
        "animals": (tiles[..., rules.C_ANIMAL] >= 0).sum(-1),
        "n_quads": np.asarray(state.n_quads[:, :, seat], np.int64),
    }


def holdings_delta(traj, final, seat: int) -> dict[str, np.ndarray]:
    pre = own_holdings(traj.state, seat)
    fin = own_holdings(jax.tree.map(lambda x: x[None], final), seat)
    return {k: post_of(pre[k], fin[k][0]) - pre[k] for k in pre}


def group_masks(
    unit_verb: np.ndarray, market_op: np.ndarray, delta: dict[str, np.ndarray], step: np.ndarray
) -> dict[str, np.ndarray]:
    """`unit_verb [T,B,U]`、`market_op [T,B,M]` ＋ 持有量差分 → 各組 `[T,B]` 遮罩。

    intent 只當必要條件；成立與否看 observation 差分（計畫 §3）。`water_intent` 沒有乾淨的
    效果欄，名字上就標 intent。
    """
    has_v = lambda op: np.any(unit_verb == op, axis=-1)  # noqa: E731
    has_m = lambda op: np.any(market_op == op, axis=-1)  # noqa: E731
    m_nonnone = market_op != rules.MO_NONE
    only_sell = has_m(rules.MO_SELL) & np.all(
        (market_op == rules.MO_NONE) | (market_op == rules.MO_SELL), axis=-1
    )
    clean_verbs = np.all(np.isin(unit_verb, CLEAN_SELL_VERBS), axis=-1)
    not_eod = (step + 1) % 24 != 0  # `_end_of_day` 在 (step+1)%24==0 那步跑
    return {
        # money 只在市場階段變動，而且只有 SELL 會加錢 ⇒ 有 SELL 意圖且淨增 ⇒ 確有成交
        "sell_exec": has_m(rules.MO_SELL) & (delta["money"] > 0),
        "buy_seed": has_m(rules.MO_BUY_SEED) & (delta["seeds"] > 0),
        "plant": has_v(rules.OP_PLANT) & (delta["planted"] > 0),
        "buy_animal": has_m(rules.MO_BUY_ANIMAL) & (delta["animals"] > 0),
        "buy_land": has_m(rules.MO_BUY_LAND) & (delta["n_quads"] > 0),
        "harvest": has_v(rules.OP_HARVEST) & (delta["products"] > 0),
        "water_intent": has_v(rules.OP_WATER),
        "noop": np.all(unit_verb == rules.OP_PASS, axis=-1) & ~np.any(m_nonnone, axis=-1),
        # M3 用：市場只有 SELL、現金淨增、unit 不動產品、非日終 ⇒ Δ現金 ＝ 成交額
        "sell_only": only_sell,
        "clean_sell": only_sell & (delta["money"] > 0) & clean_verbs & not_eod,
    }


# ------------------------------------------------------------------ 統計
def cluster_boot(
    values: np.ndarray, mask: np.ndarray, cluster: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    """env-cluster bootstrap 的 pooled mean（Σ值／Σ列數）。"""
    ids = np.unique(cluster)
    sums = np.array([values[(cluster == i) & mask].sum() for i in ids])
    cnts = np.array([((cluster == i) & mask).sum() for i in ids])
    n = int(cnts.sum())
    if n == 0:
        return {
            "n_rows": 0,
            "n_clusters_with_rows": 0,
            "mean": None,
            "median": None,
            "ci95": [None, None],
        }
    idx = rng.integers(0, len(ids), size=(BOOT_R, len(ids)))
    s, c = sums[idx].sum(1), cnts[idx].sum(1)
    ok = c > 0
    means = s[ok] / c[ok]
    return {
        "n_rows": n,
        "n_clusters_with_rows": int((cnts > 0).sum()),
        "mean": float(sums.sum() / n),
        "median": float(np.median(values[mask])),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "boot_resamples_with_rows": int(ok.sum()),
    }


def analyse(traj, final, tgt, *, seat: int, scale: float, env_offset: int) -> dict[str, Any]:
    """一座 rollout → 逐列（只留 valid）陣列＋M1 計數。"""
    valid = np.asarray(traj.valid)
    dec = decompose(traj, final, seat=seat, scale=scale)
    reward = np.asarray(tgt["reward"], np.float64)
    recomposed = dec["a_own_cash"] + dec["b_own_noncash"] + dec["c_neg_opp_cash"]
    decomp_err = float(np.max(np.abs(recomposed - reward * valid)))
    delta = holdings_delta(traj, final, seat)
    unit_verb = np.asarray(traj.decision["unit_verb"])
    market_op = np.asarray(traj.decision["market_op"])
    step = np.asarray(traj.state.step)
    masks = group_masks(unit_verb, market_op, delta, step)
    t_len, b = valid.shape
    env = np.broadcast_to(np.arange(b)[None, :] + env_offset, (t_len, b))
    rows = {
        "env": env[valid],
        "adv_raw": np.asarray(tgt["adv_raw"], np.float64)[valid],
        "reward": reward[valid],
        "a": dec["a_own_cash"][valid],
        "b": dec["b_own_noncash"][valid],
        "c": dec["c_neg_opp_cash"][valid],
        "dmoney": dec["dmoney"][valid],
        "dnoncash": dec["dnoncash"][valid],
        "is_last": dec["is_last"][valid],
        **{f"g_{k}": v[valid] for k, v in masks.items()},
    }
    vv = valid[..., None]
    m1 = {
        "unit_verb_intent": {
            rules.UNIT_OP_NAMES[op]: int(((unit_verb == op) & vv).sum())
            for op in range(len(rules.UNIT_OP_NAMES))
        },
        "market_op_intent": {
            rules.MARKET_OP_NAMES[op]: int(((market_op == op) & vv).sum())
            for op in range(len(rules.MARKET_OP_NAMES))
            if op != rules.MO_NONE
        },
        "effect_steps": {k: int((masks[k] & valid).sum()) for k in GROUPS if k != "water_intent"},
        "sell_only_revenue": int(dec["dmoney"][masks["sell_only"] & valid].sum()),
        "planted_increase_total": int(np.clip(delta["planted"], 0, None)[valid].sum()),
        "final_money_seat0": [int(x) for x in np.asarray(final.money[:, 0])],
        "final_money_seat1": [int(x) for x in np.asarray(final.money[:, 1])],
    }
    return {"rows": rows, "m1": m1, "decomp_max_abs_err": decomp_err}


def pool(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def m2_m3(rows: dict[str, np.ndarray], rng_seed: int) -> dict[str, Any]:
    env = rows["env"]
    allmask = np.ones_like(rows["adv_raw"], bool)
    m2 = {
        g: cluster_boot(rows["adv_raw"], rows[f"g_{g}"], env, np.random.default_rng(rng_seed))
        for g in GROUPS
    }
    m2["all_valid"] = cluster_boot(rows["adv_raw"], allmask, env, np.random.default_rng(rng_seed))
    gap = rows["dmoney"] + rows["dnoncash"]  # Φ 自己的變動（未乘 scale，錢）
    clean = rows["g_clean_sell"]
    removed = -rows["dnoncash"]
    ratio_ok = clean & (removed > 0)
    comp_var = {k: float(np.var(rows[k])) for k in ("a", "b", "c", "reward")}
    m3 = {
        "clean_sell_gap_money": cluster_boot(gap, clean, env, np.random.default_rng(rng_seed)),
        "sell_exec_own_phi_delta_money": cluster_boot(
            gap, rows["g_sell_exec"], env, np.random.default_rng(rng_seed)
        ),
        "clean_sell_realized_over_phi_ratio": {
            "n": int(ratio_ok.sum()),
            "median": float(np.median(rows["dmoney"][ratio_ok] / removed[ratio_ok]))
            if ratio_ok.any()
            else None,
        },
        "component_variance": comp_var,
        "c_share_of_component_var": comp_var["c"]
        / max(comp_var["a"] + comp_var["b"] + comp_var["c"], 1e-30),
        "terminal_step_b_mean": float(rows["b"][rows["is_last"]].mean())
        if rows["is_last"].any()
        else None,
        "terminal_step_reward_mean": float(rows["reward"][rows["is_last"]].mean())
        if rows["is_last"].any()
        else None,
    }
    return {"m2": m2, "m3": m3}


# ------------------------------------------------------------------ 預登記判讀
def reference(m2: dict[str, Any]) -> tuple[float, str]:
    """規則 1、2 的對照值：無動作組 mean；列數 < 30 時改全體 valid 平均（審查者 BF2）。"""
    if m2["noop"]["n_rows"] >= NOOP_MIN_ROWS:
        return m2["noop"]["mean"], "noop"
    return m2["all_valid"]["mean"], "all_valid_fallback"


def verdict(il: dict[str, Any], m1_il: dict[str, Any], m1_last: dict[str, Any]) -> dict[str, Any]:
    ref, ref_src = reference(il["m2"])
    gap = il["m3"]["clean_sell_gap_money"]
    sell = il["m2"]["sell_exec"]
    r1 = (
        gap["n_rows"] > 0
        and gap["mean"] < 0
        and gap["ci95"][1] < 0
        and sell["n_rows"] > 0
        and sell["ci95"][1] < ref
    )
    r2_groups = [
        g for g in START_GROUPS if il["m2"][g]["n_rows"] > 0 and il["m2"][g]["ci95"][1] < ref
    ]

    def cash(m1):
        s0 = np.asarray(m1["final_money_seat0"], float)
        s1 = np.asarray(m1["final_money_seat1"], float)
        return float(np.median(s0)), float(np.median(s1)), float(np.median(np.abs(s0 - s1)))

    a0, a1, ad = cash(m1_il)
    b0, b1, bd = cash(m1_last)
    drop0 = (a0 - b0) / a0 if a0 > 0 else None
    drop1 = (a1 - b1) / a1 if a1 > 0 else None
    abs_drop = ((a0 - b0) + (a1 - b1)) / 2
    r3 = (
        drop0 is not None
        and drop1 is not None
        and drop0 >= 0.2
        and drop1 >= 0.2
        and abs(bd - ad) < 0.25 * abs_drop
    )
    return {
        "reference": {"value": ref, "source": ref_src, "noop_n_rows": il["m2"]["noop"]["n_rows"]},
        "rule1_H_SELL": bool(r1),
        "rule2_H_START": bool(r2_groups),
        "rule2_groups": r2_groups,
        "rule3_H_SELF_support": bool(r3),
        "rule3_detail": {
            "il_median_cash_seat0_seat1_absdiff": [a0, a1, ad],
            "last_median_cash_seat0_seat1_absdiff": [b0, b1, bd],
            "drop_frac_seat0_seat1": [drop0, drop1],
            "mean_abs_drop": abs_drop,
        },
        "rule4_none": not (r1 or r2_groups or r3),
        "note": "規則 1、2 只用 IL rollout（審查者 BF1）；iter 末的 M2 只作輔助對照",
    }


# ------------------------------------------------------------------ 表
def fmt(x, nd=4):
    return "—" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}")


def table_md(doc: dict[str, Any]) -> str:
    lines = ["# 塌縮診斷表", "", f"source `{doc['source']['git_head']}`", ""]
    ck = list(doc["m1"].keys())
    lines += ["## M1 漂移（兩座合計；intent 與效果分開）", ""]
    keys_v = ["PLANT", "WATER", "HARVEST", "PICKUP", "PLACE", "PASS"]
    keys_m = ["BUY_SEED", "BUY_ANIMAL", "BUY_LAND", "SELL", "HIRE"]
    eff = ["sell_exec", "buy_seed", "plant", "buy_animal", "buy_land", "harvest", "noop"]
    hdr = [
        "ckpt",
        *[f"v:{k}" for k in keys_v],
        *[f"m:{k}" for k in keys_m],
        *[f"e:{k}" for k in eff],
        "sell_only收入",
        "現金中位 s0",
        "現金中位 s1",
        "|差|中位",
    ]
    lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for c in ck:
        m = doc["m1"][c]
        s0 = np.asarray(m["final_money_seat0"], float)
        s1 = np.asarray(m["final_money_seat1"], float)
        vals = [
            c,
            *[fmt(m["unit_verb_intent"].get(k, 0)) for k in keys_v],
            *[fmt(m["market_op_intent"].get(k, 0)) for k in keys_m],
            *[fmt(m["effect_steps"].get(k, 0)) for k in eff],
            fmt(m["sell_only_revenue"]),
            fmt(float(np.median(s0)), 1),
            fmt(float(np.median(s1)), 1),
            fmt(float(np.median(np.abs(s0 - s1))), 1),
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    for c, res in doc["m2m3"].items():
        lines += [
            "",
            f"## M2 advantage 歸因（{c}；adv_raw，env-cluster bootstrap 95%）",
            "",
            "| 組 | n_rows | env數 | mean | median | CI |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for g, s in res["m2"].items():
            lines.append(
                f"| {g} | {fmt(s['n_rows'])} | {fmt(s['n_clusters_with_rows'])} | "
                f"{fmt(s['mean'])} | {fmt(s['median'])} | "
                f"[{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}] |"
            )
        m3 = res["m3"]
        ratio = m3["clean_sell_realized_over_phi_ratio"]
        var_s = ", ".join(fmt(v, 6) for v in m3["component_variance"].values())
        g = m3["clean_sell_gap_money"]
        se = m3["sell_exec_own_phi_delta_money"]
        lines += [
            "",
            f"## M3 reward 分解（{c}）",
            "",
            f"* 乾淨 SELL 步缺口（錢，Δ現金−Φ移除估值）：n={fmt(g['n_rows'])}、"
            f"mean {fmt(g['mean'], 1)}、CI [{fmt(g['ci95'][0], 1)}, {fmt(g['ci95'][1], 1)}]",
            f"* 全部 sell_exec 步的自己 ΦΔ：n={fmt(se['n_rows'])}、"
            f"mean {fmt(se['mean'], 1)}、"
            f"CI [{fmt(se['ci95'][0], 1)}, {fmt(se['ci95'][1], 1)}]",
            f"* 乾淨 SELL 成交額／Φ 估值 中位：{fmt(ratio['median'])}（n={ratio['n']}）",
            f"* 分項變異 a/b/c/r：{var_s}；c 占比 {fmt(m3['c_share_of_component_var'])}",
            f"* 真局末步 b 平均 {fmt(m3['terminal_step_b_mean'])}、reward 平均 "
            f"{fmt(m3['terminal_step_reward_mean'])}",
        ]
    v = doc["verdict"]
    lines += [
        "",
        "## 預登記判讀",
        "",
        "```json",
        json.dumps(v, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default="il,iter1,iter2,iter3,iter4,iter5,iter6,iter7,iter8")
    ap.add_argument("--seats", default="0,1")
    ap.add_argument("--n-fresh", type=int, default=15)
    ap.add_argument("--t", type=int, default=rules.N_TRANSITIONS)
    ap.add_argument("--base-seed", type=int, default=9230)
    ap.add_argument("--out", required=True, help="輸出目錄（必須不存在）")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"已存在，拒絕覆蓋：{out}")
    dirty = sorted(
        set(_git_porcelain())
        & {
            "scripts/full_actions_collapse_diag.py",
            "scripts/full_actions_ppo_train.py",
            "rl/full_actions_ppo.py",
            "rl/full_actions_updater.py",
            "rl/loop4_ppo.py",
        }
    )
    if dirty:
        raise SystemExit(f"依賴檔有未提交改動：{dirty}")
    precision = require_gpu_highest()
    t0 = time.perf_counter()
    checks = Checks()
    if LEG_DIR is None:
        raise SystemExit("請以環境變數 KAGGRI_COLLAPSE_LEG_DIR 指定 PPO leg 目錄")
    leg = json.loads((LEG_DIR / "result.json").read_text())
    table = checkpoint_table(leg)
    names = args.checkpoints.split(",")
    seats = [int(s) for s in args.seats.split(",")]
    missing = [n for n in names if n not in table]
    if missing:
        raise SystemExit(f"result.json 查不到這些 checkpoint：{missing}")

    keys = diag_keys(args.base_seed)
    train_seeds = training_init_seeds(leg)
    checks.add(
        "seeds/disjoint_from_training",
        not ({keys[s]["init_seed"] for s in seats} & train_seeds),
        {"diag": [keys[s]["init_seed"] for s in seats]},
    )

    il_np, il_meta = FN.load_params(
        table["il"]["path"], caps=CAPS, expect_sha256=table["il"]["sha256"]
    )
    cfg_model = FN.config_from_meta(il_meta)
    il_params = {k: jnp.asarray(v) for k, v in il_np.items()}
    cfg = U.Cfg(**{k: leg["coefs"][k] for k in U.Cfg._fields})
    U.check_cfg(cfg)
    run = leg["run"]
    ctx = Ctx(
        cfg_model,
        cfg,
        t=args.t,
        row=int(run["row"]),
        n_fresh=args.n_fresh,
        relabel=-1,
        il_params=il_params,
    )
    checks.add(
        "teacher/same_as_leg",
        ctx.teacher_source()["teacher_state_leaves_sha256"]
        == run["teacher_source"]["teacher_state_leaves_sha256"],
    )

    init_states = {}
    for s in seats:
        st, ident = ctx.init_state(keys[s]["init_seed"])
        init_states[s] = (st, leaves_sha256(st), ident)

    doc: dict[str, Any] = {
        "schema": "collapse-diag-v1",
        "plan": "內部規格文件",
        "source": {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "matmul_precision": precision,
        "backend": jax.devices()[0].platform,
        "coefs": cfg._asdict(),
        "base_seed": args.base_seed,
        "diag_init_seeds": {f"seat{s}": keys[s]["init_seed"] for s in seats},
        "n_fresh": args.n_fresh,
        "T": args.t,
        "init_state_leaves_sha256": {f"seat{s}": init_states[s][1] for s in seats},
        "checkpoints": {},
        "m1": {},
        "m2m3": {},
    }
    rows_by_ckpt = {}
    for name in names:
        ent = table[name]
        npp, _meta = FN.load_params(ent["path"], caps=CAPS, expect_sha256=ent["sha256"])
        params = {k: jnp.asarray(v) for k, v in npp.items()}
        doc["checkpoints"][name] = {**ent, "sha256_verified": True}
        parts, m1s = [], []
        for s in seats:
            st, sha0, _ = init_states[s]
            checks.add(f"{name}/seat{s}/init_state_unchanged", leaves_sha256(st) == sha0)
            t1 = time.perf_counter()
            final, traj = ctx.rollouts[s](params, st, keys[s]["sample_key"])
            jax.block_until_ready((final, traj))
            wall = round(time.perf_counter() - t1, 3)
            tgt = U.targets(traj, final, seat=s, cfg=cfg)
            summ = PB.trace_summary(traj, final, seat=s)
            checks.add(
                f"{name}/seat{s}/padding_matches_final", bool(summ["padding_state_equals_final"])
            )
            res = analyse(
                traj, final, tgt, seat=s, scale=cfg.reward_scale, env_offset=s * (args.n_fresh + 1)
            )
            checks.add(
                f"{name}/seat{s}/decomposition_le_1e-4",
                res["decomp_max_abs_err"] <= 1e-4,
                {"max_abs_err": res["decomp_max_abs_err"]},
            )
            parts.append(res["rows"])
            m1s.append(
                {
                    **res["m1"],
                    "rollout_wall_s": wall,
                    "decomp_max_abs_err": res["decomp_max_abs_err"],
                }
            )
            del traj, final, tgt
        m1 = {
            "unit_verb_intent": {
                k: sum(m["unit_verb_intent"][k] for m in m1s) for k in m1s[0]["unit_verb_intent"]
            },
            "market_op_intent": {
                k: sum(m["market_op_intent"][k] for m in m1s) for k in m1s[0]["market_op_intent"]
            },
            "effect_steps": {
                k: sum(m["effect_steps"][k] for m in m1s) for k in m1s[0]["effect_steps"]
            },
            "sell_only_revenue": sum(m["sell_only_revenue"] for m in m1s),
            "planted_increase_total": sum(m["planted_increase_total"] for m in m1s),
            "final_money_seat0": [x for m in m1s for x in m["final_money_seat0"]],
            "final_money_seat1": [x for m in m1s for x in m["final_money_seat1"]],
            "per_seat": m1s,
        }
        doc["m1"][name] = m1
        rows_by_ckpt[name] = pool(parts)
        if name in ("il", names[-1]):
            doc["m2m3"][name] = m2_m3(rows_by_ckpt[name], rng_seed=0)
        print(
            f"[diag] {name} done rss={rss_gib()}GiB t={time.perf_counter() - t0:.1f}s", flush=True
        )

    doc["verdict"] = verdict(doc["m2m3"]["il"], doc["m1"]["il"], doc["m1"][names[-1]])
    doc["verdict"]["last_checkpoint"] = names[-1]
    doc["checks"] = checks.items
    doc["all_checks_pass"] = all(i["ok"] for i in checks.items)
    doc["wall_s"] = round(time.perf_counter() - t0, 3)
    doc["peak_rss_gib"] = rss_gib()
    out.mkdir(parents=True)
    (out / "diag.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    (out / "table.md").write_text(table_md(doc))
    print(
        json.dumps(
            {"all_checks_pass": doc["all_checks_pass"], "verdict": doc["verdict"]},
            ensure_ascii=False,
        )
    )
    return 0 if doc["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
