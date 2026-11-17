#!/usr/bin/env python
"""凍結 IL trunk 上的線性 critic 暖身（計畫 內部規格文件）。

**不更新任何 policy 參數**：V ＝ `h_glob @ value_w + value_b`，trunk 凍結 ⇒ 只剩 193 個參數，
直接用閉式 ridge regression 對 Monte Carlo return（γ=1）求解。

* fit 集（base seed 9300）→ α 以 env 切 80/20 選 → 全 fit 重解。
* held-out 集（base seed 9230，與塌縮診斷同一組）→ 三種 V（IL 未訓練／時間 baseline／ridge）的 EV，
  以及用 ridge V 重算 GAE 後的 M2 分組與預登記判讀 A／B。
"""

from __future__ import annotations

import argparse
import json
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

from rl import full_actions_head as H  # noqa: E402
from rl import full_actions_numpy as FN  # noqa: E402
from rl import full_actions_obs as FO  # noqa: E402
from rl import full_actions_updater as U  # noqa: E402
from rl import loop4_ppo as P1  # noqa: E402
from rl import rules  # noqa: E402
from scripts import full_actions_collapse_diag as D  # noqa: E402
from scripts.full_actions_il_train import _git_porcelain  # noqa: E402
from scripts.full_actions_ppo_cost import (  # noqa: E402
    Checks,
    leaves_sha256,
    require_gpu_highest,
    rss_gib,
    sha256_file,
)
from scripts.full_actions_ppo_train import Ctx  # noqa: E402

CAPS = D.CAPS
ALPHAS = (1e-4, 1e-2, 1.0, 100.0)
EV_MIN = 0.5  # 判讀 A：ridge V 的 held-out EV 下限
EV_MARGIN = 0.05  # 判讀 A：須勝時間 baseline 的幅度
VALUE_ATOL = 1e-4  # 另算的 IL V 對 rollout 存的 V


# ------------------------------------------------------------------ 純函式（tests 用）
def mc_return(reward: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """`[T, B]` → γ=1 的 reward-to-go，只累加 valid 列；無效列回 0。"""
    r = np.where(valid, reward, 0.0)
    g = np.flip(np.cumsum(np.flip(r, 0), 0), 0)
    return np.where(valid, g, 0.0)


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """最後一欄是常數項、不正則化。float64 閉式解。"""
    x = x.astype(np.float64)
    reg = np.full(x.shape[1], alpha)
    reg[-1] = 0.0
    return np.linalg.solve(x.T @ x + np.diag(reg), x.T @ y.astype(np.float64))


def explained_variance(g: np.ndarray, v: np.ndarray) -> float:
    var = float(np.var(g))
    return float(1.0 - np.var(g - v) / var) if var > 0 else float("nan")


def with_const(h: np.ndarray) -> np.ndarray:
    return np.concatenate([h.astype(np.float64), np.ones((h.shape[0], 1))], axis=1)


def gae_with(values_tb: np.ndarray, traj, reward: np.ndarray, cfg=None) -> np.ndarray:
    """與 `U.targets` 同一支 GAE（last_value=0），只換 V。

    `cfg` 給定時走 `U.advantages(cfg, …)`（與 `targets()` 同一分流；winloss＝`gae_gamma(1, 1)`）；
    未給＝舊行為（`P1.gae`）逐位不變。"""
    if cfg is not None:
        adv, _ = U.advantages(
            cfg,
            jnp.asarray(values_tb, jnp.float32),
            jnp.asarray(reward, jnp.float32),
            traj.valid,
            traj.done,
        )
        return np.asarray(adv, np.float64)
    adv, _ = P1.gae(
        jnp.asarray(values_tb, jnp.float32),
        jnp.zeros(values_tb.shape[1:], jnp.float32),
        jnp.asarray(reward, jnp.float32),
        traj.valid,
        traj.done,
    )
    return np.asarray(adv, np.float64)


def select_alpha(x: np.ndarray, y: np.ndarray, env: np.ndarray) -> dict[str, Any]:
    """env id % 5 == 4 當內部驗證（按 env 切，不按列切）。"""
    val = env % 5 == 4
    mse = {}
    for a in ALPHAS:
        w = ridge(x[~val], y[~val], a)
        mse[str(a)] = float(np.mean((x[val] @ w - y[val]) ** 2))
    best = min(ALPHAS, key=lambda a: mse[str(a)])
    return {"alpha": best, "val_mse": mse, "n_train": int((~val).sum()), "n_val": int(val.sum())}


def time_baseline(step_fit: np.ndarray, g_fit: np.ndarray):
    """每個 step 的 fit 集平均 G；held-out 遇到沒見過的 step 用全體平均。"""
    table = np.full(rules.N_TRANSITIONS + 1, float(g_fit.mean()))
    sums = np.bincount(step_fit, weights=g_fit, minlength=table.size)
    cnts = np.bincount(step_fit, minlength=table.size)
    seen = cnts > 0
    table[seen] = sums[seen] / cnts[seen]
    return lambda s: table[np.clip(s, 0, table.size - 1)]


def rule2(m2: dict[str, Any]) -> dict[str, Any]:
    ref, src = D.reference(m2)
    groups = [g for g in D.START_GROUPS if m2[g]["n_rows"] > 0 and m2[g]["ci95"][1] < ref]
    return {"holds": bool(groups), "groups": groups, "reference": ref, "reference_source": src}


# ------------------------------------------------------------------ rollout → 特徵
def make_feature_fn(cfg_model: dict[str, Any], seat: int):
    @jax.jit
    def f(params, st):
        return H.encode_tokens(params, FO.observation(st, seat, CAPS), cfg_model)["glob"]

    return f


def features(feat_fn, params, traj, chunk: int) -> np.ndarray:
    """valid 列的 `h_glob [N, d]`；固定 chunk 大小（尾批 pad）避免重編譯。"""
    valid = np.asarray(traj.valid).reshape(-1)
    idx = np.nonzero(valid)[0]
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), traj.state)
    out = []
    for s in range(0, idx.size, chunk):
        sel = idx[s : s + chunk]
        pad = np.concatenate([sel, np.full(chunk - sel.size, sel[0])]) if sel.size < chunk else sel
        st = jax.tree.map(lambda x, p=pad: x[p], flat)
        out.append(np.asarray(feat_fn(params, st))[: sel.size])
    return np.concatenate(out, axis=0)


def keys_for(base_seed: int, batch: int, seat: int) -> dict[str, Any]:
    k = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(base_seed), batch), seat)
    k_init, k_sample = jax.random.split(k)
    return {"init_seed": int(jax.random.randint(k_init, (), 0, 2**31 - 1)), "sample_key": k_sample}


def run_one(ctx, params, feat_fns, seat, keys, cfg, chunk, checks, label):
    st, _ident = ctx.init_state(keys["init_seed"])
    final, traj = ctx.rollouts[seat](params, st, keys["sample_key"])
    jax.block_until_ready((final, traj))
    tgt = U.targets(traj, final, seat=seat, cfg=cfg)
    summ = D.PB.trace_summary(traj, final, seat=seat)
    checks.add(f"{label}/padding_matches_final", bool(summ["padding_state_equals_final"]))
    reward = np.asarray(tgt["reward"], np.float64)
    valid = np.asarray(traj.valid)
    h = features(feat_fns[seat], params, traj, chunk)
    v_il_recomputed = with_const(h) @ np.concatenate(
        [np.asarray(params["value_w"], np.float64)[:, 0], np.asarray(params["value_b"], np.float64)]
    )
    v_il_stored = np.asarray(traj.value, np.float64)[valid]
    err = float(np.max(np.abs(v_il_recomputed - v_il_stored)))
    checks.add(f"{label}/il_value_recompute_le_{VALUE_ATOL}", err <= VALUE_ATOL, {"max": err})
    adv_rep = gae_with(
        np.asarray(traj.value), traj, reward, cfg if cfg.reward == "winloss" else None
    )
    checks.add(
        f"{label}/gae_reproduces_targets_bitwise",
        bool(np.array_equal(adv_rep.astype(np.float32), np.asarray(tgt["adv_raw"], np.float32))),
    )
    return {
        "traj": traj,
        "final": final,
        "tgt": tgt,
        "reward": reward,
        "valid": valid,
        "h": h,
        "g": mc_return(reward, valid)[valid],
        "step": np.asarray(traj.state.step)[valid].astype(np.int64),
    }


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-batches", type=int, default=4)
    ap.add_argument("--fit-seed", type=int, default=9300)
    ap.add_argument("--heldout-seed", type=int, default=9230)
    ap.add_argument("--seats", default="0,1")
    ap.add_argument("--n-fresh", type=int, default=15)
    ap.add_argument("--t", type=int, default=rules.N_TRANSITIONS)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"已存在，拒絕覆蓋：{out}")
    deps = {
        "scripts/full_actions_critic_warmup.py",
        "scripts/full_actions_collapse_diag.py",
        "scripts/full_actions_ppo_train.py",
        "rl/full_actions_ppo.py",
        "rl/full_actions_updater.py",
        "rl/full_actions_head.py",
        "rl/full_actions_obs.py",
        "rl/loop4_ppo.py",
    }
    dirty = sorted(set(_git_porcelain()) & deps)
    if dirty:
        raise SystemExit(f"依賴檔有未提交改動：{dirty}")
    precision = require_gpu_highest()
    t0 = time.perf_counter()
    checks = Checks()
    seats = [int(s) for s in args.seats.split(",")]

    if D.LEG_DIR is None:
        raise SystemExit("請以環境變數 KAGGRI_COLLAPSE_LEG_DIR 指定 PPO leg 目錄")
    leg = json.loads((D.LEG_DIR / "result.json").read_text())
    il = D.checkpoint_table(leg)["il"]
    il_np, il_meta = FN.load_params(il["path"], caps=CAPS, expect_sha256=il["sha256"])
    cfg_model = FN.config_from_meta(il_meta)
    params = {k: jnp.asarray(v) for k, v in il_np.items()}
    policy_sha_before = leaves_sha256(params)
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
        il_params=params,
    )
    checks.add(
        "teacher/same_as_leg",
        ctx.teacher_source()["teacher_state_leaves_sha256"]
        == run["teacher_source"]["teacher_state_leaves_sha256"],
    )
    feat_fns = {s: make_feature_fn(cfg_model, s) for s in seats}

    fit_keys = {
        (b, s): keys_for(args.fit_seed, b, s) for b in range(args.fit_batches) for s in seats
    }
    held_keys = D.diag_keys(args.heldout_seed)
    fit_seeds = {k["init_seed"] for k in fit_keys.values()}
    held_seeds = {held_keys[s]["init_seed"] for s in seats}
    train_seeds = D.training_init_seeds(leg)
    checks.add(
        "seeds/fit_heldout_training_disjoint",
        not (fit_seeds & held_seeds)
        and not (fit_seeds & train_seeds)
        and not (held_seeds & train_seeds)
        and len(fit_seeds) == len(fit_keys),
        {"fit": sorted(fit_seeds), "held": sorted(held_seeds)},
    )

    # ---- fit 集：只留特徵、G、step、env id（軌跡算完即丟）
    xs, gs, steps, envs = [], [], [], []
    b_env = args.n_fresh + 1
    for (b, s), k in fit_keys.items():
        r = run_one(ctx, params, feat_fns, s, k, cfg, args.chunk, checks, f"fit/b{b}/seat{s}")
        env_tb = np.broadcast_to(np.arange(b_env)[None, :], r["valid"].shape)[r["valid"]]
        xs.append(with_const(r["h"]))
        gs.append(r["g"])
        steps.append(r["step"])
        envs.append(env_tb + (b * len(seats) + seats.index(s)) * b_env)
        del r
        print(
            f"[critic] fit b{b} seat{s} rss={rss_gib()} t={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    x_fit, g_fit = np.concatenate(xs), np.concatenate(gs)
    step_fit, env_fit = np.concatenate(steps), np.concatenate(envs)
    del xs, gs, steps, envs
    sel = select_alpha(x_fit, g_fit, env_fit)
    w = ridge(x_fit, g_fit, sel["alpha"])
    tbase = time_baseline(step_fit, g_fit)
    fit_ev = explained_variance(g_fit, x_fit @ w)

    # ---- held-out 集：EV ＋ M2（三種 advantage）
    parts = {"untrained_il_v": [], "ridge_v": [], "mc_return_no_baseline": []}
    ev_parts = {"g": [], "il": [], "time": [], "ridge": []}
    for s in seats:
        r = run_one(
            ctx, params, feat_fns, s, held_keys[s], cfg, args.chunk, checks, f"heldout/seat{s}"
        )
        v_ridge_valid = with_const(r["h"]) @ w
        ev_parts["g"].append(r["g"])
        ev_parts["il"].append(np.asarray(r["traj"].value, np.float64)[r["valid"]])
        ev_parts["time"].append(tbase(r["step"]))
        ev_parts["ridge"].append(v_ridge_valid)
        v_ridge_tb = np.zeros(r["valid"].shape)
        v_ridge_tb[r["valid"]] = v_ridge_valid
        g_tb = mc_return(r["reward"], r["valid"])
        advs = {
            "untrained_il_v": np.asarray(r["tgt"]["adv_raw"], np.float64),
            "ridge_v": gae_with(v_ridge_tb, r["traj"], r["reward"]),
            "mc_return_no_baseline": g_tb,
        }
        for name, adv in advs.items():
            res = D.analyse(
                r["traj"],
                r["final"],
                {"reward": r["reward"], "adv_raw": adv},
                seat=s,
                scale=cfg.reward_scale,
                env_offset=s * b_env,
            )
            parts[name].append(res["rows"])
        del r
        print(
            f"[critic] heldout seat{s} rss={rss_gib()} t={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    g_h = np.concatenate(ev_parts["g"])
    ev = {k: explained_variance(g_h, np.concatenate(ev_parts[k])) for k in ("il", "time", "ridge")}
    m2 = {name: D.m2_m3(D.pool(p), rng_seed=0)["m2"] for name, p in parts.items()}

    checks.add("policy/params_unchanged", leaves_sha256(params) == policy_sha_before)
    usable = ev["ridge"] >= EV_MIN and ev["ridge"] >= ev["time"] + EV_MARGIN
    r2_untrained = rule2(m2["untrained_il_v"])
    r2_ridge = rule2(m2["ridge_v"])
    verdict = {
        "A_linear_critic_usable": bool(usable),
        "A_detail": {"heldout_ev": ev, "ev_min": EV_MIN, "margin_over_time": EV_MARGIN},
        "B_rule2_untrained_v": r2_untrained,
        "B_rule2_ridge_v": r2_ridge,
        "B_H_CRITIC_supported": (bool(usable) and r2_untrained["holds"] and not r2_ridge["holds"])
        if usable
        else None,
        "B_H_START_survives_ridge_v": r2_ridge["holds"] if usable else None,
        "note": "A 不可用時 B 只報數字不判定；mc_return_no_baseline 只作輔助",
    }
    doc = {
        "schema": "critic-warmup-v1",
        "plan": "內部規格文件",
        "source": {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "matmul_precision": precision,
        "backend": jax.devices()[0].platform,
        "il": il,
        "coefs": cfg._asdict(),
        "fit": {
            "seed": args.fit_seed,
            "batches": args.fit_batches,
            "seats": seats,
            "n_rows": int(g_fit.size),
            "alpha_selection": sel,
            "fit_ev": fit_ev,
        },
        "heldout": {
            "seed": args.heldout_seed,
            "n_rows": int(g_h.size),
            "init_seeds": sorted(held_seeds),
        },
        "m2": m2,
        "verdict": verdict,
    }
    out.mkdir(parents=True)
    np.savez(
        out / "critic_ridge.npz",
        value_w=w[:-1].astype(np.float32)[:, None],
        value_b=w[-1:].astype(np.float32),
        meta=json.dumps(
            {
                "il_sha256": il["sha256"],
                "alpha": sel["alpha"],
                "fit_seed": args.fit_seed,
                "fit_batches": args.fit_batches,
                "source": doc["source"]["git_head"],
                "target": "mc_return_gamma1",
            }
        ),
    )
    doc["critic_ridge_sha256"] = sha256_file(out / "critic_ridge.npz")
    doc["checks"] = checks.items
    doc["all_checks_pass"] = all(i["ok"] for i in checks.items)
    doc["wall_s"] = round(time.perf_counter() - t0, 3)
    doc["peak_rss_gib"] = rss_gib()
    (out / "result.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    (out / "table.md").write_text(table_md(doc))
    print(
        json.dumps(
            {"all_checks_pass": doc["all_checks_pass"], "verdict": verdict}, ensure_ascii=False
        )
    )
    return 0 if doc["all_checks_pass"] else 1


def table_md(doc: dict[str, Any]) -> str:
    v = doc["verdict"]
    ev = v["A_detail"]["heldout_ev"]
    lines = [
        "# Critic 暖身表",
        "",
        f"source `{doc['source']['git_head']}`",
        "",
        f"fit 列數 {doc['fit']['n_rows']:,}、α={doc['fit']['alpha_selection']['alpha']}、"
        f"fit EV {D.fmt(doc['fit']['fit_ev'])}；held-out 列數 {doc['heldout']['n_rows']:,}",
        "",
        "| V | held-out EV |",
        "|---|---:|",
        f"| IL 未訓練 V | {D.fmt(ev['il'])} |",
        f"| 時間 baseline | {D.fmt(ev['time'])} |",
        f"| ridge V | {D.fmt(ev['ridge'])} |",
    ]
    for name, m2 in doc["m2"].items():
        lines += [
            "",
            f"## M2（held-out IL rollout；{name}）",
            "",
            "| 組 | n_rows | env數 | mean | median | CI |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for g, s in m2.items():
            lines.append(
                f"| {g} | {D.fmt(s['n_rows'])} | {D.fmt(s['n_clusters_with_rows'])} | "
                f"{D.fmt(s['mean'])} | {D.fmt(s['median'])} | "
                f"[{D.fmt(s['ci95'][0])}, {D.fmt(s['ci95'][1])}] |"
            )
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


if __name__ == "__main__":
    raise SystemExit(main())
