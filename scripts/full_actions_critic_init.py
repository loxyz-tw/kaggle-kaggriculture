#!/usr/bin/env python
"""C2 式暖身 critic（T3 PPO 穩定化計畫 內部規格文件 §3）。

凍結 actor（新 IL）與 trunk，只以 ridge 擬合 `value_w`／`value_b` 到**完整 own MC return**
（`cfg.reward`／γ 由命令列顯式給；本計畫為 networth_own、γ1、scale 1e-4），再換算回原始
h_glob 尺度：`value_w = w/σ`、`value_b = b − Σ μ·w/σ`。

- 輸入全部顯式：`--ckpt/--expect-ckpt-sha`、`--reward/--gamma`、`--row`（教師列）、
  混池 `--pool-manifest/--expect-pool-manifest-sha`；不讀任何歷史 leg 目錄、不比舊探針 EV。
- **只用 fresh env 列**擬合／選 α／報 held-out；每批最後的教師列（row150）排除，另報其列數。
- 分組以完整 episode 為單位（`(batch, seat, env)`；兩座是獨立遊戲），fit 內 `gid % 5 == 4`
  當 validation 選 α；held-out 用另一 seed，seed 與 fit、訓練前 10 輪 init seed 皆不重疊。
- 硬檢：真的 `H.value` 前向對 ridge 預測 max_abs ≤1e-4、finite、held-out EV ≥0.4；
  policy 葉 SHA 不變。任一失敗 rc=1，不追加 fit 批次。

winloss（計畫 內部規格文件 v3.1 §3）：`--reward winloss`，目標＝整局
常數 z＝sign(own−opp)（γ1、λ1、scale 1）。held-out 改為**另一 seed 的 `--heldout-batches` 批**
（與 fit 同一輪換：第 b 批座 s 對 `(b+s)%N`，須覆蓋全部成員×座）；EV≥0.4 門改為事前凍結的
(1) held-out 目標非退化（方差 >0 且 W、L 局數皆 ≥1）、(2) held-out MSE ≤0.95×常數基準
（基準＝fit 集平均 z，只由 fit 決定）、(3) parity／finite／policy 葉不變。報告另列成員×座
W/L/T 與 MSE vs 常數、四步段 MSE，並註明整體通過是否只靠末段（不作門）。
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
from rl import full_actions_updater as U  # noqa: E402
from rl import rules  # noqa: E402
from scripts import full_actions_collapse_diag as D  # noqa: E402
from scripts import full_actions_critic_warmup as W  # noqa: E402
from scripts.full_actions_il_train import _git_porcelain  # noqa: E402
from scripts.full_actions_ppo_cost import (  # noqa: E402
    Checks,
    leaves_sha256,
    require_gpu_highest,
    rss_gib,
    sha256_file,
)
from scripts.full_actions_ppo_train import (  # noqa: E402
    SEATS,
    SOURCE_FILES,
    Ctx,
    critic_identity,
    derive_iteration_keys,
    load_pool_actors,
    load_tape_pool,
    pool_league,
)

CAPS = D.CAPS
EV_MIN = 0.4
CONVERT_ATOL = 1e-4
SCHEMA = "t3-critic-init-v2"
FNAME = "critic_own.npz"
#: winloss（計畫 v3.1 §3）：held-out MSE ≤ 此比例 × fit 常數基準（與 trainer 結果門同值）
WINLOSS_MSE_RATIO = 0.95
#: winloss 報告的局內步段（不作門）
STEP_SEGMENTS = ((0, 180), (180, 360), (360, 540), (540, 719))
#: 訓練前 N 輪的 init seed 也要與 fit／held-out 不重疊（正式最多 10 輪）
TRAIN_ITERS_CHECK = 12  # ppo150k 規格 §6 B1：檢查覆蓋本案最多 12 iters（原 10）


def fresh_mask(valid: np.ndarray, n_fresh: int) -> np.ndarray:
    """`[T, B]` valid 裡只留 fresh env（排除最後的教師列）。"""
    m = np.asarray(valid).copy()
    m[:, n_fresh:] = False
    return m


def training_init_seeds(base_seed: int, n_iters: int) -> set[int]:
    """正式 trainer 從 `PRNGKey(base_seed)` 派生的前 n 輪兩座 init seed。"""
    master = jax.random.PRNGKey(base_seed)
    out = set()
    for it in range(n_iters):
        master, per_seat = derive_iteration_keys(master, it)
        out |= {per_seat[s]["init_seed"] for s in SEATS}
    return out


def raw_value_weights(w: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    """標準化空間的 ridge 權重（末項為常數）→ 原始 h_glob 空間的 `value_w [d,1]`、`value_b [1]`。"""
    scale = w[:-1] / (sd + 1e-8)
    return scale[:, None], np.array([w[-1] - float(np.sum(mu * scale))])


def fit_standardized_ridge(x: np.ndarray, y: np.ndarray, env: np.ndarray):
    """與 `full_actions_advantage_probe.fit_ridge` 同一套：μσ 取 train split、α 以 val 選。"""
    va = env % 5 == 4
    mu, sd = x[~va].mean(0), x[~va].std(0)
    zt = W.with_const((x[~va] - mu) / (sd + 1e-8))
    zv = W.with_const((x[va] - mu) / (sd + 1e-8))
    mse, ws = {}, {}
    for a in W.ALPHAS:
        ws[a] = W.ridge(zt, y[~va], a)
        mse[a] = float(np.mean((zv @ ws[a] - y[va]) ** 2))
    a_best = min(W.ALPHAS, key=lambda a: mse[a])
    return ws[a_best], mu, sd, {"alpha": a_best, "val_mse": mse[a_best]}


def rows_of(r: dict[str, Any], n_fresh: int, gid_offset: int) -> dict[str, np.ndarray]:
    """一次 rollout → fresh 列的特徵、MC return、episode 分組 id；教師列只計數。"""
    valid = r["valid"]
    fm = fresh_mask(valid, n_fresh)
    g_tb = W.mc_return(r["reward"], valid)
    # r["h"] 是 valid 列（row-major [T, B]）的特徵；取其中 fresh 的子集
    keep = fm[valid]
    b = valid.shape[1]
    env_tb = np.broadcast_to(np.arange(b)[None, :], valid.shape)
    return {
        "h": r["h"][keep],
        "g": g_tb[fm],
        "gid": env_tb[fm] + gid_offset,
        "step": np.asarray(r["step"])[keep],
        "n_teacher_rows": int(valid[:, n_fresh:].sum()),
    }


def episode_outcomes(g: np.ndarray, gid: np.ndarray) -> dict[int, float]:
    """winloss：每局（gid）的 z（整局常數；不同步取值不一致即拒——那代表 target 不是整局常數）。"""
    out: dict[int, float] = {}
    for k in np.unique(gid):
        z = np.unique(g[gid == k])
        if z.size != 1 or float(z[0]) not in (-1.0, 0.0, 1.0):
            raise SystemExit(f"winloss 目標在局 {int(k)} 內不是單一 z∈{{-1,0,1}}：{z[:4]}")
        out[int(k)] = float(z[0])
    return out


def wlt(zs) -> dict[str, int]:
    zs = list(zs)
    return {
        "W": sum(z > 0 for z in zs),
        "L": sum(z < 0 for z in zs),
        "T": sum(z == 0 for z in zs),
        "n_episodes": len(zs),
    }


def mse_vs_const(g: np.ndarray, v: np.ndarray, const: float) -> dict[str, Any]:
    if g.size == 0:
        return {"n_rows": 0, "mse": None, "const_mse": None}
    return {
        "n_rows": int(g.size),
        "mse": float(np.mean((g - v) ** 2)),
        "const_mse": float(np.mean((g - const) ** 2)),
    }


def winloss_report(held: dict[str, np.ndarray], const: float, groups: dict) -> dict[str, Any]:
    """winloss 的 held-out 門與報告欄（計畫 v3.1 §3）。

    `held`：`g`／`v`／`step`／`gid` 逐列；`groups`：gid → (member idx, seat)。"""
    g, v, step, gid = held["g"], held["v"], held["step"], held["gid"]
    ep = episode_outcomes(g, gid)
    tot = wlt(ep.values())
    heldout_mse = float(np.mean((g - v) ** 2))
    const_mse = float(np.mean((g - const) ** 2))
    per_ms = {}
    for key in sorted(set(groups.values())):
        gids = [k for k in ep if groups[k] == key]
        m = np.isin(gid, gids)
        per_ms[f"member{key[0]}/seat{key[1]}"] = {
            **wlt(ep[k] for k in gids),
            **mse_vs_const(g[m], v[m], const),
        }
    seg = {}
    for lo, hi in STEP_SEGMENTS:
        m = (step >= lo) & (step < hi)
        seg[f"{lo}-{hi - 1}"] = mse_vs_const(g[m], v[m], const)
    early = step < STEP_SEGMENTS[-1][0]
    e = mse_vs_const(g[early], v[early], const)
    passes = heldout_mse <= WINLOSS_MSE_RATIO * const_mse
    early_pass = e["n_rows"] > 0 and e["mse"] <= WINLOSS_MSE_RATIO * e["const_mse"]
    return {
        "heldout_W": tot["W"],
        "heldout_L": tot["L"],
        "heldout_T": tot["T"],
        "heldout_n_episodes": tot["n_episodes"],
        "heldout_target_var": float(np.var(g)),
        "heldout_mse": heldout_mse,
        "const_value_fit_mean_z": const,
        "const_mse": const_mse,
        "mse_ratio": heldout_mse / const_mse if const_mse > 0 else None,
        "mse_ratio_max": WINLOSS_MSE_RATIO,
        "mse_gate_pass": bool(passes),
        "per_member_seat": per_ms,
        "step_segments": seg,
        "excluding_last_segment": {**e, "passes_same_ratio": bool(early_pass)},
        "overall_pass_relies_on_last_segment": bool(passes and not early_pass),
        "note": "報告欄不作門；常數基準只由 fit 決定、不讀 held-out 調參",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expect-ckpt-sha", required=True)
    ap.add_argument("--reward", required=True, choices=("networth_own", "winloss"))
    ap.add_argument("--gamma", required=True, type=float)
    ap.add_argument("--row", required=True, type=int, help="教師列（B 構造的最後一列）")
    ap.add_argument("--opponent", required=True, choices=("tape-pool", "il-frozen", "pool"))
    ap.add_argument(
        "--pool-actor",
        action="append",
        default=None,
        help="opponent=pool 必填（可重複；與 trainer 同序）；其他對手必空",
    )
    ap.add_argument("--expect-pool-actor-sha", action="append", default=None, help="同上逐一對應")
    ap.add_argument("--pool-manifest", default="", help="opponent=tape-pool 必填；il-frozen 必空")
    ap.add_argument("--expect-pool-manifest-sha", default="", help="同上")
    ap.add_argument("--fit-batches", type=int, default=4)
    ap.add_argument("--fit-seed", type=int, default=9300)
    ap.add_argument("--heldout-seed", type=int, default=9230)
    ap.add_argument(
        "--heldout-batches",
        type=int,
        default=0,
        help=(
            "winloss 必填（≥池長；與 fit 同輪換、另一 seed）；"
            "networth_own 必 0（舊：單一 held-out 輪）"
        ),
    )
    ap.add_argument("--train-seed", type=int, default=0, help="正式 trainer 的 --seed（核不重疊）")
    ap.add_argument("--n-fresh", type=int, default=15)
    ap.add_argument("--t", type=int, default=rules.N_TRANSITIONS)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument(
        "--allow-dirty-cpu-test", action="store_true", help=argparse.SUPPRESS
    )  # 只給 CPU tests：依賴 dirty 也跑（結果不可當正式暖身）
    ap.add_argument(
        "--relabel-step", type=int, default=-1, help=argparse.SUPPRESS
    )  # 只給 CPU tests：與 trainer 同一支教師／fresh 時間重標（讓短 T 也走到真局末）
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"已存在，拒絕覆蓋：{out}")
    deps = {
        *SOURCE_FILES,
        "scripts/full_actions_critic_init.py",
        "scripts/full_actions_critic_warmup.py",
        "scripts/full_actions_collapse_diag.py",
    }
    dirty = sorted(set(_git_porcelain()) & deps)
    if dirty and not args.allow_dirty_cpu_test:
        raise SystemExit(f"依賴檔有未提交改動：{dirty}")
    precision = require_gpu_highest()
    backend = jax.devices()[0].platform
    if (args.allow_dirty_cpu_test or args.relabel_step != -1) and backend != "cpu":
        raise SystemExit("--allow-dirty-cpu-test／--relabel-step 只准在 CPU tests 用")
    t0 = time.perf_counter()
    checks = Checks()

    ckpt = Path(args.ckpt)
    ckpt_sha = sha256_file(ckpt)
    if ckpt_sha != args.expect_ckpt_sha:
        raise SystemExit(f"ckpt SHA 不符：期望 {args.expect_ckpt_sha}、實算 {ckpt_sha}")
    il_np, il_meta = FN.load_params(ckpt, caps=CAPS, expect_sha256=ckpt_sha)
    cfg_model = FN.config_from_meta(il_meta)
    params = {k: jnp.asarray(v) for k, v in il_np.items()}
    if float(args.gamma) != 1.0:
        # rows_of 的目標是 γ1 MC return；折扣 target 未實作，不以 identity 名稱冒充
        raise SystemExit(f"本工具只支援 γ=1（得到 {args.gamma}）")
    winloss = args.reward == "winloss"
    if winloss:
        # 計畫 v3.1 §1／§2：winloss 固定 γ1、λ1、scale 1（`check_cfg` 再核一次）
        cfg = U.Cfg(
            reward=args.reward,
            gamma=args.gamma,
            lam=U.WINLOSS_LAMBDA,
            reward_scale=U.WINLOSS_SCALE,
        )
        if args.heldout_batches < 1:
            raise SystemExit("winloss 需要 --heldout-batches ≥1（須覆蓋全部成員×座）")
    else:
        cfg = U.Cfg(reward=args.reward, gamma=args.gamma)
        if args.heldout_batches:
            raise SystemExit("--heldout-batches 只給 winloss；networth_own 沿舊單一 held-out 輪")
    U.check_cfg(cfg)
    if args.opponent == "tape-pool":
        if not (args.pool_manifest and args.expect_pool_manifest_sha):
            raise SystemExit(
                "opponent=tape-pool 需要 --pool-manifest 與 --expect-pool-manifest-sha"
            )
        tape_bank, pool_man = load_tape_pool(args.pool_manifest, args.expect_pool_manifest_sha)
        pool_sha = args.expect_pool_manifest_sha
    else:  # il-frozen／pool：對手是凍結 actor（il-frozen＝--ckpt 自己），與 trainer 的 Ctx 同構
        if args.pool_manifest or args.expect_pool_manifest_sha:
            raise SystemExit(
                "--pool-manifest／--expect-pool-manifest-sha 只在 opponent=tape-pool 時使用"
            )
        tape_bank, pool_man, pool_sha = None, None, None
    pool_params, league = None, None
    if args.opponent == "pool":
        pool_params, pool_shas = load_pool_actors(
            args.pool_actor or [], args.expect_pool_actor_sha or [], il_np, cfg_model
        )
        league = pool_league(ckpt_sha, pool_shas)
        if len(set(league["members_sha256"])) != len(league["members_sha256"]):
            raise SystemExit(f"池成員（含 --ckpt）SHA 重複：{league['members_sha256']}")
        # league 覆蓋（計畫 §2）：fit 第 b 批＝第 b 輪 ⇒ 座 s 對 (b+s)%N；須每名成員×每座都出現
        n_pool = len(league["members_sha256"])
        cover = {((b + s) % n_pool, s) for b in range(args.fit_batches) for s in SEATS}
        if len(cover) != n_pool * len(SEATS):
            raise SystemExit(
                f"--fit-batches {args.fit_batches} 不足以覆蓋池 {n_pool} 名×{len(SEATS)} 座"
                f"（需 ≥{n_pool}）"
            )
    elif args.pool_actor or args.expect_pool_actor_sha:
        raise SystemExit("--pool-actor／--expect-pool-actor-sha 只在 opponent=pool 時使用")
    ident = critic_identity(
        ckpt_sha=ckpt_sha,
        cfg=cfg,
        pool_sha=pool_sha,
        il_meta=il_meta,
        opponent=args.opponent,
        league=league,
    )
    ctx = Ctx(
        cfg_model,
        cfg,
        t=args.t,
        row=args.row,
        n_fresh=args.n_fresh,
        relabel=args.relabel_step,
        il_params=params,
        opponent=args.opponent,
        pool_params=pool_params,
        tape_bank=tape_bank,
        tape_pool_manifest=pool_man,
        pool_members_sha256=league["members_sha256"] if league else None,
    )
    feat_fns = {s: W.make_feature_fn(cfg_model, s) for s in SEATS}
    fit_keys = {
        (b, s): W.keys_for(args.fit_seed, b, s) for b in range(args.fit_batches) for s in SEATS
    }
    held_keys = D.diag_keys(args.heldout_seed)
    fit_seeds = {k["init_seed"] for k in fit_keys.values()}
    held_seeds = {held_keys[s]["init_seed"] for s in SEATS}
    held_batch_keys = {}
    if winloss:
        # 另一 seed、各 heldout_batches 批輪換（第 b 批＝第 b 輪的對手索引表，與 fit 同規則）
        held_batch_keys = {
            (b, s): W.keys_for(args.heldout_seed, b, s)
            for b in range(args.heldout_batches)
            for s in SEATS
        }
        held_seeds = {k["init_seed"] for k in held_batch_keys.values()}
        if league:
            n_pool = len(league["members_sha256"])
            cover_h = {((b + s) % n_pool, s) for b in range(args.heldout_batches) for s in SEATS}
            if len(cover_h) != n_pool * len(SEATS):
                raise SystemExit(
                    f"--heldout-batches {args.heldout_batches} 不足以覆蓋池 {n_pool} 名×"
                    f"{len(SEATS)} 座（需 ≥{n_pool}）"
                )
    train_seeds = training_init_seeds(args.train_seed, TRAIN_ITERS_CHECK)
    split = {
        "fit_seed": args.fit_seed,
        "fit_batches": args.fit_batches,
        "fit_init_seeds": sorted(fit_seeds),
        "heldout_seed": args.heldout_seed,
        "heldout_init_seeds": sorted(held_seeds),
        "train_seed": args.train_seed,
        "train_iters_checked": TRAIN_ITERS_CHECK,
        "fit_tape_pool_iterations": (
            list(range(args.fit_batches)) if args.opponent == "tape-pool" else None
        ),
        "heldout_tape_pool_iteration": args.fit_batches if args.opponent == "tape-pool" else None,
        **(
            {
                "pool_league": league,
                # 第 b 批（＝第 b 輪）座 s 的對手成員 idx；held-out 用第 fit_batches 輪
                "fit_pool_index": {
                    f"b{b}/seat{s}": ctx.pool_index_at(b, s)
                    for b in range(args.fit_batches)
                    for s in SEATS
                },
                "heldout_pool_index": (
                    {
                        f"b{b}/seat{s}": ctx.pool_index_at(b, s)
                        for b in range(args.heldout_batches)
                        for s in SEATS
                    }
                    if winloss
                    else {f"seat{s}": ctx.pool_index_at(args.fit_batches, s) for s in SEATS}
                ),
            }
            if league
            else {}
        ),
        **(
            {
                "heldout_batches": args.heldout_batches,
                "heldout_rotation": "第 b 批＝第 b 輪的對手索引表（與 fit 同規則、另一 seed）",
            }
            if winloss
            else {}
        ),
        "opponent": args.opponent,
        "grouping": "(batch, seat, env) 一局一組；兩座是各自 reset 的獨立遊戲",
        "validation": "fit 內 gid % 5 == 4",
        "rows": "只用 fresh env；教師列（最後一列）排除",
    }
    checks.add(
        "split/seeds_disjoint",
        not (fit_seeds & held_seeds)
        and not (fit_seeds & train_seeds)
        and not (held_seeds & train_seeds)
        and len(fit_seeds) == len(fit_keys)
        and (not winloss or len(held_seeds) == len(held_batch_keys)),
        {"fit": sorted(fit_seeds), "held": sorted(held_seeds)},
    )
    stop = [i["name"] for i in checks.items if not i["ok"]]
    if stop:
        raise SystemExit(f"split 身分不過：{stop}")

    b_env = args.n_fresh + 1
    parts, pool_recs, n_teacher, seen_pool = [], {}, 0, set()
    for (b, s), k in fit_keys.items():
        ctx.iteration = b  # 混池 lane 規則沿輪次；fit 第 b 批 ＝ 第 b 輪的索引表
        pool_recs[f"fit/b{b}"] = ctx.tape_pool_record(b_env)
        if league:
            seen_pool.add((ctx.opp_index(s), s))
        r = W.run_one(ctx, params, feat_fns, s, k, cfg, args.chunk, checks, f"fit/b{b}/seat{s}")
        rw = rows_of(r, args.n_fresh, (b * len(SEATS) + s) * b_env)
        n_teacher += rw.pop("n_teacher_rows")
        parts.append(rw)
        del r
        print(f"[critic-init] fit b{b} seat{s} t={time.perf_counter() - t0:.1f}s", flush=True)
    fit = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if league:
        n_pool = len(league["members_sha256"])
        checks.add(
            "pool/fit_covers_all_members_x_seats",
            seen_pool == {(i, s) for i in range(n_pool) for s in SEATS},
            {"seen": sorted(seen_pool), "n_pool": n_pool},
        )
    del parts
    if not float(np.var(fit["g"])) > 0:
        raise SystemExit("fit 目標方差為 0：停止（不改報酬、不追加批次）")
    w, mu, sd, rec = fit_standardized_ridge(fit["h"], fit["g"], fit["gid"])
    value_w, value_b = raw_value_weights(w, mu, sd)
    new_params = {
        **params,
        "value_w": jnp.asarray(value_w, jnp.float32),
        "value_b": jnp.asarray(value_b, jnp.float32),
    }

    wl_report = None
    if winloss:
        # 計畫 v3.1 §3：held-out 另一 seed、heldout_batches 批輪換；
        # 逐批前向後丟特徵（只留 g／v／step）
        const = float(np.mean(fit["g"]))  # 常數基準只由 fit 決定
        hg, hv, hs, hgid, groups, conv, n_teacher_h, seen_h = [], [], [], [], {}, [], 0, set()
        for (b, s), k in held_batch_keys.items():
            ctx.iteration = b
            pool_recs[f"heldout/b{b}"] = ctx.tape_pool_record(b_env)
            midx = ctx.opp_index(s) if league else 0
            seen_h.add((midx, s))
            r = W.run_one(
                ctx, params, feat_fns, s, k, cfg, args.chunk, checks, f"heldout/b{b}/seat{s}"
            )
            off = (b * len(SEATS) + s) * b_env
            rw = rows_of(r, args.n_fresh, off)
            n_teacher_h += rw["n_teacher_rows"]
            v_head = np.asarray(H.value(new_params, jnp.asarray(rw["h"], jnp.float32)), np.float64)
            v_ridge = W.with_const((rw["h"] - mu) / (sd + 1e-8)) @ w
            conv.append(float(np.max(np.abs(v_head - v_ridge))))
            hg.append(rw["g"])
            hv.append(v_head)
            hs.append(rw["step"])
            hgid.append(rw["gid"])
            groups.update({int(x): (midx, s) for x in np.unique(rw["gid"])})
            del r, rw
            print(
                f"[critic-init] heldout b{b} seat{s} t={time.perf_counter() - t0:.1f}s", flush=True
            )
        held = {
            "g": np.concatenate(hg),
            "v": np.concatenate(hv),
            "step": np.concatenate(hs),
            "gid": np.concatenate(hgid),
        }
        g_h, v_h = held["g"], held["v"]
        ev = W.explained_variance(g_h, v_h)
        wl_report = winloss_report(held, const, groups)
        heldout = {
            "n_rows": int(g_h.size),
            "n_teacher_rows_excluded": n_teacher_h,
            "ev": ev,
            "ev_note": "winloss 只報告、不作門（門＝MSE vs 常數基準）",
            "mse": wl_report["heldout_mse"],
            "mean_bias": float(np.mean(v_h - g_h)),
            "target_var": wl_report["heldout_target_var"],
            "winloss": wl_report,
        }
        if league:
            n_pool = len(league["members_sha256"])
            checks.add(
                "pool/heldout_covers_all_members_x_seats",
                seen_h == {(i, s) for i in range(n_pool) for s in SEATS},
                {"seen": sorted(seen_h), "n_pool": n_pool},
            )
        checks.add(
            f"convert/value_head_eq_ridge_le_{CONVERT_ATOL}",
            max(conv) <= CONVERT_ATOL,
            {"max": max(conv)},
        )
        checks.add("heldout/finite", bool(np.all(np.isfinite(v_h))))
        checks.add(
            "heldout/target_nondegenerate_var_gt0_and_W_L_ge1",
            wl_report["heldout_target_var"] > 0
            and wl_report["heldout_W"] >= 1
            and wl_report["heldout_L"] >= 1,
            {
                k: wl_report[k]
                for k in ("heldout_target_var", "heldout_W", "heldout_L", "heldout_T")
            },
        )
        checks.add(
            f"heldout/mse_le_{WINLOSS_MSE_RATIO}x_fit_const",
            wl_report["mse_gate_pass"],
            {k: wl_report[k] for k in ("heldout_mse", "const_mse", "mse_ratio")},
        )
    # held-out：真的 value head 前向（fresh 列）
    per_seat, gs, vs, conv, n_teacher_h = {}, [], [], [], 0
    if not winloss:
        ctx.iteration = args.fit_batches
        pool_recs["heldout"] = ctx.tape_pool_record(b_env)
    for s in SEATS if not winloss else ():
        r = W.run_one(
            ctx, params, feat_fns, s, held_keys[s], cfg, args.chunk, checks, f"heldout/seat{s}"
        )
        rw = rows_of(r, args.n_fresh, 0)
        n_teacher_h += rw["n_teacher_rows"]
        v_head = np.asarray(H.value(new_params, jnp.asarray(rw["h"], jnp.float32)), np.float64)
        v_ridge = W.with_const((rw["h"] - mu) / (sd + 1e-8)) @ w
        conv.append(float(np.max(np.abs(v_head - v_ridge))))
        per_seat[f"seat{s}"] = {
            "n_rows": int(rw["g"].size),
            "ev": W.explained_variance(rw["g"], v_head),
            "mse": float(np.mean((rw["g"] - v_head) ** 2)),
            "mean_bias": float(np.mean(v_head - rw["g"])),
            "finite": bool(np.all(np.isfinite(v_head))),
        }
        gs.append(rw["g"])
        vs.append(v_head)
        del r
    if not winloss:
        g_h, v_h = np.concatenate(gs), np.concatenate(vs)
        ev = W.explained_variance(g_h, v_h)
        heldout = {
            "n_rows": int(g_h.size),
            "n_teacher_rows_excluded": n_teacher_h,
            "ev": ev,
            "mse": float(np.mean((g_h - v_h) ** 2)),
            "mean_bias": float(np.mean(v_h - g_h)),
            "target_var": float(np.var(g_h)),
            "per_seat": per_seat,
        }
        checks.add(
            f"convert/value_head_eq_ridge_le_{CONVERT_ATOL}",
            max(conv) <= CONVERT_ATOL,
            {"max": max(conv)},
        )
        checks.add("heldout/finite", bool(np.all(np.isfinite(v_h))) and np.isfinite(ev))
        checks.add("heldout/target_var_positive", heldout["target_var"] > 0)
        checks.add(f"ev/ge_{EV_MIN}", bool(ev >= EV_MIN), {"ev": ev})
    policy = lambda p: {k: v for k, v in p.items() if k not in ("value_w", "value_b")}  # noqa: E731
    checks.add(
        "policy/unchanged", leaves_sha256(policy(new_params)) == leaves_sha256(policy(params))
    )

    out.mkdir(parents=True)
    meta = {
        "schema": SCHEMA,
        "identity": ident,
        "split": split,
        "alpha": rec["alpha"],
        "val_mse": rec["val_mse"],
        "n_fit_rows": int(fit["g"].size),
        "n_fit_teacher_rows_excluded": n_teacher,
        "heldout_ev": ev,
        **(
            {
                "winloss_gate": {
                    k: wl_report[k]
                    for k in (
                        "heldout_mse",
                        "const_mse",
                        "const_value_fit_mean_z",
                        "heldout_target_var",
                        "heldout_W",
                        "heldout_L",
                        "heldout_T",
                        "mse_ratio_max",
                        "mse_gate_pass",
                    )
                }
            }
            if winloss
            else {}
        ),
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_cpu_test": bool(args.allow_dirty_cpu_test),
        "relabel_step": args.relabel_step,
    }
    np.savez(
        out / FNAME,
        value_w=value_w.astype(np.float32),
        value_b=value_b.astype(np.float32),
        meta=json.dumps(meta),
    )
    doc = {
        "schema": SCHEMA,
        "plan": (
            "內部規格文件 v3.1 §3"
            if winloss
            else "內部規格文件 §3"
        ),
        "meta": meta,
        "cfg": cfg._asdict(),
        "ckpt": {"path": str(ckpt.resolve()), "sha256": ckpt_sha},
        "opponent": args.opponent,
        "tape_pool": (
            {
                "manifest": str(Path(args.pool_manifest).resolve()),
                "sha256": args.expect_pool_manifest_sha,
                "records": pool_recs,
            }
            if args.opponent == "tape-pool"
            else None
        ),
        "fit_record": rec,
        "heldout": heldout,
        "precision": precision,
        "backend": backend,
        "script_sha256": sha256_file(Path(__file__)),
        "critic_sha256": sha256_file(out / FNAME),
        "checks": checks.items,
    }
    doc["all_checks_pass"] = all(i["ok"] for i in checks.items)
    doc["wall_s"] = round(time.perf_counter() - t0, 3)
    doc["peak_rss_gib"] = rss_gib()
    (out / "result.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {
                "all_checks_pass": doc["all_checks_pass"],
                "heldout_ev": ev,
                "critic_sha256": doc["critic_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if doc["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
