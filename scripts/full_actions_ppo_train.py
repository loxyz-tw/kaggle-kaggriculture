#!/usr/bin/env python
"""完整動作 PPO trainer：單輪接線與實測（規格 內部規格文件）。

**這是量測用的最小 trainer，不是正式配方**：係數由命令列給，本段只證「一個 iteration 的
接線正確、硬契約會擋、iteration 邊界能續跑」，不證任何勝率或改善。

一個 iteration
--------------
1. 凍結本輪行為 `params_old`；由 master rollout key 派生兩座的 init seed 與 sample key，
   再更新 master key（**存下一輪的 master，不重設 base seed**）。
2. 兩座各一支 static jit 的 rollout（各自 `B = n_fresh + 1`、T 步、雙方同一當輪 policy）。
   **兩座是兩組獨立遊戲**，兩次成本都算，不稱「同一局雙座」。
3. `U.targets` → `U.normalize_joint`（兩座 valid adv 合併**一次**）→ `U.batch_from`。
4. 先排完單 epoch 的 minibatch 排程，再用 `params_old` 對**每一個實際 minibatch** 預算
   分母（§49.2）；任一不過即 STOP。固定後才開始第一次更新。
5. `U.make_update` ＋ `U.accept_or_reject` 逐批硬檢後才接受候選 params／opt。
6. iteration 邊界存 checkpoint（optimizer、next master key、shuffle RNG、累計步數、
   `iteration_complete=True`）與可 NumPy 部署的 actor。**只在完整 iteration 邊界續跑。**
"""

from __future__ import annotations

import argparse
import hashlib
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

from rl import full_actions_decode as D  # noqa: E402
from rl import full_actions_numpy as FN  # noqa: E402
from rl import full_actions_ppo as PB  # noqa: E402
from rl import full_actions_updater as U  # noqa: E402
from rl import loop4_tape as T  # noqa: E402
from rl import rules  # noqa: E402
from rl import state as S  # noqa: E402
from rl import tape_pool as TP  # noqa: E402
from scripts.full_actions_il_train import (  # noqa: E402
    DEPENDENCY_FILES,
    _git_diff_sha256,
    _git_porcelain,
    load_train_state,
    restore_rng,
    save_train_state,
)
from scripts.full_actions_ppo_cost import (  # noqa: E402
    Checks,
    Sink,
    cache_size,
    device_info,
    leaves_sha256,
    require_gpu_highest,
    rss_gib,
    sha256_file,
    teacher_row,
)

CAPS = PB.CAPS
#: 本段的依賴閉包（比 IL 多 bridge／updater／reward／GAE／座位工具與本檔）。
SOURCE_FILES = (
    *DEPENDENCY_FILES,
    "rl/env.py",
    "rl/loop4_ppo.py",
    "rl/gae.py",
    "rl/reward.py",
    "rl/full_actions_ppo.py",
    "rl/full_actions_updater.py",
    "scripts/full_actions_ppo_train.py",
    "scripts/full_actions_ppo_cost.py",
    "rl/tape_pool.py",
    # `teacher_row` 內部動態 import `_read_slice`：不列進來，它的 dirty 擋不住、SHA 也沒綁身分
    "scripts/loop4_il_labels.py",
)
SEATS = (0, 1)
#: rollout 對手：`self` ＝ 自我對戰（舊行為）；`il-frozen` ＝ 對手固定為原始 IL（不更新）
OPPONENTS = ("self", "il-frozen", "pool", "tape", "tape-pool")
#: `tape` ＝ 對手座照 tape bank 出動作；env k 用 `(k + iteration·B) % n_tapes`
#: （計畫 rl-tape-opponent-leg-plan-2026-09-23）
#: `tape-pool` ＝ T3 混池（`rl/tape_pool.py`）：新 B′ 75%／舊 bank 25%，lane 規則見該模組
#: `pool` ＝ 第 i 輪 seat s 的對手是 `pool[(i + s) % len(pool)]`（計畫 rl-opponent-pool-leg-plan）
#: league（計畫 ppo-league-lrm-plan-2026-09-30 §1）：`--pool-actor` 可重複給 ⇒ 有序池 [IL, a1, …]；
#: 規則不變，只是池變長。身分綁「有序成員 SHA＋規則字串」的 SHA（`pool_league`）。
POOL_RULE = (
    "pool=[il]+pool_actors（有序）；全域 iteration i（跨 leg resume 承接、不重置）、"
    "seat s 的對手＝pool[(i+s) % len(pool)]"
)


def pool_league(il_sha: str, actor_shas: list[str]) -> dict[str, Any]:
    """有序池的身分：成員（idx0＝IL）＋規則；`sha256` 是兩者正規化 JSON 的 SHA256。"""
    doc = {"rule": POOL_RULE, "members_sha256": [il_sha, *actor_shas]}
    blob = json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()
    return {**doc, "sha256": hashlib.sha256(blob).hexdigest()}


# ----------------------------------------------------------------- key 與 seed 派生
def key_to_list(key) -> list[int]:
    """master key 存成原本的 `uint32` 陣列（不存成 seed，也不存成有號整數）。"""
    raw = key
    if getattr(key, "dtype", None) != jnp.uint32:  # typed key ⇒ 先取出底層 uint32
        raw = jax.random.key_data(key)
    return [int(x) for x in np.asarray(raw, dtype=np.uint32).reshape(-1)]


def key_from_list(vals: list[int]):
    return jnp.asarray(np.asarray(vals, dtype=np.uint32))


def derive_iteration_keys(master_key, iteration: int) -> tuple[Any, dict[int, dict[str, Any]]]:
    """由 master key 派生本輪兩座的 `init_seed` 與 `sample_key`，回**下一輪的 master**。

    `fold_in(iteration)` 只是讓派生與輪次綁定、便於核對；真正的推進是 `split`，
    所以續跑載回 master key 後接著派生，不會重放同一組 key。
    """
    next_master, k_iter = jax.random.split(master_key)
    k_iter = jax.random.fold_in(k_iter, iteration)
    per_seat: dict[int, dict[str, Any]] = {}
    for seat in SEATS:
        k_init, k_sample = jax.random.split(jax.random.fold_in(k_iter, seat))
        per_seat[seat] = {
            "init_seed": int(jax.random.randint(k_init, (), 0, 2**31 - 1)),
            "sample_key": k_sample,
        }
    return next_master, per_seat


def seed_derivation_note(n_fresh: int) -> str:
    return (
        "master ← split(master)[0]；k_iter ← fold_in(split(master)[1], iteration)；"
        "每座 fold_in(k_iter, seat) 再 split ⇒ (init_seed, sample_key)；"
        f"init_state ＝ S.reset({n_fresh}, CAPS, init_seed) ＋ 預載教師列"
    )


def device_info_with_limit() -> dict[str, Any]:
    """`cost.device_info()` ＋ **實讀** XLA 的 `bytes_limit`（不改舊 helper）。

    物理 16 GiB 不等於可用：allocator 的上限要現場讀，取不到就記 null，不猜。
    """
    info = device_info()
    limit = None
    fn = getattr(jax.devices()[0], "memory_stats", None)
    if callable(fn):
        try:
            limit = (fn() or {}).get("bytes_limit")
        except Exception:  # noqa: BLE001  取不到就 null
            limit = None
    ms = dict(info.get("memory_stats") or {})
    ms["bytes_limit"] = limit
    return {**info, "memory_stats": ms if info.get("memory_stats") or limit else None}


def stop_if_failed(checks: Checks, n_before: int, where: str) -> None:
    """**本輪的 false 就地擋下**：`Checks` 只收集，不會自己停；不能等收尾才 FAIL。

    這支在每個關卡（rollout／targets 之後、分母之後、本輪結束前）呼叫，
    一有 false 立刻丟例外 ⇒ 不會走到「已經標 `iteration_complete=True`」那一步。
    """
    bad = [i["name"] for i in checks.items[n_before:] if not i["ok"]]
    if bad:
        raise ValueError(f"{where}：本輪硬契約未過（STOP）{bad}")


# ----------------------------------------------------------------------- 診斷小工具
def _jsonable(v) -> float | None:
    """nan／inf 不是合法 JSON；EV 在 `ret` 方差 0 時沒有定義 ⇒ 寫 `null`。"""
    f = float(v)
    return f if np.isfinite(f) else None


def kl_probe(params_now, params_ref, rows, cfg_model, *, seat: int, decoders, label: str):
    """`mean_rows Σ_active (logp_now − logp_ref)`，用**一個實際小批**算。

    **Monte Carlo KL-to-IL proxy**（可為負），不是精確 KL；`fresh` 是更新前對本輪自己抽到的
    樣本，`stale-sample` 是更新後拿同一批舊樣本重算。active 逐位、teacher_drop 全 0 硬檢，
    對不上就不是同一組條件下的比較。
    """
    now = PB.recompute(
        params_now, cfg_model, rows["state"], rows["decision"], seat=seat, decoders=decoders
    )
    ref = PB.recompute(
        params_ref, cfg_model, rows["state"], rows["decision"], seat=seat, decoders=decoders
    )
    act = rows["active"]
    drop = sum(
        int(jnp.sum(v))
        for out in (now, ref)
        for grp in ("unit", "market")
        for v in out["teacher_drop"][grp].values()
    )
    d = jnp.sum(jnp.where(act, now["logp"] - ref["logp"], 0.0), axis=-1)
    return {
        "label": label,
        "kind": "Monte Carlo KL-to-IL proxy（可負，非精確 KL）",
        "seat": seat,
        "n_rows": int(act.shape[0]),
        "mean": float(jnp.mean(d)),
        "max": float(jnp.max(d)),
        "active_matches_rollout": bool(
            jnp.all(now["active"] == act) and jnp.all(ref["active"] == act)
        ),
        "teacher_drop_total": drop,
    }


def kl_probe_inputs(rows_list, probe_rows: dict[int, int], budget) -> dict[int, dict[str, Any]]:
    """KL-to-IL proxy 的輸入列：每座第一批的**前 `budget.kl_probe_rows` 列**（取 min(k, n)）。

    作者 09-26 裁定方案 (b)（mb96 煙測 KL probe 列數 STOP 後）：proxy 的樣本定義固定為第一批
    前 k 列，與 minibatch 無關——mb=32 時就是整個第一批（**直接用原物件，逐位不變**）；mb=96 時
    的第一批由 mb=32 的前三批相接，前 32 列＝mb=32 第一批。n < k 時整批照用，列數不符仍由
    `check_kl_to_il` 硬拒。`budget is None`（成本量測）照舊用整批。
    """
    out = {}
    for seat, i_ in probe_rows.items():
        rows = rows_list[i_]
        n = int(np.asarray(rows["active"]).shape[0])
        k = n if budget is None else min(int(budget.kl_probe_rows), n)
        out[seat] = rows if k == n else U.take_rows(rows, np.arange(k))
    return out


def cash_report(final, *, n_fresh: int) -> dict[str, Any]:
    """局末現金：**fresh 與教師列分開**（教師列是最後一列，起點不同不可混平均）。"""
    money = np.asarray(final.money)
    return {
        "fresh": {
            "n": int(n_fresh),
            "seat0_median": float(np.median(money[:n_fresh, 0])),
            "seat1_median": float(np.median(money[:n_fresh, 1])),
        },
        "teacher_row": {"seat0": int(money[n_fresh, 0]), "seat1": int(money[n_fresh, 1])},
        "note": "教師列由真列中途開始，與 fresh 不同起點，不合併平均",
    }


def health_summary(hist: list[dict[str, Any]]) -> dict[str, Any]:
    """只彙總 `make_loss` 已經回的欄位；**不為了報表多跑一次 forward**。"""

    def agg(name: str, how: str) -> float | None:
        vals = [h[name] for h in hist if h.get(name) is not None]
        if not vals:
            return None
        fn = {"max": np.max, "min": np.min, "median": np.median, "mean": np.mean}[how]
        return float(fn(vals))

    return {
        "n_steps": len(hist),
        "ratio_h_max": agg("ratio_h_max", "max"),
        "ratio_h_min": agg("ratio_h_min", "min"),
        "ratio_h_median_of_batch_medians": agg("ratio_h_median", "median"),
        "clip_frac_h_mean": agg("clip_frac_h", "mean"),
        "clip_frac_h_max": agg("clip_frac_h", "max"),
        "pg_mean": agg("pg", "mean"),
        "value_loss_mean": agg("value_loss", "mean"),
        "entropy_mean": agg("entropy", "mean"),
        "explained_variance_median": agg("explained_variance", "median"),
        "offmask_logp_abs_max": agg("offmask_logp_abs_max", "max"),
        "clip_frac_h_token_weighted": (
            float(
                sum(h["clip_frac_h"] * h["n_components_active"] for h in hist)
                / max(sum(h["n_components_active"] for h in hist), 1.0)
            )
            if hist and all(h.get("clip_frac_h") is not None for h in hist)
            else None
        ),
        "grad_norm_mean": agg("grad_norm", "mean"),
        "grad_norm_max": agg("grad_norm", "max"),
        "anchor_kl_rowsum_mean": agg("anchor_kl_rowsum_mean", "mean"),
        "anchor_kl_rowsum_max": agg("anchor_kl_rowsum_max", "max"),
        "anchor_kl_per_active_mean": agg("anchor_kl_per_active_mean", "mean"),
        "anchor_term_mean": agg("anchor_term", "mean"),
        "n_explained_variance_null": sum(1 for h in hist if h.get("explained_variance") is None),
        "method": "逐批 metrics 的彙總；median 是『各批 median 的 median』，不是全體元素 median",
    }


def timing_summary(hist: list[dict[str, Any]]) -> dict[str, Any]:
    """**逐座**把首呼排除後再算 steady（seat 1 的首呼也可能含編譯）。"""
    steady = [h["wall_s"] for h in hist if h["phase"] == "steady"]
    first = {f"seat{h['seat']}": h["wall_s"] for h in hist if h["phase"] == "first_call_bound"}
    return {
        "first_call_bound_s": first,
        "steady_n": len(steady),
        "steady_median_s": round(float(np.median(steady)), 5) if steady else None,
        "steady_p95_s": round(float(np.percentile(steady, 95)), 5) if steady else None,
        "steady_max_s": round(float(np.max(steady)), 5) if steady else None,
        "method": "np.median／np.percentile(95)；每座首呼單獨列，不進 steady",
    }


# --------------------------------------------------------------------------- 訓練情境
class Ctx:
    """iteration 之外只建一次的東西：decoders、兩座 rollout jit、兩座 update jit、optimizer。

    **不要每輪重建**：`PB.rollout` 每次呼叫都會重新 trace，包在一支 jit 裡才不會每輪重編。
    """

    def __init__(
        self,
        cfg_model: dict[str, Any],
        cfg: U.Cfg,
        *,
        t: int,
        row: int,
        n_fresh: int,
        relabel: int,
        il_params,
        opponent: str = "self",
        pool_params=None,
        tape_bank=None,
        tape_pool_manifest=None,
        effect_mask: bool = False,
        pool_members_sha256: list[str] | None = None,
    ):
        self.cfg_model = cfg_model
        # T3 effect-mask（t3_stabilization_v4）：update／post-check 的 enforce 集合改 score_active
        self.effect_mask = bool(effect_mask)
        if opponent not in OPPONENTS:
            raise ValueError(f"opponent 只接受 {OPPONENTS}，得到 {opponent!r}")
        self.opponent = opponent
        # 對手池：il-frozen ＝ [IL]；pool ＝ [IL, 凍結 actor…]（dict＝舊行為；list＝有序 league）
        if opponent == "pool" and pool_params is None:
            raise ValueError("opponent=pool 必須給 pool_params")
        extra = list(pool_params) if isinstance(pool_params, (list, tuple)) else [pool_params]
        if opponent == "pool" and not extra:
            raise ValueError("opponent=pool 的 pool_params 不得為空")
        self.pool = [il_params] + (extra if opponent == "pool" else [])
        # 成員 SHA（idx 對齊 self.pool）：給了才在每輪 rollout 紀錄寫出實際對手身分
        if pool_members_sha256 is not None and (
            opponent != "pool" or len(pool_members_sha256) != len(self.pool)
        ):
            raise ValueError("pool_members_sha256 只在 opponent=pool 時給，且須與池等長")
        self.pool_members_sha256 = pool_members_sha256
        uses_tape = opponent in ("tape", "tape-pool")
        if uses_tape != (tape_bank is not None):
            raise ValueError("tape_bank 只在 opponent=tape／tape-pool 時給，且這兩者必須給")
        if (opponent == "tape-pool") != (tape_pool_manifest is not None):
            raise ValueError("tape_pool_manifest 只在 opponent=tape-pool 時給，且它必須給")
        self.tape_bank = tape_bank
        self.tape_pool_manifest = tape_pool_manifest
        self.iteration = 0  # run_iteration 開頭更新；pool 依它輪換
        self.il_params = il_params  # KL-to-IL proxy 的**固定**參照，整段訓練不換
        self.cfg = cfg
        self.t = t
        self.row = row
        self.n_fresh = n_fresh
        self.relabel = relabel
        self.backend = jax.devices()[0].platform
        # 教師列**只讀一次**（含 shard 串流 SHA 實算），之後每輪只 reset fresh 再拼接：
        # 不要每輪重讀大 shard、重算大檔 hash。
        self.teacher_state, self.teacher_ident = teacher_row(row)
        # update jit 的「該座首呼」要跨輪記住，不然第二輪的第一批會被誤標成 first_call_bound
        self.seen_update_seats: set[int] = set()
        self.diag = None  # 一次性診斷 probe（`--diag-probe`）；預設 None ＝ 正式路徑不經過
        self.repro = None  # ppo150k 修復 R2 重現守門（`--repro-gate`）；預設 None ＝ 正式路徑不經過
        self.decoders = PB.make_decoders(cfg_model, CAPS)
        self.opt = U.make_optimizer(cfg)
        if opponent == "self":
            self.rollouts = {
                seat: jax.jit(
                    lambda p_, st_, k_, s_=seat: PB.rollout(
                        p_, cfg_model, st_, k_, seat=s_, t=t, decoders=self.decoders, caps=CAPS
                    )
                )
                for seat in SEATS
            }
        elif opponent in ("tape", "tape-pool"):
            # bank 與 tape_idx 當 jit 引數傳：輪換 tape 不重編
            self._rollouts_tape = {
                seat: jax.jit(
                    lambda p_, bk_, ix_, st_, k_, s_=seat: PB.rollout(
                        p_,
                        cfg_model,
                        st_,
                        k_,
                        seat=s_,
                        t=t,
                        decoders=self.decoders,
                        caps=CAPS,
                        opp_tape=(bk_, ix_),
                    )
                )
                for seat in SEATS
            }
            self.rollouts = {
                seat: (
                    lambda p_, st_, k_, s_=seat: self._rollouts_tape[s_](
                        p_, self.tape_bank, self.tape_idx(st_.step.shape[0]), st_, k_
                    )
                )
                for seat in SEATS
            }
        else:
            # 對手參數當 jit 的引數傳（不烘成常數）；對手固定是原始 IL，整段訓練不換
            self._rollouts_vs = {
                seat: jax.jit(
                    lambda p_, o_, st_, k_, s_=seat: PB.rollout(
                        p_,
                        cfg_model,
                        st_,
                        k_,
                        seat=s_,
                        t=t,
                        decoders=self.decoders,
                        caps=CAPS,
                        opp_params=o_,
                    )
                )
                for seat in SEATS
            }
            self.rollouts = {
                seat: (
                    lambda p_, st_, k_, s_=seat: self._rollouts_vs[s_](
                        p_, self.opp_for(s_), st_, k_
                    )
                )
                for seat in SEATS
            }
        # KL 錨（T3 §5）：β>0 時 update 以 jit 引數吃**固定原 IL**（`il_params`）
        ref = il_params if float(cfg.anchor_beta) > 0.0 else None
        self.steps = {
            seat: U.make_update(
                cfg_model,
                cfg,
                self.opt,
                seat=seat,
                caps=CAPS,
                ref_params=ref,
                effect_mask=self.effect_mask,
            )
            for seat in SEATS
        }
        self._post_checks: dict[int, Any] = {}

    def post_check(self, seat: int):
        """輪末全量 post-update 檢查的 jit（第一次用到才建）。"""
        if seat not in self._post_checks:
            self._post_checks[seat] = U.make_post_check(
                self.cfg_model, seat=seat, caps=CAPS, effect_mask=self.effect_mask
            )
        return self._post_checks[seat]

    def opp_index(self, seat: int) -> int:
        return self.pool_index_at(self.iteration, seat)

    def pool_index_at(self, iteration: int, seat: int) -> int:
        """選取表：第 `iteration` 輪 seat 的對手 idx（`POOL_RULE`）。"""
        return (iteration + seat) % len(self.pool)

    def tape_idx(self, b: int):
        """env k 的 tape：`tape` ＝ `(k + iteration·B) % n_tapes`；`tape-pool` ＝ 混池規則。

        兩座同一組、逐輪輪換。
        """
        if self.opponent == "tape-pool":
            ix = TP.global_tape_idx(self.tape_pool_manifest, self.iteration, b)["tape_idx"]
            return jnp.asarray(ix, jnp.int32)
        return (jnp.arange(b, dtype=jnp.int32) + self.iteration * b) % self.tape_bank.n_tapes

    def tape_pool_record(self, b: int) -> dict[str, Any] | None:
        if self.opponent != "tape-pool":
            return None
        return TP.iteration_record(self.tape_pool_manifest, self.iteration, b)

    def opp_for(self, seat: int):
        return self.pool[self.opp_index(seat)]

    def init_state(self, seed: int):
        """`B = n_fresh + 1`（最後一列是**預載的**真教師列）。`relabel >= 0` 只有 tests 會用。

        與 `full_actions_ppo_cost.init_batch` 同一個構造（`S.reset(n_fresh, CAPS, seed)` ＋
        教師列 concat），差別只在教師列來自 `Ctx` 的預載，不每輪重讀 shard。
        """
        fresh = S.reset(self.n_fresh, CAPS, seed=seed)
        state = jax.tree.map(
            lambda a, b: jnp.concatenate([a, b], axis=0), fresh, self.teacher_state
        )
        ident = {
            **self.teacher_ident,
            "n_fresh": self.n_fresh,
            "seed_derivation": f"jax.random.split(PRNGKey({seed}), {self.n_fresh})",
            "teacher_row_index_in_batch": self.n_fresh,
            "teacher_preloaded_once_in_ctx": True,
        }
        if self.relabel >= 0:
            state = state._replace(step=jnp.full_like(state.step, self.relabel))
            ident = {**ident, "relabelled_step": self.relabel, "relabel_scope": "tests only"}
        return state, ident

    def teacher_source(self) -> dict[str, Any]:
        """identity 要綁的**教師來源身分**（不只是 row 號）。"""
        return {
            "row": self.row,
            "shard": self.teacher_ident["shard"],
            "shard_sha256": self.teacher_ident["shard_sha256_actual"],
            "fixture_ref_sha256": self.teacher_ident["fixture_ref_sha256"],
            "teacher_state_leaves_sha256": self.teacher_ident["teacher_state_leaves_sha256"],
            "original_step": self.teacher_ident["original_step"],
        }


def apply_candidate(step_fn, params, opt_state, rows, *, budget, where: str):
    """跑一步更新，**先過所有 host 守衛、通過了才接受候選**。

    守衛沒過就丟例外——候選 params／opt 不會被回傳，呼叫端的 `params` 不被替換；
    `run_iteration` 因此中止，`main` 只有在它**成功 return 之後**才會寫 checkpoint。
    `budget is None` ＝ 成本量測模式：只走既有的 `accept_or_reject`，行為與舊 run 相同。

    **ratio 的口徑**：`metrics` 來自這一步**更新前**的 loss 前向（`params`，不是 `cand_p`）。
    這道門只阻止「接受這個候選」，**不保證**更新後再前向的 ratio 也在界內；不另加 forward。
    """
    cand_p, cand_o, loss, metrics, grads = step_fn(params, opt_state, rows)
    jax.block_until_ready((cand_p, cand_o, loss))
    try:
        gate = U.accept_or_reject(metrics, grads, cand_p)
        ratio_gate = None
        if budget is not None:
            ratio_gate = U.check_candidate_ratio(metrics, budget, where=where)
    except ValueError as e:
        e.step_record = {"where": where, "loss": float(loss), **step_record(metrics)}
        raise
    return cand_p, cand_o, loss, metrics, grads, gate, ratio_gate


def _step_ratio_stats(stats) -> dict[str, Any]:
    agg = U.aggregate_ratio_stats([{"batch": None, "stats": jax.device_get(stats)}])
    return {
        "bounds": agg["bounds"],
        "groups": agg["groups"],
        "hist_counts": agg["hist"]["counts"],
        "extreme_min": agg["extreme_min"],
        "extreme_max": agg["extreme_max"],
    }


def step_record(metrics) -> dict[str, Any]:
    """逐步 history 的一筆：純量 metrics ＋ 候選 ratio 分布（head 分組、直方圖計數、極值位置）。

    effect-mask（v4）：`ratio_stats`＝score_active、`ratio_stats_raw`＝policy_active，兩份各自
    完整落盤（審查者 e887eb8 審 B2）；正常 history 與候選 STOP 的 `failing_step` 走同一支。
    """
    rec = {k: _jsonable(v) for k, v in U.scalar_metrics(metrics).items()}
    if "ratio_stats" in metrics:
        rec["ratio_stats"] = _step_ratio_stats(metrics["ratio_stats"])
    if "ratio_stats_raw" in metrics:
        rec["ratio_stats_raw"] = _step_ratio_stats(metrics["ratio_stats_raw"])
        rec["ratio_stats_sets"] = {
            "ratio_stats": "score_active",
            "ratio_stats_raw": "policy_active",
        }
    return rec


def run_iteration(
    ctx: Ctx,
    params,
    opt_state,
    master_key,
    rng: np.random.Generator,
    *,
    iteration: int,
    checks: Checks,
    sink: Sink | None = None,
    budget: U.HealthBudget | None = None,
) -> dict[str, Any]:
    """跑完整一輪：rollout → targets → 排程 → 分母 → 全部 minibatch 更新。

    回 `{"params", "opt_state", "master_key", "rng_state_before_schedule", "report", …}`；
    **中途 STOP 就丟例外**，不回半套狀態冒充完整 iteration。
    """
    t_iter0 = time.perf_counter()
    n_before = len(checks.items)
    params_old = params  # 本輪行為 params：分母只能用它，更新後不得重算
    behaviour_sha = leaves_sha256(params_old)
    next_master, per_seat = derive_iteration_keys(master_key, iteration)

    ctx.iteration = iteration  # pool 對手依輪次輪換
    # --- rollout：兩座各自獨立的一組遊戲
    t_roll0 = time.perf_counter()
    tgts, trajs, roll, init_steps = {}, {}, {}, {}
    for seat in SEATS:
        t_init = time.perf_counter()
        state, ident = ctx.init_state(per_seat[seat]["init_seed"])
        jax.block_until_ready(state)
        init_s = round(time.perf_counter() - t_init, 5)
        init_steps[seat] = np.asarray(state.step).reshape(-1).copy()
        t0 = time.perf_counter()
        final, traj = ctx.rollouts[seat](params_old, state, per_seat[seat]["sample_key"])
        jax.block_until_ready((final, traj))
        dt = round(time.perf_counter() - t0, 5)
        tgts[seat] = U.targets(traj, final, seat=seat, cfg=ctx.cfg)
        trajs[seat] = (traj, final)
        roll[f"seat{seat}"] = {
            "fixture": ident,
            # 本輪 rollout **實際收到**的 params 身分：只有它能證明第二輪不是又用 IL
            "behaviour_params_leaves_sha256": behaviour_sha,
            "opponent": ctx.opponent,
            "opponent_pool_index": (
                ctx.opp_index(seat) if ctx.opponent in ("il-frozen", "pool") else None
            ),
            # league：實際對手成員 SHA（帳本逐輪核輪換、記「成員×座」曝光）；舊池不加此鍵
            **(
                {"opponent_pool_member_sha256": ctx.pool_members_sha256[ctx.opp_index(seat)]}
                if ctx.pool_members_sha256 is not None
                else {}
            ),
            "opponent_tape_idx": (
                np.asarray(ctx.tape_idx(state.step.shape[0])).tolist()
                if ctx.opponent in ("tape", "tape-pool")
                else None
            ),
            "opponent_tape_pool": ctx.tape_pool_record(state.step.shape[0]),
            "init_seed": per_seat[seat]["init_seed"],
            "sample_key": key_to_list(per_seat[seat]["sample_key"]),
            "init_state_leaves_sha256": leaves_sha256(state),
            "init_state_wall_s": init_s,
            "rollout_wall_s": dt,
            "jit_cache_size": cache_size(ctx.rollouts[seat]),
            "summary": PB.trace_summary(traj, final, seat=seat),
            "cash": cash_report(final, n_fresh=ctx.n_fresh),
        }
        summ = roll[f"seat{seat}"]["summary"]
        # `trace_summary` 已經算過 freeze 契約，這裡不再跑第二次
        checks.add(
            f"iter{iteration}/rollout/seat{seat}/padding_matches_final",
            bool(summ["padding_state_equals_final"]),
        )
        checks.add(
            f"iter{iteration}/rollout/seat{seat}/done_at_last_valid_step",
            summ["done_first_t"] == summ["last_valid_t"],
            {"done_first_t": summ["done_first_t"], "last_valid_t": summ["last_valid_t"]},
        )
        checks.add(
            f"iter{iteration}/rollout/seat{seat}/valid_plus_padding",
            summ["n_valid_rows"] + summ["n_padding_rows"] == summ["T"] * summ["B"],
            summ,
        )
    roll["two_seats_are_independent_games"] = (
        "每座各自 reset 一組 env、各自一支 scan；兩座不是同一局的兩個視角，成本兩份都算"
    )
    roll["rollout_wall_s_total"] = round(time.perf_counter() - t_roll0, 5)
    roll["peak_rss_gib_after_rollout"] = rss_gib()
    roll["device_after_rollout"] = device_info_with_limit()

    # --- targets：兩座 valid adv **合併一次** normalize
    t_tgt = time.perf_counter()
    norm = U.normalize_joint(tgts)
    batches, n_env_steps = {}, 0
    for seat in SEATS:
        traj, _final = trajs[seat]
        batch = U.batch_from(traj, tgts[seat])
        batches[seat] = {
            "batch": batch,
            "n_rows": batch["n_rows_kept"],
            "minibatch": ctx.cfg.minibatch,
        }
        n_env_steps += int(batch["n_rows_kept"])
        roll[f"seat{seat}"].update(
            {
                "n_rows_total": batch["n_rows_total"],
                "n_rows_kept": batch["n_rows_kept"],
                "n_rows_padding": batch["n_rows_total"] - batch["n_rows_kept"],
            }
        )
        traj_valid = int(np.asarray(traj.valid).sum())
        flat_len = int(np.asarray(batch["adv"]).shape[0])
        # 每個 env 的有效步 ＝ min(T, N_TRANSITIONS − 起始 step)，由**實際 init step** 推
        expected = int(
            np.minimum(ctx.t, np.maximum(rules.N_TRANSITIONS - init_steps[seat], 0)).sum()
        )
        roll[f"seat{seat}"].update(
            {
                "init_steps": [int(x) for x in init_steps[seat]],
                "n_rows_expected_from_init_steps": expected,
                "n_valid_from_traj": traj_valid,
                "flatten_len": flat_len,
            }
        )
        checks.add(
            f"iter{iteration}/rows/seat{seat}/kept_equals_valid_and_expected",
            batch["n_rows_kept"] == traj_valid == flat_len == expected,
            {
                "n_rows_kept": batch["n_rows_kept"],
                "sum_valid": traj_valid,
                "flatten_len": flat_len,
                "expected_from_init_steps": expected,
            },
        )
        checks.add(
            f"iter{iteration}/rows/seat{seat}/total_is_T_times_B",
            batch["n_rows_total"] == ctx.t * int(init_steps[seat].shape[0]),
            {"n_rows_total": batch["n_rows_total"], "T": ctx.t},
        )
        ne = np.asarray(batch["noeffect"])
        ac = np.asarray(batch["active"])
        roll[f"seat{seat}"]["noeffect"] = {
            "n_active": int(ac.sum()),
            "n_noeffect": int(ne.sum()),
            "frac_of_active": float(ne.sum() / max(int(ac.sum()), 1)),
            "n_rows_with_noeffect": int(ne.any(axis=-1).sum()),
            "rules_version": D.EFFECT_RULES_VERSION,
            "used_by_loss": bool(ctx.effect_mask),
        }
        checks.add(
            f"iter{iteration}/rows/seat{seat}/noeffect_subset_active",
            bool(np.all(~ne | ac)),
            roll[f"seat{seat}"]["noeffect"],
        )
        checks.add(
            f"iter{iteration}/rows/seat{seat}/full_batch_available",
            batch["n_rows_kept"] >= ctx.cfg.minibatch,
            {"n_rows_kept": batch["n_rows_kept"], "minibatch": ctx.cfg.minibatch},
        )
    roll["advantage_normalization"] = norm
    checks.add(
        f"iter{iteration}/targets/joint_normalized_once",
        norm["n_valid_rows_all_seats"] == sum(int(b["n_rows"]) for b in batches.values()),
        {**norm, "n_rows_per_seat": {s_: int(b["n_rows"]) for s_, b in batches.items()}},
    )
    roll["targets_flatten_wall_s"] = round(time.perf_counter() - t_tgt, 5)
    if sink:
        sink.stage(f"iter{iteration}/rollout", roll)
    stop_if_failed(checks, n_before, f"iter{iteration} rollout／targets")

    # --- 排程（這一輪排之前的 RNG 狀態就是 checkpoint 要存的那份）
    rng_state_before = {k: v for k, v in rng.bit_generator.state.items()}
    plan, plan_report = U.schedule(batches, rng, 1, start_update=iteration)
    tail = plan_report["n_rows_dropped_tail"]

    # --- 分母：**首次更新之前**，照每個實際 minibatch 用 params_old 預算一次
    t0 = time.perf_counter()
    rows_list, denom = U.precompute_denominators(
        params_old,
        ctx.cfg_model,
        batches,
        plan,
        decoders=ctx.decoders,
        caps=CAPS,
        backend=ctx.backend,
    )
    jax.block_until_ready([r["teacher_old_logp_h"] for r in rows_list])
    denom_s = round(time.perf_counter() - t0, 5)
    # 數值出口依平台：CPU 逐 token 嚴格，GPU 核零更新比率偏差預算（兩者都由 U 硬拒過）
    if ctx.backend == "cpu":
        checks.add(
            f"iter{iteration}/denominator/token_parity_cpu_strict",
            denom["max_abs_token_logp_diff"] <= U.DENOM_ATOL,
            denom,
        )
    else:
        checks.add(
            f"iter{iteration}/denominator/ratio_budget",
            denom["max_ratio_budget_abs_expm1"] <= U.RATIO_BUDGET,
            denom,
        )
    checks.add(
        f"iter{iteration}/denominator/source_is_rollout_logp",
        denom["denominator_source"] == U.DENOMINATOR_SOURCE,
        {"source": denom["denominator_source"]},
    )
    checks.add(
        f"iter{iteration}/denominator/no_teacher_drop", denom["teacher_drop_total"] == 0, denom
    )
    checks.add(
        f"iter{iteration}/denominator/one_per_planned_minibatch",
        len(rows_list) == len(plan)
        and [r["n_rows"] for r in denom["per_minibatch"]] == [len(i["idx"]) for i in plan],
        {"n_rows_list": len(rows_list), "n_plan": len(plan)},
    )
    denom_report = {
        "n_minibatches": denom["n_minibatches"],
        "max_abs_token_logp_diff": denom["max_abs_token_logp_diff"],
        "max_abs_joint_logp_diff": denom["max_abs_joint_logp_diff"],
        "teacher_drop_total": denom["teacher_drop_total"],
        "atol": denom["atol"],
        "denominator_source": denom["denominator_source"],
        "backend": denom["backend"],
        # 超診斷線的槽數照實記（GPU 不用它當出口，但永久留紀錄）
        "n_slots_over_diagnostic_line": denom["n_slots_over_diagnostic_line"],
        "max_ratio_budget_abs_expm1": denom["max_ratio_budget_abs_expm1"],
        "ratio_budget": denom["ratio_budget"],
        "wall_s": denom_s,
        "wall_s_per_minibatch": round(denom_s / max(len(plan), 1), 6),
        "rows_per_minibatch_unique": sorted({r["n_rows"] for r in denom["per_minibatch"]}),
        "note": denom["note"],
    }
    stop_if_failed(checks, n_before, f"iter{iteration} 分母（更新前）")
    if sink:
        sink.stage(
            f"iter{iteration}/behaviour_denominator",
            {
                "plan": {"n_steps": len(plan), "n_rows_dropped_tail": tail},
                "parity": denom_report,
                "peak_rss_gib_after_rollout_and_denominator": rss_gib(),
            },
        )

    # --- 一次性診斷（opt-in）：更新前先核目標事件是否重現；不符即落盤並結束
    if ctx.diag is not None and iteration == int(ctx.diag.spec["iteration"]):
        ctx.diag.check_event(roll, plan, rows_list)
    # --- 重現守門（ppo150k 修復 R2，opt-in）：零更新校驗與任何 optimizer update **之前**；
    # 不符即落盤並丟
    if ctx.repro is not None and iteration == ctx.repro.iteration:
        ctx.repro.check(roll, plan, rows_list, master_key_list=key_to_list(master_key))
        if sink:
            sink.stage(f"iter{iteration}/repro_gate", ctx.repro.report)

    # --- 更新前的 KL-to-IL proxy（實際第一個小批的前 `kl_probe_rows` 列；不另外全資料跑一次）
    probe_rows = {}
    for i_, item_ in enumerate(plan):
        probe_rows.setdefault(int(item_["seat"]), i_)
    probe_data = kl_probe_inputs(rows_list, probe_rows, budget)
    t_kl = time.perf_counter()
    kl_before = {
        f"seat{seat}": kl_probe(
            params_old,
            ctx.il_params,
            probe_data[seat],
            ctx.cfg_model,
            seat=seat,
            decoders=ctx.decoders,
            label="fresh-rollout",
        )
        for seat in sorted(probe_data)
    }
    for seat in probe_rows:
        r_ = kl_before[f"seat{seat}"]
        checks.add(
            f"iter{iteration}/kl_before/seat{seat}/active_matches_rollout",
            r_["active_matches_rollout"],
            r_,
        )
        checks.add(
            f"iter{iteration}/kl_before/seat{seat}/no_teacher_drop",
            r_["teacher_drop_total"] == 0,
            r_,
        )
    kl_before_s = round(time.perf_counter() - t_kl, 5)
    kl_gate = None
    if budget is not None:
        # 漂移警報：**本輪更新之前**，兩座 fresh proxy 的均值都要在預算內（非有限即停）
        kl_gate = U.check_kl_to_il(kl_before, budget, where=f"iter{iteration}/kl_before")
        checks.add(f"iter{iteration}/health/kl_to_il_within_budget", True, kl_gate)
    stop_if_failed(checks, n_before, f"iter{iteration} KL probe（更新前）")

    # 從零更新校驗到輪末檢查：任何 STOP 都先把本輪已有的 history／timing／VRAM 落盤再拋出
    # （審查者 09-25：B16／B64 煙測 STOP 後逐步走勢遺失）。
    hist: list[dict[str, Any]] = []
    tseg: dict[str, Any] = {
        "rollout_wall_s_total": roll["rollout_wall_s_total"],
        "targets_flatten_wall_s": roll["targets_flatten_wall_s"],
        "denominator_wall_s": denom_s,
        "kl_probe_before_wall_s": kl_before_s,
    }
    try:
        # --- 零更新校驗：**第一次接受任何更新之前**，用正式 make_update 的融合前向各座跑一批。
        # 候選 params／opt **一律丟棄**（這一步不更新），但成本計入。
        t_zero = time.perf_counter()
        zero_update = {}
        for seat, i_ in sorted(probe_rows.items()):
            cand_p, cand_o, loss_z, metrics_z, _g = ctx.steps[seat](
                params_old, opt_state, rows_list[i_]
            )
            jax.block_until_ready((cand_p, cand_o, loss_z))
            zero_update[f"seat{seat}"] = U.check_zero_update_budget(
                metrics_z, where=f"iter{iteration}/seat{seat}/zero_update"
            )
            del cand_p, cand_o  # 不接受候選
        zero_update_s = round(time.perf_counter() - t_zero, 5)
        tseg["zero_update_check_wall_s"] = zero_update_s
        for seat in probe_rows:
            r_ = zero_update[f"seat{seat}"]
            checks.add(
                f"iter{iteration}/zero_update/seat{seat}/ratio_budget",
                r_["max_abs_ratio_deviation"] <= U.RATIO_BUDGET,
                r_,
            )
        stop_if_failed(checks, n_before, f"iter{iteration} 零更新校驗（首次更新前）")

        # --- 更新：逐批 host 硬檢後才接受
        t_loop0 = time.perf_counter()
        for i, (item, rows) in enumerate(zip(plan, rows_list, strict=True)):
            seat = int(item["seat"])
            if ctx.diag is not None:
                ctx.diag.maybe_probe(i, params, ctx.il_params, hist)
            first_for_seat = seat not in ctx.seen_update_seats  # 跨輪常駐：第二輪不算首呼
            ctx.seen_update_seats.add(seat)
            t0 = time.perf_counter()
            cand_p, cand_o, loss, metrics, grads, gate, ratio_gate = apply_candidate(
                ctx.steps[seat],
                params,
                opt_state,
                rows,
                budget=budget,
                where=f"iter{iteration}/step{i}/seat{seat}",
            )
            dt = time.perf_counter() - t0
            params, opt_state = cand_p, cand_o  # 守衛都過了才接受候選
            hist.append(
                {
                    "step": i,
                    "iteration": iteration,
                    "seat": seat,
                    "minibatch": item["minibatch"],
                    "wall_s": round(dt, 5),
                    "phase": "first_call_bound" if first_for_seat else "steady",
                    "jit_cache_size_after": cache_size(ctx.steps[seat]),
                    "loss": float(loss),
                    "gate_teacher_drop_total": gate["teacher_drop_total"],
                    "ratio_gate": ratio_gate,
                    **step_record(metrics),
                }
            )
        if ctx.diag is not None:
            ctx.diag.final_probe(len(plan), params, ctx.il_params, hist)
        moved = float(
            jnp.sqrt(
                sum(
                    jnp.sum((a - b) ** 2)
                    for a, b in zip(
                        jax.tree.leaves(params), jax.tree.leaves(params_old), strict=True
                    )
                )
            )
        )
        # **整個迴圈**的牆鐘：含 708 步 Python 迴圈、host 分發與 accept_or_reject，
        # 不是逐批 GPU 算子時間的加總（那樣會低估總預算）。
        update_loop_wall_s = round(time.perf_counter() - t_loop0, 5)
        tseg["update_loop_wall_s"] = update_loop_wall_s

        # --- 更新後的 stale-sample proxy（與 fresh 同一份列，標籤分開）
        t_kl2 = time.perf_counter()
        kl_after = {
            f"seat{seat}": kl_probe(
                params,
                ctx.il_params,
                probe_data[seat],
                ctx.cfg_model,
                seat=seat,
                decoders=ctx.decoders,
                label="stale-sample",
            )
            for seat in sorted(probe_data)
        }
        for seat in probe_rows:
            r_ = kl_after[f"seat{seat}"]
            checks.add(
                f"iter{iteration}/kl_after/seat{seat}/active_matches_rollout",
                r_["active_matches_rollout"],
                r_,
            )
            checks.add(
                f"iter{iteration}/kl_after/seat{seat}/no_teacher_drop",
                r_["teacher_drop_total"] == 0,
                r_,
            )
        kl_after_s = round(time.perf_counter() - t_kl2, 5)
        tseg["kl_probe_after_wall_s"] = kl_after_s

        # --- 重現守門的事件報告（更新後、輪末檢查前；不論輪末檢查過否都留）
        if ctx.repro is not None and ctx.repro.passed and iteration == ctx.repro.iteration:
            ev_ = ctx.repro.post_report(params, ctx.il_params, ctx.cfg_model, rows_list, plan)
            if sink:
                sink.stage(f"iter{iteration}/repro_events_post_update", ev_)
        # --- 輪末全量 post-update 檢查（t3_stabilization_v1）：同輪**全部**有效 minibatch、
        # 更新後參數、原 rollout 分母；不過就 STOP，本輪不寫完整 checkpoint。
        post_rep, post_s = None, None
        if budget is not None and budget.post_update_check:
            t_post = time.perf_counter()
            per_batch = []
            for item_, rows_ in zip(plan, rows_list, strict=True):
                seat_ = int(item_["seat"])
                st_ = ctx.post_check(seat_)(params, ctx.il_params, rows_)
                per_batch.append({**jax.device_get(st_), "seat": seat_, "batch": len(per_batch)})
            post_s = round(time.perf_counter() - t_post, 5)
            tseg["post_update_check_wall_s"] = post_s
            post_rep = U.check_post_update(
                per_batch,
                budget,
                where=f"iter{iteration}/post_update",
                teacher_env_id=ctx.n_fresh,  # B＝n_fresh＋1，教師列是最後一個 env
            )
            post_rep["wall_s"] = post_s
            checks.add(f"iter{iteration}/health/post_update_full_check", True, post_rep)
    except Exception as e:  # noqa: BLE001  落盤後原樣拋出，不吞例外
        if sink:
            sink.stage(
                f"iter{iteration}/updates_partial",
                {
                    "error": f"{type(e).__name__}: {str(e)[:4000]}",
                    "n_steps_done": len(hist),
                    "n_steps_planned": len(plan),
                    "failing_step": getattr(e, "step_record", None),
                    "post_update_check": getattr(e, "report", None),
                    "history": hist,
                    "health": health_summary(hist),
                    "timing": {
                        **tseg,
                        "update_steps_sum_wall_s": round(sum(h["wall_s"] for h in hist), 5),
                        "since_iteration_start_s": round(time.perf_counter() - t_iter0, 5),
                    },
                    "kl_to_il_proxy_before": kl_before,
                    "device_at_stop": device_info_with_limit(),
                },
            )
        raise

    checks.add(f"iter{iteration}/update/n_steps_equals_plan", len(hist) == len(plan))
    checks.add(f"iter{iteration}/update/params_changed", moved > 0.0, {"l2_delta": moved})
    checks.add(
        f"iter{iteration}/update/params_finite",
        all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(params)),
    )
    report = {
        "iteration": iteration,
        "behaviour_params_leaves_sha256": behaviour_sha,
        "params_leaves_sha256_after": leaves_sha256(params),
        "rollout": roll,
        "plan": {"n_steps": len(plan), "n_rows_dropped_tail": tail},
        "denominator": denom_report,
        "health": health_summary(hist),
        "zero_update_check": {
            **zero_update,
            "wall_s": zero_update_s,
            "note": "用正式 make_update 的融合前向；候選 params／opt 全丟棄，不接受更新",
        },
        "kl_to_il_proxy": {
            "before_fresh_rollout": kl_before,
            "after_stale_sample": kl_after,
            "note": "MC proxy 可為負；負值不是發散證據，也不是精確 KL",
        },
        "timing": {
            **timing_summary(hist),
            "init_state_wall_s": {
                f"seat{s_}": roll[f"seat{s_}"]["init_state_wall_s"] for s_ in SEATS
            },
            "rollout_wall_s_total": roll["rollout_wall_s_total"],
            "targets_flatten_wall_s": roll["targets_flatten_wall_s"],
            "denominator_wall_s": denom_s,
            "kl_probe_wall_s": {"before": kl_before_s, "after": kl_after_s},
            "zero_update_check_wall_s": zero_update_s,
            "post_update_check_wall_s": post_s,
            "update_loop_wall_s": update_loop_wall_s,
            "update_steps_sum_wall_s": round(sum(h["wall_s"] for h in hist), 5),
            "iteration_wall_s": None,  # 下面補（要含本段所有診斷與落盤）
            "note": "update_loop_wall_s 含 host gate 與同步；逐批 wall 的加總另列，兩者不相等",
        },
        "peak_rss_gib_after_rollout": roll["peak_rss_gib_after_rollout"],
        "peak_rss_gib_end_of_iteration": rss_gib(),
        "device_end_of_iteration": device_info_with_limit(),
        "params_l2_delta": moved,
        "history": hist,
        "env_steps_valid": n_env_steps,
        "post_update_check": post_rep,
        "anchor": {
            "beta": float(ctx.cfg.anchor_beta),
            **(U.ANCHOR_META if ctx.cfg.anchor_beta else {}),
        },
    }
    if budget is not None:
        # **寫完成 checkpoint 之前**：輪末 clip fraction 均值要在預算內
        clip_gate = U.check_round_clip_fraction(hist, budget, where=f"iter{iteration}/round_end")
        checks.add(f"iter{iteration}/health/clip_fraction_within_budget", True, clip_gate)
        report["health_budget_gates"] = {
            "budget": U.health_budget_meta(budget),
            "kl_to_il_before": kl_gate,
            "clip_fraction_round_end": clip_gate,
            "candidate_ratio_bounds": [budget.ratio_lo, budget.ratio_hi],
            "note": "host 停損；任一超限即停當輪、不寫完成 checkpoint",
        }
    else:
        report["health_budget_gates"] = None  # 成本量測模式：不開 budget，舊 run 不改判
    report["timing"]["iteration_wall_s"] = round(time.perf_counter() - t_iter0, 5)
    if sink:
        sink.stage(f"iter{iteration}/updates", report)
    # **存 checkpoint 之前**再擋一次：本輪有任何 false 就不許被標成完整 iteration
    stop_if_failed(checks, n_before, f"iter{iteration} 更新後（存 checkpoint 之前）")
    # 本輪資料用完就放掉：epoch 1 不跨輪保留副本，下一輪必須重新 rollout
    del rows_list, batches, trajs, tgts
    return {
        "params": params,
        "opt_state": opt_state,
        "master_key": next_master,
        "rng_state_before_schedule": rng_state_before,
        "n_updates": len(plan),
        "env_steps_valid": n_env_steps,
        "report": report,
    }


# ------------------------------------------------------------------ checkpoint 與身分
class ReproMismatch(RuntimeError):
    """`--repro-gate`：重建 rollout 與凍結原件指紋不符（INVALID）。

    已落盤、未做任何 optimizer update。
    """


#: repro spec 必填欄（內部紀錄 產生）
REPRO_SPEC_KEYS = ("schema", "iteration", "source", "expect_rollout", "expect_summary", "events")
REPRO_ROLLOUT_KEYS = (
    "init_seed",
    "sample_key",
    "init_state_leaves_sha256",
    "opponent_pool_index",
    "opponent_tape_idx",
    "behaviour_params_leaves_sha256",
    "n_rows_kept",
)


class ReproGate:
    """ppo150k iter3 修復 R2：更新前事件重現守門。

    沿 `rl/t3_diag_probe.Diag.check_event` 的比對規則。

    與 `--diag-probe` 模式隔離：不做無梯度 probe 回呼、不落 NPZ、不改 verdict、不擋 checkpoint；
    允許 resume。比對：每座 rollout 身分（全等，JSON 正規化）、
    advantage 正規化摘要（n 全等、mean／std 相對 1e-6）、
    每座 cash 中位數與 summary.last_valid_t／done_first_t（全等）；事件分兩類——
    `full`（原件留有逐列指紋）：seat／env_id／turn／decision_token／active 全等、
    logp_old 以 |expm1(Δ)|≤RATIO_BUDGET、
    advantage 以 1e-3·max(1,|a|)（與 diag 同規則）；缺欄即拒；
    `episode_only`（原件只存 (seat, env_id)）：只核該局在 rollout 內存在且 summary 同
    （`row_fingerprint_available=false`）。
    任一不符 ⇒ 落盤 `REPRO_MISMATCH.json`、丟 `ReproMismatch`
    （在零更新校驗與任何 optimizer update 之前）。
    """

    def __init__(self, spec: dict[str, Any], *, out_dir: Path):
        miss = [k for k in REPRO_SPEC_KEYS if k not in spec]
        if miss:
            raise SystemExit(f"repro spec 缺欄 {miss}")
        if spec["schema"] != "ppo150k-repro-spec-v1":
            raise SystemExit(f"repro spec schema 不符：{spec['schema']}")
        kinds = [e.get("kind") for e in spec["events"]]
        if not spec["events"] or any(k not in ("full", "episode_only") for k in kinds):
            raise SystemExit(f"repro spec events 種類不合法：{kinds}")
        for e in spec["events"]:
            need = (
                (
                    "seat",
                    "env_id",
                    "turn",
                    "slot",
                    "batch",
                    "row_in_batch",
                    "decision_token",
                    "active",
                    "logp_old",
                    "advantage",
                )
                if e["kind"] == "full"
                else ("seat", "env_id")
            )
            m = [k for k in need if e.get(k) is None]
            if m:
                raise SystemExit(f"repro spec 事件 {e.get('name')} 缺欄 {m}：拒絕（缺一即拒）")
        for s_key in ("seat0", "seat1"):
            m = [
                k for k in REPRO_ROLLOUT_KEYS if k not in (spec["expect_rollout"].get(s_key) or {})
            ]
            if m:
                raise SystemExit(f"repro spec expect_rollout.{s_key} 缺欄 {m}")
        self.spec = spec
        self.iteration = int(spec["iteration"])
        self.out_dir = Path(out_dir)
        self.report: dict[str, Any] = {
            "schema": "ppo150k-repro-gate-v1",
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            "source": spec["source"],
            "events_gap_note": spec.get("events_gap_note"),
            "full_event_identity": {
                "verified": False,
                "reason": (
                    "原件未留存完整 prefix／state／合法 mask／plan idx；"
                    "只核列出的有限指紋欄（審查者 v2 B3 降格）"
                ),
            },
        }
        self.passed = False
        self.rows_index: dict[str, Any] = {}

    @staticmethod
    def _norm(x):
        return json.loads(
            json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        )

    def check(self, roll: dict, plan: list, rows_list: list, *, master_key_list=None) -> None:
        sp, mism, got = self.spec, {}, {}
        if sp.get("expect_master_key") is not None and master_key_list is not None:
            if list(master_key_list) != list(sp["expect_master_key"]):
                mism["master_key"] = {"got": master_key_list, "want": sp["expect_master_key"]}
        for s_key, want in sp["expect_rollout"].items():
            g = roll.get(s_key) or {}
            for k in REPRO_ROLLOUT_KEYS:
                if self._norm(g.get(k)) != self._norm(want[k]):
                    mism[f"rollout.{s_key}.{k}"] = {
                        "got": str(self._norm(g.get(k)))[:200],
                        "want": str(want[k])[:200],
                    }
        es = sp["expect_summary"]
        an = roll.get("advantage_normalization") or {}
        if an.get("n_valid_rows_all_seats") != es["adv_n_valid_rows_all_seats"]:
            mism["adv.n_valid_rows_all_seats"] = {
                "got": an.get("n_valid_rows_all_seats"),
                "want": es["adv_n_valid_rows_all_seats"],
            }
        for k in ("adv_mean_before", "adv_std_before"):
            g, w = an.get(k), es[k]
            ok = g is not None and np.isfinite(g) and abs(float(g) - w) <= 1e-6 * max(abs(w), 1e-12)
            got[k] = g
            if not ok:
                mism[f"adv.{k}"] = {"got": g, "want": w, "rtol": 1e-6}
        for s_key in ("seat0", "seat1"):
            g = roll.get(s_key) or {}
            if self._norm(g.get("cash")) != self._norm(es["cash"][s_key]):
                mism[f"{s_key}.cash"] = {
                    "got": str(self._norm(g.get("cash")))[:300],
                    "want": str(es["cash"][s_key])[:300],
                }
            summ = g.get("summary") or {}
            for k in ("last_valid_t", "done_first_t"):
                if self._norm(summ.get(k)) != self._norm(es["summary"][s_key][k]):
                    mism[f"{s_key}.summary.{k}"] = {
                        "got": self._norm(summ.get(k)),
                        "want": es["summary"][s_key][k],
                    }
        ev_got = []
        for e in sp["events"]:
            eg: dict[str, Any] = {"name": e["name"], "kind": e["kind"]}
            if e["kind"] == "episode_only":
                s_key = f"seat{int(e['seat'])}"
                lv = ((roll.get(s_key) or {}).get("summary") or {}).get("last_valid_t") or []
                present = 0 <= int(e["env_id"]) < len(lv)
                eg.update(
                    {
                        "seat": int(e["seat"]),
                        "env_id": int(e["env_id"]),
                        "episode_present": present,
                        "row_fingerprint_available": False,
                    }
                )
                if not present:
                    mism[f"event.{e['name']}.episode_present"] = {"got": False}
                ev_got.append(eg)
                continue
            b, r, slot = int(e["batch"]), int(e["row_in_batch"]), int(e["slot"])
            if not (0 <= b < len(plan)) or not (
                0 <= r < int(np.asarray(rows_list[b]["env_id"]).shape[0])
            ):
                mism[f"event.{e['name']}.locate"] = {"batch": b, "row": r, "n_plan": len(plan)}
                ev_got.append(eg)
                continue
            rows = rows_list[b]
            host = lambda k, rows=rows: np.asarray(jax.device_get(rows[k]))  # noqa: E731
            packed = np.asarray(jax.device_get(U._packed_decision(rows)))
            eg.update(
                {
                    "seat": int(plan[b]["seat"]),
                    "env_id": int(host("env_id")[r]),
                    "turn": int(host("turn")[r]),
                    "decision_token": int(packed[r, slot]),
                    "active": bool(host("active")[r, slot]),
                    "logp_old": float(host("logp_h")[r, slot]),
                    "advantage": float(host("adv")[r]),
                }
            )
            for k in ("seat", "env_id", "turn", "decision_token", "active"):
                if eg[k] != e[k]:
                    mism[f"event.{e['name']}.{k}"] = {"got": eg[k], "want": e[k]}
            fin = all(np.isfinite([eg["logp_old"], eg["advantage"]]))
            d_lp = abs(float(np.expm1(eg["logp_old"] - e["logp_old"]))) if fin else float("nan")
            d_ad = abs(eg["advantage"] - e["advantage"]) if fin else float("nan")
            eg.update({"logp_old_abs_expm1": d_lp, "advantage_abs_diff": d_ad})
            if not (d_lp <= U.RATIO_BUDGET):
                mism[f"event.{e['name']}.logp_old"] = {
                    "got": eg["logp_old"],
                    "want": e["logp_old"],
                    "budget": U.RATIO_BUDGET,
                }
            if not (d_ad <= 1e-3 * max(1.0, abs(e["advantage"]))):
                mism[f"event.{e['name']}.advantage"] = {
                    "got": eg["advantage"],
                    "want": e["advantage"],
                }
            ev_got.append(eg)
            self.rows_index[e["name"]] = (b, r, slot)
        self.report.update(
            {"got": {**got, "events": ev_got}, "mismatch": mism, "matched": not mism}
        )
        if mism:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / "REPRO_MISMATCH.json"
            path.write_text(json.dumps(self._norm(self.report), ensure_ascii=False, indent=1))
            raise ReproMismatch(
                f"repro gate 不符（INVALID，未做任何 update）：{sorted(mism)}；見 {path}"
            )
        self.passed = True

    def post_report(self, params, il_params, cfg_model, rows_list, plan) -> dict[str, Any]:
        """更新後（輪末檢查前）的事件報告。

        全指紋列：old／new／ref／ratio；episode_only 局：逐 head 組 ratio 極值。
        """
        from rl import (
            t3_diag_probe as DG,  # noqa: PLC0415 （只借用 make_probe 的前向；不啟用 diag 模式）
        )

        out: dict[str, Any] = {"full": {}, "episode_only": {}}
        probes = {s: DG.make_probe(cfg_model, seat=s) for s in (0, 1)}
        for name, (b, r, slot) in self.rows_index.items():
            seat = int(plan[b]["seat"])
            pr = probes[seat](params, il_params, rows_list[b])
            old = float(np.asarray(rows_list[b]["logp_h"])[r, slot])
            new = float(np.asarray(pr["logp"])[r, slot])
            ref = float(np.asarray(pr["logp_ref"])[r, slot])
            out["full"][name] = {
                "batch": b,
                "row": r,
                "slot": slot,
                **U.slot_describe(slot),
                "logp_old": old,
                "logp_new": new,
                "logp_ref": ref,
                "ratio": float(np.exp(new - old)),
            }
        for e in self.spec["events"]:
            if e["kind"] != "episode_only":
                continue
            seat, env = int(e["seat"]), int(e["env_id"])
            grp: dict[str, list[float]] = {}
            for b, item in enumerate(plan):
                if int(item["seat"]) != seat:
                    continue
                rows = rows_list[b]
                sel = np.asarray(rows["env_id"]) == env
                if not sel.any():
                    continue
                pr = probes[seat](params, il_params, rows)
                lr_ = np.asarray(pr["logp"]) - np.asarray(rows["logp_h"])
                act = np.asarray(rows["active"]).astype(bool) & sel[:, None]
                for sl in np.nonzero(act.any(axis=0))[0]:
                    d = U.slot_describe(int(sl))
                    g = f"{d['phase']}_{d['head']}"
                    v = np.exp(lr_[act[:, sl], sl])
                    lo, hi = grp.get(g, [np.inf, -np.inf])
                    grp[g] = [min(lo, float(v.min())), max(hi, float(v.max()))]
            out["episode_only"][e["name"]] = {
                "seat": seat,
                "env_id": env,
                "row_fingerprint_available": False,
                "head_group_ratio_minmax": grp,
            }
        self.report["post_update_events"] = out
        return out


def build_identity(
    *,
    cfg_model,
    caps_meta,
    cfg,
    ckpt_sha,
    init_state_meta,
    source,
    precision,
    backend,
    budget=None,
):
    """分母來源／政策與 health budget 一併進身分。

    ⇒ 舊 teacher 分母的 checkpoint 不可能靜默續接；成本量測（budget=None）與短程 leg
    （budget=v1）的 checkpoint 也不可互接。
    """
    return U.resume_identity(
        denominator_source=U.DENOMINATOR_SOURCE,
        denominator_policy_=U.denominator_policy(backend),
        health_budget=U.health_budget_meta(budget),
        cfg_model=cfg_model,
        caps_meta=caps_meta,
        cfg=cfg,
        reference_sha256=ckpt_sha,
        il_start_sha256=ckpt_sha,
        init_state=init_state_meta,
        source_files=source["files"],
        matmul_precision=precision,
        backend=backend,
    )


def save_ppo_checkpoint(
    path: Path, params, opt_state, base_meta: dict[str, Any], train: dict[str, Any]
) -> None:
    """只在**完整 iteration 邊界**呼叫：必要欄位與 `load_ppo_checkpoint` 對齊，
    寫入時一律 `iteration_complete=True`；呼叫端若明寫 False 就拒存，不幫它冒充完整。"""
    for k in ("iteration", "master_key", "rng", "identity", "opt", "env_steps_valid", "updates"):
        if k not in train:
            raise ValueError(f"checkpoint 的 train 缺 {k}，拒絕存半套 resume 狀態")
    # 呼叫端明說這輪沒跑完，就不准存成邊界 checkpoint（不覆寫成 True 冒充完整）
    if train.get("iteration_complete") is False:
        raise ValueError("train 標了 iteration_complete=False：這不是完整 iteration 邊界，拒絕存檔")
    save_train_state(path, params, opt_state, base_meta, {**train, "iteration_complete": True})


#: `--resume-migrate-lr` 允許 source_files 映射的檔
#: （ppo150k iter3 修復 R1 實際改動、且在 SOURCE_FILES 內）。
#: 具體 old／new SHA 由 `--resume-migrate-source-map` 的 JSON 綁定（受審件）；
#: 不在此集合的檔一律嚴格全等。
MIGRATE_SOURCE_WHITELIST = ("rl/full_actions_updater.py", "scripts/full_actions_ppo_train.py")


#: 回退續跑（rollback plan v1.1 B1）：`--resume-source-migrate` 只准映射 trainer 本身
#: （本版只加 --reseed-salt）；其餘 source_files 與全部身分欄嚴格全等。
SOURCE_MIGRATE_WHITELIST = ("scripts/full_actions_ppo_train.py",)


def source_migrate_identity(
    saved: dict[str, Any], current: dict[str, Any], contract: dict[str, Any], state_sha256: str
) -> dict[str, Any]:
    """一次性來源遷移契約：`contract＝{"from_state_sha256", "files": {path: {old, new}}}`。
    綁來源 state 完整 SHA；映射檔只准白名單、old＝checkpoint 記錄、new＝本次實際、old≠new；
    比較用副本把映射檔 old 換成 new 後走原 `check_resume_identity`（其餘欄全等）。
    原 checkpoint 不改。"""
    import copy  # noqa: PLC0415

    if not saved:
        raise ValueError("checkpoint 沒有 identity：拒絕 resume")
    if set(contract) != {"from_state_sha256", "files"} or not contract["files"]:
        raise ValueError(f"來源遷移契約欄位不符：{sorted(contract)}")
    if contract["from_state_sha256"] != state_sha256:
        raise ValueError(
            f"來源遷移契約綁定的 state {contract['from_state_sha256']} ≠ 本次 resume {state_sha256}"
        )
    extra = sorted(set(contract["files"]) - set(SOURCE_MIGRATE_WHITELIST))
    if extra:
        raise ValueError(f"來源遷移含白名單外的檔 {extra}（只准 {SOURCE_MIGRATE_WHITELIST}）")
    s_sf, c_sf = saved.get("source_files") or {}, current.get("source_files") or {}
    for f, m in contract["files"].items():
        if set(m) != {"old", "new"} or m["old"] == m["new"]:
            raise ValueError(f"來源遷移 {f} 須恰有 old／new 且不同：{m}")
        if s_sf.get(f) != m["old"]:
            raise ValueError(f"來源遷移 {f} 的 old ≠ checkpoint 記錄：{s_sf.get(f)} vs {m['old']}")
        if c_sf.get(f) != m["new"]:
            raise ValueError(f"來源遷移 {f} 的 new ≠ 本次實際：{c_sf.get(f)} vs {m['new']}")
    cmp = copy.deepcopy(saved)
    for f, m in contract["files"].items():
        cmp["source_files"][f] = m["new"]
    rep_ = U.check_resume_identity(cmp, current)
    return {
        **rep_,
        "source_migrate": {"from_state_sha256": state_sha256, "files": contract["files"]},
    }


def apply_reseed(master_key, salt: int) -> tuple[Any, dict[str, Any]]:
    """回退續跑（rollback v1.1）：salt＝0 ⇒ 原 key 物件原樣回傳、不 fold、不寫任何欄
    （數值行為與 4323331 相同）；salt>0 ⇒ `fold_in(載入的 master_key, salt)`
    （每個 attempt 都自同一原始 resume key 派生，不串接前一 attempt）。
    salt 是 attempt 動態資訊：記在 init stage，不入靜態 identity
    （下一 leg 以 salt0 接續不會被拒）。"""
    if not salt:
        return master_key, {}
    return jax.random.fold_in(master_key, int(salt)), {
        "reseed_salt": int(salt),
        "master_key_before_salt": key_to_list(master_key),
    }


def parse_migrate_lr(
    values: list[str] | None, *, resume: str, lr: float
) -> tuple[float, float] | None:
    """`--resume-migrate-lr FROM:TO`：只收一筆、只收 lr。

    無 --resume、重複、FROM==TO、--lr≠TO 一律拒。
    """
    if not values:
        return None
    if len(values) != 1:
        raise SystemExit(f"--resume-migrate-lr 只准給一次，得到 {values}")
    if not resume:
        raise SystemExit("--resume-migrate-lr 只能配 --resume 使用")
    parts = values[0].split(":")
    if len(parts) != 2:
        raise SystemExit(f"--resume-migrate-lr 格式是 FROM:TO（只收 lr），得到 {values[0]!r}")
    try:
        fr, to = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise SystemExit(f"--resume-migrate-lr 不是數字：{values[0]!r}") from e
    if not (np.isfinite(fr) and np.isfinite(to) and fr > 0 and to > 0) or fr == to:
        raise SystemExit(f"--resume-migrate-lr 需 FROM≠TO 且皆為正有限值：{values[0]!r}")
    if float(lr) != to:
        raise SystemExit(f"--lr {lr} 必須等於 --resume-migrate-lr 的 TO {to}")
    return fr, to


def _migrate_identity(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    lr_from: float,
    lr_to: float,
    source_map: dict[str, dict[str, str]],
    at_iteration: int,
    from_state_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """比較用副本正規化（coefs.lr FROM→TO、白名單 source_files old→new、追加遷移歷史）後走原
    `check_resume_identity`；回（新身分的 coef_migrations 歷史, 三段報告）。saved 原物件不改。"""
    import copy  # noqa: PLC0415

    if not saved:
        raise ValueError("checkpoint 沒有 identity：拒絕 resume（不靜默 fresh start）")
    hist = list(saved.get("coef_migrations") or [])
    if any(h.get("key") == "lr" for h in hist):
        raise ValueError("checkpoint 已含 lr 遷移歷史：--resume-migrate-lr 只准一次，拒絕再遷移")
    s_lr = (saved.get("coefs") or {}).get("lr")
    if s_lr is None or float(s_lr) != lr_from:
        raise ValueError(f"saved identity.coefs.lr＝{s_lr}，不是 FROM {lr_from}：拒絕遷移")
    c_lr = (current.get("coefs") or {}).get("lr")
    if c_lr is None or float(c_lr) != lr_to:
        raise ValueError(f"current identity.coefs.lr＝{c_lr}，不是 TO {lr_to}：拒絕遷移")
    extra = sorted(set(source_map) - set(MIGRATE_SOURCE_WHITELIST))
    if extra:
        raise ValueError(
            f"source 映射含白名單外的檔 {extra}：拒絕（只准 {MIGRATE_SOURCE_WHITELIST}）"
        )
    s_sf, c_sf = saved.get("source_files") or {}, current.get("source_files") or {}
    for f, m in source_map.items():
        if set(m) != {"old", "new"}:
            raise ValueError(f"source 映射 {f} 必須恰有 old／new：{m}")
        if s_sf.get(f) != m["old"]:
            raise ValueError(
                f"source 映射 {f} 的 old ≠ checkpoint 記錄：{s_sf.get(f)} vs {m['old']}"
            )
        if c_sf.get(f) != m["new"]:
            raise ValueError(f"source 映射 {f} 的 new ≠ 本次實際：{c_sf.get(f)} vs {m['new']}")
        if m["old"] == m["new"]:
            raise ValueError(f"source 映射 {f} 的 old＝new：不需要也不准映射")
    entry = {
        "key": "lr",
        "from": lr_from,
        "to": lr_to,
        "at_iteration": at_iteration,
        "from_state_sha256": from_state_sha256,
        "source_files_map": {f: dict(m) for f, m in sorted(source_map.items())},
    }
    new_hist = [*hist, entry]
    raw_diff = {
        k: sorted(f for f in set(s_sf) | set(c_sf) if s_sf.get(f) != c_sf.get(f))
        if k == "source_files"
        else {"saved": saved.get(k), "current": current.get(k)}
        for k in U.IDENTITY_KEYS
        if k != "coef_migrations" and saved.get(k) != current.get(k)
    }
    comp = copy.deepcopy(saved)
    comp["coefs"] = {**comp["coefs"], "lr": lr_to}
    comp["source_files"] = {**s_sf, **{f: m["new"] for f, m in source_map.items()}}
    comp["coef_migrations"] = new_hist
    cur = {**current, "coef_migrations": new_hist}
    rep = U.check_resume_identity(comp, cur)  # 其餘欄照原函式逐欄全等，不符即 ValueError
    return new_hist, {
        "raw_diff": raw_diff,
        "allowed_migration": entry,
        "rest_all_equal": {"compared_keys": rep["compared_keys"], "all_equal": rep["all_equal"]},
    }


def load_ppo_checkpoint(
    path: Path,
    opt,
    *,
    cfg: U.Cfg,
    identity: dict[str, Any],
    expect_key_len: int = 2,
    migrate_lr: tuple[float, float] | None = None,
    migrate_source_map: dict[str, dict[str, str]] | None = None,
    source_migrate: dict[str, Any] | None = None,
):
    """載回並硬核：`iteration_complete` 為 True、master key 型別長度對、累計欄齊、身分逐欄同。

    缺欄一律拒——不預設 0、不靜默 fresh start、不容許半途的 checkpoint 被當成邊界。

    `migrate_lr=(FROM, TO)`（ppo150k iter3 修復 R1）：三道守衛都走、都不關——
    (a) `load_train_state` 以 saved lr＝FROM／clip＝本次 clip 核 optimizer 配置；
    本次 cfg.lr 必須＝TO，
    Adam count／mu／nu 與 params 全葉載入並逐葉核 shape／dtype（lr 不在 optax 狀態內，只換超參數）；
    (b) 身分用比較副本正規化後走原 `check_resume_identity`（見 `_migrate_identity`）；
    (c) 遷移歷史寫入 `identity["coef_migrations"]`（在 `IDENTITY_KEYS`）。原 checkpoint 檔不改。
    未遷移的 resume：`coef_migrations` 承接 checkpoint 的歷史（未遷移過為 None），其餘欄嚴格全等。
    回傳第 5 項 `id_report`；遷移時另含 `coef_migration` 三段報告。
    """
    if migrate_lr is not None:
        lr_from, lr_to = migrate_lr
        if float(cfg.lr) != lr_to or float(U.opt_config(cfg)["lr"]) != lr_to:
            raise ValueError(f"本次 cfg.lr {cfg.lr} ≠ 遷移 TO {lr_to}")
        guard_cfg = {"lr": lr_from, "clip": U.opt_config(cfg)["clip"]}
        params, opt_state, meta, train = load_train_state(path, opt, opt_cfg=guard_cfg)
        if float(train["opt"]["lr"]) != lr_from:
            raise ValueError(f"checkpoint optimizer lr {train['opt']['lr']} ≠ FROM {lr_from}")
        want = jax.tree.leaves(opt.init(params))
        got = jax.tree.leaves(opt_state)
        if len(want) != len(got) or any(
            np.shape(a) != np.shape(b) or jnp.asarray(a).dtype != jnp.asarray(b).dtype
            for a, b in zip(want, got, strict=True)
        ):
            raise ValueError("optimizer 葉數／shape／dtype 與 TO optimizer 的 init 不符：拒絕遷移")
    else:
        params, opt_state, meta, train = load_train_state(path, opt, opt_cfg=U.opt_config(cfg))
    if train.get("iteration_complete") is not True:
        raise ValueError(
            f"{path} 的 iteration_complete ＝ {train.get('iteration_complete')!r}："
            "不是完整 iteration 邊界，拒絕當 resume 用"
        )
    mk = train.get("master_key")
    if (
        not isinstance(mk, list)
        or len(mk) != expect_key_len
        or not all(isinstance(v, int) and 0 <= v < 2**32 for v in mk)
    ):
        raise ValueError(
            f"{path} 的 master_key 不是長度 {expect_key_len} 的 uint32 list："
            f"{mk!r} ⇒ 拒載（續跑會重放或亂掉 rollout key）"
        )
    missing = [k for k in ("iteration", "env_steps_valid", "updates") if k not in train]
    if missing:
        raise ValueError(f"{path} 的 train 缺 {missing}：累計欄不預設 0，拒載")
    saved = train.get("identity")
    if migrate_lr is not None:
        new_hist, mig = _migrate_identity(
            saved,
            identity,
            lr_from=migrate_lr[0],
            lr_to=migrate_lr[1],
            source_map=migrate_source_map or {},
            at_iteration=int(train["iteration"]),
            from_state_sha256=sha256_file(path),
        )
        identity["coef_migrations"] = new_hist
        id_report = {**mig["rest_all_equal"], "coef_migration": mig}
        return params, opt_state, meta, train, id_report
    if (
        saved
        and saved.get("coef_migrations") is not None
        and identity.get("coef_migrations") is None
    ):
        # 承接：新 checkpoint 帶完整遷移歷史；無旗標 resume 沿用之，其餘欄仍嚴格全等（含 coefs.lr）
        identity["coef_migrations"] = saved["coef_migrations"]
    if source_migrate is not None:
        id_report = source_migrate_identity(saved, identity, source_migrate, sha256_file(path))
        return params, opt_state, meta, train, id_report
    id_report = U.check_resume_identity(saved, identity)
    return params, opt_state, meta, train, id_report


def critic_identity(
    *,
    ckpt_sha: str,
    cfg: U.Cfg,
    pool_sha: str | None,
    il_meta: dict[str, Any],
    opponent: str = "tape-pool",
    league: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """critic 與 trainer 必須相同的身分欄（`apply_critic_init(expect_identity=…)` 逐欄核）。

    ppo150k（2026-09-28；ops 副臂草案 A1）：非 tape-pool 才加 `opponent` 欄——tape-pool 的鍵集逐
    位不變
    （既有 C2 等 critic 仍可核），il-frozen critic 多一鍵，與 tape-pool run 互拒。il-frozen 的對
    手權重＝
    `--ckpt`，已由 `il_sha256` 綁定。
    league（2026-09-30）：opponent=pool 必給 `league`（`pool_league` 輸出），加有序成員 SHA 與
    規則 SHA 兩欄——池的成員或順序一改，critic 與 run 即互拒；其他對手不得給 league。
    """
    if (opponent == "pool") != (league is not None):
        raise ValueError("league 只在 opponent=pool 時給，且 pool 必須給")
    ident = {
        "il_sha256": ckpt_sha,
        "reward": cfg.reward,
        "gamma": float(cfg.gamma),
        "reward_scale": float(cfg.reward_scale),
        "tape_pool_manifest_sha256": pool_sha,
        "caps": CAPS.name,
        "obs_version": il_meta["obs"]["obs_version"],
        "codec": il_meta["action_space"]["codec"],
        "target": critic_target(cfg),
    }
    if cfg.reward == "winloss":
        # winloss（計畫 v3.1 §3）才加 λ 欄；舊 reward 的鍵集逐位不變（既有 critic 仍可核）
        ident["lam"] = float(cfg.lam)
    if opponent != "tape-pool":
        ident["opponent"] = opponent
    if league is not None:
        if league["members_sha256"][0] != ckpt_sha:
            raise ValueError("league 成員 idx0 必須是 --ckpt（IL）")
        ident["pool_members_sha256"] = list(league["members_sha256"])
        ident["pool_rule_sha256"] = league["sha256"]
    return ident


#: winloss critic 的目標名（計畫 v3.1 §3：γ1、λ1、整局常數 z）；與 networth 系 critic 互拒
CRITIC_TARGET_WINLOSS = "winloss_z_gamma1"
#: winloss critic 的結果門（計畫 v3.1 §3，事前凍結）：held-out MSE ≤ 0.95 × fit 平均 z 常數基準
CRITIC_WINLOSS_MSE_RATIO = 0.95


def critic_target(cfg: U.Cfg) -> str:
    if cfg.reward == "winloss":
        return CRITIC_TARGET_WINLOSS
    return "mc_return_own_gamma1" if float(cfg.gamma) == 1.0 else "mc_return_gamma"


def winloss_gate_ok(gate: dict | None) -> tuple[bool, str]:
    """winloss critic 結果門（trainer 端重算，不只信 result.json 的旗標）。"""
    g = gate or {}
    num = ("heldout_mse", "const_mse", "heldout_target_var")
    vals = {k: g.get(k) for k in num}
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in vals.values()):
        return False, f"winloss_gate 數值缺或非有限：{vals}"
    w, lo = g.get("heldout_W"), g.get("heldout_L")
    if not (isinstance(w, int) and isinstance(lo, int) and w >= 1 and lo >= 1):
        return False, f"held-out W／L 局數須皆 ≥1：W={w}、L={lo}"
    if not vals["heldout_target_var"] > 0:
        return False, f"held-out 目標方差須 >0：{vals['heldout_target_var']}"
    if g.get("mse_ratio_max") != CRITIC_WINLOSS_MSE_RATIO:
        return False, f"mse_ratio_max {g.get('mse_ratio_max')} ≠ 凍結值 {CRITIC_WINLOSS_MSE_RATIO}"
    if not vals["heldout_mse"] <= CRITIC_WINLOSS_MSE_RATIO * vals["const_mse"]:
        return False, (
            f"held-out MSE {vals['heldout_mse']} > {CRITIC_WINLOSS_MSE_RATIO}×常數基準 "
            f"{vals['const_mse']}"
        )
    return True, "ok"


#: 有 critic 身分契約的對手（ppo150k 規格 §6 B1：非 tape-pool 也要核）；其餘對手帶 --critic-init
#:  一律拒。league（2026-09-30）加 pool：身分含有序成員＋規則 SHA。
CRITIC_CHECKED = ("tape-pool", "il-frozen", "pool")
CRITIC_EV_MIN = 0.4


def critic_result_gate(path: Path, meta: dict, sha: str) -> None:
    """審查者 終審：正式消費 critic 前必核**結果**，不只 identity／SHA／shape。

    拒 CPU 測試標記（`dirty_cpu_test`、`relabel_step`）、held-out EV < 0.4 或非有限，並要求同目錄
    `result.json` 的 `all_checks_pass` 為真、`critic_sha256` 等於實算 SHA。

    winloss critic（identity.target＝`CRITIC_TARGET_WINLOSS`，計畫 v3.1 §3）：EV≥0.4 門由
    `winloss_gate_ok` 取代（W、L 各 ≥1、方差 >0、held-out MSE ≤0.95×fit 常數基準）。
    """
    if meta.get("dirty_cpu_test") is not False or meta.get("relabel_step") != -1:
        raise SystemExit(
            f"critic 帶測試標記（dirty_cpu_test={meta.get('dirty_cpu_test')}、"
            f"relabel_step={meta.get('relabel_step')}）：不可用於訓練"
        )
    if (meta.get("identity") or {}).get("target") == CRITIC_TARGET_WINLOSS:
        ok, why = winloss_gate_ok(meta.get("winloss_gate"))
        if not ok:
            raise SystemExit(f"winloss critic 結果門未過：{why}：不可用於訓練")
    else:
        ev = meta.get("heldout_ev")
        if not isinstance(ev, (int, float)) or not np.isfinite(ev) or ev < CRITIC_EV_MIN:
            raise SystemExit(f"critic held-out EV {ev} 未達 {CRITIC_EV_MIN}：不可用於訓練")
    rp = path.parent / "result.json"
    if not rp.exists():
        raise SystemExit(f"critic 缺結果檔 {rp}：不可只憑 NPZ 使用")
    doc = json.loads(rp.read_text())
    if doc.get("all_checks_pass") is not True or doc.get("critic_sha256") != sha:
        raise SystemExit(
            f"critic 結果未通過或 SHA 不符：all_checks_pass={doc.get('all_checks_pass')}、"
            f"result.critic_sha256={doc.get('critic_sha256')}"
        )


def load_pool_actors(
    paths: list[str], shas: list[str], il_np: dict[str, np.ndarray], cfg_model: dict[str, Any]
) -> tuple[list[dict], list[str]]:
    """`--pool-actor`／`--expect-pool-actor-sha` 逐一載入：數量相等且 ≥1、SHA 逐一實核、
    架構（config_from_meta）與葉集合／shape 須與 IL 相同——同架構才共用同一支 rollout jit。"""
    if not paths or len(paths) != len(shas) or not all(paths) or not all(shas):
        raise SystemExit(
            "opponent=pool 需要等量且非空的 --pool-actor 與 --expect-pool-actor-sha"
            f"（得到 {len(paths)} 個 actor、{len(shas)} 個 SHA）"
        )
    out = []
    for pth, sha in zip(paths, shas, strict=True):
        pnp, pmeta = FN.load_params(Path(pth), caps=CAPS, expect_sha256=sha)
        if FN.config_from_meta(pmeta) != cfg_model:
            raise SystemExit(f"池 actor {pth} 架構與 --ckpt 不同：{FN.config_from_meta(pmeta)}")
        bad = sorted(
            k
            for k in set(pnp) | set(il_np)
            if k not in pnp or k not in il_np or tuple(pnp[k].shape) != tuple(il_np[k].shape)
        )
        if bad:
            raise SystemExit(f"池 actor {pth} 的參數葉／shape 與 --ckpt 不同：{bad[:6]}")
        out.append({k: jnp.asarray(v) for k, v in pnp.items()})
    return out, list(shas)


def load_tape_pool(path: str, expect_sha: str):
    """混池 manifest → 記憶體內拼接的 `TapeBank`（原 bank 不改寫）＋ manifest。"""
    try:
        arrays, man = TP.load_pool(Path(path), expect_sha)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    bank = T.TapeBank(**{k: jnp.asarray(v) for k, v in arrays.items()}, source=())
    return bank, man


# ------------------------------------------------------------------------ 暖身 critic
CRITIC_KEYS = ("value_w", "value_b")


def apply_critic_init(
    il_params: dict, path: str, expect_sha: str, expect_identity: dict | None = None
):
    """只替換 `value_w`／`value_b`；其餘（policy 與 trunk）逐位不動，錯 SHA／shape 一律拒。

    計畫 內部規格文件 §2。沒給 `path` 就原樣回 IL。
    `expect_identity` 給定時（T3 穩定化計畫 §3），critic meta 的 `identity` 必須逐欄相等：
    IL／reward／γ／scale／混池 manifest／caps／obs／codec 任一不同都拒——value 頭 shape
    相同不代表可混用。
    """
    if not path:
        if expect_sha:
            raise SystemExit("給了 --expect-critic-sha 卻沒給 --critic-init")
        return il_params, None
    got = sha256_file(Path(path))
    if got != expect_sha:
        raise SystemExit(f"critic SHA 不符：期望 {expect_sha!r}、實算 {got}")
    z = np.load(path, allow_pickle=False)
    new = dict(il_params)
    for k in CRITIC_KEYS:
        arr = np.asarray(z[k], np.float32)
        if arr.shape != tuple(il_params[k].shape):
            raise SystemExit(f"critic {k} shape {arr.shape} ≠ IL {tuple(il_params[k].shape)}")
        new[k] = jnp.asarray(arr)
    policy = lambda p: {k: v for k, v in p.items() if k not in CRITIC_KEYS}  # noqa: E731
    if leaves_sha256(policy(new)) != leaves_sha256(policy(il_params)):
        raise SystemExit("替換 critic 後 policy 葉 SHA 改變：拒絕")
    meta = json.loads(str(z["meta"])) if "meta" in z.files else None
    if expect_identity is not None:
        got_id = (meta or {}).get("identity")
        if not isinstance(got_id, dict):
            raise SystemExit("critic 缺 identity（舊格式 critic 不可用於本配方）")
        diff = {
            k: {"critic": got_id.get(k), "run": v}
            for k, v in expect_identity.items()
            if got_id.get(k) != v
        }
        if diff or set(got_id) != set(expect_identity):
            extra = sorted(set(got_id) ^ set(expect_identity))
            raise SystemExit(f"critic identity 與本 run 不符：{diff or extra}")
        critic_result_gate(Path(path), meta, got)
    return new, {
        "path": str(Path(path).resolve()),
        "sha256": got,
        "replaced_keys": list(CRITIC_KEYS),
        "policy_leaves_sha256": leaves_sha256(policy(new)),
        "meta": meta,
    }


# -------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="IL checkpoint（npz）")
    ap.add_argument("--expect-ckpt-sha", required=True)
    ap.add_argument("--row", type=int, default=150)
    ap.add_argument("--n-fresh", type=int, default=15)
    ap.add_argument("--t", type=int, default=rules.N_TRANSITIONS)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minibatch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument(
        "--relabel-step",
        type=int,
        default=-1,
        help="**只給 tests 用**的教師時間重標；-1 ＝ 不重標（正式量測一律 -1）",
    )
    ap.add_argument(
        "--health-budget",
        choices=("off", *U.BUDGETS),
        default="off",
        help=(
            "`off` ＝ 成本量測模式（舊 run 同語義）；`v1` ＝ 短程 leg 的 host 停損預算；"
            "`t3_stabilization_v1` ＝ T3 §6.2（ratio [0.2,5]、輪末全量 post-update、anchor KL）"
        ),
    )
    ap.add_argument("--resume", default="", help="完整 iteration 邊界的 PPO checkpoint")
    ap.add_argument(
        "--resume-source-migrate",
        default="",
        help="回退續跑一次性來源遷移契約 JSON（rollback v1.1 B1）："
        "綁來源 state SHA，只准 trainer 檔 old→new",
    )
    ap.add_argument(
        "--reseed-salt",
        type=int,
        default=0,
        help="回退續跑（ppo-league-lrm3 rollback v1）："
        ">0 ⇒ 載入 master key 後 fold_in(master_key, salt)；"
        "0＝逐位不變（不 fold、不寫額外欄）。只准 0–1000，且 >0 須配 --resume",
    )
    ap.add_argument("--gamma", type=float, default=1.0, help="GAE 的 γ；1.0 ＝ 原版 P1.gae")
    ap.add_argument(
        "--anchor-beta",
        type=float,
        default=0.0,
        help="T3 KL 錨 β（row-sum nats；0 ＝ 不加錨）；reference 固定為 --ckpt 的原 IL",
    )
    ap.add_argument(
        "--critic-init",
        default="",
        help="暖身 critic npz（只含 value_w／value_b）；iteration 0 前只替換這兩個葉",
    )
    ap.add_argument("--expect-critic-sha", default="", help="--critic-init 的預期 SHA256")
    ap.add_argument("--opponent", choices=OPPONENTS, default="self", help="rollout 的對手")
    ap.add_argument(
        "--pool-actor",
        action="append",
        default=None,
        help="opponent=pool 的凍結 actor；可重複給 ⇒ 有序池 [--ckpt, 第1個, 第2個, …]（league）",
    )
    ap.add_argument(
        "--expect-pool-actor-sha", action="append", default=None, help="與 --pool-actor 逐一對應"
    )
    ap.add_argument(
        "--reward",
        choices=("networth", "networth_own", "winloss"),
        default="networth",
        help=(
            "networth ＝ 自己淨值變動 − 對手現金變動（舊行為）；networth_own 刪掉對手項；"
            "winloss ＝ 終局 sign(own−opp) 只在最後有效步（須 --lam 1 --reward-scale 1）"
        ),
    )
    ap.add_argument(
        "--lam",
        type=float,
        default=U.Cfg().lam,
        help="GAE λ；非 winloss 只准 P1.LAMBDA（0.95，預設）；winloss 只准 1.0",
    )
    ap.add_argument(
        "--reward-scale",
        type=float,
        default=U.Cfg().reward_scale,
        help="reward 尺度；非 winloss 預設 1e-4（舊行為）；winloss 只准 1.0",
    )
    ap.add_argument("--tape-bank", default="", help="opponent=tape 的 tape bank npz")
    ap.add_argument("--expect-tape-bank-sha", default="")
    ap.add_argument("--pool-manifest", default="", help="opponent=tape-pool 的混池 manifest")
    ap.add_argument("--expect-pool-manifest-sha", default="")
    ap.add_argument(
        "--diag-probe",
        default="",
        help="一次性診斷 spec JSON（opt-in）：只跑 --iters 1、不 resume、不寫 checkpoint／actor",
    )
    ap.add_argument(
        "--resume-migrate-lr",
        action="append",
        default=None,
        metavar="FROM:TO",
        help=(
            "ppo150k iter3 修復 R1：resume 時把 lr 由 FROM 遷移為 TO"
            "（只收一次、只收 lr；--lr 必須＝TO）"
        ),
    )
    ap.add_argument(
        "--resume-migrate-source-map",
        default="",
        help="--resume-migrate-lr 的 source_files 白名單映射 JSON {path: {old, new}}（必填）",
    )
    ap.add_argument(
        "--repro-gate",
        default="",
        help="ppo150k iter3 修復 R2：更新前事件重現守門 spec JSON（不符即落盤並停）",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.diag_probe and (args.iters != 1 or args.resume):
        raise SystemExit("--diag-probe 只准 --iters 1 且不 resume")
    if args.diag_probe and args.repro_gate:
        raise SystemExit("--diag-probe 與 --repro-gate 不可同時使用（模式隔離）")
    if not 0 <= args.reseed_salt <= 1000:
        raise SystemExit(f"--reseed-salt 只准 0–1000：{args.reseed_salt}")
    if args.reseed_salt and not args.resume:
        raise SystemExit("--reseed-salt >0 只能配 --resume（回退續跑）")
    if args.resume_source_migrate and (not args.resume or args.resume_migrate_lr):
        raise SystemExit("--resume-source-migrate 只能配 --resume，且不與 --resume-migrate-lr 併用")
    source_migrate = (
        json.loads(Path(args.resume_source_migrate).read_text())
        if args.resume_source_migrate
        else None
    )
    migrate_lr = parse_migrate_lr(args.resume_migrate_lr, resume=args.resume, lr=args.lr)
    if bool(migrate_lr) != bool(args.resume_migrate_source_map):
        raise SystemExit("--resume-migrate-lr 與 --resume-migrate-source-map 必須一起給")
    migrate_source_map = (
        json.loads(Path(args.resume_migrate_source_map).read_text())
        if args.resume_migrate_source_map
        else None
    )

    out_path = Path(args.out)
    for pth in (out_path, out_path.parent):
        if pth.exists():
            raise SystemExit(f"已存在，拒絕覆蓋既有原件：{pth}")

    ckpt = Path(args.ckpt)
    ckpt_sha = sha256_file(ckpt)
    if ckpt_sha != args.expect_ckpt_sha:
        raise SystemExit(f"ckpt SHA 不符：期望 {args.expect_ckpt_sha}、實算 {ckpt_sha}")
    np_params, il_meta = FN.load_params(ckpt, caps=CAPS)
    cfg_model = FN.config_from_meta(il_meta)
    il_params = {k: jnp.asarray(v) for k, v in np_params.items()}
    pool_params, pool_sha, league = None, None, None
    pool_paths, pool_shas = args.pool_actor or [], args.expect_pool_actor_sha or []
    if args.opponent == "pool":
        pool_params, pool_shas = load_pool_actors(pool_paths, pool_shas, np_params, cfg_model)
        # 單一 actor：身分欄與舊行為逐位相同（字串）；≥2：有序清單＋league 身分
        pool_sha = pool_shas[0] if len(pool_shas) == 1 else list(pool_shas)
        league = pool_league(ckpt_sha, pool_shas)
        if len(set(league["members_sha256"])) != len(league["members_sha256"]):
            raise SystemExit(f"池成員（含 --ckpt）SHA 重複：{league['members_sha256']}")
        if len(pool_params) == 1:
            pool_params = pool_params[0]
    elif pool_paths or pool_shas:
        raise SystemExit("--pool-actor 只在 opponent=pool 時使用")
    tape_bank, tape_sha = None, None
    if args.opponent == "tape":
        if not args.tape_bank or not args.expect_tape_bank_sha:
            raise SystemExit("opponent=tape 需要 --tape-bank 與 --expect-tape-bank-sha")
        tape_sha = sha256_file(Path(args.tape_bank))
        if tape_sha != args.expect_tape_bank_sha:
            raise SystemExit(
                f"tape bank SHA 不符：期望 {args.expect_tape_bank_sha}、實算 {tape_sha}"
            )
        tape_bank = T.load_bank(Path(args.tape_bank))._replace(source=())
    elif args.tape_bank or args.expect_tape_bank_sha:
        raise SystemExit("--tape-bank 只在 opponent=tape 時使用")
    tape_pool_man, tape_pool_sha = None, None
    if args.opponent == "tape-pool":
        if not args.pool_manifest or not args.expect_pool_manifest_sha:
            raise SystemExit(
                "opponent=tape-pool 需要 --pool-manifest 與 --expect-pool-manifest-sha"
            )
        tape_bank, tape_pool_man = load_tape_pool(args.pool_manifest, args.expect_pool_manifest_sha)
        tape_pool_sha = args.expect_pool_manifest_sha
    elif args.pool_manifest or args.expect_pool_manifest_sha:
        raise SystemExit("--pool-manifest 只在 opponent=tape-pool 時使用")
    cfg = U.Cfg(
        minibatch=args.minibatch,
        lr=args.lr,
        gamma=args.gamma,
        reward=args.reward,
        lam=args.lam,
        reward_scale=args.reward_scale,
        anchor_beta=args.anchor_beta,
    )
    U.check_cfg(cfg)
    # 暖身 critic 必須是同 IL／reward／γ／對手（tape-pool 另核混池）做出來的（身分逐欄核；ops 草
    # 案 A2）
    critic_expect = (
        critic_identity(
            ckpt_sha=ckpt_sha,
            cfg=cfg,
            pool_sha=tape_pool_sha,
            il_meta=il_meta,
            opponent=args.opponent,
            league=league,
        )
        if args.opponent in CRITIC_CHECKED
        else None
    )
    if args.critic_init and critic_expect is None:
        raise SystemExit(f"opponent={args.opponent} 沒有 critic 身分契約：不接受 --critic-init")
    start_params, critic_meta = apply_critic_init(
        il_params, args.critic_init, args.expect_critic_sha, critic_expect
    )

    dirty_paths = _git_porcelain()
    hit = sorted(set(dirty_paths) & set(SOURCE_FILES))
    if hit:
        raise SystemExit(f"依賴集合裡有未提交的改動，拒絕啟動：{hit}")
    precision = require_gpu_highest()
    backend = jax.devices()[0].platform
    if args.iters <= 0:
        raise SystemExit(f"--iters 必須是正整數，得到 {args.iters}")
    if U.Cfg().epochs != 1 or cfg.epochs != 1:
        raise SystemExit(f"本段只做 epoch 1，cfg.epochs ＝ {cfg.epochs}")
    if args.relabel_step != -1 and backend != "cpu":
        raise SystemExit(
            f"--relabel-step {args.relabel_step} 只准在 CPU tests 用；"
            f"現在 backend ＝ {backend}，量測一律 -1（不重標）"
        )
    source = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": {f: sha256_file(ROOT / f) for f in SOURCE_FILES},
        "dirty_paths": dirty_paths,
        "dirty_diff_sha256": _git_diff_sha256(),
        "dependency_set": "ppo-train（bridge／updater／decoder／reward／checkpoint 全鏈）",
    }
    caps_meta = {
        "name": CAPS.name,
        "max_units": CAPS.max_units,
        "max_orders": CAPS.max_orders,
        "market_steps": CAPS.market_steps,
        "max_order_qty": CAPS.max_order_qty,
    }
    # identity 只放**靜態的 init 來源與派生策略**；每輪實際 seed／key 記在 train／history，
    # 不然 resume 會被動態值弄成假不符。
    init_state_meta = {
        "row": args.row,
        "n_fresh": args.n_fresh,
        "T": args.t,
        "base_seed": args.seed,
        "relabel_step": args.relabel_step,
        "seed_derivation": seed_derivation_note(args.n_fresh),
        # 暖身 critic 進 identity：不同 critic 起點的 checkpoint 不可互接
        "critic_init_sha256": critic_meta["sha256"] if critic_meta else None,
        "opponent": args.opponent,
        "pool_actor_sha256": pool_sha,
        # league（≥2 個 pool actor）才加：有序成員＋規則 SHA 進 identity（換池／換序即不可互接）
        **({"pool_league": league} if isinstance(pool_sha, list) else {}),
        "tape_bank_sha256": tape_sha,
        "tape_pool_manifest_sha256": tape_pool_sha,
        # KL 錨方向／估計器／聚合進身分：β 在 cfg（coefs）裡，這裡記語義
        "anchor": U.ANCHOR_META if args.anchor_beta > 0 else None,
        "tape_pool_sampling": tape_pool_man["sampling"] if tape_pool_man else None,
        "tape_pool_bank_sha256": (
            {n: tape_pool_man["pools"][n]["sha256"] for n in TP.POOL_NAMES}
            if tape_pool_man
            else None
        ),
        "reward": args.reward,
    }
    budget = U.BUDGETS.get(args.health_budget)
    head = {
        "schema": "full-actions-ppo-train-v1",
        "kind": (
            "ppo-short-leg（host 停損已開）"
            if budget
            else "ppo-trainer-measurement（量測用係數，非正式配方）"
        ),
        "matmul_precision": precision,
        "backend": backend,
        "device": device_info_with_limit(),
        "source": source,
        "weights": {"il_start": str(ckpt.resolve()), "sha256": ckpt_sha, "config": cfg_model},
        "reference": {"same_as_il_start": True, "sha256": ckpt_sha},
        "critic_init": critic_meta,
        "caps": caps_meta,
        "coefs": cfg._asdict(),
        "denominator": {
            "source": U.DENOMINATOR_SOURCE,
            "policy": U.denominator_policy(backend),
        },
        "health_budget": U.health_budget_meta(budget),
        "ratio_gate_scope": "候選接受前、更新前前向的 ratio；不保證更新後狀態的 ratio",
        "run": {
            **init_state_meta,
            "iters": args.iters,
            "resume": args.resume or None,
            **({"diag_probe": args.diag_probe} if args.diag_probe else {}),
            **(
                {
                    "resume_source_migrate": {
                        "path": args.resume_source_migrate,
                        "sha256": sha256_file(Path(args.resume_source_migrate)),
                        "contract": source_migrate,
                    }
                }
                if source_migrate
                else {}
            ),
            **(
                {
                    "resume_migrate_lr": {"from": migrate_lr[0], "to": migrate_lr[1]},
                    "resume_migrate_source_map": {
                        "path": args.resume_migrate_source_map,
                        "sha256": sha256_file(Path(args.resume_migrate_source_map)),
                        "map": migrate_source_map,
                    },
                }
                if migrate_lr
                else {}
            ),
            **(
                {
                    "repro_gate": {
                        "path": args.repro_gate,
                        "sha256": sha256_file(Path(args.repro_gate)),
                    }
                }
                if args.repro_gate
                else {}
            ),
        },
    }
    sink = Sink(out_path, head)
    checks = Checks()
    t_start = time.perf_counter()

    diag = None
    if args.diag_probe:
        from rl import t3_diag_probe as DG

        diag = DG.Diag(json.loads(Path(args.diag_probe).read_text()), cfg_model)
    ctx = Ctx(
        cfg_model,
        cfg,
        t=args.t,
        row=args.row,
        n_fresh=args.n_fresh,
        relabel=args.relabel_step,
        il_params=il_params,
        opponent=args.opponent,
        pool_params=pool_params,
        tape_bank=tape_bank,
        tape_pool_manifest=tape_pool_man,
        effect_mask=U.uses_effect_mask(budget),
        pool_members_sha256=(league["members_sha256"] if isinstance(pool_sha, list) else None),
    )
    ctx.diag = diag
    if args.repro_gate:
        ctx.repro = ReproGate(
            json.loads(Path(args.repro_gate).read_text()), out_dir=out_path.parent
        )
    # identity 綁的是**教師來源身分**（shard ＋ SHA ＋ 原 step），不只 row 號
    init_state_meta = {**init_state_meta, "teacher_source": ctx.teacher_source()}
    sink.doc["run"] = {**sink.doc["run"], "teacher_source": ctx.teacher_source()}
    sink.flush()
    identity = build_identity(
        cfg_model=cfg_model,
        caps_meta=caps_meta,
        cfg=cfg,
        ckpt_sha=ckpt_sha,
        init_state_meta=init_state_meta,
        source=source,
        precision=precision,
        backend=backend,
        budget=budget,
    )
    ppo_meta = {
        **{k: v for k, v in il_meta.items() if k not in ("source", "train")},
        "kind": "ppo",
        "source": source,
        "il_start_sha256": ckpt_sha,
    }

    if args.resume:
        params, opt_state, _meta_l, train_l, id_report = load_ppo_checkpoint(
            Path(args.resume),
            ctx.opt,
            cfg=cfg,
            identity=identity,
            migrate_lr=migrate_lr,
            migrate_source_map=migrate_source_map,
            source_migrate=source_migrate,
        )
        master_key = key_from_list(train_l["master_key"])
        rng = restore_rng(train_l["rng"])
        start_iter = int(train_l["iteration"])
        cum = {
            "env_steps_valid": int(train_l["env_steps_valid"]),
            "updates": int(train_l["updates"]),
        }
        checks.add("resume/identity_compared", bool(id_report["all_equal"]), id_report)
        if migrate_lr:
            checks.add(
                "resume/coef_migration",
                bool(id_report["coef_migration"]["rest_all_equal"]["all_equal"]),
                id_report["coef_migration"],
            )
    else:
        params, opt_state = start_params, None
        master_key = jax.random.PRNGKey(args.seed)
        rng = np.random.default_rng(args.seed)
        start_iter, cum = 0, {"env_steps_valid": 0, "updates": 0}
        id_report = {"resumed": False}
    if opt_state is None:
        opt_state = ctx.opt.init(params)
    # 健康 STOP 後自同一 resume state 換 key 重跑：只換隨機流，
    # params／Adam／critic／NumPy RNG／累計欄不動
    master_key, salt_info = apply_reseed(master_key, int(args.reseed_salt))
    sink.stage(
        "init",
        {
            **salt_info,
            "resumed": bool(args.resume),
            "start_iteration": start_iter,
            "identity": id_report,
            "master_key": key_to_list(master_key),
            "cumulative": cum,
            "wall_s": round(time.perf_counter() - t_start, 5),
        },
    )

    ckpt_paths = []
    for it in range(start_iter, start_iter + args.iters):
        try:
            res = run_iteration(
                ctx,
                params,
                opt_state,
                master_key,
                rng,
                iteration=it,
                checks=checks,
                sink=sink,
                budget=budget,
            )
        except Exception as e:
            # 計畫 winloss v3.1 §5 第 1 項／審查者 實作審 B1：只有明確型別的健康門才標 health_stop
            # （runner 據此 record-stop reason＝health_stop）；其他一律 abnormal_stop 落盤留證據
            info = U.health_stop_info(e)
            key = "health_stop" if info is not None else "abnormal_stop"
            sink.doc[key] = {
                "iteration": it,
                "type": type(e).__name__,
                "error": str(e)[:2000],
                **(info or {"kind": "abnormal"}),
            }
            sink.flush()
            if diag is not None:
                rep_ = getattr(e, "report", None)
                diag.set_final(
                    {
                        "outcome": type(e).__name__,
                        "error": str(e)[:2000],
                        "post_update_failed": (rep_ or {}).get("failed"),
                        "post_update_ratio_minmax": [
                            (rep_ or {}).get("ratio_active_min"),
                            (rep_ or {}).get("ratio_active_max"),
                        ],
                        "candidate_written": False,
                    }
                )
            raise
        if diag is not None:
            # 診斷模式：輪末檢查照舊跑完（過了也）**不寫** checkpoint／actor、不輸出候選
            post_ = res["report"].get("post_update_check") or {}
            diag.set_final(
                {
                    "outcome": "post_check_pass",
                    "post_update_failed": post_.get("failed"),
                    "post_update_ratio_minmax": [
                        post_.get("ratio_active_min"),
                        post_.get("ratio_active_max"),
                    ],
                    "candidate_written": False,
                }
            )
            sink.doc["checks"] = checks.report()
            sink.doc["verdict"] = "DIAG_ONLY"
            sink.doc["wall_s_total"] = round(time.perf_counter() - t_start, 5)
            sink.doc["peak_rss_gib"] = rss_gib()
            sink.flush()
            return 0
        params, opt_state, master_key = res["params"], res["opt_state"], res["master_key"]
        cum["env_steps_valid"] += res["env_steps_valid"]
        cum["updates"] += res["n_updates"]
        train = {
            "kind": "ppo",
            "iter": it + 1,  # `load_train_state` 要的欄位：已完成的輪數
            "iteration": it + 1,
            "master_key": key_to_list(master_key),
            # 下一輪**排程之前**的 RNG：resume 要用它重建下一輪，不是沿用記憶體 plan
            "rng": {k: v for k, v in rng.bit_generator.state.items()},
            "opt": U.opt_config(cfg),
            "identity": identity,
            "env_steps_valid": cum["env_steps_valid"],
            "updates": cum["updates"],
            "next_update_index": it + 1,
        }
        t0 = time.perf_counter()
        state_path = out_path.parent / f"ppo_state_iter{it + 1}.npz"
        save_ppo_checkpoint(state_path, params, opt_state, ppo_meta, train)
        actor_path = out_path.parent / f"ppo_actor_iter{it + 1}.npz"
        FN.save_params(
            actor_path,
            {k: np.asarray(v) for k, v in params.items()},
            {**ppo_meta, "kind": "ppo-actor"},
        )
        ckpt_paths.append(
            {
                "iteration": it + 1,
                "state": str(state_path),
                "state_sha256": sha256_file(state_path),
                "actor": str(actor_path),
                "actor_sha256": sha256_file(actor_path),
                "wall_s": round(time.perf_counter() - t0, 5),
            }
        )
        sink.stage(f"iter{it}/checkpoint", ckpt_paths[-1])

    # 存回去的 checkpoint 必須真的載得起來（含 iteration_complete 與 master key）
    p_l, o_l, meta_l, train_l, id_back = load_ppo_checkpoint(
        Path(ckpt_paths[-1]["state"]), ctx.opt, cfg=cfg, identity=identity
    )
    checks.add("checkpoint/reloads", bool(id_back["all_equal"]))
    checks.add(
        "checkpoint/meta_kind_ppo", meta_l.get("kind") == "ppo", {"kind": meta_l.get("kind")}
    )
    checks.add(
        "checkpoint/no_il_corpus_inputs",
        "inputs" not in (meta_l.get("source") or {}),
        {"source_keys": sorted(meta_l.get("source") or {})},
    )
    checks.add(
        "checkpoint/master_key_roundtrip",
        train_l["master_key"] == key_to_list(master_key),
        {"saved": train_l["master_key"], "current": key_to_list(master_key)},
    )
    checks.add(
        "checkpoint/rng_roundtrip",
        restore_rng(train_l["rng"]).bit_generator.state == rng.bit_generator.state,
    )
    checks.add(
        "checkpoint/params_bitwise",
        all(
            bool(jnp.all(a == b))
            for a, b in zip(jax.tree.leaves(p_l), jax.tree.leaves(params), strict=True)
        ),
    )
    checks.add(
        "checkpoint/opt_state_bitwise",
        all(
            bool(jnp.all(a == b))
            for a, b in zip(jax.tree.leaves(o_l), jax.tree.leaves(opt_state), strict=True)
        ),
    )
    _actor_params, actor_meta_back = FN.load_params(Path(ckpt_paths[-1]["actor"]), caps=CAPS)
    checks.add(
        "checkpoint/actor_loads_as_ppo_actor",
        actor_meta_back.get("kind") == "ppo-actor",
        {"kind": actor_meta_back.get("kind")},
    )
    sink.stage(
        "checkpoint",
        {
            "written": ckpt_paths,
            "cumulative": cum,
            "train_keys": sorted(train_l),
            "iteration_complete": train_l.get("iteration_complete"),
        },
    )

    sink.doc["checks"] = checks.report()
    sink.doc["verdict"] = sink.doc["checks"]["verdict"]
    sink.doc["wall_s_total"] = round(time.perf_counter() - t_start, 5)
    sink.doc["peak_rss_gib"] = rss_gib()
    sink.flush()
    print(json.dumps({k: v for k, v in sink.doc.items() if k != "stages"}, ensure_ascii=False))
    return 0 if not checks.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
