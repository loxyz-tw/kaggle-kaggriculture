#!/usr/bin/env python
"""完整動作 PPO 第 2 項：rollout 成本與正確性量測（審查者 §45 PASS 的規格）。

規格：內部規格文件。**只量成本與驗契約**，
不改 decoder／不做 updater／不宣稱任何競爭力。

為什麼要這一支
--------------
updater 的帽與 PPO 的短程估算都要真數字：**編譯**與 **steady** 分開、
decode 與 `env.step` 分開（不把環境時間藏進 actor）、完整 719 步 scan 一次實測
（不拿兩座微基準相加冒稱實測）。

分段落盤
--------
每個 stage 一完成就寫 `result.json`（覆蓋同一檔，逐段加欄）。1800 s 硬帽若在
完整 scan 砍下去，前面的分段成本仍在檔裡；**不自動重跑**。

口徑
----
* 兩次完整 trace 是 **cold／warm 的成本量測**，不是兩個新樣本；warm 只核結果相同。
* 固定 key 的 10 次穩態重複是**微基準**，不是訓練、不是獨立對局。
* CPU 沒有 device peak 就記 `null`，不冒稱 0。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl import full_actions_numpy as FN  # noqa: E402
from rl import full_actions_ppo as PB  # noqa: E402
from rl import loop4_ppo as P1  # noqa: E402
from rl import rules  # noqa: E402
from rl import state as S  # noqa: E402

CAPS = PB.CAPS
#: 真教師列的定位檔（JSON：`shard`、`shard_sha256`）；以環境變數 `KAGGRI_FIXTURE_REFS` 指定。
FIXREF = Path(os.environ.get("KAGGRI_FIXTURE_REFS") or ROOT / "fixtures/full_actions_fixture_refs.json")
#: 身分閉包：這些檔任何一支變了，成本數字就不能跨 commit 比較。
SOURCE_FILES = (
    "rl/full_actions_ppo.py",
    "rl/full_actions_decode.py",
    "rl/full_actions_head.py",
    "rl/full_actions_obs.py",
    "rl/full_actions_numpy.py",
    "rl/loop4_ppo.py",
    "rl/loop4_obs.py",
    "rl/loop4_model.py",
    "rl/env.py",
    "rl/kernels.py",
    "rl/state.py",
    "rl/rules.py",
    "rl/caps.py",
    "rl/pricing.py",
    "rl/actions.py",
    "rl/gae.py",
    "rl/reward.py",
    "rl/ordered_v3_codec.py",
    "rl/ordered_v3_contract.py",
    "scripts/full_actions_ppo_cost.py",
)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rss_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


def device_info() -> dict[str, Any]:
    """platform ＋（可用時）device peak。CPU 沒有 `memory_stats` 就明記 null。"""
    d = jax.devices()[0]
    stats = None
    fn = getattr(d, "memory_stats", None)
    if callable(fn):
        try:
            got = fn()
        except Exception:  # noqa: BLE001  取不到就記 null，不猜
            got = None
        if got:
            stats = {
                "peak_bytes_in_use": got.get("peak_bytes_in_use"),
                "bytes_in_use": got.get("bytes_in_use"),
            }
    return {"platform": d.platform, "device_kind": d.device_kind, "memory_stats": stats}


def cache_size(fn: Any) -> int | None:
    """jit 的 cache entry 數：每多一筆就是一次重編。取不到就 null，不猜。"""
    for name in ("_cache_size", "cache_info"):
        got = getattr(fn, name, None)
        if callable(got):
            try:
                out = got()
            except Exception:  # noqa: BLE001
                return None
            return (
                int(out)
                if isinstance(out, int)
                else int(getattr(out, "hits", 0) + getattr(out, "misses", 0))
            )
    return None


def matmul_precision() -> str:
    """實際生效的 matmul 精度（不是「我們設了什麼」，是 config 現在是什麼）。"""
    raw = jax.config.jax_default_matmul_precision
    if raw is None:
        return "default"
    name = getattr(raw, "name", None)
    return (name or str(raw)).split(".")[-1].lower()


def require_gpu_highest() -> str:
    """GPU 上必須是 `highest`。

    實測（`p2-ppo-cost-gpu-f61efc3`）：TF32 預設讓 sample 與 teacher 重算的逐 token
    logp 差到 1.45e-3，`highest` 降到 1.97e-6。`rl/export_numpy.py` 檔頭對同一張
    RTX 5080 也記過 3.3e-4 → 1.2e-7。所以這不是偏好，是跑之前就該擋下的前提。
    """
    prec = matmul_precision()
    if jax.devices()[0].platform != "cpu" and prec != "highest":
        raise SystemExit(
            f"GPU 上 matmul 精度是 {prec!r}，必須是 'highest'："
            "請以 JAX_DEFAULT_MATMUL_PRECISION=highest 啟動（不放寬 1e-5 容差）"
        )
    return prec


def ready(tree: Any) -> Any:
    """所有 device 輸出 `block_until_ready` 後才停錶。"""
    jax.block_until_ready(tree)
    return tree


def timed(fn, *a, **kw) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = ready(fn(*a, **kw))
    return out, time.perf_counter() - t0


def steady(fn, *a, n: int = 10, **kw) -> dict[str, Any]:
    ts = []
    for _ in range(n):
        _, dt = timed(fn, *a, **kw)
        ts.append(dt)
    arr = np.asarray(ts, dtype=np.float64)
    return {
        "n": n,
        # 明寫算法：偶數樣本的 median 是中間兩筆的平均（不是取上中位）
        "median_s": round(float(np.median(arr)), 5),
        "p95_s": round(float(np.percentile(arr, 95, method="linear")), 5),
        "min_s": round(float(arr.min()), 5),
        "max_s": round(float(arr.max()), 5),
        "method": "np.median / np.percentile(95, linear)",
        "samples_s": [round(float(x), 5) for x in arr],
    }


# ------------------------------------------------------------------ 起始狀態
def sha256_stream(path: Path, chunk: int = 1 << 22) -> str:
    """822 MiB 的 shard 不整包讀進記憶體：分塊串流算 SHA。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def leaves_sha256(tree: Any) -> str:
    """state 全葉的內容 SHA（葉序 ＝ `_fields` 序），給核收與 reuse 對帳。"""
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def teacher_row(row: int):
    """真教師列（**保留原 step，不重標**）。缺定位檔或缺 shard 直接 raise。

    定位檔記的 `shard_sha256` **實算比對**，不是引用就算數。
    """
    if not FIXREF.exists():
        raise FileNotFoundError(
            f"缺 fixture 定位檔：{FIXREF}（請以環境變數 KAGGRI_FIXTURE_REFS 指定）"
        )
    ref = json.loads(FIXREF.read_text())
    shard = ROOT / ref["shard"]
    if not shard.exists():
        raise FileNotFoundError(f"缺 shard：{shard}")
    actual = sha256_stream(shard)
    if actual != ref["shard_sha256"]:
        raise ValueError(f"shard SHA 不符：定位檔 {ref['shard_sha256']}，實算 {actual}（{shard}）")
    from scripts.loop4_il_labels import _read_slice  # noqa: PLC0415

    z = zipfile.ZipFile(shard)
    st = S.SimState(
        **{f: jnp.asarray(_read_slice(z, f"s/{f}.npy", row, row + 1)) for f in S.SimState._fields}
    )
    ident = {
        "shard": ref["shard"],
        "shard_sha256_ref": ref["shard_sha256"],
        "shard_sha256_actual": actual,
        "fixture_ref_sha256": sha256_file(FIXREF),
        "npz_row": row,
        "original_step": int(st.step[0]),
        "data_schema_version": ref.get("data_schema_version"),
        "teacher_state_leaves_sha256": leaves_sha256(st),
    }
    return st, ident


def init_batch(row: int, seed: int, n_fresh: int = 1):
    """`B = n_fresh + 1`：前 `n_fresh` 列 fresh、最後一列真教師列（**原 step**）。

    **seed 派生**：`S.reset(n, caps, seed)` 內部是 `jax.random.split(PRNGKey(seed), n)`，
    所以 `n` 不同時每一列的 key 都不同——`reset(15, seed=0)[0]` **不等於**
    `reset(1, seed=0)[0]`。因此 `n_fresh=1` 走的是與既有基準**完全相同的呼叫**
    （`S.reset(1, CAPS, seed=seed)`），B=2 的舊結果仍可逐位比對；`n_fresh>1` 是
    另一組 env，屬新基準，不與 B=2 混報。
    """
    if n_fresh < 1:
        raise ValueError(f"--n-fresh 至少 1，得到 {n_fresh}")
    fresh = S.reset(n_fresh, CAPS, seed=seed)
    teach, ident = teacher_row(row)
    state = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), fresh, teach)
    ident = {
        **ident,
        "n_fresh": n_fresh,
        "seed_derivation": f"jax.random.split(PRNGKey({seed}), {n_fresh})；"
        "n_fresh=1 與既有 B=2 基準同一呼叫",
        "teacher_row_index_in_batch": n_fresh,
    }
    return state, ident


# ------------------------------------------------------------------ 契約驗收
def trace_contract(traj, final, flat: dict[str, Any], init_steps: np.ndarray, t: int) -> dict:
    valid = np.asarray(traj.valid)
    b = valid.shape[1]
    last = np.asarray(P1.last_valid_step(traj.valid))
    done_first = [
        int(np.argmax(np.asarray(traj.done)[:, e]))
        if bool(np.any(np.asarray(traj.done)[:, e]))
        else -1
        for e in range(b)
    ]
    want_valid = rules.N_TRANSITIONS - init_steps  # 各 env 的有效步數
    env_id = flat["env_id"]
    turn = flat["turn"]
    keep = np.asarray(valid).reshape(-1)
    return {
        "T": t,
        "B": b,
        "n_valid": int(valid.sum()),
        "n_padding": int((~valid).sum()),
        "valid_plus_padding": int(valid.size),
        "valid_plus_padding_expected": t * b,
        "per_env_valid": [int(x) for x in valid.sum(axis=0)],
        "per_env_valid_expected": [int(x) for x in want_valid],
        "per_env_valid_ok": bool(np.array_equal(valid.sum(axis=0), want_valid)),
        "final_step": [int(x) for x in np.asarray(final.step)],
        "final_step_all_719": bool(np.all(np.asarray(final.step) == rules.N_TRANSITIONS)),
        "last_valid_t": [int(x) for x in last],
        "done_first_t": done_first,
        "done_equals_last_valid": bool(done_first == [int(x) for x in last]),
        "padding_state_equals_final": PB.padding_matches_final(traj, final),
        "flatten_kept": int(flat["n_rows_kept"]),
        "flatten_kept_equals_sum_valid": int(flat["n_rows_kept"]) == int(valid.sum()),
        "env_id_ok": bool(
            np.array_equal(env_id, np.broadcast_to(np.arange(b), (t, b)).reshape(-1)[keep])
        ),
        "turn_ok": bool(
            np.array_equal(turn, np.broadcast_to(np.arange(t)[:, None], (t, b)).reshape(-1)[keep])
        ),
        "game_step_ok": bool(
            np.array_equal(flat["game_step"], np.asarray(traj.state.step).reshape(-1)[keep])
        ),
    }


def traces_identical(traj_a, traj_b, final_a, final_b) -> dict[str, Any]:
    """warm 與 cold 必須逐位元相同：`traj` 的**全部葉**（含 state 與 decision）
    ＋ `final` 的全部葉。只比幾個欄位等於沒比。"""
    bad: list[str] = []
    la, lb = jax.tree.leaves(traj_a), jax.tree.leaves(traj_b)
    if len(la) != len(lb):
        return {"bitwise_equal": False, "mismatched": ["traj leaf count"], "n_leaves": len(la)}
    for i, (x, y) in enumerate(zip(la, lb, strict=True)):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            bad.append(f"traj_leaf[{i}]")
    fa, fb = jax.tree.leaves(final_a), jax.tree.leaves(final_b)
    for name, x, y in zip(type(final_a)._fields, fa, fb, strict=True):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            bad.append(f"final.{name}")
    return {
        "bitwise_equal": not bad,
        "mismatched": bad,
        "n_traj_leaves": len(la),
        "n_final_leaves": len(fa),
        "traj_leaves_sha256": leaves_sha256(traj_a),
        "final_leaves_sha256": leaves_sha256(final_a),
    }


def recompute_at(params, cfg, traj, t_idx: int, *, seat: int, decoders) -> dict[str, Any]:
    """**保持 B=2 同形狀**在第 `t_idx` 步重算（不縮成 B=1，容差才守得住 1e-5）。"""
    state = jax.tree.map(lambda x: x[t_idx], traj.state)
    decision = {k: v[t_idx] for k, v in traj.decision.items()}
    out = PB.recompute(params, cfg, state, decision, seat=seat, decoders=decoders)
    sampled = {"logp": traj.logp_h[t_idx], "active": traj.active[t_idx]}
    rep = PB.recompute_report(sampled, out)
    rep["t"] = int(t_idx)
    rep["batch_shape"] = list(np.asarray(out["logp"]).shape)
    rep["token_diff_within_1e_5"] = rep["max_abs_token_logp_diff"] <= 1e-5
    return rep


# ------------------------------------------------------------------ 硬檢收集
class Checks:
    """契約檢查收集器：任何一條 False 就是 **FAIL**，`main()` 以非零退出。

    不是「印個 bool 就回 0」——成本量測的價值全靠這些契約，沒守住就不該被當成
    可引用的數字。
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: Any = None) -> bool:
        self.items.append({"name": name, "ok": bool(ok), "detail": detail if not ok else None})
        return bool(ok)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [i for i in self.items if not i["ok"]]

    def report(self) -> dict[str, Any]:
        return {
            "n_checks": len(self.items),
            "n_failed": len(self.failures),
            "verdict": "PASS" if not self.failures else "FAIL",
            "failed": self.failures,
            "names": [i["name"] for i in self.items],
        }


# ------------------------------------------------------------------ reward 帳
def unscaled_cash_margin(money: Any, seat: int) -> np.ndarray:
    """**未乘 scale** 的現金差（int64）：直接從 state 的整數 money 算，不用 scaled 值回除。"""
    m = np.asarray(money, dtype=np.int64)
    own, opp = m[..., seat], m[..., 1 - seat]
    return own - opp


def unscaled_networth_margin(state, final_money, seat: int, *, terminal: bool) -> np.ndarray:
    """未乘 scale 的 Φ − 對手現金。終局只認現金（與 `networth_rewards` 同定義）。"""
    if terminal:
        m = np.asarray(final_money, dtype=np.float64)
        return m[..., seat] - m[..., 1 - seat]
    phi = np.asarray(P1.asset_value(state, seat), dtype=np.float64)
    opp = np.asarray(P1.opponent_money(state, seat), dtype=np.float64)
    return phi - opp


def unscaled_phi_series(traj, final, seat: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """**未乘 scale** 的勢函數序列 `phi_pre[T, B]` 與各 env 的終值 `phi_T[B]`。

    全部從**原 state** 算（`money` 是整數、`asset_value`／`opponent_money` 直接呼叫
    P1 的同一支），不從 scaled reward 回除。終值與 `networth_rewards` 同定義：
    局末只認現金。
    """
    if kind == "dense":
        pre = np.asarray(traj.state.money, dtype=np.float64)
        phi_pre = pre[..., seat] - pre[..., 1 - seat]
    else:
        phi_pre = np.asarray(
            jax.vmap(lambda st: P1.asset_value(st, seat) - P1.opponent_money(st, seat))(traj.state),
            dtype=np.float64,
        )
    fm = np.asarray(final.money, dtype=np.float64)
    return phi_pre, fm[..., seat] - fm[..., 1 - seat]


def unscaled_telescope(traj, final, seat: int, kind: str) -> dict[str, Any]:
    """未縮放的相鄰差 → valid 遮罩 → host float64 求和，對 `phi_T − phi_0`。

    整數且每個點 `|·| < 2^24` 才要求 **恰等**；否則只報 diff。
    """
    phi_pre, phi_T = unscaled_phi_series(traj, final, seat, kind)
    valid = np.asarray(traj.valid)
    last = np.asarray(P1.last_valid_step(traj.valid))
    t = phi_pre.shape[0]
    post = np.concatenate([phi_pre[1:], phi_T[None, :]], axis=0)
    # 各 env 的真局末那一步，post ＝ 終值（不是 scan 的末步）
    is_last = np.arange(t)[:, None] == last[None, :]
    post = np.where(is_last, np.broadcast_to(phi_T[None, :], post.shape), post)
    step_diff = (post - phi_pre) * valid.astype(np.float64)
    total = step_diff.sum(axis=0, dtype=np.float64)
    target = phi_T - phi_pre[0]
    points = np.concatenate([phi_pre.reshape(-1), phi_T, post.reshape(-1)])
    integral = bool(np.all(points == np.round(points)) and np.all(np.abs(points) < 2**24))
    diff = total - target
    return {
        "kind": kind,
        "sum_unscaled_step_diffs": [float(x) for x in total],
        "phi_T_minus_phi_0": [float(x) for x in target],
        "abs_diff": [float(x) for x in np.abs(diff)],
        "all_points_integral_lt_2_24": integral,
        "exact": bool(np.all(diff == 0.0)) if integral else None,
        "note": "未乘 scale，直接由原 state 的勢函數序列算；不由 scaled reward 回除",
    }


def reward_checks(traj, final, *, seat: int, checks: Checks) -> dict[str, Any]:
    """只驗獎勵管線：padding 0、telescoping、GAE 真末步與 padding。

    telescoping 分兩層：

    * **未乘 scale**：從原 state 的整數 money（Φ 版是 `asset_value`／`opponent_money`）
      直接算勢差。整數且 < 2^24 時**恰等**是硬檢，不是報告。
    * **乘 1e-4 之後**：host float64 累加**已存的 float32** reward，對真勢差記
      `max_abs_diff`。**不預設門檻**，只照實記（浮點量化誤差本來就存在）。
    """
    valid = traj.valid
    pad = ~np.asarray(valid)
    scale = P1.REWARD_SCALE
    s0 = jax.tree.map(lambda x: x[0], traj.state)
    out: dict[str, Any] = {}

    for name, r in (
        ("dense", P1.dense_rewards(traj, final, seat, valid)),
        ("networth", P1.networth_rewards(traj, final, seat, valid=valid)),
    ):
        r_np = np.asarray(r)
        total = np.asarray(r_np, dtype=np.float64).sum(axis=0)  # [B]，float64 累加 float32
        if name == "dense":
            phi_0 = unscaled_cash_margin(s0.money, seat)
            phi_T = unscaled_cash_margin(final.money, seat)
        else:
            phi_0 = unscaled_networth_margin(s0, final.money, seat, terminal=False)
            phi_T = unscaled_networth_margin(None, final.money, seat, terminal=True)
        unscaled_diff = np.asarray(phi_T, dtype=np.float64) - np.asarray(phi_0, dtype=np.float64)
        scaled_expected = unscaled_diff * scale
        max_abs_diff = float(np.abs(total - scaled_expected).max())
        integral = bool(
            np.all(unscaled_diff == np.round(unscaled_diff))
            and np.all(np.abs(unscaled_diff) < 2**24)
        )
        # §46 的 scaled 容差：n_valid × 2^-24 × max|累積尺度|。**口徑明列**——只看
        # `|Σr|` 會被正負抵銷假裝尺度很小，所以把逐步累積的最大絕對值與勢函數兩端
        # （乘 scale 後）都納入取 max。
        cum = np.cumsum(np.asarray(r_np, dtype=np.float64), axis=0)
        n_valid_e = np.asarray(valid, dtype=np.int64).sum(axis=0)
        scale_terms = {
            "max_abs_cumulative_reward": np.abs(cum).max(axis=0),
            "abs_sum_rewards": np.abs(np.asarray(r_np, dtype=np.float64).sum(axis=0)),
            "abs_phi_0_scaled": np.abs(np.asarray(phi_0, dtype=np.float64) * scale),
            "abs_phi_T_scaled": np.abs(np.asarray(phi_T, dtype=np.float64) * scale),
        }
        max_scale = np.max(np.stack(list(scale_terms.values()), axis=0), axis=0)
        tol = n_valid_e.astype(np.float64) * (2.0**-24) * max_scale
        entry = {
            "padding_reward_all_zero": bool(np.all(r_np[pad] == 0.0)),
            "rewards_finite": bool(np.all(np.isfinite(r_np))),
            "sum_rewards_float64": [float(x) for x in total],
            "unscaled_phi_0": [float(x) for x in np.asarray(phi_0, dtype=np.float64)],
            "unscaled_phi_T": [float(x) for x in np.asarray(phi_T, dtype=np.float64)],
            "unscaled_potential_diff": [float(x) for x in unscaled_diff],
            "unscaled_potential_is_integral_lt_2_24": integral,
            "scaled_expected": [float(x) for x in scaled_expected],
            "max_abs_diff_after_scale": max_abs_diff,
            "per_env_abs_diff": [float(x) for x in np.abs(total - scaled_expected)],
            "scale": scale,
            "tolerance": {
                "formula": "n_valid × 2^-24 × max(|cumsum r|max, |Σr|, |φ0·scale|, |φT·scale|)",
                "n_valid_per_env": [int(x) for x in n_valid_e],
                "terms": {k: [float(x) for x in v] for k, v in scale_terms.items()},
                "max_scale_per_env": [float(x) for x in max_scale],
                "tol_per_env": [float(x) for x in tol],
            },
        }
        within = bool(np.all(np.abs(total - scaled_expected) <= tol))
        entry["scaled_within_quantization_tolerance"] = within
        checks.add(f"reward/{name}/scaled_within_tolerance", within, entry)
        checks.add(f"reward/{name}/padding_zero", entry["padding_reward_all_zero"], entry)
        checks.add(f"reward/{name}/finite", entry["rewards_finite"], None)
        # 未縮放那一層**自己算一遍序列**（不由 scaled reward 回除）：整數且每點 < 2^24
        # 就要求恰等，這才是「原 phi 恰等」。
        tele = unscaled_telescope(traj, final, seat, name)
        entry["unscaled_telescope"] = tele
        if tele["all_points_integral_lt_2_24"]:
            checks.add(f"reward/{name}/unscaled_telescope_exact", bool(tele["exact"]), tele)

        # `last_value=0` 只有在每個 env 的最後有效步真的 done 時才對（否則等於把
        # 未完成的局當終局）。這裡**明檢**，不靠「配方大概會跑到局末」。
        last_idx = np.asarray(P1.last_valid_step(valid))
        done_np = np.asarray(traj.done)
        last_done = bool(np.all(done_np[(last_idx, np.arange(len(last_idx)))]))
        checks.add(
            f"gae/{name}/last_valid_step_is_done",
            last_done,
            {
                "last_valid_t": [int(x) for x in last_idx],
                "done_at_last_valid": [bool(done_np[last_idx[e], e]) for e in range(len(last_idx))],
            },
        )
        adv, ret = P1.gae(
            traj.value, jnp.zeros((traj.value.shape[1],), jnp.float32), r, valid, traj.done
        )
        adv_np, ret_np = np.asarray(adv), np.asarray(ret)
        last = np.asarray(P1.last_valid_step(valid))
        v_np = np.asarray(traj.value)
        idx = (last, np.arange(len(last)))
        want_adv = r_np[idx] - v_np[idx]
        got_adv, got_ret = adv_np[idx], ret_np[idx]
        gae_entry = {
            "last_valid_adv": [float(x) for x in got_adv],
            "last_valid_adv_expected_r_minus_v": [float(x) for x in want_adv],
            "last_valid_adv_max_abs_diff": float(np.abs(got_adv - want_adv).max()),
            "last_valid_ret": [float(x) for x in got_ret],
            "last_valid_ret_expected_r": [float(x) for x in r_np[idx]],
            "last_valid_ret_max_abs_diff": float(np.abs(got_ret - r_np[idx]).max()),
            "padding_adv_all_zero": bool(np.all(adv_np[pad] == 0.0)),
            "padding_ret_all_zero": bool(np.all(ret_np[pad] == 0.0)),
            "adv_ret_finite": bool(np.all(np.isfinite(adv_np)) and np.all(np.isfinite(ret_np))),
        }
        checks.add(
            f"gae/{name}/last_valid_adv_equals_r_minus_v",
            gae_entry["last_valid_adv_max_abs_diff"] <= 1e-5,
            gae_entry,
        )
        checks.add(
            f"gae/{name}/last_valid_ret_equals_r",
            gae_entry["last_valid_ret_max_abs_diff"] <= 1e-5,
            gae_entry,
        )
        checks.add(f"gae/{name}/padding_adv_zero", gae_entry["padding_adv_all_zero"], None)
        checks.add(f"gae/{name}/padding_ret_zero", gae_entry["padding_ret_all_zero"], None)
        checks.add(f"gae/{name}/finite", gae_entry["adv_ret_finite"], None)
        entry["gae"] = gae_entry
        out[name] = entry
    return out


# ------------------------------------------------------------------ 落盤
class Sink:
    """分段即落盤：每呼叫一次就把目前的 result 整份寫出去（超帽也留得住前段）。"""

    def __init__(self, path: Path, head: dict[str, Any]) -> None:
        self.path = path
        self.doc: dict[str, Any] = {**head, "stages": {}}
        path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()

    def stage(self, name: str, payload: dict[str, Any]) -> None:
        self.doc["stages"][name] = {**payload, "peak_rss_gib": rss_gib(), "device": device_info()}
        self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.doc, ensure_ascii=False, indent=1))


def save_trace(traj, final, path: Path, ident: dict[str, Any]) -> dict[str, Any]:
    """原始 trace 存 NPZ（給核收與後續 reuse）。**先估大小再寫**，過帽即拒。"""
    arrays: dict[str, np.ndarray] = {}
    for name, leaf in zip(
        [f"state/{f}" for f in type(traj.state)._fields], jax.tree.leaves(traj.state), strict=True
    ):
        arrays[name] = np.asarray(leaf)
    for name, leaf in zip(
        [f"final/{f}" for f in type(final)._fields], jax.tree.leaves(final), strict=True
    ):
        arrays[name] = np.asarray(leaf)
    for k, v in traj.decision.items():
        arrays[f"decision/{k}"] = np.asarray(v)
    for k in ("logp", "logp_h", "active", "value", "done", "seat", "valid", "env_id", "step"):
        arrays[k] = np.asarray(getattr(traj, k))
    est = sum(a.nbytes for a in arrays.values())
    info = {"estimated_bytes": int(est), "estimated_gib": round(est / 1073741824, 3), "path": None}
    if est > ident["trace_max_bytes"]:
        info["written"] = False
        info["reason"] = f"估算 {info['estimated_gib']} GiB 超過 --trace-max-gib"
        return info
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **arrays,
        identity=np.frombuffer(json.dumps(ident, ensure_ascii=False).encode(), dtype=np.uint8),
    )
    info["written"] = True
    info["path"] = str(path)
    info["file_bytes"] = path.stat().st_size
    info["file_sha256"] = sha256_file(path)
    return info


# ------------------------------------------------- decoder 參照原件（審查者 §46）
#
# 「改 decoder 前先存一份、改後同形狀比」——沒有這份原件，entropy 那個改動就只能
# 用「新 sample 對新 teacher 互比」自證，而那只證明兩邊一致、不證明語義沒變。
REF_ROWS = (150, 729)
REF_SCHEMA = "full-actions-decoder-reference-v1"


def _np_state_from(state, i: int):
    """JAX state 的第 `i` 列 → NumPy 路徑的 `NpState`（canonical，自己＝索引 0）。"""
    from rl import loop4_numpy as LNP  # noqa: PLC0415

    return LNP.NpState(
        **{f: np.asarray(getattr(state, f))[i] for f in LNP.NpState.__slots__ if hasattr(state, f)}
    )


#: `swap_seats` 只反轉這些欄位的 axis 1（座位軸）；`key`、`market_inv`、`shop_seq`、`overflow` 等原樣。
SEAT_FIELDS = (
    "money",
    "tiles",
    "unit_x",
    "unit_y",
    "hires_today",
    "n_quads",
    "shed",
    "seeds",
    "inv",
    "inv_seq",
    "inv_ctr",
)


def swap_seats(state):
    """把 :data:`SEAT_FIELDS` 的 axis 1（座位軸）反轉，其餘欄位原樣。

    對合：``swap_seats(swap_seats(s))`` 逐欄 bitwise 等於 ``s``。
    """
    return state._replace(**{f: getattr(state, f)[:, ::-1] for f in SEAT_FIELDS})


def reference_payload(params, np_params, cfg, *, seed: int) -> dict[str, Any]:
    """真 rows 150／729 × 兩座 × sample／greedy／teacher 的 decision／logp／active。

    * teacher 的 labels **來自同一組 sample 的 decision**（不是另外造標）。
    * NumPy 是**獨立第二參照**：seat 1 先 `swap_seats` 成 canonical 再 evaluate。
    """
    from rl import full_actions_decode as D  # noqa: PLC0415

    rows = []
    idents = []
    for row in REF_ROWS:
        st, ident = teacher_row(row)
        rows.append(st)
        idents.append(ident)
    state = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *rows)
    swapped = swap_seats(state)
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"rows": list(REF_ROWS), "fixtures": idents, "entries": []}

    for seat in (0, 1):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), seat)
        smp = D.decode("sample")(params, cfg, state, seat, CAPS, key=key)
        grd = D.decode("greedy")(params, cfg, state, seat, CAPS)
        tch = D.decode("teacher")(
            params, cfg, state, seat, CAPS, labels={k: smp["decision"][k] for k in PB.LABEL_FIELDS}
        )
        for mode, out in (("sample", smp), ("greedy", grd), ("teacher", tch)):
            pre = f"{mode}/seat{seat}"
            arrays[f"{pre}/logp"] = np.asarray(out["logp"])
            arrays[f"{pre}/active"] = np.asarray(out["active"])
            arrays[f"{pre}/value"] = np.asarray(out["value"])
            for k, v in out["decision"].items():
                arrays[f"{pre}/decision/{k}"] = np.asarray(v)
            meta["entries"].append(pre)
        # NumPy 第二參照：對 sample 抽到的 decision 重算（seat 1 走 swapped canonical）
        src = state if seat == 0 else swapped
        for i, row in enumerate(REF_ROWS):
            dec = {k: np.asarray(v)[i] for k, v in smp["decision"].items() if k != "market_length"}
            ev = FN.evaluate(np_params, cfg, _np_state_from(src, i), CAPS, dec)
            arrays[f"numpy/seat{seat}/row{row}/logp"] = np.asarray(ev["logp"])
            arrays[f"numpy/seat{seat}/row{row}/active"] = np.asarray(ev["active"])
            meta["entries"].append(f"numpy/seat{seat}/row{row}")
    meta["state_leaves_sha256"] = leaves_sha256(state)
    return {"arrays": arrays, "meta": meta}


def write_reference(path: Path, payload: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": REF_SCHEMA, **head, "reference": payload["meta"]}
    np.savez(
        path,
        **payload["arrays"],
        identity=np.frombuffer(json.dumps(doc, ensure_ascii=False).encode(), dtype=np.uint8),
    )
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "n_arrays": len(payload["arrays"]),
        "entries": payload["meta"]["entries"],
    }


def compare_reference(old_npz: Path, payload: dict[str, Any], checks: Checks) -> dict[str, Any]:
    """改 decoder 後：同形狀對原件 **decision／active 逐位元**、**logp ≤ 1e-5**。"""
    old = np.load(old_npz, allow_pickle=False)
    new = payload["arrays"]
    missing = sorted(set(k for k in old.files if k != "identity") - set(new))
    extra = sorted(set(new) - set(old.files))
    report: dict[str, Any] = {
        "old_npz": str(old_npz),
        "old_sha256": sha256_file(old_npz),
        "missing_in_new": missing,
        "extra_in_new": extra,
        "worst_logp_diff": 0.0,
        "per_key": {},
    }
    checks.add("reference/key_set_matches", not missing and not extra, report)
    worst = 0.0
    for k in sorted(set(new) & set(old.files)):
        a, b_ = np.asarray(old[k]), np.asarray(new[k])
        if a.shape != b_.shape:
            checks.add(f"reference/{k}/shape", False, {"old": a.shape, "new": b_.shape})
            continue
        if "/logp" in k or k.endswith("/value"):
            d = float(np.max(np.abs(a.astype(np.float64) - b_.astype(np.float64))))
            worst = max(worst, d)
            report["per_key"][k] = {"max_abs_diff": d}
            checks.add(f"reference/{k}/logp_le_1e-5", d <= 1e-5, {"max_abs_diff": d})
        else:  # decision／active：逐位元
            same = bool(np.array_equal(a, b_))
            report["per_key"][k] = {"bitwise_equal": same}
            checks.add(f"reference/{k}/bitwise", same, None)
    report["worst_logp_diff"] = worst
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="IL checkpoint（npz）")
    ap.add_argument("--expect-ckpt-sha", required=True)
    ap.add_argument("--row", type=int, default=150, help="真教師列（保留原 step）")
    ap.add_argument("--seed", type=int, default=0, help="fresh env 的 seed，也是 PRNG key")
    ap.add_argument("--t", type=int, default=rules.N_TRANSITIONS)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--steady-n", type=int, default=10)
    ap.add_argument(
        "--n-fresh",
        type=int,
        default=1,
        help="fresh env 數（B = n_fresh + 1 的真教師列）；1 ＝ 既有 B=2 基準",
    )
    ap.add_argument("--trace-max-gib", type=float, default=8.0)
    ap.add_argument(
        "--mode",
        choices=("cost", "reference", "compare"),
        default="cost",
        help="cost＝成本量測；reference＝存 decoder 參照原件；compare＝對既有原件比對",
    )
    ap.add_argument("--reference-npz", help="mode=compare 時要比對的舊原件")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    trace_path = out_path.parent / "trace_cold.npz"
    ref_path = out_path.parent / "decoder_reference.npz"
    must_be_new = [out_path, out_path.parent]
    must_be_new.append(trace_path if args.mode == "cost" else ref_path)
    for pth in must_be_new:
        if pth.exists():
            raise SystemExit(f"已存在，拒絕覆蓋既有原件：{pth}")

    ckpt = Path(args.ckpt)
    ckpt_sha = sha256_file(ckpt)
    if ckpt_sha != args.expect_ckpt_sha:
        raise SystemExit(f"ckpt SHA 不符：期望 {args.expect_ckpt_sha}，實算 {ckpt_sha}")
    np_params, meta = FN.load_params(ckpt, caps=CAPS)
    params = {k: jnp.asarray(v) for k, v in np_params.items()}
    cfg = PB.config_for(CAPS)
    # checkpoint 的 cfg 不是 `meta["config"]`（沒有這個鍵）——正式來源是
    # `config_from_meta`（`n_heads` 只寫在 metadata，權重 shape 看不出來）。
    ckpt_cfg = FN.config_from_meta(meta)
    if cfg != ckpt_cfg:
        raise SystemExit(f"cfg 與 checkpoint metadata 不符：{cfg} vs {ckpt_cfg}")

    if args.mode == "cost":
        state, fix_ident = init_batch(args.row, args.seed, args.n_fresh)
        init_steps = np.asarray(state.step)
        b = int(state.step.shape[0])
        run_extra = {
            "B": b,
            "T": args.t,
            "n_fresh": args.n_fresh,
            "init_steps": [int(x) for x in init_steps],
            "init_state_leaves_sha256": leaves_sha256(state),
            "trace_npz_estimate_note": "實測 B=2／T=719 的 NPZ ＝ 30,157,090 B ⇒ "
            "每 (T, B) 槽約 20.97 KB；B=16 約 241 MB（實際大小仍以 stage trace_npz 為準）",
        }
    else:  # 參照模式只吃真 rows，不建 fresh env、不跑 scan
        state, init_steps, b = None, None, len(REF_ROWS)
        fix_ident = {"mode": args.mode, "rows": list(REF_ROWS)}
        run_extra = {"B": b, "T": None, "init_steps": None}
    head = {
        "schema": "full-actions-ppo-cost-v1",
        "source": {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "files": {f: sha256_file(ROOT / f) for f in SOURCE_FILES},
        },
        "weights": {"path": str(ckpt.resolve()), "sha256": ckpt_sha, "config": ckpt_cfg},
        "caps": {
            "name": CAPS.name,
            "max_units": CAPS.max_units,
            "max_orders": CAPS.max_orders,
            "market_steps": CAPS.market_steps,
            "max_order_qty": CAPS.max_order_qty,
        },
        "fixture": fix_ident,
        "run": {
            **run_extra,
            "mode": args.mode,
            "seat": args.seat,
            "key_seed": args.seed,
            "steady_n": args.steady_n,
            "note": (
                "兩次完整 trace ＝ cold／warm 成本量測，不是兩個新樣本；"
                "固定 key 的重複是微基準，不是訓練或獨立對局"
                if args.mode == "cost"
                else "參照模式：只存 decoder 原件，不跑 scan、不量成本；"
                "verdict 在此模式沒有硬檢（compare 模式才有）"
            ),
        },
        "device": device_info(),
        "matmul_precision": require_gpu_highest(),
    }
    sink = Sink(out_path, head)
    checks = Checks()

    if args.mode in ("reference", "compare"):
        # 參照原件只吃真 rows（不含 fresh env），所以 head 的 fixture／init 欄位
        # 與 cost 模式不同；身分一樣全記。
        payload = reference_payload(params, np_params, cfg, seed=args.seed)
        if args.mode == "reference":
            sink.stage("decoder_reference", write_reference(ref_path, payload, head))
        else:
            if not args.reference_npz:
                raise SystemExit("mode=compare 要給 --reference-npz")
            sink.stage(
                "decoder_reference_compare",
                compare_reference(Path(args.reference_npz), payload, checks),
            )
        sink.doc["checks"] = checks.report()
        sink.doc["verdict"] = sink.doc["checks"]["verdict"]
        sink.doc["peak_rss_gib"] = rss_gib()
        sink.flush()
        print(json.dumps({k: v for k, v in sink.doc.items() if k != "stages"}, ensure_ascii=False))
        return 0 if not checks.failures else 1

    decoders = PB.make_decoders(cfg, CAPS)
    key = jax.random.PRNGKey(args.seed)
    k0, k1 = jax.random.split(key, 2)

    # --- stage 1：兩座 decode 的 compile+first 與 steady（**每座量完就落盤**）
    dec_stage: dict[str, Any] = {}
    outs = {}
    for seat in (0, 1):
        fn = decoders["sample"][seat]
        kk = k0 if seat == 0 else k1
        got, first_s = timed(fn, params, state, key=kk)
        outs[seat] = got
        dec_stage[f"seat{seat}"] = {
            "compile_plus_first_s": round(first_s, 5),
            "steady": steady(fn, params, state, key=kk, n=args.steady_n),
            "jit_cache_size_after": cache_size(fn),
        }
        sink.stage("decode", dict(dec_stage))

    # --- stage 2：step_once 獨立 jit（含 joint_action／merge），輸入用真 decision
    step_jit = jax.jit(lambda st, d0, d1: PB.step_once(st, d0, d1, CAPS))
    d0, d1 = outs[0]["decision"], outs[1]["decision"]
    _, step_first = timed(step_jit, state, d0, d1)
    sink.stage(
        "step_once",
        {
            "compile_plus_first_s": round(step_first, 5),
            "steady": steady(step_jit, state, d0, d1, n=args.steady_n),
            "jit_cache_size_after": cache_size(step_jit),
        },
    )

    # --- stage 3：完整 T 步 scan。**同一支 jit 呼叫兩次**（cold ＝ 含編譯，
    # warm ＝ 已編譯），不是每次重建 closure 再冒稱 warm。
    rollout_jit = jax.jit(
        lambda p_, st_, k_: PB.rollout(
            p_, cfg, st_, k_, seat=args.seat, t=args.t, decoders=decoders, caps=CAPS
        )
    )
    (final_c, traj_c), cold_s = timed(rollout_jit, params, state, key)
    n_valid = int(np.asarray(traj_c.valid).sum())
    cache_after_cold = cache_size(rollout_jit)
    sink.stage(
        "rollout_cold",
        {
            "compile_plus_first_s": round(cold_s, 5),
            "T": args.t,
            "B": b,
            "raw_scan_slots": args.t * b,
            "raw_scan_slots_per_s": round(args.t * b / cold_s, 3),
            "valid_rows": n_valid,
            "valid_rows_per_s": round(n_valid / cold_s, 3),
            "jit_cache_size_after": cache_after_cold,
        },
    )
    (final_w, traj_w), warm_s = timed(rollout_jit, params, state, key)
    cache_after_warm = cache_size(rollout_jit)
    same = traces_identical(traj_c, traj_w, final_c, final_w)
    no_retrace = cache_after_cold is None or cache_after_cold == cache_after_warm
    sink.stage(
        "rollout_warm",
        {
            "wall_s": round(warm_s, 5),
            "raw_scan_slots_per_s": round(args.t * b / warm_s, 3),
            "valid_rows": n_valid,
            "valid_rows_per_s": round(n_valid / warm_s, 3),
            "jit_cache_size_after": cache_after_warm,
            "no_retrace_between_cold_and_warm": no_retrace,
            "identical_to_cold": same,
            "note": "warm 是同一支 jit 的第二次呼叫，只量成本、不算新樣本；"
            "raw_scan_slots 含 padding，有效吞吐看 valid_rows_per_s",
        },
    )
    checks.add("rollout/warm_bitwise_equal_to_cold", same["bitwise_equal"], same)
    checks.add(
        "rollout/no_retrace_between_cold_and_warm",
        no_retrace,
        {"cold": cache_after_cold, "warm": cache_after_warm},
    )

    # --- stage 4：trace 契約
    flat = PB.flatten_bridge(traj_c)
    contract = trace_contract(traj_c, final_c, flat, init_steps, args.t)
    for name in (
        "per_env_valid_ok",
        "final_step_all_719",
        "done_equals_last_valid",
        "padding_state_equals_final",
        "flatten_kept_equals_sum_valid",
        "env_id_ok",
        "turn_ok",
        "game_step_ok",
    ):
        checks.add(f"trace/{name}", bool(contract[name]), contract)
    checks.add(
        "trace/valid_plus_padding",
        contract["valid_plus_padding"] == contract["valid_plus_padding_expected"],
        contract,
    )
    sink.stage(
        "trace_contract",
        {**contract, "summary": PB.trace_summary(traj_c, final_c, seat=args.seat)},
    )

    # --- stage 5：原權重 teacher 重算（同 B 同形狀）
    last = np.asarray(P1.last_valid_step(traj_c.valid))
    # fresh 與教師兩列在**同一個 B=2 batch** 裡，所以 t=0 這一次同時覆蓋兩種開始；
    # 另外取兩個 env 各自最後有效行所在的 t（相同就只算一次）。
    t_points: dict[str, int] = {"all_starts_t0": 0}
    for name, t_i in (("last_valid_min", int(last.min())), ("last_valid_max", int(last.max()))):
        if t_i not in t_points.values():
            t_points[name] = t_i
    rec: dict[str, Any] = {}
    for name, t_i in sorted(t_points.items(), key=lambda kv: kv[1]):
        r = recompute_at(params, cfg, traj_c, t_i, seat=args.seat, decoders=decoders)
        rec[name] = r
        checks.add(f"recompute/{name}/active_bitwise_equal", r["active_bitwise_equal"], r)
        checks.add(f"recompute/{name}/teacher_drop_zero", r["teacher_drop_total"] == 0, r)
        checks.add(f"recompute/{name}/inactive_logp_zero", r["inactive_logp_all_zero"], r)
        checks.add(f"recompute/{name}/token_diff_le_1e-5", r["token_diff_within_1e_5"], r)
        checks.add(f"recompute/{name}/batch_shape_kept", r["batch_shape"][0] == b, r)
    sink.stage(
        "recompute",
        {
            "t_points": t_points,
            "results": rec,
            "tolerance": {"token": 1e-5, "note": "同 B 同形狀；1e-4 只適用異形狀比較"},
        },
    )

    # --- stage 6：reward／GAE 帳
    sink.stage("rewards", reward_checks(traj_c, final_c, seat=args.seat, checks=checks))

    # --- stage 7：原始 trace 落盤
    sink.stage(
        "trace_npz",
        save_trace(
            traj_c,
            final_c,
            trace_path,
            {**head, "trace_max_bytes": int(args.trace_max_gib * 1073741824)},
        ),
    )

    sink.doc["checks"] = checks.report()
    sink.doc["verdict"] = sink.doc["checks"]["verdict"]
    sink.doc["peak_rss_gib"] = rss_gib()
    sink.doc["device_final"] = device_info()
    sink.flush()
    print(json.dumps({k: v for k, v in sink.doc.items() if k != "stages"}, ensure_ascii=False))
    return 0 if not checks.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
