"""T3 PPO 一次性診斷 probe（A／seed1 iter0，審查者 beta02-stop 審 §3 第一順位；opt-in）。

只由 `scripts/full_actions_ppo_train.py --diag-probe SPEC.json` 啟用；預設路徑完全不經過本檔。

- **事件重現核對**（更新前）：rollout 身分（init seed／sample key／init state SHA／對手 tape 抽樣／
  behaviour params SHA）與目標列的 seat／env／turn／decision／active／原 rollout logp／adv 須和
  A 原件**已留存的有限欄位**（fingerprint）一致；不一致或非有限就落盤並丟 `DiagNotReproduced`
  （不換 seed、不重試）。A 原件沒有留存完整 prefix／state／合法 mask，所以「完整事件同一」
  **無法獨立驗證**，欄位 `full_event_identity.verified=false` 明示此限制（審查者 9ee140d 審 B1）。
- **目標列原件落盤**（B2）：state 各葉的該列、完整 decision、分母／active／adv、規則與 decoder
  來源 SHA 寫成小 NPZ＋SHA，供 GPU 後唯讀判讀引擎效果；對手當回合動作不在 rows 中，明列缺失。
- **合法域核對**（B1）：每個 probe 點核 θ／IL 合法 mask 逐元素相等（整批）、rollout 所選 token
  在 θ mask 內合法（整批 active 槽）、teacher_drop＝0、active 與 rollout 相同、logp／KL 有限；
  任一不符就落盤並丟 `DiagAnomaly` 停止。
- **無梯度 probe**：在指定更新序號**之前**與迴圈結束後，對原 minibatch（原形狀、原分母
  `logp_h`）做 teacher 前向（dist＝True），記目標列指定槽與同訂單其他槽的 logp／ratio／
  IL 側 logp／逐槽 exact KL(IL‖θ)／合法 support 大小／active，以及整批各 head 組的 ratio 極值。
- 不產生梯度、不改 params／opt state，也不改正式 loss／decoder／帳本。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rl import full_actions_decode as D
from rl import full_actions_ppo as PB
from rl import full_actions_updater as U

F32 = jnp.float32
SPEC_KEYS = (
    "iteration",
    "batch",
    "row",
    "slots",
    "every",
    "out",
    "expect_row",
    "expect_rollout",
)


class DiagNotReproduced(RuntimeError):
    """目標事件在重建的 rollout 中未重現：診斷如實結束。"""


class DiagAnomaly(RuntimeError):
    """probe 發現合法域／active／有限性異常：落盤後停止，不據以判讀機制。"""


#: 引擎效果判讀需要的來源檔（落盤其 SHA）
SOURCE_FILES = (
    "rl/rules.py",
    "rl/full_actions_decode.py",
    "rl/full_actions_ppo.py",
    "rl/caps.py",
)


def make_probe(cfg_model: dict[str, Any], *, seat: int, caps=PB.CAPS):
    """單批無梯度 teacher 前向（θ 與 IL 各一次），回逐槽量；與 `make_post_check` 同一前向。"""
    teacher = D.decode("teacher", dist=True)

    @jax.jit
    def probe(params, ref, rows):
        labels = {k: rows["decision"][k] for k in PB.LABEL_FIELDS}
        out = teacher(params, cfg_model, rows["state"], seat, caps, labels=labels)
        out_ref = teacher(ref, cfg_model, rows["state"], seat, caps, labels=labels)
        act = rows["active"]
        kl = U.anchor_kl(out, out_ref, act)
        return {
            "logp": out["logp"],
            "logp_ref": out_ref["logp"],
            "active_new": out["active"],
            "support": jnp.sum(out["dist"]["mask"], axis=-1),
            "support_ref": jnp.sum(out_ref["dist"]["mask"], axis=-1),
            "mask": out["dist"]["mask"],
            "mask_equal": jnp.all(out["dist"]["mask"] == out_ref["dist"]["mask"]),
            "teacher_drop": sum(
                jnp.sum(v)
                for o in (out, out_ref)
                for g in ("unit", "market")
                for v in o["teacher_drop"][g].values()
            ),
            "kl_slot": kl["slot"],
            "kl_row_sum": kl["row_sum"],
        }

    return probe


def _norm(x):
    return json.loads(
        json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    )


def _f(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


class Diag:
    """診斷狀態；`spec` 由 JSON 讀入（見 `SPEC_KEYS`）。"""

    def __init__(self, spec: dict[str, Any], cfg_model: dict[str, Any]):
        missing = [k for k in SPEC_KEYS if k not in spec]
        if missing:
            raise ValueError(f"diag spec 缺欄位 {missing}")
        self.spec = spec
        self.cfg_model = cfg_model
        self.out = Path(spec["out"])
        if self.out.exists():
            raise SystemExit(f"diag 輸出已存在，拒絕覆蓋：{self.out}")
        self.doc: dict[str, Any] = {
            "schema": "t3-diag-probe-v2",
            "spec": spec,
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            "index_base": (
                "更新序號 i 為 0 基準（plan 順序）；probe『before i』＝第 i 次更新之前的 params"
            ),
            "event": None,
            "probes": [],
            "final": None,
        }
        self._probe = None
        self.seat = None
        self.rows = None
        self.points = set()

    # ------------------------------------------------------------ 事件核對
    def check_event(self, roll: dict, plan: list, rows_list: list) -> None:
        sp = self.spec
        b, r = int(sp["batch"]), int(sp["row"])
        exp_r, exp_ro = sp["expect_row"], sp["expect_rollout"]
        seat = int(plan[b]["seat"])
        rows = rows_list[b]
        mism: dict[str, Any] = {}
        # rollout 身分
        for s_key, want in exp_ro.items():
            got = roll.get(s_key) or {}
            for k, v in want.items():
                # 兩邊都走 JSON 正規化（numpy 標量／tuple → list），再逐值比較
                g = _norm(got.get(k))
                if g != _norm(v):
                    mism[f"rollout.{s_key}.{k}"] = {"got": str(g)[:200], "want": str(v)[:200]}

        # 目標列
        def host(k):
            return np.asarray(jax.device_get(rows[k]))

        slot0 = int(sp["slots"][0])
        got_row = {
            "seat": seat,
            "env_id": int(host("env_id")[r]),
            "turn": int(host("turn")[r]),
            "active": bool(host("active")[r, slot0]),
        }
        packed = np.asarray(jax.device_get(U._packed_decision(rows)))
        got_row["decision_token"] = int(packed[r, slot0])
        for k in ("seat", "env_id", "turn", "decision_token", "active"):
            if got_row[k] != exp_r[k]:
                mism[f"row.{k}"] = {"got": got_row[k], "want": exp_r[k]}
        lp_old = float(host("logp_h")[r, slot0])
        adv = float(host("adv")[r])
        finite = all(np.isfinite([lp_old, adv, exp_r["logp_old"], exp_r["advantage"]]))
        dev_lp = abs(float(np.expm1(lp_old - exp_r["logp_old"]))) if finite else float("nan")
        dev_adv = abs(adv - exp_r["advantage"]) if finite else float("nan")
        if not finite:
            mism["row.non_finite"] = {"logp_old": lp_old, "advantage": adv}
        got_row.update(
            {
                "logp_old": lp_old,
                "advantage": adv,
                "logp_old_ratio_abs_expm1_vs_A": dev_lp,
                "advantage_abs_diff_vs_A": dev_adv,
            }
        )
        # 以「不在界內」判（NaN 也算不過），不 fail-open
        if not (dev_lp <= U.RATIO_BUDGET):
            mism["row.logp_old"] = {
                "got": lp_old,
                "want": exp_r["logp_old"],
                "budget": U.RATIO_BUDGET,
            }
        if not (dev_adv <= 1e-3 * max(1.0, abs(exp_r["advantage"]))):
            mism["row.advantage"] = {"got": adv, "want": exp_r["advantage"]}
        # 同一動作（market 訂單 6 槽／unit 7 槽）的 token 與整數數量（big-endian U31）
        order = self._order_slots(slot0)
        heads = [U.slot_describe(s)["head"] for s in order]
        toks = [int(packed[r, s]) for s in order]
        qb = [t for t, h in zip(toks, heads, strict=True) if h.startswith("qty_")]
        d0 = U.slot_describe(slot0)
        oi = d0["index"]
        dec = {k: np.asarray(jax.device_get(v)) for k, v in rows["decision"].items()}
        extra = (
            {
                "market_op": int(dec["market_op"][r, oi]),
                "market_item": int(dec["market_item"][r, oi]),
                "qty_bytes_field": [int(x) for x in dec["market_qty_bytes"][r, oi]],
            }
            if d0["phase"] == "market"
            else {
                "unit_verb": int(dec["unit_verb"][r, oi]),
                "unit_item": int(dec["unit_item"][r, oi]),
                "qty_bytes_field": [int(x) for x in dec["unit_qty_bytes"][r, oi]],
            }
        )
        got_row["order"] = {
            "phase": d0["phase"],
            "index": oi,
            "slots": order,
            "heads": heads,
            "tokens": toks,
            "active": [bool(host("active")[r, s]) for s in order],
            "logp_old": [float(host("logp_h")[r, s]) for s in order],
            "qty_bytes": qb,
            "qty_u31_big_endian": int(sum(int(x) << (8 * (3 - k)) for k, x in enumerate(qb))),
            **extra,
        }
        # 本次固定 decision／prefix（整列）與目標兩槽 active：之後每點比對用
        self.fixed_tokens = packed[r].copy()
        got_row["fixed_decision_row_tokens_sha256"] = hashlib.sha256(
            packed[r].astype(np.int64).tobytes()
        ).hexdigest()
        got_row["target_slots_active"] = {
            str(int(s)): bool(host("active")[r, int(s)]) for s in sp["slots"]
        }
        got_row["row_npz"] = self._save_row(rows, r, seat, packed)
        self.doc["event"] = {
            "batch": b,
            "row": r,
            "seat": seat,
            "got": got_row,
            "mismatch": mism,
            "a_fingerprint_matched": not mism,
            "a_fingerprint_fields": sorted(
                [f"rollout.{s_}.{k}" for s_, w in exp_ro.items() for k in w]
                + [f"row.{k}" for k in exp_r]
            ),
            "full_event_identity": {
                "verified": False,
                "reason": (
                    "A 原件未留存 turn48 的完整 prefix／state／合法 mask，無法獨立比對；"
                    "本次值是新生成的，不當作 A 的舊證據"
                ),
            },
        }
        self.flush()
        if mism:
            raise DiagNotReproduced(f"未重現該事件：{sorted(mism)}")
        self.seat, self.rows = seat, rows
        self._probe = make_probe(self.cfg_model, seat=seat)
        n = len(plan)
        ev = int(sp["every"])
        self.points = set(range(0, n, ev)) | {b, b + 1}

    def _save_row(self, rows, r: int, seat: int, packed: np.ndarray) -> dict[str, Any]:
        """目標列原件（B2）：state 各葉該列、完整 decision、分母／active／adv 等，小 NPZ＋SHA。"""
        arrs: dict[str, np.ndarray] = {}
        leaves, _ = jax.tree_util.tree_flatten_with_path(rows["state"])
        for path, leaf in leaves:
            arrs["state" + jax.tree_util.keystr(path)] = np.asarray(jax.device_get(leaf))[r]
        for k, v in rows["decision"].items():
            arrs[f"decision.{k}"] = np.asarray(jax.device_get(v))[r]
        for k in ("logp_h", "active", "adv", "ret", "value", "seat", "env_id", "turn", "game_step"):
            if k in rows:
                arrs[f"row.{k}"] = np.asarray(jax.device_get(rows[k]))[r]
        arrs["row.packed_decision_tokens"] = packed[r]
        path = self.out.with_name(self.out.stem + "_event_row.npz")
        if path.exists():
            raise SystemExit(f"diag 目標列 NPZ 已存在，拒絕覆蓋：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrs)
        root = Path(__file__).resolve().parents[1]
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "keys": sorted(arrs),
            "seat": seat,
            "caps": {
                k: int(v) if isinstance(v, int | np.integer) else str(v)
                for k, v in PB.CAPS._asdict().items()
            },
            "source_sha256": {
                f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in SOURCE_FILES
            },
            "missing": [
                "對手當回合（turn48）的動作不在 rows 中，未取得：交錯結算下的實際成交無法核",
                "官方引擎實際成交量未記錄：只能判當前觀測下的合法性／可能效果",
            ],
        }

    @staticmethod
    def _order_slots(slot: int) -> list[int]:
        d = U.slot_describe(slot)
        if d["phase"] == "unit":
            return list(range(d["index"] * 7, d["index"] * 7 + 7))
        n_u = PB.CAPS.max_units * 7
        base = n_u + d["index"] * 6
        return list(range(base, base + 6))

    # ------------------------------------------------------------ probe
    def maybe_probe(self, i: int, params, il_params, hist: list[dict]) -> None:
        if self._probe is None or i not in self.points:
            return
        self._record(f"before_step{i}", i, params, il_params, hist)

    def final_probe(self, n_steps: int, params, il_params, hist: list[dict]) -> None:
        if self._probe is not None:
            self._record(f"after_all_step{n_steps}", n_steps, params, il_params, hist)

    def _record(self, label, i, params, il_params, hist) -> None:
        sp = self.spec
        r = int(sp["row"])
        o = jax.device_get(self._probe(params, il_params, self.rows))
        lp_h = np.asarray(jax.device_get(self.rows["logp_h"]))
        act = np.asarray(jax.device_get(self.rows["active"]))
        lp = np.asarray(o["logp"])
        mask = np.asarray(o["mask"])
        act_now = np.asarray(o["active_new"])
        packed = np.asarray(jax.device_get(U._packed_decision(self.rows)))
        bi, si = np.nonzero(act)
        tok = packed[bi, si]
        in_range = (tok >= 0) & (tok < mask.shape[-1])
        legal = np.zeros_like(in_range)
        legal[in_range] = mask[bi[in_range], si[in_range], tok[in_range]]
        anomalies = {}
        if not bool(o["mask_equal"]):
            anomalies["theta_il_mask_not_equal"] = True
        if not legal.all():
            anomalies["selected_token_illegal"] = int((~legal).sum())
        if float(o["teacher_drop"]) != 0.0:
            anomalies["teacher_drop"] = float(o["teacher_drop"])
        if not np.array_equal(act_now, act):
            anomalies["active_differs_from_rollout"] = int((act_now != act).sum())
        if not np.array_equal(packed[int(sp["row"])], self.fixed_tokens):
            anomalies["fixed_decision_changed"] = True
        if not (np.isfinite(lp[act]).all() and np.isfinite(np.asarray(o["kl_slot"])[act]).all()):
            anomalies["non_finite"] = True
        slots = {}
        for s in self._order_slots(int(sp["slots"][0])):
            slots[str(s)] = {
                "head": U.slot_describe(s)["head"],
                "logp": _f(lp[r, s]),
                "logp_ref": _f(o["logp_ref"][r, s]),
                "ratio": _f(np.exp(lp[r, s] - lp_h[r, s])),
                "kl_il_theta": _f(o["kl_slot"][r, s]),
                "support": int(o["support"][r, s]),
                "support_ref": int(o["support_ref"][r, s]),
                "active_rollout": bool(act[r, s]),
                "active_now": bool(np.asarray(o["active_new"])[r, s]),
            }
        # 整批：各 head 組 active ratio 極值；目標列 active 槽的 ratio 最低槽
        rat = np.where(act, np.exp(lp - lp_h), np.nan)
        groups = {}
        for gi, g in enumerate(U.HEAD_GROUPS):
            m = (U.SLOT_GROUP == gi)[None, :] & act
            groups[g] = (
                [_f(np.nanmin(np.where(m, rat, np.nan))), _f(np.nanmax(np.where(m, rat, np.nan)))]
                if m.any()
                else None
            )
        row_rat = np.where(act[r], rat[r], np.inf)
        j = int(np.argmin(row_rat))
        prev = hist[i - 1] if 0 < i <= len(hist) else None
        upd = None
        if prev is not None:
            gn = prev.get("grad_norm")
            upd = {
                "step": prev.get("step"),
                "minibatch": prev.get("minibatch"),
                "seat": prev.get("seat"),
                **{
                    k: prev.get(k)
                    for k in ("loss", "pg", "value_loss", "entropy", "anchor_term", "grad_norm")
                },
                "clip_multiplier": None if gn is None else min(1.0, 1.0 / max(float(gn), 1e-12)),
            }
        self.doc["probes"].append(
            {
                "label": label,
                "i": i,
                "slots": slots,
                "row_kl_row_sum": _f(o["kl_row_sum"][r]),
                "legal_domain": {
                    "theta_il_mask_equal_batch": bool(o["mask_equal"]),
                    "selected_tokens_legal_batch": bool(legal.all()),
                    "n_active_checked": int(legal.size),
                    "teacher_drop": float(o["teacher_drop"]),
                    "active_equal_rollout_batch": bool(np.array_equal(act_now, act)),
                    "row_theta_mask_sha256": hashlib.sha256(
                        np.packbits(mask[r]).tobytes()
                    ).hexdigest(),
                },
                "anomalies": anomalies,
                "row_min_ratio_slot": {**U.slot_describe(j), "ratio": _f(row_rat[j])},
                "batch_group_ratio_minmax": groups,
                "last_update_before_probe": upd,
            }
        )
        self.flush()
        if anomalies:
            raise DiagAnomaly(f"{label} 合法域／有限性異常（停止，不判讀機制）：{anomalies}")

    def set_final(self, info: dict[str, Any]) -> None:
        self.doc["final"] = info
        self.flush()

    def flush(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc, ensure_ascii=False, indent=1, default=str))
        tmp.replace(self.out)
