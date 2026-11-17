#!/usr/bin/env python
"""schema-3 shard 的 **raw overlay**：把 sidecar 的 `pre_sanitize_tokens` 疊回 `d/*`。

背景（審查者 §17、2026-09-21 更正）
--------------------------------------
`rl/ordered_v3_data.py:727` 在寫檔前跑
``decision = sanitize_unit_slots(enc.decision, outside)``，`outside` 由
`unit_slots_outside_mask` 依**原 state**（不是 prefix）的遮罩判定，把不合格的 unit slot
改成 **PASS ＋ 其餘欄 0** 才進 `d/*`。所以 `d/*` 不是原始完整請求，原請求在 sidecar 的
`raw_action`／`canonical_action`／`components[].pre_sanitize_tokens`。

這支做什麼
----------
`build`：**一趟**掃一片 sidecar，同時產出

1. `t/*`：被 sanitize 的 (列, 槽) 與其原始 token；
2. `k/*`：keep 重算所需的**原始輸入**（逐列 `supervisable[29]`、`unknown`、`overdraw`、
   `market_keep_eligible`、`sidecar_ok`）＋ 依目前規則算出的 `unit_keep`／`market_keep`；
3. `census`：審查者 §17 的那組計數。

**存輸入而不只存結論**，是因為 keep 的最終式子 審查者 還在核（prefix reconstructible）；
規則改了只要重算這幾個 bool 陣列，不必再掃一次 478 MB 的 sidecar。

紀律
----
* **原 shard 與原 `shards_manifest.json` 一個 byte 不動**；綁定寫在另一份
  `overlay_manifest.json`（derived），裡面放原 manifest／npz／sidecar／輸出檔的 SHA256。
  overlay 自己的 `meta` **不含自己的檔案 SHA**（那是循環）。
* sidecar 是 **pretty-print 的 JSON 陣列**（2 空白縮排）；**逐列串流**，一次只 `json.loads`
  一個列物件，峰值 RSS 與檔案大小無關。不 `json.load` 整檔、不加 `ijson` 相依。
* `pre_sanitize_tokens` 是 **JSON object**（`_unit_slot_tokens` 回 dict），不是 array——
  判斷用 `isinstance(dict)`，不是看 `[`。
* **沒有 `market_compacted` 這個 sidecar 欄**：`market_keep_eligible` 已內含
  `not market_compacted`（codec `_market_keep_eligible`），不硬讀不存在的欄。
* 不 import jax（`build`／`census` 純 NumPy）；`effect-example` 才 lazy import。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.ordered_v3_codec import BUCKET_ENGINE_NOOP  # noqa: E402
from rl.ordered_v3_contract import BUCKET_C_DIFF, BUCKET_EXACT, MAX_UNITS  # noqa: E402

OVERLAY_SCHEMA = "full-actions-raw-overlay-v1"
SAFETY = "SAFETY_PLACEHOLDER"
OVERFLOW = "UNITS_EXCEED_N_ALIVE"
COMPACTED = "MARKET_COMPACTED"
#: **可重建（reconstructible）≠ 可監督**（審查者 §17.1）。投影只要求「請求已知」：
#: exact／c_diff／engine_noop（引擎會 no-op，但請求本身已知，kernel 照引擎語意處理）／
#: overlay 還原回來的 outside。
RECON_BUCKETS = (BUCKET_EXACT, BUCKET_C_DIFF, BUCKET_ENGINE_NOOP)
KEEP_RULES = {
    "source": "審查者 §17.1（2026-09-21）",
    "reconstructible_slot": (
        f"kind=='unit' ∧ primary_bucket ∈ {RECON_BUCKETS} ∧ '{SAFETY}' ∉ reasons "
        f"∧ '{OVERFLOW}' ∉ reasons ∧ ¬prefix_crop_unknown（該列）"
    ),
    "unit": "該 slot 可重建（含 engine_noop：教師真的下了這個請求，合不合法由 decode 的 "
    "prefix mask 判，off-mask 就 inactive＋計數）",
    "market": (
        "market_keep_eligible ∧ ¬market_compacted ∧ order_sidecar_ok ∧ ¬prefix_crop_unknown "
        "∧ 全部活 slot 可重建 ∧ ¬overflow_has_plant"
    ),
    "overflow": (
        "審查者 §23：官方 `_farmer_position` 對超出現有 hands 的 idx 回 None → 直接 no-op，"
        "**唯一例外是原子 PLANT 驗證會把多餘 hand 的 PLANT 需求也算進去**（我們的 kernel "
        "只算 n_alive 槽）。所以只有「多餘 hand 含 PLANT」不可重建；單純有 unit_overflow 不擋。"
    ),
    "overdraw": "**不擋**：引擎的原子 PLANT 預檢就是 kernel 的一部分，整組投影仍忠實；"
    "超領的 token 由 decode 的 prefix mask 落 off-mask",
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown:{e}"


# ------------------------------------------------------------------ sidecar 串流
def iter_sidecar(path: Path) -> Iterator[dict[str, Any]]:
    """逐列吐出 sidecar 的列物件。**一次只在記憶體裡放一列**。

    檔案格式是 `json.dump(..., indent=2)` 的陣列：第一行 `[`、每列由縮排 2 的 `  {` 開始、
    縮排 2 的 `  }`／`  },` 結束。用縮排當邊界比自己數括號穩，因為字串裡可能有括號。
    """
    buf: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not buf:
                if line.startswith("  {"):
                    buf.append(line)
                continue
            buf.append(line)
            if line.startswith("  }"):
                text = "".join(buf).rstrip().rstrip(",")
                buf = []
                yield json.loads(text)
    if buf:
        raise ValueError(f"{path}：最後一列沒有收尾，檔案可能被截斷")


def supervisable(rec: dict[str, Any]) -> bool:
    """`rl/ordered_v3_data.py::_unit_supervisable` 的 sidecar 版（條件逐項相同）。"""
    return (
        rec.get("kind") == "unit"
        and rec.get("primary_bucket") in (BUCKET_EXACT, BUCKET_C_DIFF)
        and SAFETY not in rec.get("reasons", ())
    )


def _req(row: dict[str, Any], key: str) -> Any:
    """具名讀 sidecar 欄位；**缺欄就 raise**，不套預設值。"""
    if key not in row:
        raise KeyError(f"sidecar 列 {row.get('row_id')} 缺欄位 {key}")
    return row[key]


class RowIndex:
    """sidecar 列 → npz 列號。**只認 `npz_row`**；它是 null 時才由 (capture_id, call_index) 推。

    `row_id` **不得**當成數字 index：它是字串（`_reject_sidecar` 的型別註記就是 `str`），
    長得像數字只是巧合。推出來的列號一律再用 capture_id／teacher_seat／call_index／step
    對原資料核一次，對不上就 raise。
    """

    def __init__(self, captures: list[dict[str, Any]], rows_per_capture: int):
        self.rows_per_capture = rows_per_capture
        self.captures = captures
        self.by_id = {str(c["capture_id"]): i for i, c in enumerate(captures)}
        if len(self.by_id) != len(captures):
            raise ValueError("captures 有重複的 capture_id")
        # 每個 capture 的列數都必須等長，`ci * rows_per_capture + call` 這條式子才成立
        bad = [
            (i, c.get("n_npz_rows"))
            for i, c in enumerate(captures)
            if int(c["n_npz_rows"]) != rows_per_capture
        ]
        if bad:
            raise ValueError(f"captures 的 n_npz_rows 不等於 {rows_per_capture}：{bad[:4]}")

    def of(self, row: dict[str, Any]) -> int:
        cid = str(_req(row, "capture_id"))
        call = int(_req(row, "call_index"))
        if cid not in self.by_id:
            raise ValueError(f"sidecar 的 capture_id {cid!r} 不在 npz 的 captures 裡")
        ci = self.by_id[cid]
        derived = ci * self.rows_per_capture + call
        v = row.get("npz_row")
        if v is None:
            return derived  # writer 沒回填時才用推的
        i = int(v)
        if i != derived:
            raise ValueError(f"npz_row {i} 與 (capture {cid}, call {call}) 推出的 {derived} 不符")
        return i

    def check(self, i: int, row: dict[str, Any]) -> None:
        """列號對原資料的身分核對：capture／seat／call 三項。"""
        cap = self.captures[i // self.rows_per_capture]
        if str(cap["capture_id"]) != str(row["capture_id"]):
            raise ValueError(f"列 {i}：capture_id {row['capture_id']} ≠ {cap['capture_id']}")
        if int(cap["teacher_seat"]) != int(_req(row, "teacher_seat")):
            raise ValueError(f"列 {i}：teacher_seat {row['teacher_seat']} ≠ {cap['teacher_seat']}")
        if i % self.rows_per_capture != int(row["call_index"]):
            raise ValueError(f"列 {i}：call_index {row['call_index']} 與列內位置不符")
        if str(cap["capture_sha256"]) != str(_req(row, "capture_sha256")):
            raise ValueError(
                f"列 {i}：capture_sha256 不符 "
                f"（sidecar {row['capture_sha256']} ≠ meta {cap['capture_sha256']}）"
            )


# ------------------------------------------------------------------ build（含 census）
def _dataset_captures(shard: Path) -> list[dict[str, Any]]:
    """npz 的 `__definitions_v2_ordered_v3_data__`（uint8 的 JSON）→ captures。"""
    from rl.ordered_v3_contract import DATASET_META_KEY  # noqa: PLC0415

    with zipfile.ZipFile(shard) as z, z.open(f"{DATASET_META_KEY}.npy") as f:
        meta = json.loads(bytes(np.load(f, allow_pickle=False).tolist()).decode("utf-8"))
    return list(meta["captures"])


def _load_npz_arrays(shard: Path) -> dict[str, np.ndarray]:
    """只讀需要的成員（都很小：step/row_ids 是 [N]，unit token 是 [N,29(,4)]）。"""
    want = (
        "row_ids",
        "s/step",
        "d/unit_verb",
        "d/unit_crop",
        "d/unit_item",
        "d/unit_qty_bytes",
        "k/unit_keep",
        "k/market_keep",
    )
    out = {}
    with zipfile.ZipFile(shard) as z:
        for name in want:
            with z.open(f"{name}.npy") as f:
                out[name] = np.load(f, allow_pickle=False)
    return out


def cmd_build(args) -> int:
    shard, sidecar = Path(args.shard).resolve(), Path(args.sidecar).resolve()
    manifest = Path(args.manifest).resolve()
    out_npz = Path(args.out).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    npz_sha = sha256_file(shard)
    man = json.loads(manifest.read_text())
    audits = {
        str(Path(sh["audit"]["npz_path"]).resolve()): sh["audit"]
        for sh in man.get("shards", [])
        if "audit" in sh
    }
    audit = audits.get(str(shard))
    if audit is None:
        raise SystemExit(f"{shard} 不在 manifest 裡：拒絕產 overlay")
    if audit["npz_sha256"] != npz_sha:
        raise SystemExit(f"{shard} SHA 不符 manifest：{npz_sha} ≠ {audit['npz_sha256']}")
    # sidecar 也要**實算並對 manifest 的 audit.sidecar_path／sidecar_sha256**，不是只記錄下來
    want_sidecar = Path(audit["sidecar_path"]).resolve()
    if want_sidecar != sidecar:
        raise SystemExit(f"--sidecar {sidecar} 不是 manifest 記的 {want_sidecar}")
    sidecar_sha = sha256_file(sidecar)
    if audit["sidecar_sha256"] != sidecar_sha:
        raise SystemExit(f"{sidecar} SHA 不符 manifest：{sidecar_sha} ≠ {audit['sidecar_sha256']}")
    t_sha = time.time() - t0

    a = _load_npz_arrays(shard)
    n = int(a["row_ids"].shape[0])
    u = MAX_UNITS
    captures = _dataset_captures(shard)
    if n % len(captures):
        raise SystemExit(f"{shard}：n_rows {n} 不是 capture 數 {len(captures)} 的整數倍")
    index = RowIndex(captures, n // len(captures))

    recon = np.zeros((n, u), bool)
    sup_old = np.zeros((n, u), bool)  # 舊規則（可監督），只為了報差異
    unknown = np.zeros((n,), bool)
    overdraw = np.zeros((n,), bool)
    mk_elig = np.zeros((n,), bool)
    sidecar_ok = np.zeros((n,), bool)
    compacted = np.zeros((n,), bool)
    overflow = np.zeros((n,), bool)
    overflow_plant = np.zeros((n,), bool)
    n_alive_arr = np.zeros((n,), np.int32)
    seen = np.zeros((n,), bool)
    t_npz_row: list[int] = []
    t_slot: list[int] = []
    t_verb: list[int] = []
    t_crop: list[int] = []
    t_item: list[int] = []
    t_qty: list[list[int]] = []
    buckets: dict[str, int] = {}
    n_unit_components = 0
    n_safety = 0
    n_rejects = 0

    t1 = time.time()
    limit = int(getattr(args, "limit_rows", 0) or 0)
    n_scanned = 0
    for row in iter_sidecar(sidecar):
        n_scanned += 1
        if limit and n_scanned > limit:
            break
        if _req(row, "schema_reject"):
            n_rejects += 1
            continue
        i = index.of(row)
        if not 0 <= i < n:
            raise SystemExit(f"sidecar 的 npz 列號 {i} 超出 npz 的 {n} 列")
        index.check(i, row)
        if int(a["s/step"][i]) != int(_req(row, "step")):
            raise SystemExit(f"列 {i}：sidecar step {row['step']} ≠ npz {int(a['s/step'][i])}")
        seen[i] = True
        unknown[i] = bool(_req(row, "prefix_crop_unknown"))
        overdraw[i] = bool(_req(row, "joint_seed_overdraw"))
        mk_elig[i] = bool(_req(row, "market_keep_eligible"))
        sidecar_ok[i] = bool(_req(row, "order_sidecar_ok"))
        n_alive_arr[i] = int(_req(row, "n_alive"))
        # `market_compacted` 不保證是頂層欄；不在就從 market component 的 reasons 取
        compacted[i] = bool(row.get("market_compacted", False))
        for rec in _req(row, "components"):
            kind = rec.get("kind")
            reasons = tuple(rec.get("reasons", ()))
            if kind == "unit_overflow" or OVERFLOW in reasons:
                overflow[i] = True
                # 多餘 hand 的 raw 動詞；只有 PLANT 會經由原子預檢影響活單位
                raw = rec.get("raw")
                if isinstance(raw, list) and raw and str(raw[0]).upper() == "PLANT":
                    overflow_plant[i] = True
            if COMPACTED in reasons:  # 不限 kind：這個 reason 不保證掛在 market component 上
                compacted[i] = True
            if kind != "unit":
                continue
            n_unit_components += 1
            bucket = rec.get("primary_bucket")
            buckets[str(bucket)] = buckets.get(str(bucket), 0) + 1
            if SAFETY in reasons:
                n_safety += 1
            slot = int(rec["index"])
            sup_old[i, slot] = bucket in (BUCKET_EXACT, BUCKET_C_DIFF) and SAFETY not in reasons
            recon[i, slot] = (
                bucket in RECON_BUCKETS and SAFETY not in reasons and OVERFLOW not in reasons
            )
            pre = rec.get("pre_sanitize_tokens")
            if isinstance(pre, dict):  # object，不是 array
                t_npz_row.append(i)
                t_slot.append(slot)
                t_verb.append(int(pre["unit_verb"]))
                t_crop.append(int(pre["unit_crop"]))
                t_item.append(int(pre["unit_item"]))
                t_qty.append([int(x) for x in pre["unit_qty_bytes"]])
    t_scan = time.time() - t1
    if limit:
        out = {
            "partial": True,
            "scanned_rows": min(n_scanned, limit),
            "wall_s": {"sha256": round(t_sha, 1), "sidecar_scan": round(t_scan, 1)},
            "n_sanitized_slots_in_prefix": len(t_npz_row),
            "note": "--limit-rows：只量成本，**沒有**寫 overlay",
        }
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0
    # **不因為有 schema_reject 就放寬**（審查者 §18）：npz 只收非 reject 的列，所以每一個
    # npz 列都必須在 sidecar 出現過；覆蓋不足代表索引或資料對不上，不能默默產 overlay。
    if not bool(seen.all()):
        miss = np.flatnonzero(~seen)
        raise SystemExit(
            f"有 {miss.size} 個 npz 列沒被 sidecar 覆蓋（前幾個：{miss[:8].tolist()}）；"
            f"schema_reject={n_rejects}。拒絕產 overlay。"
        )

    # --- 審查者 §17.1 的 keep ---
    recon &= (~unknown)[:, None]
    alive_mask = np.arange(u)[None, :] < n_alive_arr[:, None]
    all_alive_recon = np.all(recon | ~alive_mask, axis=1)
    unit_keep_new = recon
    market_keep_new = (
        mk_elig & (~compacted) & sidecar_ok & (~unknown) & all_alive_recon & (~overflow_plant)
    )
    unit_keep_old = np.asarray(a["k/unit_keep"], bool)
    market_keep_old = np.asarray(a["k/market_keep"], bool)

    census = {
        "n_rows": n,
        "n_schema_rejects": n_rejects,
        "n_unit_components": n_unit_components,
        "primary_bucket_counts": dict(sorted(buckets.items())),
        "n_safety_placeholder": n_safety,
        "n_rows_unit_overflow": int(overflow.sum()),
        "n_rows_overflow_has_plant": int(overflow_plant.sum()),
        "n_rows_prefix_crop_unknown": int(unknown.sum()),
        "n_rows_joint_seed_overdraw": int(overdraw.sum()),
        "n_rows_market_compacted": int(compacted.sum()),
        "n_sanitized_slots": len(t_npz_row),
        "n_rows_with_sanitized_slot": int(np.unique(t_npz_row).size) if t_npz_row else 0,
        "alive_slots": int(alive_mask.sum()),
        "unit_keep_old_true": int(unit_keep_old.sum()),
        "unit_keep_new_true": int((unit_keep_new & alive_mask).sum()),
        "unit_slots_regained": int((unit_keep_new & alive_mask & ~unit_keep_old).sum()),
        "unit_slots_lost": int((~unit_keep_new & alive_mask & unit_keep_old).sum()),
        "market_keep_old_true": int(market_keep_old.sum()),
        "market_keep_new_true": int(market_keep_new.sum()),
        "market_rows_regained": int((market_keep_new & ~market_keep_old).sum()),
        "market_rows_lost": int((~market_keep_new & market_keep_old).sum()),
        "old_rule_supervisable_slots": int((sup_old & alive_mask).sum()),
    }

    meta = {
        "overlay_schema": OVERLAY_SCHEMA,
        "shard_path": str(shard),
        "shard_npz_sha256": npz_sha,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": sidecar_sha,
        "source_manifest_path": str(manifest),
        "git_head": git_head(),
        "keep_rules": KEEP_RULES,
        "census": census,
        "wall_s": {"sha256": round(t_sha, 1), "sidecar_scan": round(t_scan, 1)},
    }
    payload = {
        # 鍵是 **npz 列號**（不是 sidecar 的字串 `row_id`）
        "t/npz_row": np.asarray(t_npz_row, np.int64),
        "t/slot": np.asarray(t_slot, np.int32),
        "t/unit_verb": np.asarray(t_verb, np.int32),
        "t/unit_crop": np.asarray(t_crop, np.int32),
        "t/unit_item": np.asarray(t_item, np.int32),
        "t/unit_qty_bytes": np.asarray(t_qty, np.int32).reshape(-1, 4),
        # keep 的**原始輸入**（規則再改也不必重掃 478 MB 的 sidecar）
        "k/reconstructible": recon,
        "k/supervisable_old_rule": sup_old,
        "k/prefix_crop_unknown": unknown,
        "k/joint_seed_overdraw": overdraw,
        "k/market_keep_eligible": mk_elig,
        "k/order_sidecar_ok": sidecar_ok,
        "k/market_compacted": compacted,
        "k/unit_overflow": overflow,
        "k/overflow_has_plant": overflow_plant,
        "k/n_alive": n_alive_arr,
        # 審查者 §17.1 規則下的結論（改規則時重算上面那幾個 bool 就好）
        "k/unit_keep": unit_keep_new,
        "k/market_keep": market_keep_new,
        "meta": np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
    }
    tmp = out_npz.with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    tmp.replace(out_npz)

    # derived manifest：**不改原 manifest**，綁定四個 SHA（overlay 自己的檔案 SHA 放這裡）
    dm = Path(args.overlay_manifest or out_npz.with_name("overlay_manifest.json"))
    rows = []
    if dm.exists():
        prev = json.loads(dm.read_text())
        if prev.get("schema") != "full-actions-overlay-manifest-v1":
            raise SystemExit(f"{dm} 不是 overlay manifest（schema={prev.get('schema')!r}）")
        rows = prev["overlays"]
    rows = [r for r in rows if r["shard_npz_sha256"] != npz_sha]
    rows.append(
        {
            "overlay_schema": OVERLAY_SCHEMA,
            "shard_path": str(shard),
            "shard_npz_sha256": npz_sha,
            "sidecar_sha256": sidecar_sha,
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": sha256_file(manifest),
            "overlay_path": str(out_npz),
            "overlay_sha256": sha256_file(out_npz),
            "git_head": meta["git_head"],
            "census": census,
        }
    )
    dm.write_text(
        json.dumps(
            {"schema": "full-actions-overlay-manifest-v1", "overlays": rows},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(
        json.dumps(
            {"out": str(out_npz), "manifest": str(dm), **meta["wall_s"], **census},
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


# ------------------------------------------------------------------ 單列證據
def cmd_example(args) -> int:
    """一列的 raw／canonical／pre-sanitize／sanitized 四欄對照（給 B2 當證據）。"""
    shard, sidecar = Path(args.shard).resolve(), Path(args.sidecar).resolve()
    a = _load_npz_arrays(shard)
    captures = _dataset_captures(shard)
    index = RowIndex(captures, int(a["row_ids"].shape[0]) // len(captures))
    target = int(args.row)
    hit = None
    for row in iter_sidecar(sidecar):
        if row.get("schema_reject"):
            continue
        if index.of(row) == target:
            index.check(target, row)
            hit = row
            break
    if hit is None:
        raise SystemExit(f"sidecar 裡找不到 npz 列 {target}")
    pre = [
        {
            "slot": int(r["index"]),
            "pre_sanitize_tokens": r["pre_sanitize_tokens"],
            "reasons": r["reasons"],
            "data_reasons": r["data_reasons"],
            "primary_bucket": r["primary_bucket"],
            "raw": r["raw"],
            "canonical": r["canonical"],
        }
        for r in hit["components"]
        if r.get("kind") == "unit" and isinstance(r.get("pre_sanitize_tokens"), dict)
    ]
    out = {
        "npz_row": target,
        "capture_id": hit["capture_id"],
        "teacher_seat": hit["teacher_seat"],
        "step": hit["step"],
        "day": hit["day"],
        "n_alive": hit["n_alive"],
        "raw_action": hit["raw_action"],
        "canonical_action": hit["canonical_action"],
        "sanitized_slots": pre,
        "sanitized_d_tokens": [
            {
                "slot": p["slot"],
                "unit_verb": int(a["d/unit_verb"][target, p["slot"]]),
                "unit_crop": int(a["d/unit_crop"][target, p["slot"]]),
                "unit_item": int(a["d/unit_item"][target, p["slot"]]),
                "unit_qty_bytes": [int(x) for x in a["d/unit_qty_bytes"][target, p["slot"]]],
            }
            for p in pre
        ],
        "unit_keep_shard": [bool(x) for x in a["k/unit_keep"][target]],
        "market_keep_shard": bool(a["k/market_keep"][target]),
        "prefix_crop_unknown": hit["prefix_crop_unknown"],
        "joint_seed_overdraw": hit["joint_seed_overdraw"],
        "market_keep_eligible": hit["market_keep_eligible"],
        "order_sidecar_ok": hit["order_sidecar_ok"],
        "git_head": git_head(),
    }
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    print(txt)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
    return 0


def cmd_effect_example(args) -> int:
    """同一列的**效果**：原 token 在真 prefix 之後是否合法／做得成（例：WATER）。

    這是「依原 state 遮罩判不合格、但依 prefix 其實做得成」的直接反例。只有這個子命令吃 jax。
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from rl import caps as caps_mod  # noqa: PLC0415
    from rl import full_actions_decode as D  # noqa: PLC0415
    from rl import obs as rl_obs  # noqa: PLC0415
    from rl import state as S  # noqa: PLC0415
    from scripts.full_actions_il_train import RowSlicer  # noqa: PLC0415

    caps = caps_mod.BOUNDED_TRAINING
    shard, sidecar = Path(args.shard).resolve(), Path(args.sidecar).resolve()
    target = int(args.row)
    captures = _dataset_captures(shard)
    with zipfile.ZipFile(shard) as z, z.open("row_ids.npy") as f:
        n_rows = int(np.load(f, allow_pickle=False).shape[0])
    index = RowIndex(captures, n_rows // len(captures))
    hit = None
    for row in iter_sidecar(sidecar):
        if not row.get("schema_reject") and index.of(row) == target:
            index.check(target, row)
            hit = row
            break
    if hit is None:
        raise SystemExit(f"sidecar 裡找不到 npz 列 {target}")
    seat = int(hit["teacher_seat"])
    rs = RowSlicer(shard)
    st = S.SimState(
        **{f: jnp.asarray(rs.read(f"s/{f}.npy", target, target + 1)) for f in S.SimState._fields}
    )
    fields = {
        k: jnp.asarray(rs.read(f"d/{k}.npy", target, target + 1))
        for k in ("unit_verb", "unit_crop", "unit_item", "unit_qty_bytes")
    }
    sanitized = {k: np.asarray(v).copy() for k, v in fields.items()}
    restored = {k: v.copy() for k, v in sanitized.items()}
    slots = []
    for rec in hit["components"]:
        pre = rec.get("pre_sanitize_tokens")
        if rec.get("kind") == "unit" and isinstance(pre, dict):
            s_i = int(rec["index"])
            slots.append(s_i)
            restored["unit_verb"][0, s_i] = pre["unit_verb"]
            restored["unit_crop"][0, s_i] = pre["unit_crop"]
            restored["unit_item"][0, s_i] = pre["unit_item"]
            restored["unit_qty_bytes"][0, s_i] = pre["unit_qty_bytes"]
    rj = {k: jnp.asarray(v) for k, v in restored.items()}

    out: dict[str, Any] = {"npz_row": target, "teacher_seat": seat, "slots": slots, "per_slot": []}
    leg0 = rl_obs.legality(st, caps)
    for s_i in slots:
        verb = int(restored["unit_verb"][0, s_i])
        pre_st = D.prefix_state(st, rj, seat, caps, s_i)  # 前面的槽做完之後
        legp = rl_obs.legality(pre_st, caps)
        after = D.prefix_state(st, rj, seat, caps, caps.max_units)
        x = int(np.asarray(st.unit_x)[0, seat, s_i])
        y = int(np.asarray(st.unit_y)[0, seat, s_i])
        flat = y * 10 + x
        out["per_slot"].append(
            {
                "slot": s_i,
                "verb": verb,
                "legal_on_original_state": bool(np.asarray(leg0["unit_ops"])[0, seat, s_i, verb]),
                "legal_after_true_prefix": bool(np.asarray(legp["unit_ops"])[0, seat, s_i, verb]),
                "tile_xy": [x, y],
                "tile_after_full_projection": [
                    int(v) for v in np.asarray(after.tiles)[0, seat, flat]
                ],
            }
        )
    out["git_head"] = git_head()
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    print(txt)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
    return 0


def cmd_canonical_rows(args) -> int:
    """抽指定幾列的 `canonical_action`（一趟掃完），寫成**小**的 fixture 檔給測試用。

    測試不該自己去掃 478 MB 的 sidecar；raw→tokens→JSON 的對帳需要的只有這幾列的
    canonical，所以在這裡先抽出來、帶 sidecar SHA 存檔。
    """
    shard, sidecar = Path(args.shard).resolve(), Path(args.sidecar).resolve()
    want = {int(r) for r in args.row}
    with zipfile.ZipFile(shard) as z, z.open("row_ids.npy") as f:
        n_rows = int(np.load(f, allow_pickle=False).shape[0])
    captures = _dataset_captures(shard)
    index = RowIndex(captures, n_rows // len(captures))
    out: dict[str, Any] = {
        "shard_path": str(shard),
        "sidecar_path": str(sidecar),
        "sidecar_sha256": sha256_file(sidecar),
        "git_head": git_head(),
        "rows": {},
    }
    for row in iter_sidecar(sidecar):
        if row.get("schema_reject"):
            continue
        i = index.of(row)
        if i not in want:
            continue
        index.check(i, row)
        out["rows"][str(i)] = {
            "capture_id": row["capture_id"],
            "teacher_seat": row["teacher_seat"],
            "step": row["step"],
            "n_alive": row["n_alive"],
            "canonical_action": row["canonical_action"],
            "raw_action": row["raw_action"],
            "sanitized_slots": [
                {"slot": int(r["index"]), "pre_sanitize_tokens": r["pre_sanitize_tokens"]}
                for r in row["components"]
                if r.get("kind") == "unit" and isinstance(r.get("pre_sanitize_tokens"), dict)
            ],
        }
        if len(out["rows"]) == len(want):
            break
    missing = sorted(want - {int(k) for k in out["rows"]})
    if missing:
        raise SystemExit(f"sidecar 裡找不到這些 npz 列：{missing}")
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt)
    print(json.dumps({"out": args.out, "rows": sorted(int(k) for k in out["rows"])}))
    return 0


# ------------------------------------------------------------------ keep 重算（v1 → v2）
OVERLAY_SCHEMA_V2 = "full-actions-raw-overlay-v2"


def cmd_recompute(args) -> int:
    """用**已存的 `k/*`** 重算 keep，產 v2 overlay ＋ v2 manifest。**不重掃 sidecar。**

    規則變更（審查者 §23）：`market_keep'` 去掉整列 `¬unit_overflow`，改成 `¬overflow_has_plant`。
    v1 的 overlay 沒有逐列的 `overflow_has_plant`（那時 builder 還沒算），所以這裡**明示用
    `--pin-overflow-has-plant 0` 把它釘成 0**，並要求 `--evidence` 寫清楚依據；
    未來由 builder 逐 record 算出來的 v2 就不需要釘。v1 原件一個 byte 不動。
    """
    man_path = Path(args.overlay_manifest).resolve()
    man = json.loads(man_path.read_text())
    if man.get("schema") != "full-actions-overlay-manifest-v1":
        raise SystemExit(f"{man_path} 不是 overlay manifest v1")
    if args.pin_overflow_has_plant != 0:
        raise SystemExit("目前只支援釘成 0（有非零證據時請改用 builder 重跑）")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, totals = [], {}
    for e in man["overlays"]:
        src = Path(e["overlay_path"]).resolve()
        got = sha256_file(src)
        if got != e["overlay_sha256"]:
            raise SystemExit(f"{src} SHA 不符 manifest：{got} ≠ {e['overlay_sha256']}")
        shard = Path(e["shard_path"]).resolve()
        shard_sha = sha256_file(shard)
        if shard_sha != e["shard_npz_sha256"]:
            raise SystemExit(f"{shard} SHA 不符 manifest：{shard_sha} ≠ {e['shard_npz_sha256']}")
        z = dict(np.load(src, allow_pickle=False))
        meta = json.loads(str(z["meta"]))
        if meta.get("overlay_schema") != OVERLAY_SCHEMA:
            raise SystemExit(f"{src} 不是 v1 overlay")

        recon = np.asarray(z["k/reconstructible"], bool)
        n_alive_arr = np.asarray(z["k/n_alive"], np.int32)
        alive_mask = np.arange(recon.shape[1])[None, :] < n_alive_arr[:, None]
        all_alive_recon = np.all(recon | ~alive_mask, axis=1)
        ohp = np.zeros((recon.shape[0],), bool)  # 釘成 0（見 --evidence）
        unit_keep_new = recon
        market_keep_new = (
            np.asarray(z["k/market_keep_eligible"], bool)
            & ~np.asarray(z["k/market_compacted"], bool)
            & np.asarray(z["k/order_sidecar_ok"], bool)
            & ~np.asarray(z["k/prefix_crop_unknown"], bool)
            & all_alive_recon
            & ~ohp
        )
        with zipfile.ZipFile(shard) as zz:
            with zz.open("k/unit_keep.npy") as f:
                unit_keep_old = np.asarray(np.load(f, allow_pickle=False), bool)
            with zz.open("k/market_keep.npy") as f:
                market_keep_old = np.asarray(np.load(f, allow_pickle=False), bool)

        census = dict(meta["census"])
        census.update(
            {
                "n_rows_overflow_has_plant": int(ohp.sum()),
                "unit_keep_new_true": int((unit_keep_new & alive_mask).sum()),
                "unit_slots_regained": int((unit_keep_new & alive_mask & ~unit_keep_old).sum()),
                "unit_slots_lost": int((~unit_keep_new & alive_mask & unit_keep_old).sum()),
                "market_keep_new_true": int(market_keep_new.sum()),
                "market_rows_regained": int((market_keep_new & ~market_keep_old).sum()),
                "market_rows_lost": int((~market_keep_new & market_keep_old).sum()),
            }
        )
        meta2 = {
            **meta,
            "overlay_schema": OVERLAY_SCHEMA_V2,
            "keep_rules": KEEP_RULES,
            "census": census,
            "derived_from": {
                "overlay_path": str(src),
                "overlay_sha256": got,
                "overlay_schema": OVERLAY_SCHEMA,
                "source_overlay_manifest": str(man_path),
                "source_overlay_manifest_sha256": sha256_file(man_path),
            },
            "overflow_has_plant": {
                "source": "pinned",
                "value": 0,
                "evidence": args.evidence,
                "note": "v1 沒有逐列欄位；未來 builder 會逐 record 算，屆時不需釘",
            },
            "git_head": git_head(),
        }
        z["k/overflow_has_plant"] = ohp
        z["k/unit_keep"] = unit_keep_new
        z["k/market_keep"] = market_keep_new
        z["meta"] = np.asarray(json.dumps(meta2, ensure_ascii=False, sort_keys=True))
        dst = out_dir / (src.stem + "_v2.npz")
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, **z)
        tmp.replace(dst)
        entries.append(
            {
                **{k: v for k, v in e.items() if k != "census"},
                "overlay_schema": OVERLAY_SCHEMA_V2,
                "overlay_path": str(dst),
                "overlay_sha256": sha256_file(dst),
                "git_head": meta2["git_head"],
                "derived_from_overlay_sha256": got,
                "census": census,
            }
        )
        for k, v in census.items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v

    out_man = Path(args.out_manifest).resolve()
    out_man.write_text(
        json.dumps(
            {"schema": "full-actions-overlay-manifest-v1", "overlays": entries},
            ensure_ascii=False,
            indent=1,
        )
    )
    summary = {
        "n_shards": len(entries),
        "out_manifest": str(out_man),
        "out_manifest_sha256": sha256_file(out_man),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "evidence": args.evidence,
        "totals": totals,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="一趟掃一片 sidecar：產 overlay ＋ census")
    b.add_argument("--shard", required=True)
    b.add_argument("--sidecar", required=True)
    b.add_argument("--manifest", required=True, help="shards_manifest.json")
    b.add_argument("--out", required=True)
    b.add_argument("--overlay-manifest", default=None)
    b.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="只掃前 N 列（純診斷成本）；**不寫 overlay、不寫 manifest**，"
        "也不做完整覆蓋檢查，輸出標 partial——不得拿來當可訓練的 overlay",
    )
    for name, fn in (("example", cmd_example), ("effect-example", cmd_effect_example)):
        s = sub.add_parser(name)
        s.add_argument("--shard", required=True)
        s.add_argument("--sidecar", required=True)
        s.add_argument("--row", type=int, required=True)
        s.add_argument("--out", default=None)
        s.set_defaults(func=fn)
    cr = sub.add_parser("canonical-rows", help="抽幾列的 canonical_action 當測試 fixture")
    cr.add_argument("--shard", required=True)
    cr.add_argument("--sidecar", required=True)
    cr.add_argument("--row", action="append", required=True)
    cr.add_argument("--out", required=True)
    cr.set_defaults(func=cmd_canonical_rows)
    rc = sub.add_parser("recompute", help="用已存的 k/* 重算 keep → v2 overlay（不重掃 sidecar）")
    rc.add_argument("--overlay-manifest", required=True, help="v1 的 combined manifest")
    rc.add_argument("--out-dir", required=True)
    rc.add_argument("--out-manifest", required=True)
    rc.add_argument("--out", default=None, help="總表 JSON")
    rc.add_argument("--pin-overflow-has-plant", type=int, required=True)
    rc.add_argument("--evidence", required=True, help="釘住該值的依據（會寫進 v2 meta）")
    rc.set_defaults(func=cmd_recompute)
    b.set_defaults(func=cmd_build)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
