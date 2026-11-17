#!/usr/bin/env python
"""full-actions 第一段的 IL（teacher forcing）訓練。

規格：內部規格文件 §第一段精確實作規格。

做什麼
------
從 **schema-3 shard** 串流 `s/*`（SimState 20 欄）＋ `d/*`（DecisionV3 八欄）
＋ `k/*`（keep）＋ `marked`／`n_alive`／`row_ids`，用 `rl/full_actions_decode.decode("teacher")`
算逐 token NLL，unit 與 market 各佔 0.5，**空分母跳該組並報數**。

**⚠ `d/*` 不是未改動的原始請求（2026-09-21 更正）**
-------------------------------------------------------------------
`rl/ordered_v3_data.py:727` 寫檔前做了
``decision = sanitize_unit_slots(enc.decision, outside)``：`outside` 由
`unit_slots_outside_mask` 依**原 state**（不是 prefix）的遮罩判定，把不合格的 unit slot
整組改成 **PASS＋其餘欄 0** 才進 `d/*`。真正的原請求在 sidecar 的
`raw_action`／`canonical_action`／`components[].pre_sanitize_tokens`。

**解法（已落地）**：`scripts/full_actions_raw_overlay.py` 產的 raw overlay，把
`pre_sanitize_tokens` 疊回 `d/*`、keep 換成 審查者 §17.1 的重算值。
**正式路徑一定要 `--overlay` ＋ `--overlay-manifest`（核 expected SHA），沒給就拒**；
`--diagnostic-sanitized` 只給診斷，不得當訓練結果，**沒有靜默 fallback**。

SHA 鏈（審查者 §23.1）：訓練時**不重算 478 MB–1.4 GB 的 sidecar 位元**（那是 builder 產出
overlay 當下實核過的）。讀取時核的是**鏈**：原 `shards_manifest.json` 實算 SHA →
`audit.npz_sha256`／`audit.sidecar_sha256`（記錄）→ overlay entry 的 `shard_npz_sha256`／
`sidecar_sha256`／`source_manifest_sha256` → overlay 檔實算 SHA → overlay payload 內
`meta` 的自述欄。保證的是「這份 overlay 出自 manifest 指名的那片 shard 與那份 sidecar」，
不是讀取時重驗 sidecar 位元。

沒有 overlay 時的已知影響：

1. 被 sanitize 的 slot `k/unit_keep` 是 False，所以它**不會**以假 PASS 進 loss 分母——
   這點目前是安全的。
2. 但 `keeps_from_encode` 的 `market_keep` 要求 `all_alive`（該列每個活 slot 都 keep），
   所以**一個 slot 被 sanitize，整列的市場監督就沒了**；`unknown`／`overdraw` 更是整列清空。
3. **teacher off-mask 計數會低估**：與我們 prefix 遮罩不合的教師 token，上游已經先換成
   PASS（PASS 永遠合法），所以 `teacher_drop` 讀到 0 不代表兩邊遮罩一致。
4. 用 `d/*` 做的 roundtrip／投影／JSON 對帳，證明的是 **token→部署出口**的保真，
   **不是 raw→部署出口**。

紀律（規格與 審查者 §14.1 的落點）
--------------------------------
* **不重編語料**：只讀既有 shard 的欄位，不從 Loop 4 的 labels 反推。
* **不把整個成員載進 RAM**：用 `scripts/loop4_il_labels._read_slice` 逐列切
  （`s/tiles.npy` 在完整 shard 是 GB 級）。
* **dev/heldout 不混 train**：shard 目錄名帶 split，`--split` 對不上直接拒。
* **核身分**：npz SHA256 對 `shards_manifest.json` 的 `audit.npz_sha256`；
  逐列核 `marked` 的座位 ＝ captures 的 `teacher_seat`、`s/step` ＝ `row_id % rows_per_capture`。
* **teacher off-mask 不進分母**：落在共用遮罩外的 token 與其 children 退出 loss，
  逐 head／逐 op 計數輸出；raw token 與真 prefix 投影都保留，不改標、不換 PASS。
* 一個 batch 只取**同一個 teacher 座位**的列（`decode` 的 `seat` 是 Python int；
  混座位要嘛靜默算錯、要嘛得跑兩次）。列取連續區塊——`_read_slice` 是從成員開頭掃到
  `lo`，隨機逐列 gather 會把 I/O 放大到不能用。

**長跑建議經 `scripts/run_capped.sh`**（核親和、RSS 帽、wall timeout）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl import caps as caps_mod  # noqa: E402
from rl import full_actions_decode as D  # noqa: E402
from rl import full_actions_head as H  # noqa: E402
from rl import full_actions_numpy as FN  # noqa: E402
from rl import ordered_v3_contract as C  # noqa: E402
from rl import state as S  # noqa: E402

CAPS = caps_mod.BOUNDED_TRAINING
LABEL_KEYS = (
    "unit_verb",
    "unit_crop",
    "unit_item",
    "unit_qty_bytes",
    "market_op",
    "market_item",
    "market_qty_bytes",
)
SPLITS = ("train", "dev", "heldout")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class RowSlicer:
    """精確讀 npz 成員的第 `lo:hi` 列——**不重讀前綴**。

    為什麼需要：`scripts/loop4_il_labels._read_slice` 是 `f.read(lo * row)` 把前綴整塊讀出來
    再丟掉。schema-3 shard 的 `s/tiles.npy` 是 **ZIP_STORED、822.8 MiB**，每個 update 隨機
    `lo` 都重讀一次前綴，等於每步搬幾百 MB——那不叫串流。

    做法：ZIP_STORED 的成員在 npz 裡是**原樣連續存放**的，所以算出
    「local header 結束位置 ＋ NPY header 長度」就能對**原檔**直接 seek 到那一列。
    原檔一個 byte 都沒改，manifest 的 SHA binding 照樣成立。壓縮成員不能 seek，
    退回**有界 chunk** 跳過（一次 8 MiB，不一次配置整個前綴）。

    舊 helper 不動（歷史結果要能重現）。
    """

    CHUNK = 1 << 23  # 壓縮成員 fallback 的跳過區塊

    def __init__(self, path: Path):
        self.path = Path(path)
        self.zf = zipfile.ZipFile(self.path)
        self.fh = self.path.open("rb")
        self._info: dict[str, dict[str, Any]] = {}

    def info(self, name: str) -> dict[str, Any]:
        if name in self._info:
            return self._info[name]
        from numpy.lib import format as npformat  # noqa: PLC0415

        zi = self.zf.getinfo(name)
        with self.zf.open(name) as f:
            major, minor = npformat.read_magic(f)
            reader = {
                (1, 0): npformat.read_array_header_1_0,
                (2, 0): npformat.read_array_header_2_0,
            }[(major, minor)]
            shape, fortran, dtype = reader(f)
            npy_header = f.tell()
            head_row = np.frombuffer(
                f.read(int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize), dtype=dtype
            )
        if fortran:
            raise ValueError(f"{name} 是 Fortran order，這支只處理 C order")
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        stored = zi.compress_type == zipfile.ZIP_STORED
        data_off = None
        if stored:
            self.fh.seek(zi.header_offset)
            fixed = self.fh.read(30)
            n_name, n_extra = struct.unpack("<HH", fixed[26:30])
            data_off = zi.header_offset + 30 + n_name + n_extra + npy_header
            # 自我核對：同一列用兩條路各讀一次，對不上就不准用 seek 路徑
            self.fh.seek(data_off)
            probe = np.frombuffer(self.fh.read(row_bytes), dtype=dtype)
            if not np.array_equal(probe, head_row):
                raise ValueError(f"{name}：算出的 data offset 對不上，拒絕用 seek 讀")
        self._info[name] = {
            "shape": shape,
            "dtype": dtype,
            "row_bytes": row_bytes,
            "stored": stored,
            "data_off": data_off,
            "npy_header": npy_header,
        }
        return self._info[name]

    def read(self, name: str, lo: int, hi: int) -> np.ndarray:
        i = self.info(name)
        row, shape, dtype = i["row_bytes"], i["shape"], i["dtype"]
        if lo < 0 or hi > shape[0] or hi < lo:
            raise ValueError(f"{name}[{lo}:{hi}] 超出 {shape[0]} 列")
        if i["stored"]:
            self.fh.seek(i["data_off"] + lo * row)
            buf = self.fh.read((hi - lo) * row)
        else:  # 壓縮成員：有界 chunk 跳過（不一次配置整個前綴）
            with self.zf.open(name) as f:
                f.read(i["npy_header"])
                left = lo * row
                while left:
                    n = f.read(min(self.CHUNK, left))
                    if not n:
                        raise ValueError(f"{name}：跳過前綴時提早結束")
                    left -= len(n)
                buf = f.read((hi - lo) * row)
        return np.frombuffer(buf, dtype=dtype).reshape((hi - lo, *shape[1:]))

    def close(self) -> None:
        self.fh.close()
        self.zf.close()


OVERLAY_SCHEMAS = ("full-actions-raw-overlay-v1", "full-actions-raw-overlay-v2")


class Overlay:
    """`d/*` 的 raw overlay：把 sanitize 掉的 unit token 換回原請求、keep 換成 §17.1 的重算值。

    **正式路徑一律要 overlay**（`d/*` 是 sanitize 過的，見檔頭）。載入時必須通過：
    overlay 檔實算 SHA ＝ `overlay_manifest.json` 記的 `overlay_sha256`、
    overlay 綁的 `shard_npz_sha256` ＝ 這片 shard 實算的 SHA。任一不符就拒，**不 fallback**。
    """

    def __init__(
        self,
        npz_path: Path,
        manifest_path: Path,
        shard_sha256: str,
        *,
        audit_sidecar_sha256: str,
        source_manifest_sha256: str,
    ):
        self.path = Path(npz_path).resolve()
        self.sha256 = sha256_file(self.path)
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.source_manifest_sha256 = source_manifest_sha256
        man = json.loads(self.manifest_path.read_text())
        if man.get("schema") != "full-actions-overlay-manifest-v1":
            raise ValueError(f"{manifest_path} 不是 overlay manifest")
        rows = [
            r for r in man["overlays"] if str(Path(r["overlay_path"]).resolve()) == str(self.path)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"{manifest_path} 裡對 {self.path} 的紀錄有 {len(rows)} 筆，要剛好 1 筆"
            )
        entry = rows[0]
        if entry["overlay_sha256"] != self.sha256:
            raise ValueError(
                f"overlay SHA 不符 manifest：{self.sha256} ≠ {entry['overlay_sha256']}"
            )
        if entry["shard_npz_sha256"] != shard_sha256:
            raise ValueError(
                f"overlay 綁的是別片 shard：{entry['shard_npz_sha256']} ≠ {shard_sha256}"
            )
        # 鏈到**原** shards_manifest：這條是防「換掉 overlay manifest 的條目」，
        # 不是在訓練時重算 478 MB–1.4 GB 的 sidecar 位元（那個 builder 產出時已實核）。
        if entry["sidecar_sha256"] != audit_sidecar_sha256:
            raise ValueError(
                f"overlay entry 的 sidecar SHA 不是原 manifest 記的："
                f"{entry['sidecar_sha256']} ≠ {audit_sidecar_sha256}"
            )
        if entry["source_manifest_sha256"] != source_manifest_sha256:
            raise ValueError(
                f"overlay 出自別份 shards_manifest：{entry['source_manifest_sha256']} "
                f"≠ {source_manifest_sha256}"
            )
        z = np.load(self.path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        if self.meta.get("overlay_schema") not in OVERLAY_SCHEMAS:
            raise ValueError(f"overlay schema 是 {self.meta.get('overlay_schema')!r}")
        # payload 自述要與 manifest 條目一致（payload 本身的 SHA 上面已核）
        if self.meta.get("shard_npz_sha256") != shard_sha256:
            raise ValueError("overlay meta 的 shard_npz_sha256 與實算的 shard 不符")
        if self.meta.get("sidecar_sha256") != audit_sidecar_sha256:
            raise ValueError("overlay meta 的 sidecar_sha256 與原 manifest 不符")
        self.entry = entry
        self.npz_row = np.asarray(z["t/npz_row"], np.int64)
        self.slot = np.asarray(z["t/slot"], np.int64)
        self.tok = {
            "unit_verb": np.asarray(z["t/unit_verb"], np.int32),
            "unit_crop": np.asarray(z["t/unit_crop"], np.int32),
            "unit_item": np.asarray(z["t/unit_item"], np.int32),
            "unit_qty_bytes": np.asarray(z["t/unit_qty_bytes"], np.int32),
        }
        self.unit_keep = np.asarray(z["k/unit_keep"], bool)
        self.market_keep = np.asarray(z["k/market_keep"], bool)
        # row → 該列被還原的 (位置索引) 清單；M 是百級，整份進 RAM 無所謂
        self.by_row: dict[int, list[int]] = {}
        for j, r in enumerate(self.npz_row.tolist()):
            self.by_row.setdefault(int(r), []).append(j)

    def identity(self) -> dict[str, Any]:
        return {
            "overlay_path": str(self.path),
            "overlay_sha256": self.sha256,
            "overlay_manifest_path": str(self.manifest_path),
            "overlay_manifest_sha256": self.manifest_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "overlay_schema": self.meta["overlay_schema"],
            "sidecar_sha256": self.entry["sidecar_sha256"],
            "keep_rules": self.meta["keep_rules"],
            "census": self.meta["census"],
            "built_at_git_head": self.entry["git_head"],
        }

    def apply(self, idx: np.ndarray, labels: dict, unit_keep, market_keep) -> int:
        """就地把 `labels` 的 unit token 換回原請求、keep 換成重算值。回還原的 slot 數。"""
        n = 0
        for b, r in enumerate(idx.tolist()):
            for j in self.by_row.get(int(r), ()):
                sl = int(self.slot[j])
                for k, arr in self.tok.items():
                    labels[k][b, sl] = arr[j]
                n += 1
        unit_keep[:] = self.unit_keep[idx]
        market_keep[:] = self.market_keep[idx]
        return n


class ShardReader:
    """一片 schema-3 shard 的串流讀取器（只在需要時讀那幾列的 byte）。"""

    def __init__(
        self,
        path: Path,
        *,
        split: str,
        manifest: dict[str, dict[str, str]],
        overlay: Overlay | None = None,
        allow_sanitized: bool = False,
    ):
        self.path = Path(path)
        self.split = split
        if overlay is None and not allow_sanitized:
            raise ValueError(
                f"{path}：正式路徑一定要 overlay（`d/*` 是 sanitize 過的）。"
                "要用原始 sanitized 欄位只能明示 --diagnostic-sanitized，且不得拿去訓練。"
            )
        self.overlay = overlay
        self.allow_sanitized = allow_sanitized
        self.rs = RowSlicer(self.path)
        self.zf = self.rs.zf
        with self.zf.open(f"{C.DATASET_META_KEY}.npy") as f:  # 這個成員很小，整份讀
            arr = np.load(f, allow_pickle=False)  # uint8 的 JSON
        self.meta = json.loads(bytes(arr.tolist()).decode("utf-8"))
        self.captures = self.meta["captures"]
        self.n_rows = int(self.rs.info("row_ids.npy")["shape"][0])
        if self.n_rows % len(self.captures):
            raise ValueError(f"{path}：n_rows {self.n_rows} 不是 capture 數的整數倍")
        self.rows_per_capture = self.n_rows // len(self.captures)
        self.seat_of_capture = np.asarray([int(c["teacher_seat"]) for c in self.captures], np.int32)
        # 身分：**一定要**在 manifest 裡找得到這片，且 SHA 相同。找不到 binding 就拒——
        # 「manifest 沒寫就放行」等於沒核。
        want = manifest.get(str(self.path))
        if want is None:
            raise ValueError(f"{self.path} 不在 manifest 的 shards[].audit.npz_path 裡：拒絕使用")
        self.sha256 = sha256_file(self.path)
        if want["npz_sha"] != self.sha256:
            raise ValueError(f"{path} SHA256 不符 manifest：{self.sha256} ≠ {want['npz_sha']}")
        self.audit_sidecar_sha256 = want["sidecar_sha"]

    def rows(self, idx: Any) -> dict[str, Any]:
        """**任意列**（跨 capture 隨機抽也行）→ state／labels／keep／n_alive／row_ids／seat。

        連號的部分會合併成一次 read（`RowSlicer` 是 seek 讀，跨 capture 抽列是毫秒級）。
        一個 batch 只能有一個 teacher 座位——`decode` 的 `seat` 是 static。
        """
        idx = np.sort(np.asarray(idx, np.int64))
        if idx.size == 0:
            raise ValueError("空 batch")
        runs: list[tuple[int, int]] = []
        for i in idx.tolist():
            if runs and i == runs[-1][1]:
                runs[-1] = (runs[-1][0], i + 1)
            else:
                runs.append((i, i + 1))

        def gather(name: str) -> np.ndarray:
            return np.concatenate([self.rs.read(name, a, b) for a, b in runs], axis=0)

        t0 = time.time()
        st = S.SimState(**{f: jnp.asarray(gather(f"s/{f}.npy")) for f in S.SimState._fields})
        labels = {k: jnp.asarray(gather(f"d/{k}.npy")) for k in LABEL_KEYS}
        unit_keep = gather("k/unit_keep.npy")
        market_keep = gather("k/market_keep.npy")
        marked = gather("marked.npy")
        n_alive = gather("n_alive.npy")
        row_ids = gather("row_ids.npy")
        t_load = time.time() - t0
        n_overlaid = 0
        if self.overlay is not None:
            lab_np = {k: np.asarray(v).copy() for k, v in labels.items()}
            n_overlaid = self.overlay.apply(idx, lab_np, unit_keep, market_keep)
            labels = {k: jnp.asarray(v) for k, v in lab_np.items()}

        seats = np.argmax(marked.astype(np.int32), axis=-1)
        if len(set(seats.tolist())) != 1:
            raise ValueError(f"{self.path}：一個 batch 混了兩個 teacher 座位 {set(seats.tolist())}")
        seat = int(seats[0])
        # 身分：marked 的座位 ＝ capture 的 teacher_seat；step ＝ row_id % rows_per_capture
        cap_idx = row_ids // self.rows_per_capture
        if not np.array_equal(self.seat_of_capture[cap_idx], seats.astype(np.int32)):
            raise ValueError(f"{self.path}：marked 座位與 captures.teacher_seat 不符")
        step = np.asarray(st.step)
        if not np.array_equal(step, row_ids % self.rows_per_capture):
            raise ValueError(f"{self.path}：s/step 與 row_id 的回合位置不符")
        return {
            "state": st,
            "labels": labels,
            "seat": seat,
            "unit_keep": unit_keep,
            "market_keep": market_keep,
            "n_alive": n_alive,
            "row_ids": row_ids,
            "day": step // 24,
            "t_load_s": t_load,
            "n_captures_in_batch": int(np.unique(cap_idx).size),
            "n_overlaid_slots": n_overlaid,
        }

    def block(self, lo: int, n: int) -> dict[str, Any]:
        """連續 `n` 列（只給煙測／診斷；訓練用 `rows` 跨 capture 抽）。"""
        return self.rows(np.arange(lo, lo + n))


def split_of(path: Path) -> str:
    name = path.parent.parent.name  # .../shards/<shardNN_split_lo_hi>/audit/official_schema3.npz
    for s in SPLITS:
        if f"_{s}_" in name:
            return s
    raise ValueError(f"從 {name} 認不出 split（要有 _train_／_dev_／_heldout_）")


def manifest_sha_map(manifest_path: Path) -> dict[str, dict[str, str]]:
    """`shards_manifest.json` → ``{絕對路徑: {npz_sha, sidecar_sha}}``；兩邊都 `resolve()` 後比。"""
    m = json.loads(manifest_path.read_text())
    return {
        str(Path(s["audit"]["npz_path"]).resolve()): {
            "npz_sha": s["audit"]["npz_sha256"],
            "sidecar_sha": s["audit"]["sidecar_sha256"],
        }
        for s in m.get("shards", [])
        if "audit" in s and "npz_sha256" in s["audit"]
    }


# ------------------------------------------------------------------ loss
def keep_masks(unit_keep: np.ndarray, market_keep: np.ndarray, u: int) -> jax.Array:
    """`k/*` → 263 格的 keep 遮罩（unit 逐單位展成 7 格、market 整列展成 60 格）。"""
    b = unit_keep.shape[0]
    uk = np.repeat(unit_keep[:, :u, None], D.UNIT_SLOTS, axis=2).reshape(b, u * D.UNIT_SLOTS)
    mk = np.broadcast_to(market_keep[:, None], (b, C.MAX_ORDERS * D.MARKET_SLOTS))
    return jnp.asarray(np.concatenate([uk, mk], axis=1))


def loss_and_stats(params, cfg, seat: int, state, labels, keep):
    """unit／market 各 0.5 的 active-token 平均 NLL；空分母該組跳過（權重記 0）。

    `seat` 是 **Python int（static）**：`full_actions_obs.observation` 與 `rl/obs.legality`
    都用 scalar 索引（`state.money[:, seat]`、`[seat, 1 - seat]` 重排），把它 trace 成
    array 會靜默算成別的東西。所以逐座位各建一支 jit，不是把 seat 塞進 batch。
    """
    out = D.decode("teacher")(params, cfg, state, seat, CAPS, labels=labels)
    u = cfg["max_units"] * D.UNIT_SLOTS
    # T3 §5.1.3：loss 只取 policy_active（＝active，含 teacher eligibility）∩ keep ∩ il_mask。
    # il_loss_mask 只退直接 loss 與分母；不改 raw、prefix、後續 hidden 或 policy_active。
    pol = out["active"] & keep
    use = pol & out["il_loss_mask"]
    lp = jnp.where(use, out["logp"], 0.0)
    cnt_u = jnp.sum(use[:, :u])
    cnt_m = jnp.sum(use[:, u:])
    nll_u = -jnp.sum(lp[:, :u]) / jnp.maximum(cnt_u, 1.0)
    nll_m = -jnp.sum(lp[:, u:]) / jnp.maximum(cnt_m, 1.0)
    loss = 0.5 * jnp.where(cnt_u > 0, nll_u, 0.0) + 0.5 * jnp.where(cnt_m > 0, nll_m, 0.0)
    drop = out["teacher_drop"]
    stats = {
        "loss": loss,
        "nll_unit": nll_u,
        "nll_market": nll_m,
        "n_unit": cnt_u,
        "n_market": cnt_m,
        "empty_unit": (cnt_u == 0).astype(jnp.int32),
        "empty_market": (cnt_m == 0).astype(jnp.int32),
        "drop_unit_verb": jnp.sum(drop["unit"]["verb"]),
        "drop_unit_crop": jnp.sum(drop["unit"]["crop"]),
        "drop_unit_item": jnp.sum(drop["unit"]["item"]),
        "drop_unit_qty": jnp.sum(drop["unit"]["qty"]),
        "drop_market_op": jnp.sum(drop["market"]["op"]),
        "drop_market_item": jnp.sum(drop["market"]["item"]),
        "drop_market_qty": jnp.sum(drop["market"]["qty"]),
        "drop_unit_verb_by_op": jnp.sum(drop["unit_verb_by_op"], axis=0),
        "drop_market_op_by_code": jnp.sum(drop["market_op_by_code"], axis=0),
        # T3 I2：policy_active 總量、有效 IL 量、分原因遮掉量（keep 範圍內）
        "policy_active_unit": jnp.sum(pol[:, :u]),
        "policy_active_market": jnp.sum(pol[:, u:]),
        **_il_mask_counts(out["il_mask_info"], keep, cfg["max_units"]),
    }
    return loss, stats


def _il_mask_counts(info, keep, n_units: int) -> dict:
    """分原因的遮罩數（決策點數，不是 token 數），只算 keep 的單位／列。"""
    ukeep = keep[:, : n_units * D.UNIT_SLOTS : D.UNIT_SLOTS]  # 每單位 verb 槽的 keep
    mkeep = keep[:, n_units * D.UNIT_SLOTS][:, None]  # 市場整列同一個 keep
    return {
        "il_mask_a5_units": jnp.sum(info["a5"] & ukeep),
        "il_mask_pickup_empty_units": jnp.sum(info["pickup_empty"] & ukeep),
        "il_mask_m2_orders": jnp.sum(info["m2"] & mkeep),
        "il_mask_m2_unknown_orders": jnp.sum(info["m2_unknown"] & mkeep),
    }


def make_step(cfg, opt, seat: int):
    """**逐座位**一支 jit（`seat` 關進 closure，永遠是 static）。"""

    def step(params, opt_state, state, labels, keep):
        (loss, stats), grads = jax.value_and_grad(loss_and_stats, has_aux=True)(
            params, cfg, seat, state, labels, keep
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        stats["grad_norm"] = optax.global_norm(grads)
        return params, opt_state, loss, stats

    return jax.jit(step)


def make_eval(cfg, seat: int):
    def ev(params, state, labels, keep):
        return loss_and_stats(params, cfg, seat, state, labels, keep)[0]

    return jax.jit(ev)


# ------------------------------------------------------------------ checkpoint
def save_train_state(path: Path, params, opt_state, meta: dict[str, Any], train: dict) -> None:
    """``p/<key>`` ＋ ``o/<i>`` ＋ `meta`；params 的部分與 `FN.save_params` 同格式。

    `train` 記 **resume 需要的一切**：iteration、RNG 狀態、optimizer 配置、資料設定。
    沒有它，resume 會從 iter 1、seed 0 的 RNG 重跑同一段資料，還會在 `o/*` 缺席時
    默默 `opt.init` 把 Adam 動量歸零——那正是 Orbit Wars `--resume-opt-state` 的舊 bug。
    """
    if opt_state is None:
        raise ValueError("不得存沒有 optimizer 狀態的訓練 checkpoint")
    meta = {
        **meta,
        "params": FN.params_signature({k: np.asarray(v) for k, v in params.items()}),
        "train": train,
    }
    d: dict[str, Any] = {f"p/{k}": np.asarray(v) for k, v in params.items()}
    d.update({f"o/{i}": np.asarray(v) for i, v in enumerate(jax.tree.leaves(opt_state))})
    d["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(path)


def make_opt(lr: float, clip: float):
    """訓練用 optimizer（global-norm clip → Adam）；resume／lr 遷移測試共用同一個建法。"""
    return optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))


def load_train_state(path: Path, opt, *, opt_cfg: dict | None = None, lr_from: float | None = None):
    """用 `FN.load_params` 做 schema／cfg 導出 shape／簽章校驗，再還原 optimizer 與 RNG。

    **不完整的 resume 一律拒**：缺 `o/*`、缺 `meta.train`、optimizer 配置與現在給的不同，
    都丟 `ValueError`，不默默重置動量、不默默換 lr。
    `lr_from`（只由 `--resume-migrate-lr OLD:NEW` 給）：checkpoint 的 lr 改核＝OLD（本次 lr＝NEW），
    clip 照舊須相同；Adam 的 count／mu／nu 與 lr 無關，逐葉原樣還原。
    """
    np_params, meta = FN.load_params(path)
    params = {k: jnp.asarray(v) for k, v in np_params.items()}
    z = np.load(path, allow_pickle=False)
    n = sum(1 for k in z.files if k.startswith("o/"))
    if n == 0:
        raise ValueError(f"{path} 沒有 o/* optimizer 狀態：拒絕當 resume 用（會把 Adam 動量歸零）")
    train = meta.get("train")
    if not train or "iter" not in train or "rng" not in train:
        raise ValueError(f"{path} 的 meta.train 缺 iter／rng：resume 會重放同一段資料")
    if opt_cfg is not None:
        for k in ("lr", "clip"):
            want = lr_from if (k == "lr" and lr_from is not None) else opt_cfg[k]
            if float(train["opt"][k]) != float(want):
                tag = "（--resume-migrate-lr 的 OLD）" if k == "lr" and lr_from is not None else ""
                raise ValueError(
                    f"optimizer 配置不同：checkpoint {k}={train['opt'][k]}、本次 {want}{tag}"
                )
    _leaves, treedef = jax.tree.flatten(opt.init(params))
    if n != len(_leaves):
        raise ValueError(f"{path} 的 o/* 有 {n} 葉，opt.init 需要 {len(_leaves)}")
    opt_state = jax.tree.unflatten(treedef, [jnp.asarray(z[f"o/{i}"]) for i in range(n)])
    return params, opt_state, meta, train


def restore_rng(state: dict) -> np.random.Generator:
    """還原 `np.random.Generator` 的 bit generator 狀態（不是重造同 seed 的新 RNG）。"""
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


def git_head() -> str:
    import subprocess  # noqa: PLC0415

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown:{e}"


#: **本線的依賴集合**：這些檔案髒掉，訓練身分就不可信（identity 也會逐支記 SHA）。
#: 前六支是原本就記 SHA 的；後面四支是 decode／numpy 路徑實際 import 的，以及啟動器本身。
DEPENDENCY_FILES = (
    "rl/ordered_v3_codec.py",
    "rl/ordered_v3_contract.py",
    "rl/full_actions_obs.py",
    "rl/full_actions_head.py",
    "rl/full_actions_decode.py",
    "rl/full_actions_numpy.py",
    "rl/kernels.py",
    "rl/obs.py",
    "rl/loop4_numpy.py",
    # 審查者 §28：decode／obs／投影路徑再往下的一層，改了同樣會改訓練語意
    "rl/state.py",
    "rl/rules.py",
    "rl/caps.py",
    "rl/pricing.py",
    "rl/actions.py",
    "rl/loop4_model.py",
    "rl/loop4_obs.py",
    "scripts/full_actions_il_train.py",
    "scripts/run_capped.sh",
)


def device_stats() -> dict[str, Any]:
    """量測平台與 device 峰值記憶體，寫進 `result.json`。

    **正式 GPU 成本量的是這支訓練腳本**（不是 smoke），所以 helper 放這裡，smoke 直接 import。
    CPU backend 沒有 `memory_stats`，`peak_bytes_in_use` 會是 `None`——可接受，那一欄是
    給 GPU 量 VRAM 峰值用的。
    """
    d = jax.devices()[0]
    try:
        ms = d.memory_stats() or {}
    except Exception:  # noqa: BLE001
        ms = {}
    return {
        "platform": d.platform,
        "device_kind": getattr(d, "device_kind", None),
        "n_devices": len(jax.devices()),
        "peak_bytes_in_use": ms.get("peak_bytes_in_use"),
        "bytes_limit": ms.get("bytes_limit"),
    }


def _git_porcelain() -> list[str]:
    """`git status --porcelain` 的**路徑**清單（含 untracked；rename 取新舊兩邊）。"""
    import subprocess  # noqa: PLC0415

    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    paths: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        rest = line[3:].strip().strip('"')
        paths.extend(part.strip().strip('"') for part in rest.split(" -> "))
    return sorted(set(paths))


def _git_diff_sha256() -> str:
    """tracked 變更的 SHA256。用 `git diff HEAD`——**staged 與 unstaged 都涵蓋**；
    只用 `git diff` 會漏掉已 `git add` 的改動，那正好是最容易被忽略的一種髒。"""
    import subprocess  # noqa: PLC0415

    out = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    return hashlib.sha256(out).hexdigest()


def check_dirty(dirty_paths: list[str]) -> None:
    """**硬檢**：髒檔與依賴集合有交集就拒啟動。

    與本線無關的髒檔（作者 既有未提交的腳本）照實記 `dirty=true` 就好——為了訓練身分乾淨
    去 stash／代提交別人的東西，不是必要正確性（審查者 §26.1）。
    """
    hit = sorted(set(dirty_paths) & set(DEPENDENCY_FILES))
    if hit:
        raise SystemExit(
            f"依賴集合裡有未提交的改動，拒絕啟動（訓練身分不可信）：{hit}。commit 或還原後再跑。"
        )


def source_identity(readers: list[ShardReader]) -> dict[str, Any]:
    """`main()` 開頭一次記下的來源身分：git head ＋ 依賴檔 SHA ＋ 髒檔 ＋ 輸入 shard。"""
    dirty_paths = _git_porcelain()
    return {
        "git_head": git_head(),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "dirty_diff_sha256": _git_diff_sha256(),
        "dependency_sha256": {f: sha256_file(ROOT / f) for f in DEPENDENCY_FILES},
        "qty_codec": FN.ACTION_CODEC,
        "il_mask_contract": FN.IL_MASK_CONTRACT,
        "source_manifest_sha256": readers[0].overlay.source_manifest_sha256
        if readers and readers[0].overlay is not None
        else None,
        "inputs": [
            {
                "path": str(r.path),
                "split": r.split,
                "npz_sha256": r.sha256,
                "overlay": r.overlay.identity() if r.overlay is not None else None,
                "diagnostic_sanitized": r.allow_sanitized,
                "source_closure_sha256": r.meta.get("source_closure_sha256"),
                "n_rows": r.n_rows,
                "rows_per_capture": r.rows_per_capture,
            }
            for r in readers
        ],
    }


def data_identity(source: dict[str, Any]) -> list[dict[str, Any]]:
    """從 `source_identity()` 的輸出抽出**資料身分**（只有內容 SHA，沒有路徑字串）。

    `git_head`／`dirty`／模組 SHA 都**不比**：repo 改了但資料沒變，resume 是合法的。
    路徑也不比——同一份資料換個掛載點仍是同一份資料；身分只看內容 hash。

    **順序保留、不排序**：`rng` 是用 `trains` 的索引抽片的（`rng.integers(len(trains))`），
    換了輸入順序就等於換了續跑的抽樣序列，那不是同一條 run。
    """
    out = []
    for i in source["inputs"]:
        ov = i.get("overlay") or {}
        out.append(
            {
                "split": i["split"],
                "npz_sha256": i["npz_sha256"],
                "diagnostic_sanitized": bool(i["diagnostic_sanitized"]),
                "overlay_sha256": ov.get("overlay_sha256"),
                "overlay_manifest_sha256": ov.get("overlay_manifest_sha256"),
                "source_manifest_sha256": ov.get("source_manifest_sha256"),
            }
        )
    return out


#: 命令列可覆寫的 trunk 架構欄（不給＝沿 `H.DEFAULT_ARCH`）
ARCH_FLAGS = ("d_model", "n_layers", "n_heads", "d_ff")


def arch_config(args: argparse.Namespace) -> dict[str, Any]:
    """`--d-model／--n-layers／--n-heads／--d-ff` → cfg；都不給時與 `H.config(CAPS)` 相同。"""
    over = {k: getattr(args, k) for k in ARCH_FLAGS if getattr(args, k) is not None}
    cfg = H.config(CAPS, **over)
    bad = {k: cfg[k] for k in ARCH_FLAGS if cfg[k] <= 0}
    if bad:
        raise SystemExit(f"架構整數必須 > 0：{bad}；拒絕")
    if cfg["d_model"] % cfg["n_heads"]:
        raise SystemExit(f"--n-heads {cfg['n_heads']} 不能整除 --d-model {cfg['d_model']}：拒絕")
    return cfg


def check_trunk(ckpt_meta: dict[str, Any], cfg: dict[str, Any], flag: str) -> None:
    """checkpoint 的 `meta.trunk` 四欄必須等於本次架構，否則拒（`n_heads` 不在任何 shape 裡）。"""
    trunk = ckpt_meta.get("trunk") or {}
    bad = {k: (trunk.get(k), cfg[k]) for k in ARCH_FLAGS if trunk.get(k) != cfg[k]}
    if bad:
        raise SystemExit(f"{flag} 的架構與本次不符（checkpoint, 本次）：{bad}；拒收")


def load_init_params(
    path: Path, expect_sha: str, template: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """只載權重：SHA 讀檔前硬核；架構四欄、葉名、shape、dtype 與新初始化的 template 相同才收。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    got = sha256_file(path)
    if got != expect_sha:
        raise SystemExit(f"--init-params SHA 不符：期望 {expect_sha}、實算 {got}")
    loaded, _meta = FN.load_params(path, caps=CAPS, expect_sha256=expect_sha)
    check_trunk(_meta, cfg, "--init-params")
    bad = [
        k
        for k, v in template.items()
        if k not in loaded
        or loaded[k].shape != np.shape(v)
        or loaded[k].dtype != np.asarray(v).dtype
    ]
    if bad:
        raise SystemExit(f"--init-params 與模型葉不符：{bad[:5]}")
    return {k: jnp.asarray(loaded[k]) for k in template}


#: `--resume-migrate-data` 的對應表 schema（外部 JSON；見 `check_resume_identity`）
DATA_MIGRATION_MAP_SCHEMA = "il-data-migration-map-v1"


def parse_migrate_data(values: list[str] | None, *, resume: str | None) -> tuple[str, str] | None:
    """`--resume-migrate-data OLD_MANIFEST_SHA:NEW_MANIFEST_SHA`：只收一筆；無 --resume、格式錯、
    非 64 位小寫 hex、OLD＝NEW 一律拒。

    SHA 指 shards_manifest（overlay 的 `source_manifest_sha256`）。
    """
    if not values:
        return None
    if len(values) != 1:
        raise SystemExit(f"--resume-migrate-data 只准給一次，得到 {values}")
    if not resume:
        raise SystemExit("--resume-migrate-data 只能配 --resume 使用")
    parts = values[0].split(":")
    hexok = lambda x: len(x) == 64 and all(c in "0123456789abcdef" for c in x)  # noqa: E731
    if len(parts) != 2 or not all(hexok(x) for x in parts):
        raise SystemExit(f"--resume-migrate-data 格式是 OLD_SHA:NEW_SHA（64 hex）：{values[0]!r}")
    if parts[0] == parts[1]:
        raise SystemExit("--resume-migrate-data 的 OLD 與 NEW 相同：不需要遷移")
    return parts[0], parts[1]


def parse_migrate_lr(
    values: list[str] | None, *, resume: str | None, lr: float
) -> tuple[float, float] | None:
    """`--resume-migrate-lr OLD_LR:NEW_LR`：只收一筆；無 --resume、格式錯、非有限正數、OLD＝NEW、
    `--lr` ≠ NEW 一律拒。OLD 在 resume 時對 checkpoint 的 `train.opt.lr` 硬核
    （見 `load_train_state`）。
    """
    if not values:
        return None
    if len(values) != 1:
        raise SystemExit(f"--resume-migrate-lr 只准給一次，得到 {values}")
    if not resume:
        raise SystemExit("--resume-migrate-lr 只能配 --resume 使用")
    parts = values[0].split(":")
    try:
        old, new = (float(x) for x in parts) if len(parts) == 2 else (None, None)
    except ValueError:
        old = new = None
    if old is None or not all(np.isfinite(x) and x > 0 for x in (old, new)):
        raise SystemExit(f"--resume-migrate-lr 格式是 OLD_LR:NEW_LR（有限正數）：{values[0]!r}")
    if old == new:
        raise SystemExit("--resume-migrate-lr 的 OLD 與 NEW 相同：不需要遷移")
    if float(lr) != new:
        raise SystemExit(f"--resume-migrate-lr 的 NEW={new} 與 --lr {lr} 不同")
    return old, new


def lr_migration_record(
    ck_meta: dict[str, Any], train: dict[str, Any], migrate: tuple[float, float] | None, *,
    from_ckpt_sha256: str | None = None,
) -> dict[str, Any]:  # fmt: skip
    """lr 遷移歷史：`lr_migrations`（新 meta 要記的完整歷史）與 `lr_migration_this_run`。

    歷史最後一筆的 `to_lr` 必須＝checkpoint 的 `train.opt.lr`（竄改或不連續即拒）；
    無旗標時原樣承接（沒有歷史就是 None，meta／result 不加欄位）。
    """
    hist = list((ck_meta or {}).get("lr_migrations") or [])
    if hist and float(hist[-1]["to_lr"]) != float(train["opt"]["lr"]):
        raise ValueError("checkpoint 的 lr_migrations 最後一筆與其 train.opt.lr 不符：拒絕 resume")
    if migrate is None:
        return {"lr_migrations": hist or None, "migration": None}
    rec = {
        "from_lr": migrate[0],
        "to_lr": migrate[1],
        "at_iter": int(train["iter"]),
        "effective_from_iter": int(train["iter"]) + 1,
        "from_ckpt_sha256": from_ckpt_sha256,
        "adam_state": "count/mu/nu 原樣沿用（未重置）",
    }
    return {"lr_migrations": hist + [rec], "migration": rec}


def _canon_sha(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_resume_identity(
    ckpt_meta: dict[str, Any],
    source: dict[str, Any],
    *,
    migrate: tuple[str, str] | None = None,
    migrate_map: dict[str, Any] | None = None,
    at_iter: int | None = None,
    from_ckpt_sha256: str | None = None,
    map_sha256: str | None = None,
) -> dict[str, Any]:
    """resume 的資料身分必須與 checkpoint 當時完全一致，否則拒。

    沒有這一條，`--resume` 指到另一版 overlay（v1／v2）會**靜默**用不同的 keep 續訓——
    同一條 run 的前後半段吃到不同分母，事後從 checkpoint 看不出來。

    **資料遷移**（`migrate=(FROM, TO)`＋外部對應表 `migrate_map`；L8 用 09-27／28 新資料續訓）：
    對應表**逐項綁** checkpoint 的資料身分（`from_data_identity`）
    與本次的資料身分（`to_data_identity`），
    兩邊 source manifest SHA 必須分別＝FROM／TO；比較用副本把資料身分由舊換新後仍走原嚴格全等。
    遷移歷史 `data_migrations` 進新 checkpoint 的 meta 並承接：之後無旗標 resume 沿用歷史，
    且歷史最後一筆的 `to_data_identity_sha256` 必須＝當下資料身分（不符即拒）。
    optimizer 配置、rows、架構的硬核不在此（照原路徑），Adam／RNG／iteration 照原樣載入。
    回報告 dict：`data_migrations`（新 meta 要記的完整歷史）與
    `migration`（遷移時的三段報告，否則 None）。
    """
    prev = (ckpt_meta or {}).get("source")
    if not prev or "inputs" not in prev:
        raise ValueError("checkpoint 的 meta.source.inputs 不存在：無法核資料身分，拒絕 resume")
    a, b = data_identity(prev), data_identity(source)
    hist = list((ckpt_meta or {}).get("data_migrations") or [])
    if migrate is None:
        if a != b:
            diff = [(x, y) for x, y in zip_longest(a, b) if x != y]
            raise ValueError(
                "resume 的資料身分與 checkpoint 不同（overlay／shard／manifest 任一 SHA 不符）："
                f"{diff[:2]}"
            )
        if hist and hist[-1].get("to_data_identity_sha256") != _canon_sha(b):
            raise ValueError(
                "checkpoint 的 data_migrations 最後一筆與當下資料身分不符：拒絕 resume"
            )
        return {"data_migrations": hist or None, "migration": None}
    fr, to = migrate
    m = migrate_map or {}
    if m.get("schema") != DATA_MIGRATION_MAP_SCHEMA:
        raise ValueError(
            f"資料遷移對應表 schema 不是 {DATA_MIGRATION_MAP_SCHEMA}：{m.get('schema')!r}"
        )
    if m.get("from_source_manifest_sha256") != fr or m.get("to_source_manifest_sha256") != to:
        raise ValueError("資料遷移對應表的 FROM／TO 與 --resume-migrate-data 不符")
    if a != m.get("from_data_identity"):
        raise ValueError("checkpoint 的資料身分≠對應表 from_data_identity（FROM 不符）：拒絕遷移")
    if b != m.get("to_data_identity"):
        raise ValueError("本次的資料身分≠對應表 to_data_identity：拒絕遷移")
    if not a or any(x.get("source_manifest_sha256") != fr for x in a):
        raise ValueError(f"checkpoint 輸入的 source manifest 不全是 FROM {fr[:12]}…：拒絕遷移")
    if not b or any(x.get("source_manifest_sha256") != to for x in b):
        raise ValueError(f"本次輸入的 source manifest 不全是 TO {to[:12]}…：拒絕遷移")
    if hist and hist[-1].get("to_data_identity_sha256") != _canon_sha(a):
        raise ValueError("checkpoint 的 data_migrations 最後一筆與其資料身分不符：拒絕遷移")
    comp = list(m["to_data_identity"])  # 比較用副本：舊資料身分換成對應表的新身分
    if comp != b:  # 仍走嚴格全等
        raise ValueError("比較用副本與本次資料身分不相等：拒絕遷移")
    entry = {
        "from_source_manifest_sha256": fr,
        "to_source_manifest_sha256": to,
        "from_data_identity_sha256": _canon_sha(a),
        "to_data_identity_sha256": _canon_sha(b),
        "n_inputs": {"from": len(a), "to": len(b)},
        "at_iter": at_iter,
        "from_ckpt_sha256": from_ckpt_sha256,
        "map_sha256": map_sha256,
    }
    new_hist = [*hist, entry]
    return {
        "data_migrations": new_hist,
        "migration": {
            "raw_diff": {
                "n_inputs_from": len(a),
                "n_inputs_to": len(b),
                "n_positions_differ": sum(1 for x, y in zip_longest(a, b) if x != y),
            },
            "allowed_migration": entry,
            "rest_all_equal": (
                "optimizer 配置（lr／clip）、rows、trunk 架構照原路徑硬核；"
                "Adam／RNG／iteration 原樣載入"
            ),
        },
    }


# ------------------------------------------------------------------ 主流程
def sample_rows(rng, reader: ShardReader, n: int, seat: int) -> np.ndarray:
    """**跨 capture 隨機抽 `n` 列**，全部來自 teacher 座位 ＝ `seat` 的 capture。

    先前的「同一 capture 連續 8 回合」相關性太高、有效樣本極小；`RowSlicer` 是 seek 讀，
    一列 `s/tiles` 約 12 KB，跨 capture 隨機抽是毫秒級，沒有理由用連續區塊換 I/O。
    """
    caps_of_seat = np.flatnonzero(reader.seat_of_capture == seat)
    if caps_of_seat.size == 0:
        raise RuntimeError(f"{reader.path} 沒有 teacher_seat={seat} 的 capture")

    ci = rng.choice(caps_of_seat, size=n, replace=True)
    off = rng.integers(0, reader.rows_per_capture, size=n)
    return ci.astype(np.int64) * reader.rows_per_capture + off


def draw_step(rng, trains, devs, it: int, rows: int, u: int, dev_every: int, dev_batches: int):
    """第 `it` 步要用的**全部**抽樣與讀列，照原同步迴圈的 rng 取用順序：

    訓練批（抽片 → 座位 → `sample_rows` → `rows` → `keep_masks`）→ rng 快照 A →
    若 `it` 是 dev 步，再抽 dev 批 → rng 快照 B。

    快照 A ＝ 原版在 iter `it` 存 `actor_i*.npz` 時的 rng（原版存檔在 dev 抽樣之前）；
    快照 B ＝ 抽完第 `it` 步全部、抽第 `it+1` 步之前的 rng（`final.npz` 用它）。
    預取執行緒是 rng 的唯一使用者；主迴圈不得再碰 rng。
    """
    t0 = time.time()
    reader = trains[int(rng.integers(len(trains)))]
    # 座位輪替：兩支 jit 各拿一半，batch 內座位一致；該片沒有那一座就用另一座
    seats_here = set(reader.seat_of_capture.tolist())
    seat = it % 2 if (it % 2) in seats_here else next(iter(seats_here))
    idx = sample_rows(rng, reader, rows, seat)
    batch = reader.rows(idx)
    keep = keep_masks(batch["unit_keep"], batch["market_keep"], u)
    rng_after_train = rng.bit_generator.state
    t_bg = time.time() - t0
    dev = []
    if dev_every and devs and it % dev_every == 0:
        for _ in range(dev_batches):
            dr = devs[int(rng.integers(len(devs)))]
            dseats = sorted(set(dr.seat_of_capture.tolist()))
            db = dr.rows(sample_rows(rng, dr, rows, int(rng.choice(dseats))))
            dk = keep_masks(db["unit_keep"], db["market_keep"], u)
            dev.append((db, dk))
    return {
        "it": it,
        "reader": reader,
        "batch": batch,
        "keep": keep,
        "idx_sha16": hashlib.sha256(np.asarray(idx, np.int64).tobytes()).hexdigest()[:16],
        "rng_after_train": rng_after_train,
        "rng_after_step": rng.bit_generator.state,
        "dev": dev,
        "t_bg": t_bg,
        "t_dev_bg": time.time() - t0 - t_bg,
    }


class Prefetcher:
    """單一生產者執行緒、深度 1：主迴圈取走第 `it` 步後才送出第 `it+1` 步，
    所以任何時刻最多只領先一步；依序送出、依序取回，消費順序＝原迴圈順序。
    生產者的例外在 `get` 原樣拋出（主迴圈 rc≠0），不吞、不產生空批。"""

    def __init__(self, fn, first: int, last: int):
        self.fn, self.last = fn, last
        self.ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="il-prefetch")
        self.fut = self.ex.submit(fn, first) if first <= last else None

    def get(self, it: int) -> tuple[dict[str, Any], float]:
        tw = time.time()
        item = self.fut.result()  # 生產者例外在這裡拋出
        wait = time.time() - tw
        if item["it"] != it:
            raise RuntimeError(f"預取順序錯：要 iter {it}，拿到 {item['it']}")
        self.fut = self.ex.submit(self.fn, it + 1) if it < self.last else None
        return item, wait

    def close(self) -> None:
        self.ex.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", action="append", required=True, help="train shard 的 npz 路徑")
    ap.add_argument("--dev-shard", action="append", default=[])
    ap.add_argument("--manifest", default=None, help="shards_manifest.json（必填）")
    ap.add_argument("--split", default="train", choices=SPLITS)
    ap.add_argument("--rows", type=int, default=64, help="每個 batch 的列數（跨 capture 隨機抽）")
    ap.add_argument("--steps", type=int, default=100, help="這次要再跑幾步（resume 時是增量）")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--dev-every", type=int, default=0)
    ap.add_argument("--dev-batches", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--resume-migrate-data",
        action="append",
        default=None,
        metavar="OLD_MANIFEST_SHA:NEW_MANIFEST_SHA",
        help="資料遷移續訓：checkpoint 的 shards_manifest SHA → 本次（只收一筆；需對應表）",
    )
    ap.add_argument(
        "--resume-migrate-data-map",
        default=None,
        help=f"資料遷移對應表 JSON（schema {DATA_MIGRATION_MAP_SCHEMA}；from／to 身分逐項綁）",
    )
    ap.add_argument(
        "--resume-migrate-lr",
        action="append",
        default=None,
        metavar="OLD_LR:NEW_LR",
        help="lr 遷移續訓：checkpoint 的 lr（硬核）→ 本次 --lr；Adam／RNG／iteration 全沿用"
        "（只收一筆）",
    )
    ap.add_argument(
        "--init-params",
        default=None,
        help="只載權重的起點（不載 optimizer／RNG；計畫 rl-adv12-distill-probe 2(b)）",
    )
    ap.add_argument("--expect-init-sha", default=None, help="--init-params 的 SHA256，讀檔前硬核")
    ap.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="每片 shard 的 raw overlay npz（順序對 --shard，再接 --dev-shard）；正式路徑必填",
    )
    ap.add_argument(
        "--overlay-manifest",
        action="append",
        default=[],
        help="對應的 overlay_manifest.json（給一份就套用到全部）",
    )
    ap.add_argument("--d-model", type=int, default=None, help="trunk 寬度（不給＝DEFAULT_ARCH）")
    ap.add_argument("--n-layers", type=int, default=None, help="trunk 層數（不給＝DEFAULT_ARCH）")
    ap.add_argument("--n-heads", type=int, default=None, help="注意力頭數，須整除 d_model")
    ap.add_argument("--d-ff", type=int, default=None, help="FFN 寬度（不給＝DEFAULT_ARCH）")
    ap.add_argument(
        "--diagnostic-sanitized",
        action="store_true",
        help="**只給診斷**：不套 overlay，直接吃 sanitize 過的 d/*。產物不得當訓練結果",
    )
    args = ap.parse_args()
    # 架構與起點在 mkdir／讀資料／init 之前就定：整數 ≤0、heads 不整除、resume／init-params 互斥、
    # init-params SHA、兩者的架構不符，都在這裡拒（rc≠0、不建輸出目錄）
    cfg = arch_config(args)
    migrate_data = parse_migrate_data(args.resume_migrate_data, resume=args.resume)
    migrate_lr = parse_migrate_lr(args.resume_migrate_lr, resume=args.resume, lr=args.lr)
    if bool(migrate_data) != bool(args.resume_migrate_data_map):
        raise SystemExit("--resume-migrate-data 與 --resume-migrate-data-map 必須一起給")
    if args.init_params and args.resume:
        raise SystemExit("--init-params 與 --resume 不可同時給")
    if args.init_params and not args.expect_init_sha:
        raise SystemExit("--init-params 需要 --expect-init-sha")
    if args.init_params:
        got = sha256_file(Path(args.init_params))
        if got != args.expect_init_sha:
            raise SystemExit(f"--init-params SHA 不符：期望 {args.expect_init_sha}、實算 {got}")
    for flag, ck in (("--resume", args.resume), ("--init-params", args.init_params)):
        if ck:
            with np.load(ck, allow_pickle=False) as z:
                if "meta" not in z.files:
                    raise SystemExit(f"{ck} 沒有 meta：無法核架構，拒絕 {flag}")
                check_trunk(json.loads(str(z["meta"])), cfg, flag)

    if not args.manifest:
        raise SystemExit("缺 --manifest（shards_manifest.json）")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve()
    sha_map = manifest_sha_map(manifest_path)
    source_manifest_sha256 = sha256_file(manifest_path)  # 原 manifest 實算一次

    all_shards = [Path(q).resolve() for q in args.shard] + [
        Path(q).resolve() for q in args.dev_shard
    ]
    if args.diagnostic_sanitized:
        print("!! --diagnostic-sanitized：吃的是 sanitize 過的 d/*，產物不得當訓練結果", flush=True)
        overlays: list[Overlay | None] = [None] * len(all_shards)
    else:
        if len(args.overlay) != len(all_shards):
            raise SystemExit(
                f"--overlay 要給 {len(all_shards)} 個（train 再接 dev），得到 {len(args.overlay)}"
            )
        if not args.overlay_manifest:
            raise SystemExit("--overlay-manifest 必填（用來核 overlay 的 expected SHA）")
        mans = args.overlay_manifest
        if len(mans) == 1:
            mans = mans * len(all_shards)
        if len(mans) != len(all_shards):
            raise SystemExit("--overlay-manifest 要給 1 份或與 shard 等量")
        overlays = [
            Overlay(
                Path(o).resolve(),
                Path(m).resolve(),
                sha256_file(sh),
                audit_sidecar_sha256=sha_map[str(sh)]["sidecar_sha"],
                source_manifest_sha256=source_manifest_sha256,
            )
            for o, m, sh in zip(args.overlay, mans, all_shards, strict=True)
        ]

    trains, devs = [], []
    for k, path in enumerate(all_shards):
        sp = split_of(path)
        reader = ShardReader(
            path,
            split=sp,
            manifest=sha_map,
            overlay=overlays[k],
            allow_sanitized=args.diagnostic_sanitized,
        )
        if k < len(args.shard):
            if sp != args.split:
                raise SystemExit(f"{path} 是 {sp} split，與 --split {args.split} 不符（不得混用）")
            trains.append(reader)
        else:
            if sp == args.split:
                raise SystemExit(f"dev shard {path} 的 split 與 train 相同（{sp}）：拒絕混用")
            devs.append(reader)

    opt_cfg = {"lr": args.lr, "clip": args.clip}
    opt = make_opt(args.lr, args.clip)
    # source 身分在最開頭記一次（跑到一半改檔也不會改寫這份紀錄）
    source = source_identity(trains + devs)
    check_dirty(source["dirty_paths"])  # 依賴集合髒掉就拒（無關的髒檔照實記）
    base_meta = {**H.meta(cfg, seed=args.seed), "source": source}

    if args.resume:
        params, opt_state, ck_meta, train = load_train_state(
            Path(args.resume), opt, opt_cfg=opt_cfg, lr_from=migrate_lr[0] if migrate_lr else None
        )
        lr_rep = lr_migration_record(
            ck_meta,
            train,
            migrate_lr,
            from_ckpt_sha256=sha256_file(Path(args.resume)) if migrate_lr else None,
        )
        mig_map_path = Path(args.resume_migrate_data_map) if migrate_data else None
        id_rep = check_resume_identity(  # 資料身分不同就拒（git_head 不同可以）；遷移見函式說明
            ck_meta,
            source,
            migrate=migrate_data,
            migrate_map=json.loads(mig_map_path.read_text()) if mig_map_path else None,
            at_iter=int(train["iter"]),
            from_ckpt_sha256=sha256_file(Path(args.resume)) if migrate_data else None,
            map_sha256=sha256_file(mig_map_path) if mig_map_path else None,
        )
        if id_rep["data_migrations"]:
            base_meta = {**base_meta, "data_migrations": id_rep["data_migrations"]}
        if id_rep["migration"]:
            base_meta = {**base_meta, "data_migration_this_run": id_rep["migration"]}
        if lr_rep["lr_migrations"]:
            base_meta = {**base_meta, "lr_migrations": lr_rep["lr_migrations"]}
        if lr_rep["migration"]:
            base_meta = {**base_meta, "lr_migration_this_run": lr_rep["migration"]}
            lo, ln = migrate_lr
            print(f"LR_MIGRATE {lo:g}→{ln:g} at_iter={train['iter']}", flush=True)
        rng = restore_rng(train["rng"])
        start = int(train["iter"]) + 1
        if int(train["rows"]) != args.rows:
            raise SystemExit(f"resume 的 --rows 與 checkpoint 不同：{train['rows']} ≠ {args.rows}")
    else:
        params = H.init_params(cfg, seed=args.seed)
        if args.init_params:
            params = load_init_params(Path(args.init_params), args.expect_init_sha, params, cfg)
            base_meta = {
                **base_meta,
                "init_params": {"path": args.init_params, "sha256": args.expect_init_sha},
            }
        opt_state = opt.init(params)
        rng = np.random.default_rng(args.seed)
        start = 1
    step_fns = {seat: make_step(cfg, opt, seat) for seat in (0, 1)}
    eval_fns = {seat: make_eval(cfg, seat) for seat in (0, 1)}

    def train_meta(it: int, rng_state: dict[str, Any]) -> dict[str, Any]:
        # rng 由預取執行緒持有；存檔用該步抽樣當下的快照（見 draw_step），不讀 live rng
        return {
            "iter": it,
            "rng": rng_state,
            "opt": opt_cfg,
            "rows": args.rows,
            "seed": args.seed,
        }

    hist: list[dict[str, Any]] = []
    totals = dict.fromkeys(
        (
            "empty_unit",
            "empty_market",
            "n_unit",
            "n_market",
            "drop_unit_verb",
            "drop_unit_crop",
            "drop_unit_item",
            "drop_unit_qty",
            "drop_market_op",
            "drop_market_item",
            "drop_market_qty",
            # T3 I2：policy_active／有效 IL／分原因遮罩數
            "policy_active_unit",
            "policy_active_market",
            "il_mask_a5_units",
            "il_mask_pickup_empty_units",
            "il_mask_m2_orders",
            "il_mask_m2_unknown_orders",
        ),
        0,
    )
    by_op = np.zeros((H.N_UNIT_OPS,), np.int64)
    by_code = np.zeros((H.N_MARKET_OPS,), np.int64)
    t0 = time.time()
    it = start - 1
    rng_final = rng.bit_generator.state  # steps=0 時 final 照原版存當下 rng
    pf = Prefetcher(
        lambda i: draw_step(
            rng, trains, devs, i, args.rows, cfg["max_units"], args.dev_every, args.dev_batches
        ),
        start,
        start + args.steps - 1,
    )
    try:
        for it in range(start, start + args.steps):
            item, t_wait = pf.get(it)  # 取走第 it 步的同時已送出第 it+1 步
            reader, batch, keep = item["reader"], item["batch"], item["keep"]
            tu = time.time()
            params, opt_state, loss, stats = step_fns[batch["seat"]](
                params, opt_state, batch["state"], batch["labels"], keep
            )
            jax.block_until_ready(loss)  # 不 block 的話 t_update_s 只量到 dispatch
            row = {k: float(np.asarray(v)) for k, v in stats.items() if np.asarray(v).ndim == 0}
            row.update(
                {
                    "iter": it,
                    "shard": reader.path.name,
                    "seat": batch["seat"],
                    "n_captures": batch["n_captures_in_batch"],
                    "n_overlaid_slots": batch["n_overlaid_slots"],
                    # t_load_s：主迴圈等下一批就緒的時間（預取後）；
                    # t_load_bg_s：生產者備這批的實際時間（抽樣＋讀列＋keep）；
                    # t_rows_s：其中 `rows` 讀列（＝預取前 t_load_s 的口徑）
                    "t_load_s": round(t_wait, 4),
                    "t_update_s": round(time.time() - tu, 4),
                    "t_load_bg_s": round(item["t_bg"], 4),
                    "t_rows_s": round(batch["t_load_s"], 4),
                    "batch_idx_sha16": item["idx_sha16"],
                }
            )
            hist.append(row)
            for k in totals:
                totals[k] += int(row[k])
            by_op += np.asarray(stats["drop_unit_verb_by_op"], np.int64)
            by_code += np.asarray(stats["drop_market_op_by_code"], np.int64)
            if it % args.log_every == 0 or it == start:
                print(
                    f"iter {it:5d} loss {float(loss):.4f} unit {row['nll_unit']:.4f}"
                    f" market {row['nll_market']:.4f} n_u {int(row['n_unit'])}"
                    f" n_m {int(row['n_market'])}"
                    f" load {row['t_load_s']:.3f}s update {row['t_update_s']:.3f}s"
                    f" wall {time.time() - t0:.1f}s",
                    flush=True,
                )
            if args.ckpt_every and it % args.ckpt_every == 0:
                save_train_state(
                    out / f"actor_i{it:05d}.npz",
                    params,
                    opt_state,
                    base_meta,
                    train_meta(it, item["rng_after_train"]),
                )
            if item["dev"]:
                d_losses = [
                    float(eval_fns[db["seat"]](params, db["state"], db["labels"], dk))
                    for db, dk in item["dev"]
                ]
                hist[-1]["dev_loss"] = float(np.mean(d_losses))
                hist[-1]["t_dev_bg_s"] = round(item["t_dev_bg"], 4)
                print(f"  dev loss {hist[-1]['dev_loss']:.4f}", flush=True)
            rng_final = item["rng_after_step"]
    finally:
        pf.close()

    save_train_state(out / "final.npz", params, opt_state, base_meta, train_meta(it, rng_final))
    result = {
        "schema": FN.SCHEMA,
        "cfg": cfg,
        "source": source,
        "data_migrations": base_meta.get("data_migrations"),
        "data_migration_this_run": base_meta.get("data_migration_this_run"),
        "iters": [start, it],
        "rows_per_batch": args.rows,
        "opt": opt_cfg,
        "device": device_stats(),
        "wall_s": round(time.time() - t0, 1),
        "t_load_s_total": round(sum(h["t_load_s"] for h in hist), 2),
        "t_update_s_total": round(sum(h["t_update_s"] for h in hist), 2),
        "t_load_bg_s_total": round(sum(h["t_load_bg_s"] for h in hist), 2),
        "t_rows_s_total": round(sum(h["t_rows_s"] for h in hist), 2),
        "loader": {
            "prefetch": "single producer thread, depth 1",
            "t_load_s": "main-loop wait for next batch",
            "t_load_bg_s": "producer sample+rows+keep time",
            "t_rows_s": "reader.rows time (pre-prefetch t_load_s meaning)",
        },
        "train_shards": [r.path.name for r in trains],
        "dev_shards": [r.path.name for r in devs],
        "totals": totals,
        "teacher_drop_unit_verb_by_op": by_op.tolist(),
        "teacher_drop_market_op_by_code": by_code.tolist(),
        "history": hist,
    }
    for k in ("lr_migrations", "lr_migration_this_run"):  # 無 lr 遷移時不加欄位（result 逐位不變）
        if base_meta.get(k):
            result[k] = base_meta[k]
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({k: result[k] for k in ("wall_s", "iters", "totals")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
