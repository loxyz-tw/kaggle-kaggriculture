"""T3 PPO 穩定化的 tape 混池（計畫 內部規格文件 §4）。

兩個既有 bank（新 B′ 與舊 5 家族）**原檔不改寫**：本模組只做索引與 manifest 包裝，
執行期把兩者的陣列在記憶體裡接成一個 ``TapeBank``（新池在前、舊池在後，以 offset 定位）。

抽樣規則（固定，進 resume 身分）：

- global lane ``g = iteration × B + k``（``k`` 是 batch 內 env 序號，含最後的教師列）。
- ``g % 4 == 3`` → 舊池，其餘 → 新池（3:1）。
- 池內用 ``np.random.default_rng(perm_seed)`` 對該池列號做一次凍結排列，按「該池此前已分到的
  lane 數」循環取：舊池第 ``g // 4`` 次、新池第 ``g − g // 4`` 次。
- 兩個 learner 座用同一張索引表（同一個 ``k`` 對到同一條 tape）。

只當本地訓練對手；不進 checkpoint、不進提交包。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "t3-tape-pool-v1"
SAMPLING = "t3-pool-mod4-3new1old-perm-v1"
FIELDS = ("unit_op", "unit_arg", "unit_qty", "order_op", "order_item", "order_qty")
POOL_NAMES = ("new", "old")
PERIOD = 4
OLD_RESIDUE = 3


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pool_of_lane(g: np.ndarray) -> np.ndarray:
    """0 ＝ 新池、1 ＝ 舊池。"""
    return (np.asarray(g) % PERIOD == OLD_RESIDUE).astype(np.int64)


def ordinal_in_pool(g: np.ndarray) -> np.ndarray:
    """lane ``g`` 是其所屬池的第幾次抽取（0 起算）。"""
    g = np.asarray(g, np.int64)
    n_old_before = g // PERIOD  # [0, g) 內 ≡3 (mod 4) 的個數
    return np.where(pool_of_lane(g) == 1, n_old_before, g - n_old_before)


def lane_rows(iteration: int, b: int, perms: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """本輪 ``b`` 個 env 的 (池, 池內列號)。``perms`` 是兩池的凍結排列。"""
    if iteration < 0 or b <= 0:
        raise ValueError(f"iteration／b 不合法：{iteration}, {b}")
    g = iteration * b + np.arange(b, dtype=np.int64)
    pool = pool_of_lane(g)
    ordn = ordinal_in_pool(g)
    row = np.empty(b, np.int64)
    for pid, name in enumerate(POOL_NAMES):
        m = pool == pid
        perm = perms[name]
        row[m] = perm[ordn[m] % perm.size]
    return {"lane": g, "pool": pool, "row": row}


def frozen_perms(n_new: int, n_old: int, perm_seed: int) -> dict[str, np.ndarray]:
    """每池各一個 ``default_rng(perm_seed)`` 排列（兩池各自新建 Generator，互不牽動）。"""
    return {
        "new": np.random.default_rng(perm_seed).permutation(n_new),
        "old": np.random.default_rng(perm_seed).permutation(n_old),
    }


def _load_bank_rows(bank_path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    man = json.loads(Path(manifest_path).read_text())
    z = np.load(bank_path)
    src = [str(s) for s in z["source"]]
    tapes = man["tapes"]
    if [t["source"] for t in tapes] != src:
        raise ValueError(f"bank 的 source 與 manifest 不一致：{bank_path}")
    return tapes, {"n_tapes": len(src), "caps": man.get("caps")}


def build_pool_manifest(
    *,
    new_bank: Path,
    new_manifest: Path,
    old_bank: Path,
    old_manifest: Path,
    perm_seed: int = 0,
) -> dict[str, Any]:
    """讀兩池（不改寫），核 SHA 與 manifest 一致、排除 lossy，寫出混池身分。"""
    pools = {}
    for name, bank, manp in (("new", new_bank, new_manifest), ("old", old_bank, old_manifest)):
        bank, manp = Path(bank), Path(manp)
        man = json.loads(manp.read_text())
        sha = sha256_file(bank)
        if man.get("sha256") != sha:
            raise ValueError(f"{name} bank SHA {sha} ≠ manifest {man.get('sha256')}")
        tapes, info = _load_bank_rows(bank, manp)
        lossy = [t["source"] for t in tapes if t.get("lossy")]
        if lossy:
            raise ValueError(
                f"{name} 池含 lossy 席次 {len(lossy)} 條：預設池不收（T1 規格第 3 點）"
            )
        pools[name] = {
            "bank": str(bank.resolve()),
            "sha256": sha,
            "manifest": str(manp.resolve()),
            "manifest_sha256": sha256_file(manp),
            "n_tapes": info["n_tapes"],
            "caps": info["caps"],
            "sources": [t["source"] for t in tapes],
            "episode_seat": [
                [t.get("episodeId"), t.get("seat")] if "episodeId" in t else None for t in tapes
            ],
            "version_counts": _count(t.get("version", "n/a") for t in tapes),
        }
    if pools["new"]["caps"] != pools["old"]["caps"]:
        raise ValueError(f"兩池 caps 不同：{pools['new']['caps']} ≠ {pools['old']['caps']}")
    dup = sorted(set(pools["new"]["sources"]) & set(pools["old"]["sources"]))
    return {
        "schema": SCHEMA,
        "sampling": SAMPLING,
        "rule": (
            "g = iteration×B + k；g%4==3 → old，其餘 → new；池內 default_rng(perm_seed) 凍結排列，"
            "old 取第 g//4 次、new 取第 g−g//4 次（循環）；兩座同一張表"
        ),
        "weights": {"new": 0.75, "old": 0.25},
        "perm_seed": perm_seed,
        "concat_order": list(POOL_NAMES),
        "offsets": {"new": 0, "old": pools["new"]["n_tapes"]},
        "n_tapes_total": pools["new"]["n_tapes"] + pools["old"]["n_tapes"],
        "caps": pools["new"]["caps"],
        "duplicates_new_old": dup,
        "pools": pools,
        "note": (
            "原 bank 不改寫；執行期在記憶體接成一個 TapeBank。不為任何隊／家族設收錄上限。"
            "B′ 席次為 09-23／24 每日包內版本（version=unknown），不保證現役。"
        ),
    }


def _count(it) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in it:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def load_pool(manifest_path: Path, expect_sha256: str):
    """讀混池 manifest（核 SHA），重核兩池 bank SHA，回（拼接後的 numpy 陣列, manifest）。"""
    manifest_path = Path(manifest_path)
    got = sha256_file(manifest_path)
    if got != expect_sha256:
        raise ValueError(f"pool manifest SHA 不符：期望 {expect_sha256}、實算 {got}")
    man = json.loads(manifest_path.read_text())
    if man.get("schema") != SCHEMA or man.get("sampling") != SAMPLING:
        raise ValueError(f"pool manifest schema／sampling 不符：{man.get('schema')}")
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in FIELDS}
    for name in man["concat_order"]:
        p = man["pools"][name]
        if sha256_file(Path(p["bank"])) != p["sha256"]:
            raise ValueError(f"{name} bank SHA 已變：{p['bank']}")
        z = np.load(p["bank"])
        if [str(s) for s in z["source"]] != p["sources"]:
            raise ValueError(f"{name} bank source 與 pool manifest 不一致")
        for k in FIELDS:
            arrays[k].append(np.asarray(z[k]))
    cat = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    if cat["unit_op"].shape[0] != man["n_tapes_total"]:
        raise ValueError("拼接後列數與 manifest 不符")
    return cat, man


def global_tape_idx(man: dict[str, Any], iteration: int, b: int) -> dict[str, np.ndarray]:
    """本輪 env → 拼接後 bank 的列號（含池別與池內列號，供記帳）。"""
    perms = frozen_perms(
        man["pools"]["new"]["n_tapes"], man["pools"]["old"]["n_tapes"], man["perm_seed"]
    )
    lr = lane_rows(iteration, b, perms)
    offs = np.array([man["offsets"][n] for n in POOL_NAMES], np.int64)
    return {**lr, "tape_idx": offs[lr["pool"]] + lr["row"]}


def iteration_record(man: dict[str, Any], iteration: int, b: int) -> dict[str, Any]:
    """每輪實際比例與 (episode, seat)：寫進 rollout 報告。"""
    ix = global_tape_idx(man, iteration, b)
    rows = []
    for k in range(b):
        name = POOL_NAMES[int(ix["pool"][k])]
        r = int(ix["row"][k])
        rows.append(
            {
                "env": k,
                "lane": int(ix["lane"][k]),
                "pool": name,
                "pool_row": r,
                "tape_idx": int(ix["tape_idx"][k]),
                "source": man["pools"][name]["sources"][r],
                "episode_seat": man["pools"][name]["episode_seat"][r],
            }
        )
    n_old = int(ix["pool"].sum())
    return {
        "sampling": man["sampling"],
        "n_new": b - n_old,
        "n_old": n_old,
        "frac_new": (b - n_old) / b,
        "lanes": rows,
    }
