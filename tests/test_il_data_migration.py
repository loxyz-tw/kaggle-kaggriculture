"""IL trainer 資料遷移續訓（`--resume-migrate-data`；L8 用 09-27／28 新資料續訓）。

正反例：無旗標資料不同仍硬拒；FROM／TO 與對應表、checkpoint、本次逐項核；遷移歷史承接與竄改即拒；
只收一筆；CLI 在讀資料前就拒。
既有 resume 測試（test_il_loader_prefetch／test_full_actions_arch_flags）另跑。
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import full_actions_il_train as IL  # noqa: E402

OLD, NEW, NEW2 = "a" * 64, "b" * 64, "c" * 64


def _src(man: str, tag: str, n: int = 2) -> dict:
    def inp(i, split):
        return {
            "path": f"/x/{tag}{i}.npz",
            "split": split,
            "npz_sha256": f"{tag}{i}".ljust(64, "0"),
            "overlay": {
                "overlay_sha256": f"ov{tag}{i}".ljust(64, "1"),
                "overlay_manifest_sha256": f"om{tag}".ljust(64, "2"),
                "source_manifest_sha256": man,
            },
            "diagnostic_sanitized": False,
        }

    return {"inputs": [inp(i, "train") for i in range(n)] + [inp(9, "dev")]}


def _map(fr_src, to_src, fr=OLD, to=NEW):
    return {
        "schema": IL.DATA_MIGRATION_MAP_SCHEMA,
        "from_source_manifest_sha256": fr,
        "to_source_manifest_sha256": to,
        "from_data_identity": IL.data_identity(fr_src),
        "to_data_identity": IL.data_identity(to_src),
    }


def test_plain_resume_same_data_ok_and_different_data_rejected():
    s = _src(OLD, "o")
    rep = IL.check_resume_identity({"source": s}, copy.deepcopy(s))
    assert rep == {"data_migrations": None, "migration": None}
    with pytest.raises(ValueError, match="資料身分與 checkpoint 不同"):
        IL.check_resume_identity({"source": s}, _src(NEW, "n"))
    with pytest.raises(ValueError, match="inputs 不存在"):
        IL.check_resume_identity({}, s)


def test_parse_migrate_data():
    assert IL.parse_migrate_data(None, resume="r") is None
    assert IL.parse_migrate_data([f"{OLD}:{NEW}"], resume="r") == (OLD, NEW)
    for vals, resume in (
        ([f"{OLD}:{NEW}", f"{OLD}:{NEW}"], "r"),  # 只收一筆
        ([f"{OLD}:{NEW}"], None),  # 無 --resume
        ([f"{OLD}:{OLD}"], "r"),  # OLD＝NEW
        ([f"{OLD}"], "r"),  # 格式
        ([f"{OLD}:{NEW[:-1]}"], "r"),  # 長度
        ([f"{OLD.upper()}:{NEW}"], "r"),  # 非小寫 hex
    ):
        with pytest.raises(SystemExit):
            IL.parse_migrate_data(vals, resume=resume)


def test_migrate_positive_history_and_carry():
    old, new = _src(OLD, "o"), _src(NEW, "n", n=3)
    m = _map(old, new)
    rep = IL.check_resume_identity(
        {"source": old},
        new,
        migrate=(OLD, NEW),
        migrate_map=m,
        at_iter=128000,
        from_ckpt_sha256="d" * 64,
        map_sha256="e" * 64,
    )
    hist = rep["data_migrations"]
    assert len(hist) == 1 and hist[0]["from_source_manifest_sha256"] == OLD
    assert hist[0]["to_source_manifest_sha256"] == NEW and hist[0]["at_iter"] == 128000
    assert hist[0]["n_inputs"] == {"from": 3, "to": 4}
    assert rep["migration"]["allowed_migration"] == hist[0]
    # 新 checkpoint（meta 帶歷史）⇒ 同資料無旗標 resume 承接歷史
    ck2 = {"source": new, "data_migrations": hist}
    rep2 = IL.check_resume_identity(ck2, copy.deepcopy(new))
    assert rep2["data_migrations"] == hist and rep2["migration"] is None
    # 歷史被竄改（最後一筆 to 身分不符）⇒ 拒
    bad = copy.deepcopy(hist)
    bad[-1]["to_data_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data_migrations"):
        IL.check_resume_identity({"source": new, "data_migrations": bad}, new)
    # 無旗標、換到第三份資料 ⇒ 仍硬拒
    with pytest.raises(ValueError, match="資料身分與 checkpoint 不同"):
        IL.check_resume_identity(ck2, _src(NEW2, "z"))
    # 串接第二次遷移（NEW→NEW2）：FROM 必須＝上次的 TO，歷史累加
    new2 = _src(NEW2, "z")
    rep3 = IL.check_resume_identity(
        ck2, new2, migrate=(NEW, NEW2), migrate_map=_map(new, new2, NEW, NEW2)
    )
    assert len(rep3["data_migrations"]) == 2


def test_migrate_rejections():
    old, new = _src(OLD, "o"), _src(NEW, "n")
    m = _map(old, new)
    ok = {"migrate": (OLD, NEW), "migrate_map": m}
    IL.check_resume_identity({"source": old}, new, **ok)
    cases = [
        # 旗標 FROM 與對應表不符
        ({"source": old}, new, {"migrate": (NEW2, NEW), "migrate_map": m}, "FROM／TO"),
        # 對應表 schema 錯
        ({"source": old}, new, {**ok, "migrate_map": {**m, "schema": "x"}}, "schema"),
        # checkpoint 資料身分≠對應表 from（FROM 不符）
        ({"source": _src(OLD, "q")}, new, ok, "FROM 不符"),
        # 本次資料身分≠對應表 to
        ({"source": old}, _src(NEW, "q"), ok, "to_data_identity"),
    ]
    for ck, cur, kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            IL.check_resume_identity(ck, cur, **kw)
    # checkpoint 的 source manifest 不是 FROM（對應表也照它寫）⇒ 拒
    oth = _src(NEW2, "o")
    with pytest.raises(ValueError, match="不全是 FROM"):
        IL.check_resume_identity(
            {"source": oth}, new, migrate=(OLD, NEW), migrate_map=_map(oth, new)
        )
    # 本次 source manifest 不是 TO ⇒ 拒
    cur_bad = _src(NEW2, "n")
    with pytest.raises(ValueError, match="不全是 TO"):
        IL.check_resume_identity(
            {"source": old}, cur_bad, migrate=(OLD, NEW), migrate_map=_map(old, cur_bad)
        )
    # checkpoint 已有遷移歷史但最後一筆不是它自己的資料 ⇒ 拒
    with pytest.raises(ValueError, match="data_migrations"):
        IL.check_resume_identity(
            {"source": old, "data_migrations": [{"to_data_identity_sha256": "0" * 64}]},
            new,
            **ok,
        )


def _run(tmp_path, *extra):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts/full_actions_il_train.py"),
           "--shard", str(tmp_path / "missing.npz"), "--out-dir", str(out), *extra]  # fmt: skip
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300), out


def test_cli_refuses_before_reading_data(tmp_path):
    r, out = _run(tmp_path, "--resume-migrate-data", f"{OLD}:{NEW}")
    assert r.returncode != 0 and "--resume" in r.stderr and not out.exists()
    r, out = _run(
        tmp_path, "--resume", "x.npz", "--resume-migrate-data", f"{OLD}:{NEW}"
    )  # 缺對應表
    assert r.returncode != 0 and "--resume-migrate-data-map" in r.stderr and not out.exists()
    r, out = _run(
        tmp_path,
        "--resume",
        "x.npz",
        "--resume-migrate-data",
        f"{OLD}:{NEW}",
        "--resume-migrate-data",
        f"{OLD}:{NEW}",
        "--resume-migrate-data-map",
        "m.json",
    )
    assert r.returncode != 0 and "只准給一次" in r.stderr and not out.exists()
