"""IL trainer 第二次資料遷移（LRM3：起點已帶 data＋lr 遷移各一筆，再遷移資料一次）。

預登記 內部規格文件 v1.1 §1–§2：自 LRM `actor_i132000`
（data_migrations 一筆 8d24→3892、lr_migrations 一筆 3e-4→3e-5、兩個 this_run 都在 meta）
以**只帶資料遷移、不帶 lr 遷移旗標**續訓。驗：資料歷史變兩筆且第一筆原樣、新一筆 FROM／TO／
at_iter／from_ckpt 正確且與第一筆首尾相接；lr 歷史仍一筆、無 lr this_run；Adam／RNG／iteration／
rows／clip 在遷移瞬間原樣承接。反例：FROM 不符、TO 不符、歷史末筆與 checkpoint 資料身分不符即拒。
trainer 本身未改（de81e995）；此處照 main() 的呼叫順序直接呼叫函式。
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl import caps as caps_mod  # noqa: E402
from rl import full_actions_head as H  # noqa: E402
from scripts import full_actions_il_train as IL  # noqa: E402

CAPS = caps_mod.BOUNDED_TRAINING
OLD_LR, NEW_LR = 3e-4, 3e-5
M1, M2, M3 = "a" * 64, "b" * 64, "c" * 64  # 11 片 → 合併 16 片 → 合併＋09-29
IT1, IT2 = 128000, 132000


def _src(man: str, tag: str, n: int) -> dict:
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

    return {"inputs": [inp(i, "train") for i in range(n)] + [inp(99, "dev")]}


SRC1, SRC2, SRC3 = _src(M1, "o", 10), _src(M2, "n", 15), _src(M3, "t", 16)


def _map(fr, to, sfr, sto):
    return {
        "schema": IL.DATA_MIGRATION_MAP_SCHEMA,
        "from_source_manifest_sha256": fr,
        "to_source_manifest_sha256": to,
        "from_data_identity": IL.data_identity(sfr),
        "to_data_identity": IL.data_identity(sto),
    }


MAP12, MAP23 = _map(M1, M2, SRC1, SRC2), _map(M2, M3, SRC2, SRC3)


def _leaves(t):
    return [np.asarray(x) for x in jax.tree.leaves(t)]


def _grads(key, params):
    leaves, tdef = jax.tree.flatten(params)
    ks = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        tdef, [jax.random.normal(k, x.shape) * 1e-2 for k, x in zip(ks, leaves, strict=True)]
    )


@pytest.fixture(scope="module")
def lrm_ckpt(tmp_path_factory):
    """仿 LRM actor_i132000：M2 資料、lr NEW、iter 132000、兩份遷移歷史各一筆、this_run 在 meta。

    先建 128k 起點（M1、lr OLD），照 main() 順序做 LRM 的同段兩遷移拿到兩份歷史，
    再以非零 Adam、前進過的 RNG 存成 132k checkpoint。
    """
    d = tmp_path_factory.mktemp("ck")
    cfg = H.config(CAPS)
    params = H.init_params(cfg, seed=0)
    opt = IL.make_opt(OLD_LR, 1.0)
    st = opt.init(params)
    key = jax.random.PRNGKey(0)
    for _ in range(3):
        key, sub = jax.random.split(key)
        upd, st = opt.update(_grads(sub, params), st, params)
        params = jax.tree.map(lambda p, u: p + u, params, upd)
    rng = np.random.default_rng(7)
    rng.integers(0, 1000, size=17)
    t1 = {"iter": IT1, "rng": rng.bit_generator.state, "opt": {"lr": OLD_LR, "clip": 1.0},
          "rows": 64, "seed": 0}  # fmt: skip
    p1 = d / "leg32.npz"
    IL.save_train_state(p1, params, st, {**H.meta(cfg, seed=0), "source": SRC1}, t1)
    sha1 = IL.sha256_file(p1)
    _p, st1, meta1, train1 = IL.load_train_state(
        p1, IL.make_opt(NEW_LR, 1.0), opt_cfg={"lr": NEW_LR, "clip": 1.0}, lr_from=OLD_LR
    )
    lr_rep = IL.lr_migration_record(meta1, train1, (OLD_LR, NEW_LR), from_ckpt_sha256=sha1)
    id_rep = IL.check_resume_identity(
        meta1, SRC2, migrate=(M1, M2), migrate_map=MAP12, at_iter=IT1,
        from_ckpt_sha256=sha1, map_sha256="7" * 64,
    )  # fmt: skip
    meta2 = {**H.meta(cfg, seed=0), "source": SRC2,
             "data_migrations": id_rep["data_migrations"],
             "data_migration_this_run": id_rep["migration"],
             "lr_migrations": lr_rep["lr_migrations"],
             "lr_migration_this_run": lr_rep["migration"]}  # fmt: skip
    rng.integers(0, 1000, size=5)
    t2 = {**t1, "iter": IT2, "rng": rng.bit_generator.state, "opt": {"lr": NEW_LR, "clip": 1.0}}
    p2 = d / "actor_i132000.npz"
    IL.save_train_state(p2, params, st1, meta2, t2)
    return p2, st1, t2, meta2


def _resume_data_only(path, *, migrate=(M2, M3), mmap=MAP23, src=SRC3):
    """照 main() 順序、只帶資料遷移：load（lr_from=None）→ lr 紀錄（migrate=None）→ 資料遷移。"""
    opt = IL.make_opt(NEW_LR, 1.0)
    params, st, meta, train = IL.load_train_state(path, opt, opt_cfg={"lr": NEW_LR, "clip": 1.0})
    sha = IL.sha256_file(path)
    lr_rep = IL.lr_migration_record(meta, train, None)
    id_rep = IL.check_resume_identity(
        meta, src, migrate=migrate, migrate_map=mmap, at_iter=int(train["iter"]),
        from_ckpt_sha256=sha, map_sha256="9" * 64,
    )  # fmt: skip
    return st, meta, train, lr_rep, id_rep


def test_second_migration_histories(lrm_ckpt):
    path, _st0, _t0, meta0 = lrm_ckpt
    _st, _meta, _train, lr_rep, id_rep = _resume_data_only(path)
    dm0, dm1 = id_rep["data_migrations"]  # 恰兩筆
    assert dm0 == meta0["data_migrations"][0]  # 第一筆原樣
    assert dm1["from_source_manifest_sha256"] == M2 and dm1["to_source_manifest_sha256"] == M3
    assert dm1["at_iter"] == IT2 and dm1["n_inputs"] == {"from": 16, "to": 17}
    assert dm1["from_ckpt_sha256"] == IL.sha256_file(path) and dm1["map_sha256"] == "9" * 64
    assert dm1["from_data_identity_sha256"] == dm0["to_data_identity_sha256"]  # 首尾相接
    assert id_rep["migration"]["allowed_migration"] == dm1  # data this_run＝新一筆
    assert lr_rep["lr_migrations"] == meta0["lr_migrations"] and len(lr_rep["lr_migrations"]) == 1
    assert lr_rep["migration"] is None  # 不帶 lr 旗標 ⇒ 無 lr this_run


def test_second_migration_state_carried(lrm_ckpt):
    """遷移瞬間：Adam 逐位、RNG、iteration 132000、rows、opt（lr 3e-5／clip 1.0）原樣。"""
    path, st0, t0, _m = lrm_ckpt
    st, _meta, train, _l, _i = _resume_data_only(path)
    assert all(np.array_equal(a, b) for a, b in zip(_leaves(st0), _leaves(st), strict=True))
    assert train["iter"] == IT2 and train["rows"] == 64
    assert train["opt"] == {"lr": NEW_LR, "clip": 1.0}
    assert IL.restore_rng(train["rng"]).bit_generator.state == t0["rng"]


def test_third_resume_carries_two_entries(lrm_ckpt):
    """遷移後的 checkpoint 再無旗標 resume：兩筆資料歷史＋一筆 lr 歷史原樣承接；回 M2 資料即拒。"""
    _st, meta, train, lr_rep, id_rep = _resume_data_only(lrm_ckpt[0])
    meta3 = {**meta, "source": SRC3, "data_migrations": id_rep["data_migrations"],
             "lr_migrations": lr_rep["lr_migrations"]}  # fmt: skip
    for k in ("data_migration_this_run", "lr_migration_this_run"):
        meta3.pop(k, None)
    rep = IL.check_resume_identity(meta3, copy.deepcopy(SRC3))
    assert rep["data_migrations"] == id_rep["data_migrations"] and rep["migration"] is None
    t3 = {**train, "iter": IT2 + 4000}
    assert IL.lr_migration_record(meta3, t3, None)["lr_migrations"] == lr_rep["lr_migrations"]
    with pytest.raises(ValueError, match="資料身分與 checkpoint 不同"):
        IL.check_resume_identity(meta3, SRC2)


def test_second_migration_rejects_from_mismatch(lrm_ckpt):
    path = lrm_ckpt[0]
    with pytest.raises(ValueError, match="FROM／TO"):  # 旗標 FROM 與對應表不符
        _resume_data_only(path, migrate=(M1, M3))
    bad = _map(M1, M3, SRC1, SRC3)  # 拿舊 11 片身分當 FROM（checkpoint 實為 M2）
    with pytest.raises(ValueError, match="FROM 不符"):
        _resume_data_only(path, migrate=(M1, M3), mmap=bad)


def test_second_migration_rejects_to_mismatch(lrm_ckpt):
    path = lrm_ckpt[0]
    with pytest.raises(ValueError, match="FROM／TO"):  # 旗標 TO 與對應表不符
        _resume_data_only(path, migrate=(M2, "d" * 64))
    with pytest.raises(ValueError, match="to_data_identity"):  # 本次資料≠對應表 TO
        _resume_data_only(path, src=_src(M3, "q", 16))


def test_second_migration_rejects_tampered_last_entry(lrm_ckpt, tmp_path):
    path, st0, t0, meta0 = lrm_ckpt
    opt = IL.make_opt(NEW_LR, 1.0)
    params, st, _meta, _t = IL.load_train_state(path, opt, opt_cfg={"lr": NEW_LR, "clip": 1.0})
    bad = copy.deepcopy(meta0)
    bad["data_migrations"][-1]["to_data_identity_sha256"] = "0" * 64
    p = tmp_path / "tampered.npz"
    IL.save_train_state(p, params, st, bad, t0)
    with pytest.raises(ValueError, match="最後一筆與其資料身分不符"):
        _resume_data_only(p)
