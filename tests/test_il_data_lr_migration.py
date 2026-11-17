"""IL trainer 資料遷移＋lr 遷移同段（`--resume-migrate-data` ＋ `--resume-migrate-lr`；L8 LRM）。

預登記 內部規格文件 v2 §2、§6：同一次 resume 同時換資料與 lr 時，
Adam（count／mu／nu）逐位沿用、新 lr 生效、兩份遷移歷史各自正確並承接、RNG／iteration 承接、
資料 FROM→TO 綁定、clip 與 rows 契約不因旗標放寬。trainer 本身未改（主流程兩段檢查彼此獨立）。
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
OLD_M, NEW_M = "a" * 64, "b" * 64
IT = 128000


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

    return {"inputs": [inp(i, "train") for i in range(n)] + [inp(9, "dev")]}


OLD_SRC, NEW_SRC = _src(OLD_M, "o", 10), _src(NEW_M, "n", 15)
MAP = {
    "schema": IL.DATA_MIGRATION_MAP_SCHEMA,
    "from_source_manifest_sha256": OLD_M,
    "to_source_manifest_sha256": NEW_M,
    "from_data_identity": IL.data_identity(OLD_SRC),
    "to_data_identity": IL.data_identity(NEW_SRC),
}


def _leaves(t):
    return [np.asarray(x) for x in jax.tree.leaves(t)]


def _grads(key, params):
    leaves, tdef = jax.tree.flatten(params)
    ks = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        tdef, [jax.random.normal(k, x.shape) * 1e-2 for k, x in zip(ks, leaves, strict=True)]
    )


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """d192＋非零 Adam（3 步）＋舊資料身分＋lr OLD＋rows 64＋前進過的 RNG 的完整 checkpoint。"""
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
    rng.integers(0, 1000, size=17)  # RNG 狀態非初始
    train = {"iter": IT, "rng": rng.bit_generator.state, "opt": {"lr": OLD_LR, "clip": 1.0},
             "rows": 64, "seed": 0}  # fmt: skip
    path = tmp_path_factory.mktemp("ck") / "ck.npz"
    IL.save_train_state(path, params, st, {**H.meta(cfg, seed=0), "source": OLD_SRC}, train)
    return path, params, st, train


def _resume_both(path):
    """照 main() 的順序：load（lr_from）→ lr 遷移紀錄 → 資料身分／遷移。"""
    opt = IL.make_opt(NEW_LR, 1.0)
    params, st, meta, train = IL.load_train_state(
        path, opt, opt_cfg={"lr": NEW_LR, "clip": 1.0}, lr_from=OLD_LR
    )
    sha = IL.sha256_file(path)
    lr_rep = IL.lr_migration_record(meta, train, (OLD_LR, NEW_LR), from_ckpt_sha256=sha)
    id_rep = IL.check_resume_identity(
        meta, NEW_SRC, migrate=(OLD_M, NEW_M), migrate_map=MAP, at_iter=int(train["iter"]),
        from_ckpt_sha256=sha, map_sha256="e" * 64,
    )  # fmt: skip
    return opt, params, st, meta, train, lr_rep, id_rep


def test_both_adam_rng_iter_and_histories(ckpt):
    path, _p, st0, train0 = ckpt
    _opt, _params, st, _meta, train, lr_rep, id_rep = _resume_both(path)
    assert all(np.array_equal(a, b) for a, b in zip(_leaves(st0), _leaves(st), strict=True))
    assert train["iter"] == IT and train["rows"] == 64
    assert IL.restore_rng(train["rng"]).bit_generator.state == train0["rng"]  # RNG 原樣承接
    (lm,) = lr_rep["lr_migrations"]
    assert (lm["from_lr"], lm["to_lr"], lm["at_iter"], lm["effective_from_iter"]) == (
        OLD_LR, NEW_LR, IT, IT + 1,
    )  # fmt: skip
    (dm,) = id_rep["data_migrations"]
    assert dm["from_source_manifest_sha256"] == OLD_M and dm["to_source_manifest_sha256"] == NEW_M
    assert dm["at_iter"] == IT and dm["n_inputs"] == {"from": 11, "to": 16}
    assert dm["from_ckpt_sha256"] == lm["from_ckpt_sha256"] == IL.sha256_file(path)


def test_both_new_lr_applies_same_as_lr_only(ckpt):
    """同段兩旗標與只換 lr 對同一梯度：更新量逐位相同、動量更新逐位相同；對 OLD lr 為 0.1 倍。"""
    path = ckpt[0]
    opt, params, st, *_ = _resume_both(path)
    o_lr = IL.make_opt(NEW_LR, 1.0)  # 只換 lr（LR 的路徑）
    p_lr, s_lr, *_ = IL.load_train_state(
        path, o_lr, opt_cfg={"lr": NEW_LR, "clip": 1.0}, lr_from=OLD_LR
    )
    o_old = IL.make_opt(OLD_LR, 1.0)
    p_o, s_o, *_ = IL.load_train_state(path, o_old, opt_cfg={"lr": OLD_LR, "clip": 1.0})
    g = _grads(jax.random.PRNGKey(1), params)
    u, s2 = opt.update(g, st, params)
    u_lr, s_lr2 = o_lr.update(g, s_lr, p_lr)
    u_o, s_o2 = o_old.update(g, s_o, p_o)
    assert all(np.array_equal(a, b) for a, b in zip(_leaves(u), _leaves(u_lr), strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(_leaves(s2), _leaves(s_o2), strict=True))
    flat = lambda t: np.concatenate([np.ravel(x) for x in _leaves(t)])  # noqa: E731
    np.testing.assert_allclose(flat(u), flat(u_o) * (NEW_LR / OLD_LR), rtol=1e-5, atol=1e-12)


def test_both_histories_carry_and_tamper(ckpt):
    """新 checkpoint（新資料、lr NEW、兩份歷史）⇒ 無旗標 resume 兩份都承接；任一份竄改即拒。"""
    _o, _p, _s, meta, train, lr_rep, id_rep = _resume_both(ckpt[0])
    meta2 = {**meta, "source": NEW_SRC, "data_migrations": id_rep["data_migrations"],
             "lr_migrations": lr_rep["lr_migrations"]}  # fmt: skip
    train2 = {**train, "iter": IT + 4000, "opt": {"lr": NEW_LR, "clip": 1.0}}
    assert IL.lr_migration_record(meta2, train2, None)["lr_migrations"] == lr_rep["lr_migrations"]
    rep = IL.check_resume_identity(meta2, copy.deepcopy(NEW_SRC))
    assert rep["data_migrations"] == id_rep["data_migrations"]
    bad = copy.deepcopy(meta2)
    bad["data_migrations"][-1]["to_data_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data_migrations"):
        IL.check_resume_identity(bad, NEW_SRC)
    with pytest.raises(ValueError, match="lr_migrations"):  # lr 歷史與 checkpoint lr 不連續
        IL.lr_migration_record(meta2, {**train2, "opt": {"lr": OLD_LR, "clip": 1.0}}, None)
    with pytest.raises(ValueError, match="資料身分與 checkpoint 不同"):  # 無旗標回舊資料仍拒
        IL.check_resume_identity(meta2, OLD_SRC)


def test_both_contract_rejections(ckpt):
    path = ckpt[0]
    with pytest.raises(ValueError, match="clip"):  # clip 不因兩旗標放寬
        IL.load_train_state(
            path, IL.make_opt(NEW_LR, 0.5), opt_cfg={"lr": NEW_LR, "clip": 0.5}, lr_from=OLD_LR
        )
    with pytest.raises(ValueError, match="optimizer 配置不同"):  # 只帶資料遷移、不帶 lr 遷移就換 lr
        IL.load_train_state(path, IL.make_opt(NEW_LR, 1.0), opt_cfg={"lr": NEW_LR, "clip": 1.0})
    meta = IL.load_train_state(
        path, IL.make_opt(NEW_LR, 1.0), opt_cfg={"lr": NEW_LR, "clip": 1.0}, lr_from=OLD_LR
    )[2]
    with pytest.raises(ValueError, match="FROM／TO"):  # 資料 FROM 錯
        IL.check_resume_identity(meta, NEW_SRC, migrate=("c" * 64, NEW_M), migrate_map=MAP)
    with pytest.raises(ValueError, match="to_data_identity"):  # 本次資料≠對應表 TO
        IL.check_resume_identity(
            meta, _src(NEW_M, "q", 15), migrate=(OLD_M, NEW_M), migrate_map=MAP
        )
