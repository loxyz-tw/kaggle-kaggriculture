"""IL trainer lr 遷移續訓（`--resume-migrate-lr OLD:NEW`；L8 低 lr 續訓）。

正反例：旗標解析；checkpoint lr＝OLD 才放行、clip 仍須相同、無旗標換 lr 仍拒；
Adam count／mu／nu 逐葉原樣還原；新 lr 實際生效（同梯度下更新量＝NEW/OLD 倍、動量更新逐位相同）；
遷移歷史承接與竄改即拒；無旗標無歷史時不加任何欄位；CLI 在讀資料前就拒。
"""

from __future__ import annotations

import os
import subprocess
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
OLD, NEW = 3e-4, 3e-5


def _train(it=128000, lr=OLD):
    rng = np.random.default_rng(7)
    return {"iter": it, "rng": rng.bit_generator.state, "opt": {"lr": lr, "clip": 1.0},
            "rows": 64, "seed": 0}  # fmt: skip


def _grads(key, params):
    leaves, tdef = jax.tree.flatten(params)
    ks = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        tdef, [jax.random.normal(k, x.shape) * 1e-2 for k, x in zip(ks, leaves, strict=True)]
    )


def _flat(tree):
    return np.concatenate([np.ravel(x) for x in _leaves(tree)])


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """d192 隨機權重＋非零 Adam 狀態（對隨機梯度先走 3 步）的完整訓練 checkpoint，lr＝OLD。"""
    cfg = H.config(CAPS)
    params = H.init_params(cfg, seed=0)
    opt = IL.make_opt(OLD, 1.0)
    st = opt.init(params)
    key = jax.random.PRNGKey(0)
    for _ in range(3):
        key, sub = jax.random.split(key)
        g = _grads(sub, params)
        upd, st = opt.update(g, st, params)
        params = jax.tree.map(lambda p, u: p + u, params, upd)
    path = tmp_path_factory.mktemp("ck") / "ck.npz"
    IL.save_train_state(path, params, st, H.meta(cfg, seed=0), _train())
    return path, params, st


def _leaves(st):
    return [np.asarray(x) for x in jax.tree.leaves(st)]


def test_parse_migrate_lr():
    assert IL.parse_migrate_lr(None, resume="r", lr=NEW) is None
    assert IL.parse_migrate_lr([f"{OLD}:{NEW}"], resume="r", lr=NEW) == (OLD, NEW)
    for vals, resume, lr in (
        ([f"{OLD}:{NEW}", f"{OLD}:{NEW}"], "r", NEW),  # 只收一筆
        ([f"{OLD}:{NEW}"], None, NEW),  # 無 --resume
        ([f"{OLD}:{OLD}"], "r", OLD),  # OLD＝NEW
        ([f"{OLD}"], "r", NEW),  # 格式
        (["x:3e-5"], "r", NEW),  # 非數
        ([f"{OLD}:-3e-5"], "r", -3e-5),  # 非正
        ([f"{OLD}:nan"], "r", NEW),  # 非有限
        ([f"{OLD}:{NEW}"], "r", OLD),  # --lr ≠ NEW
    ):
        with pytest.raises(SystemExit):
            IL.parse_migrate_lr(vals, resume=resume, lr=lr)


def test_load_with_lr_from_restores_adam_exactly(ckpt):
    path, _p, st = ckpt
    opt_new = IL.make_opt(NEW, 1.0)
    params, st2, _meta, train = IL.load_train_state(
        path, opt_new, opt_cfg={"lr": NEW, "clip": 1.0}, lr_from=OLD
    )
    a, b = _leaves(st), _leaves(st2)
    assert len(a) == len(b)
    assert all(x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert int(train["iter"]) == 128000 and float(train["opt"]["lr"]) == OLD


def test_load_rejections(ckpt):
    path = ckpt[0]
    opt_new = IL.make_opt(NEW, 1.0)
    with pytest.raises(ValueError, match="optimizer 配置不同"):  # 無旗標換 lr 仍拒
        IL.load_train_state(path, opt_new, opt_cfg={"lr": NEW, "clip": 1.0})
    with pytest.raises(ValueError, match="OLD"):  # OLD 與 checkpoint 不符
        IL.load_train_state(path, opt_new, opt_cfg={"lr": NEW, "clip": 1.0}, lr_from=1e-3)
    with pytest.raises(ValueError, match="clip"):  # clip 不同仍拒
        IL.load_train_state(
            path, IL.make_opt(NEW, 0.5), opt_cfg={"lr": NEW, "clip": 0.5}, lr_from=OLD
        )
    # 無旗標、同 lr：照舊放行（既有行為）
    IL.load_train_state(path, IL.make_opt(OLD, 1.0), opt_cfg={"lr": OLD, "clip": 1.0})


def test_new_lr_actually_applies(ckpt):
    """同一狀態、同一梯度：NEW 的更新量＝OLD 的 NEW/OLD 倍；count／mu／nu 更新逐位相同。"""
    path, _p, _st = ckpt
    o_new, o_old = IL.make_opt(NEW, 1.0), IL.make_opt(OLD, 1.0)
    p_n, s_n, *_ = IL.load_train_state(path, o_new, opt_cfg={"lr": NEW, "clip": 1.0}, lr_from=OLD)
    p_o, s_o, *_ = IL.load_train_state(path, o_old, opt_cfg={"lr": OLD, "clip": 1.0})
    g = _grads(jax.random.PRNGKey(1), p_n)
    u_n, s_n2 = o_new.update(g, s_n, p_n)
    u_o, s_o2 = o_old.update(g, s_o, p_o)
    assert all(np.array_equal(x, y) for x, y in zip(_leaves(s_n2), _leaves(s_o2), strict=True))
    un, uo = _flat(u_n), _flat(u_o)
    assert np.max(np.abs(uo)) > 0
    np.testing.assert_allclose(un, uo * (NEW / OLD), rtol=1e-5, atol=1e-12)


def test_history_record_carry_and_tamper():
    tr = _train()
    rep = IL.lr_migration_record({}, tr, (OLD, NEW), from_ckpt_sha256="d" * 64)
    h = rep["lr_migrations"]
    assert len(h) == 1 and rep["migration"] == h[0]
    assert h[0]["from_lr"] == OLD and h[0]["to_lr"] == NEW
    assert h[0]["at_iter"] == 128000 and h[0]["effective_from_iter"] == 128001
    # 新 checkpoint（lr＝NEW、meta 帶歷史）⇒ 無旗標 resume 承接
    tr2 = _train(132000, NEW)
    rep2 = IL.lr_migration_record({"lr_migrations": h}, tr2, None)
    assert rep2["lr_migrations"] == h and rep2["migration"] is None
    # 歷史與 checkpoint lr 不連續 ⇒ 拒
    with pytest.raises(ValueError, match="lr_migrations"):
        IL.lr_migration_record({"lr_migrations": h}, _train(132000, OLD), None)
    # 無歷史、無旗標 ⇒ None（meta／result 不加欄位）
    assert IL.lr_migration_record({}, tr, None) == {"lr_migrations": None, "migration": None}


def _run(tmp_path, *extra):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts/full_actions_il_train.py"),
           "--shard", str(tmp_path / "missing.npz"), "--out-dir", str(out), *extra]  # fmt: skip
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300), out


def test_cli_refuses_before_reading_data(tmp_path):
    r, out = _run(tmp_path, "--lr", str(NEW), "--resume-migrate-lr", f"{OLD}:{NEW}")
    assert r.returncode != 0 and "--resume" in r.stderr and not out.exists()
    # --lr 預設 3e-4 ≠ NEW
    r, out = _run(tmp_path, "--resume", "x.npz", "--resume-migrate-lr", f"{OLD}:{NEW}")
    assert r.returncode != 0 and "--lr" in r.stderr and not out.exists()
