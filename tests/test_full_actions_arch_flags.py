"""`scripts/full_actions_il_train.py` 的架構旗標（S1 d384 成本煙測前置）。


- d192（現況）／d384（L6／h6／ff1536）隨機權重（seed 0）：JAX 與 NumPy 部署路徑的 trunk 輸出與
  逐 263 token logp 逐元素比對，門檻沿用既有 d192 parity 測試（f32 ≤ 1e-4）。
- 不給旗標時 cfg 與 `H.config(CAPS)` 相同（架構 meta 欄相同）。
- 架構整數 ≤0、heads 不整除 d_model 拒；`--resume`／`--init-params` 的 checkpoint 架構
  與本次不符，在讀資料與建輸出目錄前拒（rc≠0）。d512 本檔不做 parity（煙測只量成本）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl import caps as caps_mod  # noqa: E402
from rl import full_actions_decode as D  # noqa: E402
from rl import full_actions_head as H  # noqa: E402
from rl import full_actions_numpy as FN  # noqa: E402
from rl import full_actions_obs as FO  # noqa: E402
from rl import state as S  # noqa: E402
from scripts import full_actions_il_train as IL  # noqa: E402

CAPS = caps_mod.BOUNDED_TRAINING
D384 = {"d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536}
ARCHS = {"d192": {}, "d384": D384}


def _np_state(st):
    from rl import loop4_numpy as LNP  # noqa: PLC0415

    return LNP.NpState(
        **{f: np.asarray(getattr(st, f))[0] for f in LNP.NpState.__slots__ if hasattr(st, f)}
    )


@pytest.fixture(scope="module", params=list(ARCHS), ids=list(ARCHS))
def arch_params(request):
    cfg = H.config(CAPS, **ARCHS[request.param])
    return request.param, cfg, H.init_params(cfg, seed=0)


def test_arch_cfg_and_param_shapes(arch_params):
    name, cfg, p = arch_params
    want = {**H.DEFAULT_ARCH, **ARCHS[name]}
    assert {k: cfg[k] for k in IL.ARCH_FLAGS} == want
    assert p["l0_ff1_w"].shape == (cfg["d_model"], cfg["d_ff"])
    assert f"l{cfg['n_layers'] - 1}_ff1_w" in p and f"l{cfg['n_layers']}_ff1_w" not in p


def test_numpy_jax_trunk_parity(arch_params):
    """trunk 輸出（unit／prod／glob）逐元素：f32 ≤ 1e-4。"""
    _name, cfg, p = arch_params
    st = S.reset(1, CAPS, seed=0)
    pnp = {k: np.asarray(v) for k, v in p.items()}
    hj = H.encode_tokens(p, FO.observation(st, 0, CAPS), cfg)
    hn = FN.encode_tokens(pnp, FN.observation(_np_state(st), CAPS), cfg)
    for k in ("unit", "prod", "glob"):
        a, b = np.asarray(hj[k])[0], np.asarray(hn[k])
        b = b[0] if b.ndim == a.ndim + 1 else b
        assert a.shape == b.shape, k
        assert np.max(np.abs(a - b)) < 1e-4, k


def test_numpy_jax_per_token_logp_parity_arch(arch_params):
    """greedy 決策八欄（含 market_qty_bytes／market_length）相同；
    263 槽 active 相同且 inactive 為 0；logp、value ≤ 1e-4（門檻同既有 d192 測試）。"""
    _name, cfg, p = arch_params
    st = S.reset(1, CAPS, seed=0)
    pnp = {k: np.asarray(v) for k, v in p.items()}
    got = FN.act(pnp, cfg, _np_state(st), CAPS, greedy=True)
    jx = D.decode("greedy")(p, cfg, st, 0, CAPS)
    for k in (
        "unit_verb",
        "unit_crop",
        "unit_item",
        "unit_qty_bytes",
        "market_op",
        "market_item",
        "market_qty_bytes",
    ):
        assert np.array_equal(np.asarray(got["decision"][k]), np.asarray(jx["decision"][k])[0]), k
    jx_len = int(np.asarray(jx["decision"]["market_length"])[0])
    assert int(got["decision"]["market_length"]) == jx_len
    assert got["logp"].shape == got["active"].shape == (263,) == (FN.N_LOGP_SLOTS,)
    assert np.asarray(jx["logp"]).shape[-1] == 263
    assert np.array_equal(got["active"], np.asarray(jx["active"])[0])
    assert (got["logp"][~got["active"]] == 0).all(), "inactive 位置不得有 logp"
    assert np.max(np.abs(got["logp"] - np.asarray(jx["logp"])[0])) < 1e-4
    assert abs(got["value"] - float(np.asarray(jx["value"])[0])) < 1e-4


def test_numpy_evaluate_scores_jax_sample_arch(arch_params):
    """JAX 抽樣的 decision 交給 NumPy `evaluate`，逐 token 分數 ≤ 1e-4。"""
    _name, cfg, p = arch_params
    st = S.reset(1, CAPS, seed=2)
    pnp = {k: np.asarray(v) for k, v in p.items()}
    jx = D.decode("sample")(p, cfg, st, 0, CAPS, key=jax.random.PRNGKey(7))
    dec = {k: np.asarray(v)[0] for k, v in jx["decision"].items() if k != "market_length"}
    got = FN.evaluate(pnp, cfg, _np_state(st), CAPS, dec)
    assert np.array_equal(got["active"], np.asarray(jx["active"])[0])
    assert np.max(np.abs(got["logp"] - np.asarray(jx["logp"])[0])) < 1e-4


def test_meta_roundtrip_carries_arch(arch_params, tmp_path):
    """旗標值進 `H.meta`：存檔再以 `load_params`／`config_from_meta` 讀回，四欄＝實際值。"""
    name, cfg, p = arch_params
    path = tmp_path / f"{name}.npz"
    pnp = {k: np.asarray(v) for k, v in p.items()}
    FN.save_params(path, pnp, H.meta(cfg, seed=0, params=pnp))
    _loaded, meta = FN.load_params(path, caps=CAPS)
    back = FN.config_from_meta(meta)
    assert {k: back[k] for k in IL.ARCH_FLAGS} == {k: cfg[k] for k in IL.ARCH_FLAGS}
    IL.check_trunk(meta, cfg, "--resume")  # 同架構不拒


def _ns(**kw):
    return argparse.Namespace(**{k: kw.get(k) for k in IL.ARCH_FLAGS})


def test_no_flags_is_default_config():
    assert IL.arch_config(_ns()) == H.config(CAPS)


def test_heads_not_dividing_rejected():
    with pytest.raises(SystemExit, match="不能整除"):
        IL.arch_config(_ns(d_model=384, n_heads=5))


@pytest.mark.parametrize("field", IL.ARCH_FLAGS)
@pytest.mark.parametrize("value", [0, -1])
def test_arch_ints_must_be_positive(field, value):
    with pytest.raises(SystemExit, match="> 0"):
        IL.arch_config(_ns(**{field: value}))


def test_check_trunk_rejects_each_field():
    cfg = H.config(CAPS, **D384)
    meta = H.meta(cfg, seed=0)
    for k in IL.ARCH_FLAGS:
        bad = json.loads(json.dumps(meta))
        bad["trunk"][k] = bad["trunk"][k] // 2
        with pytest.raises(SystemExit, match="拒收"):
            IL.check_trunk(bad, cfg, "--init-params")


def _run_trainer(tmp_path, *extra):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
    out = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/full_actions_il_train.py"),
        "--shard",
        str(tmp_path / "missing.npz"),
        "--out-dir",
        str(out),
        *extra,
    ]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    return r, out


@pytest.fixture(scope="module")
def d192_ckpt(tmp_path_factory):
    """只含 meta 的 d192 checkpoint：架構檢查在讀任何資料前就發生，只需要 meta。"""
    path = tmp_path_factory.mktemp("ck") / "d192.npz"
    np.savez(path, meta=np.asarray(json.dumps(H.meta(H.config(CAPS), seed=0))))
    return path


def test_resume_arch_mismatch_rc_nonzero(tmp_path, d192_ckpt):
    flags = ["--d-model", "384", "--n-layers", "6", "--d-ff", "1536"]
    r, out = _run_trainer(tmp_path, "--resume", str(d192_ckpt), *flags)
    assert r.returncode != 0
    assert "拒收" in r.stderr and "--resume" in r.stderr
    assert not out.exists(), "拒收時不得建輸出目錄"


def test_resume_heads_only_mismatch_rc_nonzero(tmp_path, d192_ckpt):
    """`n_heads` 不在任何 shape 裡：只改 heads 也必須拒。"""
    r, out = _run_trainer(tmp_path, "--resume", str(d192_ckpt), "--n-heads", "4")
    assert r.returncode != 0 and "拒收" in r.stderr
    assert not out.exists(), "拒收時不得建輸出目錄"


@pytest.fixture(scope="module")
def d192_init(tmp_path_factory):
    """合法的 d192 權重檔（`FN.save_params`＋完整 meta），給 `--init-params` 的真 CLI 測試。"""
    import hashlib  # noqa: PLC0415

    cfg = H.config(CAPS)
    p = {k: np.asarray(v) for k, v in H.init_params(cfg, seed=0).items()}
    path = tmp_path_factory.mktemp("init") / "d192_init.npz"
    FN.save_params(path, p, H.meta(cfg, seed=0, params=p))
    FN.load_params(path, caps=CAPS)  # 確認是合法 checkpoint
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "flags",
    [["--d-model", "384", "--n-layers", "6", "--d-ff", "1536"], ["--n-heads", "4"]],
    ids=["width", "heads_only"],
)
def test_init_params_arch_mismatch_rc_nonzero(tmp_path, d192_init, flags):
    """合法 checkpoint＋正確 SHA：架構不符在讀資料前拒（假 shard 不得搶先失敗），不建輸出目錄。"""
    path, sha = d192_init
    r, out = _run_trainer(tmp_path, "--init-params", str(path), "--expect-init-sha", sha, *flags)
    assert r.returncode != 0
    assert "拒收" in r.stderr and "--init-params" in r.stderr
    assert "missing.npz" not in r.stderr, "假 shard 不得先於架構檢查失敗"
    assert not out.exists(), "拒收時不得建輸出目錄"


def test_init_params_same_arch_passes_arch_check(tmp_path, d192_init):
    path, sha = d192_init
    r, _out = _run_trainer(tmp_path, "--init-params", str(path), "--expect-init-sha", sha)
    assert r.returncode != 0  # 之後因假 shard 失敗
    assert "拒收" not in r.stderr and "SHA 不符" not in r.stderr


def test_arch_int_nonpositive_rc_nonzero(tmp_path):
    r, out = _run_trainer(tmp_path, "--n-heads", "0")
    assert r.returncode != 0 and "> 0" in r.stderr and "ZeroDivisionError" not in r.stderr
    assert not out.exists()


def test_resume_same_arch_passes_arch_check(tmp_path, d192_ckpt):
    """不給旗標 resume d192：架構檢查放行（之後因假 shard 失敗，但不是架構拒收）。"""
    r, _out = _run_trainer(tmp_path, "--resume", str(d192_ckpt))
    assert r.returncode != 0
    assert "拒收" not in r.stderr and "不能整除" not in r.stderr


def test_heads_not_dividing_rc_nonzero(tmp_path):
    r, out = _run_trainer(tmp_path, "--d-model", "384", "--n-heads", "5")
    assert r.returncode != 0 and "不能整除" in r.stderr
    assert not out.exists()
