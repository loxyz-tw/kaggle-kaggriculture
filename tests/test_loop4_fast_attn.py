"""`rl/loop4_model.py` 的訓練期快速前向路徑（任務 25 第 4 點）。

`fast="f32"`／`"bf16"` 是**訓練期**的路徑：token 補到 256、注意力換 cuDNN 融合核。
這裡釘住兩件事：

1. **預設路徑沒被改到**——`fast=None` 與補 pad 前的 token 佈局仍然是 head 讀得到的那一份。
2. 兩個 fast 變體對預設路徑的數值差在 審查者 量過的範圍內（各 head argmax 一致 ≥ 0.99、
   log-softmax max|Δ| ≤ 0.1（f32）／≤ 0.5（bf16））。

cuDNN 核只有 GPU 有，**沒有 GPU 就 skip**。狀態優先用教師 bank 的前 64 列（審查者 的探針
用的就是教師狀態）；bank 不在（未隨 repo 公開）就退回 `S.reset` 的 64 局開局狀態。
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl import caps as caps_mod
from rl import loop4_model as M
from rl import loop4_obs as O
from rl import rules
from rl import state as S

CAPS = caps_mod.BOUNDED_TRAINING
B = 64
BANK = Path(__file__).resolve().parent.parent / "fixtures/teacher_state_bank.npz"
IL = Path(__file__).resolve().parent.parent / "fixtures/il_all_perhead_best_dev.npz"

needs_gpu = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="cuDNN 融合注意力只有 GPU 有"
)


def _states():
    if BANK.exists():
        z = np.load(BANK)
        if all(f"s/{f}" in z.files for f in S.SimState._fields):
            return S.SimState(**{f: jnp.asarray(z[f"s/{f}"][:B]) for f in S.SimState._fields})
    return S.reset(B, CAPS, seed=0)


def _params(cfg):
    if IL.exists():
        z = np.load(IL)
        return {k: jnp.asarray(z[k]) for k in z.files}
    return M.init_params(cfg, 0)


def _setup():
    cfg = M.Config.default(CAPS)
    return cfg, _params(cfg), O.observation(_states(), 0, CAPS)


def test_pad_keeps_the_three_read_points():
    """補 pad 後：自己的格仍在前 100、glob 仍是最後一個、pad 全遮罩且插在 glob 之前。"""
    cfg, params, obs = _setup()
    x, mask = M.build_tokens(params, obs, cfg)
    t = x.shape[1]
    xp, mp = M.pad_tokens_for_fast(x, mask)
    pad = xp.shape[1] - t
    assert xp.shape[1] % M.FAST_PAD_MULTIPLE == 0 and pad > 0
    m = np.asarray(mp)
    assert m[:, -1].all()  # glob token 仍然有效
    assert not m[:, -1 - pad : -1].any()  # 插進去的 pad 全遮罩
    assert np.allclose(np.asarray(xp[:, : rules.N_TILES]), np.asarray(x[:, : rules.N_TILES]))
    assert np.allclose(np.asarray(xp[:, -1]), np.asarray(x[:, -1]))
    n_units = cfg["max_units"]
    lo = M.N_TILE_TOKENS + n_units
    assert np.allclose(
        np.asarray(xp[:, lo : lo + rules.N_PRODUCTS]), np.asarray(x[:, lo : lo + rules.N_PRODUCTS])
    )


@needs_gpu
@pytest.mark.parametrize(("fast", "tol"), [("f32", 0.1), ("bf16", 0.5)])
def test_fast_forward_matches_default(fast, tol):
    cfg, params, obs = _setup()
    base = M.forward(params, obs, cfg)
    got = M.forward(params, obs, cfg, fast=fast)
    for k, v in base.items():
        if k == "value":
            continue
        lp = np.asarray(jax.nn.log_softmax(v, axis=-1))
        lq = np.asarray(jax.nn.log_softmax(got[k], axis=-1))
        agree = float((lp.argmax(-1) == lq.argmax(-1)).mean())
        assert agree >= 0.99, f"{k} argmax 一致率 {agree}"
        d = float(np.abs(lp - lq).max())
        assert d <= tol, f"{k} log-softmax max|Δ| {d}"
