"""`rl/loop4_model.py::init_params` 的 key 配額與各變體形狀（任務 18）。

`init_params` 原本固定切 64 把 key（前綴 6 ＋ 每層 4 ＋ head 5）。任務 14b 的變體 E
是 16 層，要 75 把，`next(keys)` 會在 `ff1_w` 拋 `StopIteration`（改前的碼實測：
12 層仍在 64 把之內、**16 層才炸**，任務書寫的「12 層就炸」與實測不符）。修法是讓
`n_keys` 依層數自己長，**但下限保 64**——`jax.random.split(key, n)` 的輸出整批取決於 `n`，
n 一變所有 key 都變，4 層 d192 就對不回 `il_all` 訓練時的初始化。

所以這支測試守的是兩件事：
1. 4 層 d192 seed 0 的初始化**逐位**不變（對改前存檔，以及一個不依賴外部存檔的雜湊）。
2. 6／8／12／16 層 × d ∈ {128, 192} 都能 init 並 forward。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from rl import caps as caps_mod
from rl import loop4_model as M
from rl import loop4_obs as O
from rl import state as S

CAPS = caps_mod.BOUNDED_TRAINING
ROOT = Path(__file__).resolve().parent.parent

#: 改前（commit 66fe566^，固定 64 把 key）的 4 層 d192 seed 0 初始化。
#: 改前存檔不隨 repo 公開，所以這裡放雜湊，存檔比對另外做（見下面兩個測試）。
INIT_L4_D192_SEED0_SHA256 = "52f1e05e974c397b18eec616da6e4fdb08eae7fc4a217c7f34e6aa6b3d12e2c5"
BEFORE_NPZ = ROOT / "fixtures/init_before_l4_d192.npz"


def _digest(params: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(params):
        v = np.asarray(params[k], dtype=np.float32)
        h.update(k.encode())
        h.update(str(v.shape).encode())
        h.update(v.tobytes())
    return h.hexdigest()


def test_four_layer_d192_init_is_bit_identical_to_the_recorded_one():
    params = M.init_params(M.Config.default(CAPS, d_model=192, n_layers=4), 0)
    assert M.param_count(params) == 1_836_595
    assert _digest(params) == INIT_L4_D192_SEED0_SHA256


@pytest.mark.skipif(not BEFORE_NPZ.exists(), reason=f"沒有 {BEFORE_NPZ}（未隨 repo 公開）")
def test_four_layer_d192_init_matches_the_pre_fix_npz_array_by_array():
    ref = np.load(BEFORE_NPZ)
    params = M.init_params(M.Config.default(CAPS, d_model=192, n_layers=4), 0)
    assert sorted(ref.files) == sorted(params)
    for k in params:
        assert np.array_equal(ref[k], np.asarray(params[k])), k


@pytest.mark.parametrize("n_layers", [6, 8, 12, 16])
@pytest.mark.parametrize("d_model", [128, 192])
def test_init_and_forward_scale_with_depth_and_width(d_model: int, n_layers: int):
    cfg = M.Config.default(CAPS, d_model=d_model, n_layers=n_layers)
    assert cfg["n_heads"] == (4 if d_model == 128 else 6)
    assert d_model % cfg["n_heads"] == 0
    params = M.init_params(cfg, 0)  # 改前在 16 層這裡會 StopIteration
    obs = O.observation(S.reset(2, CAPS, seed=0), 0, CAPS)
    out = M.forward(params, obs, cfg)
    assert out["task"].shape == (2, 100, M.A.N_TASKS)
    assert out["value"].shape == (2,)
    assert np.all(np.isfinite(np.asarray(out["value"])))


#: 任務 14b 任務書的表。**只有 A 是實測基準**；其餘五個是用「每層 12d²＋9d、非 block 57,139」
#: 推算的，與實際 `init_params` 對不上（差 −6,105 到 ＋3,072）。任務 18 只列事實、不改表，
#: 所以這裡記的是**實測值**，右欄附上表的數字供對照。
PARAM_COUNTS = [
    # 變體, 層, d, 實測參數總數, 14b 表的數字
    ("A", 4, 192, 1_836_595, 1_836_595),
    ("F", 6, 192, 2_724_787, 2_721_715),
    ("C", 8, 192, 3_612_979, 3_616_051),
    ("B", 6, 128, 1_226_739, 1_227_724),
    ("D", 12, 128, 2_413_299, 2_417_356),
    ("E", 16, 128, 3_204_339, 3_210_444),
]


@pytest.mark.parametrize(("name", "n_layers", "d_model", "actual", "table"), PARAM_COUNTS)
def test_variant_param_counts(name: str, n_layers: int, d_model: int, actual: int, table: int):
    cfg = M.Config.default(CAPS, d_model=d_model, n_layers=n_layers)
    assert M.param_count(M.init_params(cfg, 0)) == actual
    if name != "A":
        assert actual != table  # 表與實測的差還在；修好表之前不要把這行改成相等


def test_key_budget_covers_every_draw():
    """每層 4 把、前綴 6 把、head 5 把；下限 64 只在 14 層以下有作用。"""
    for n_layers in (4, 6, 8, 12, 13, 14, 16):
        cfg = M.Config.default(CAPS, d_model=128, n_layers=n_layers)
        need = 6 + 4 * n_layers + 5
        assert max(64, need) >= need
        M.init_params(cfg, 0)
