"""top-k 選格的排序鍵：新舊兩種在精確算術下同序，舊的只是會放大浮點雜訊（任務 21）。

背景：`scripts/loop4_eval.py::make_policy_step` 與 `rl/loop4_numpy.py::decode` 原本按
``want = log1p(-exp(logp_NOOP))`` 降冪取前 k 格。``log(1 - P)`` 對 ``P`` 單調遞減，
所以它與「按 ``logp_NOOP`` 升冪」同序——**換鍵不換決策定義**。

但 IL 權重把幾乎全部機率給 NOOP（教師標籤裡 NOOP 佔 98.7%），``1 - P(NOOP)`` 落在 1e-5 量級，
舊鍵等於在取這個小量的對數，**把 logits 上 1e-7 的差放大約 1e5 倍**。任務 20 實測：
JAX 端兩格是同一個 float32 值（完全平手），NumPy 端差 6.3e-3 就翻轉，自走 parity 只剩 97.5%。

這支測試守兩件事：

1. 隨機 logits 下，兩種鍵選出的 top-k **集合相同**（邊界近平手的案例排除，見 ``TIE``）。
2. 放大效應是真的：同一組 logits 動 1e-7，舊鍵的變動比新鍵大好幾個數量級。
"""

from __future__ import annotations

import numpy as np
import pytest

from rl import loop4_numpy as NP

F32 = np.float32
#: 邊界上兩格的 ``logp_NOOP`` 差小於這個值就當成平手，不要求兩種鍵選一樣
#: （任務書：差 ≥ 1e-4 的才算失敗；這裡取更嚴的 1e-6）。
TIE = 1e-6


def _case(rng, n_tiles=100, n_tasks=17, p_noop_hi=True):
    """一組隨機 logits ＋ 合法遮罩。``p_noop_hi`` 模擬 IL 權重那種 NOOP 壓倒性的分布。"""
    logits = rng.normal(0.0, 1.0, (n_tiles, n_tasks)).astype(F32)
    if p_noop_hi:
        logits[:, NP.T_NOOP] += rng.uniform(8.0, 14.0, n_tiles).astype(F32)
    mask = rng.random((n_tiles, n_tasks)) < 0.5
    mask[:, NP.T_NOOP] = True
    return logits, mask


def _select(logits, mask, n_alive, new_key: bool):
    task_lg = np.where(mask, logits, F32(-np.inf))
    logp = NP._log_softmax(task_lg, axis=-1)
    noop = logp[:, NP.T_NOOP]
    want = -noop if new_key else np.log1p(-np.exp(np.clip(noop, -30.0, -1e-6)))
    want = np.where(np.any(mask[:, 1:], axis=-1), want, -np.inf)
    order = np.argsort(-want, axis=-1, kind="stable")
    rank = np.argsort(order, axis=-1, kind="stable")
    return set(np.flatnonzero(rank < n_alive).tolist()), noop


def _boundary_gap(noop, mask, n_alive):
    """排序邊界（第 k-1 與第 k 名）的 ``logp_NOOP`` 差；平手就回 0。"""
    live = np.flatnonzero(np.any(mask[:, 1:], axis=-1))
    if len(live) <= n_alive:
        return np.inf
    s = np.sort(noop[live])  # 升冪＝最不像 NOOP 的在前
    return float(abs(s[n_alive] - s[n_alive - 1]))


@pytest.mark.parametrize("p_noop_hi", [True, False])
def test_old_and_new_key_pick_the_same_tiles(p_noop_hi: bool):
    rng = np.random.default_rng(0)
    skipped = 0
    for i in range(200):
        logits, mask = _case(rng, p_noop_hi=p_noop_hi)
        n_alive = int(rng.integers(1, 12))
        old, noop = _select(logits, mask, n_alive, new_key=False)
        new, _ = _select(logits, mask, n_alive, new_key=True)
        if _boundary_gap(noop, mask, n_alive) < TIE:
            skipped += 1
            continue
        assert old == new, (
            f"case {i}: 只有舊鍵選到 {sorted(old - new)}、只有新鍵選到 {sorted(new - old)}"
        )
    assert skipped < 60, f"跳過的近平手案例太多（{skipped}/200），這支測試就沒在測東西了"


def test_old_key_amplifies_float_noise_and_new_key_does_not():
    """P(NOOP)→1 時，logits 上 **1 個 ULP** 的擾動在舊鍵上被放大好幾個數量級。

    用 ``nextafter`` 而不是加一個小常數：量級 10 的 float32 的 ULP 約 1e-6，
    加 1e-7 會被直接吃掉、兩邊都不動，那樣測不到東西（第一版就是這樣寫錯的）。
    一個 ULP 是 JAX 與 NumPy 之間**真的會有**的差距——兩邊的求和順序不同。
    """
    rng = np.random.default_rng(1)
    logits, mask = _case(rng, p_noop_hi=True)
    bumped = np.nextafter(logits, F32(np.inf)).astype(F32)
    assert not np.array_equal(logits, bumped)

    def keys(lg):
        task_lg = np.where(mask, lg, F32(-np.inf))
        noop = NP._log_softmax(task_lg, axis=-1)[:, NP.T_NOOP]
        return -noop, np.log1p(-np.exp(np.clip(noop, -30.0, -1e-6)))

    new_a, old_a = keys(logits)
    new_b, old_b = keys(bumped)
    d_new = float(np.max(np.abs(new_a - new_b)))
    d_old = float(np.max(np.abs(old_a - old_b)))
    assert d_new > 0.0, "擾動被 float32 吃掉了，這支測試就沒在測東西"
    assert d_old > 100 * d_new, f"舊鍵變動 {d_old:.3e}、新鍵變動 {d_new:.3e}"


def test_deploy_decode_uses_the_new_key():
    """`rl/loop4_numpy.py::decode` 走的就是新鍵——直接對它的輸出比。"""
    rng = np.random.default_rng(2)
    logits, mask = _case(rng, p_noop_hi=True)
    n_alive = 5
    out = {
        "task": logits[None],
        "tile_qty": rng.normal(size=(1, 100, NP.N_QTY_BUCKETS)).astype(F32),
        "sell": rng.normal(size=(1, 9, NP.N_SELL)).astype(F32),
        "buy_seed": rng.normal(size=(1, 5, NP.N_BUY_SEED)).astype(F32),
        "buy_product": rng.normal(size=(1, 2, NP.N_BUY_PRODUCT)).astype(F32),
        "buy_animal": rng.normal(size=(1, 3, NP.N_BUY_ANIMAL)).astype(F32),
        "hire": rng.normal(size=(1, NP.N_HIRE)).astype(F32),
        "buy_land": rng.normal(size=(1, NP.N_BUY_LAND)).astype(F32),
    }
    dec = NP.decode(out, mask, n_alive)
    picked = set(np.flatnonzero(dec.tile_task != NP.T_NOOP).tolist())
    expected, _ = _select(logits, mask, n_alive, new_key=True)
    # 被選中但該格只有 NOOP 合法時不會出現在 picked，所以用子集關係兩邊夾
    assert picked <= expected
    assert len(picked) == min(n_alive, len(expected))
