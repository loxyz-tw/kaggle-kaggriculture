"""IL trainer 預取（il-loader-prefetch-0928）：順序、深度 1、例外、rng 快照與 resume 等價。

參照實作 `_orig_sync` 逐行照抄 0dd6789 `main()` 主迴圈的抽樣部分（訓練批 → actor ckpt → dev 批），
用假 reader（只有抽樣要用的欄位）比對 `draw_step`＋`Prefetcher` 的 rng 取用序列。
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts import full_actions_il_train as T  # noqa: E402
from scripts.full_actions_il_train import (  # noqa: E402
    Prefetcher,
    draw_step,
    restore_rng,
    sample_rows,
)

U = 4


class FakeReader:
    def __init__(self, name: str, seats: list[int], rpc: int = 24):
        self.path = Path(f"/fake/{name}.npz")
        self.seat_of_capture = np.asarray(seats, np.int32)
        self.rows_per_capture = rpc

    def rows(self, idx):
        idx = np.sort(np.asarray(idx, np.int64))
        b = idx.size
        cap = idx // self.rows_per_capture
        return {
            "seat": int(self.seat_of_capture[cap[0]]),
            "unit_keep": np.ones((b, U), bool),
            "market_keep": np.ones((b,), bool),
            "t_load_s": 0.0,
            "n_captures_in_batch": int(np.unique(cap).size),
            "n_overlaid_slots": 0,
            "row_ids": idx,
        }


def _readers():
    trains = [
        FakeReader("a", [0, 1, 0, 1, 1]),
        FakeReader("b", [1, 1, 1]),
        FakeReader("c", [0, 0, 1, 0]),
    ]
    devs = [FakeReader("d0", [0, 1]), FakeReader("d1", [1, 0, 1])]
    return trains, devs


def _sha(idx):
    import hashlib

    return hashlib.sha256(np.asarray(idx, np.int64).tobytes()).hexdigest()[:16]


def _orig_sync(rng, trains, devs, start, steps, rows, dev_every, dev_batches):
    """0dd6789 主迴圈的抽樣順序（無 GPU）。

    回傳逐步 (idx sha, seat, shard)、actor ckpt 當下 rng、結束 rng。
    """
    seq, ck_rng = [], {}
    for it in range(start, start + steps):
        reader = trains[int(rng.integers(len(trains)))]
        seats_here = set(reader.seat_of_capture.tolist())
        seat = it % 2 if (it % 2) in seats_here else next(iter(seats_here))
        idx = sample_rows(rng, reader, rows, seat)
        batch = reader.rows(idx)
        seq.append((_sha(idx), batch["seat"], reader.path.name))
        ck_rng[it] = rng.bit_generator.state  # 原版 actor_i 存檔點（dev 之前）
        if dev_every and devs and it % dev_every == 0:
            for _ in range(dev_batches):
                dr = devs[int(rng.integers(len(devs)))]
                dseats = sorted(set(dr.seat_of_capture.tolist()))
                dr.rows(sample_rows(rng, dr, rows, int(rng.choice(dseats))))
    return seq, ck_rng, rng.bit_generator.state


def _prefetch(rng, trains, devs, start, steps, rows, dev_every, dev_batches):
    pf = Prefetcher(
        lambda i: draw_step(rng, trains, devs, i, rows, U, dev_every, dev_batches),
        start,
        start + steps - 1,
    )
    seq, ck_rng, final = [], {}, rng.bit_generator.state
    try:
        for it in range(start, start + steps):
            item, _ = pf.get(it)
            seq.append((item["idx_sha16"], item["batch"]["seat"], item["reader"].path.name))
            ck_rng[it] = item["rng_after_train"]
            final = item["rng_after_step"]
    finally:
        pf.close()
    return seq, ck_rng, final


@pytest.mark.parametrize("dev_every", [0, 7, 10])
def test_sequence_and_rng_snapshots_match_original_sync_loop(dev_every):
    trains, devs = _readers()
    a = _orig_sync(np.random.default_rng(0), trains, devs, 1, 60, 8, dev_every, 2)
    b = _prefetch(np.random.default_rng(0), trains, devs, 1, 60, 8, dev_every, 2)
    assert a[0] == b[0]  # 逐步 batch 序列
    assert a[1] == b[1]  # 每步 actor ckpt 的 rng 快照
    assert a[2] == b[2]  # final 的 rng（含最後一步的 dev 抽樣）


def test_resume_from_final_snapshot_continues_identically():
    trains, devs = _readers()
    full = _prefetch(np.random.default_rng(0), trains, devs, 1, 40, 8, 10, 2)
    first = _prefetch(np.random.default_rng(0), trains, devs, 1, 20, 8, 10, 2)
    second = _prefetch(restore_rng(first[2]), trains, devs, 21, 20, 8, 10, 2)
    assert first[0] + second[0] == full[0]
    assert second[2] == full[2]


def test_resume_from_actor_snapshot_matches_original_semantics():
    """actor_i 快照在 dev 抽樣之前（照原版）：從它續跑與原版從同一快照續跑相同。"""
    trains, devs = _readers()
    _, ck_new, _ = _prefetch(np.random.default_rng(0), trains, devs, 1, 20, 8, 10, 2)
    _, ck_old, _ = _orig_sync(np.random.default_rng(0), trains, devs, 1, 20, 8, 10, 2)
    assert ck_new[20] == ck_old[20]
    a = _orig_sync(restore_rng(ck_old[20]), trains, devs, 21, 10, 8, 10, 2)
    b = _prefetch(restore_rng(ck_new[20]), trains, devs, 21, 10, 8, 10, 2)
    assert a == b


def test_depth_one_never_more_than_one_ahead():
    calls, lock = [], threading.Lock()

    def fn(i):
        with lock:
            calls.append(i)
        return {"it": i}

    pf = Prefetcher(fn, 1, 30)
    try:
        for it in range(1, 31):
            pf.get(it)
            pf.fut.result() if pf.fut else None  # 讓生產者跑到它能跑的最遠
            assert max(calls) <= it + 1
        assert calls == list(range(1, 31))  # 依序、不多抽
    finally:
        pf.close()


def test_producer_exception_propagates():
    def fn(i):
        if i == 3:
            raise OSError("讀列失敗")
        return {"it": i}

    pf = Prefetcher(fn, 1, 10)
    try:
        pf.get(1)
        pf.get(2)
        with pytest.raises(OSError, match="讀列失敗"):
            pf.get(3)
    finally:
        pf.close()


def test_order_mismatch_is_rejected():
    pf = Prefetcher(lambda i: {"it": i + 1}, 1, 5)
    try:
        with pytest.raises(RuntimeError, match="預取順序錯"):
            pf.get(1)
    finally:
        pf.close()


def test_main_loop_does_not_touch_rng():
    """主迴圈裡 rng 只出現在交給預取執行緒的 lambda 與起跑前的 rng_final 快照。"""
    import inspect

    src = inspect.getsource(T.main)
    loop = src[src.index("pf = Prefetcher(") : src.index('save_train_state(out / "final.npz"')]
    body = loop[loop.index("try:") :]
    assert "rng" not in body.replace("rng_final", "").replace("rng_after_train", "").replace(
        "rng_after_step", ""
    )
