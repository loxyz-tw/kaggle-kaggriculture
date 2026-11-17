"""任務 12 A3：教師原始動作 → Loop 4 標籤（每格任務 ＋ 市場桶），並量映射不到的比例。

映射方向與訓練時的解碼**必須是同一組定義**，所以桶的整數一律呼叫 `rl/loop4_actions.py`
的同一批函式，不在這裡另寫一份。

兩段映射：

* **單位動作 → 每格任務**：教師的每個單位動作帶著它當下的座標，所以「這個單位在這格做了
  什麼」可以直接落到格上。移動（NORTH／SOUTH／EAST／WEST）與 PASS **不產生標籤**——
  走路在 Loop 4 是執行器的事，不是 policy 的輸出（v4 §5）。同一格被兩個單位指派到不同任務時
  記 `tile_conflict`，取 flat 索引小的那個單位。
* **市場訂單 → 桶**：每個 (op, item) 取教師該回合的**第一筆**訂單（任務 11 量到現任第一名
  同回合同 (op,item) 重複是 0 筆；教師語料是 0.19%），數量對桶做精確比對，對不上記
  `unrepresentable`，**不進 loss**。

SELL 的桶整數要用「單位相位之後的 shed」（任務 11：教師 48.2% 的 SELL 數量超過回合開始時的
庫存）。語料的 `s/*` 是**回合開始**的狀態，所以本腳本用同一份狀態與動作跑一次
`kernels.apply_unit_actions` 把單位相位補上，再比 SELL 的桶——這和訓練時
`loop4_actions.loop4_step` 的時點完全一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.lib import format as npformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl import caps as caps_mod  # noqa: E402
from rl import kernels, rules  # noqa: E402
from rl import loop4_actions as A  # noqa: E402
from rl.actions import SimAction  # noqa: E402
from rl.state import SimState  # noqa: E402

CAPS = caps_mod.EXACT_REFERENCE
CALLS = rules.N_TRANSITIONS

STATE_FIELDS = (
    "step",
    "money",
    "tiles",
    "unit_x",
    "unit_y",
    "hires_today",
    "n_quads",
    "shed",
    "seeds",
    "inv",
    "inv_seq",
    "inv_ctr",
    "market_inv",
    "shop_seq",
    "n_shops",
    "key",
    "overflow",
    "market_trips",
    "market_units",
    "market_closed",
)


def _read(zf, name):
    with zf.open(name) as f:
        return npformat.read_array(f)


BIG_FIELDS = ("tiles", "inv", "inv_seq")


def open_chunks(zf, name, chunk):
    """**循序**串流一個未壓縮 npz 成員，一次 ``chunk`` 列。

    不要用「每次重開成員再 seek 到 lo」的寫法：zip 成員不能隨機定位，``f.read(lo * row)``
    是真的把前面那些 byte 全讀一遍，逐 capture 讀整份 shard 就變成 O(n²)——實測 400 個
    capture 的 ``s/tiles.npy``（3.45 GB）會讀掉約 688 GB，跑不完。``_read_slice`` 只留給
    「只讀前面少數 capture」的探針用。
    """
    f = zf.open(name)
    major, minor = npformat.read_magic(f)
    reader = {
        (1, 0): npformat.read_array_header_1_0,
        (2, 0): npformat.read_array_header_2_0,
    }[(major, minor)]
    shape, _fortran, dtype = reader(f)
    row = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
    while True:
        buf = f.read(chunk * row)
        if not buf:
            f.close()
            return
        yield np.frombuffer(buf, dtype=dtype).reshape((-1, *shape[1:]))


def _state_from_parts(small, big, lo, hi):
    """用整份的小欄位與本 capture 的大欄位區塊組一個 SimState。"""
    out = {k: (big[k] if k in BIG_FIELDS else small[k][lo:hi]) for k in STATE_FIELDS}
    return SimState(**{k: jnp.asarray(v) for k, v in out.items()})


def _read_slice(zf, name, lo, hi):
    with zf.open(name) as f:
        major, minor = npformat.read_magic(f)
        reader = {
            (1, 0): npformat.read_array_header_1_0,
            (2, 0): npformat.read_array_header_2_0,
        }[(major, minor)]
        shape, _fortran, dtype = reader(f)
        row = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        f.read(lo * row)
        buf = f.read((hi - lo) * row)
    return np.frombuffer(buf, dtype=dtype).reshape((hi - lo, *shape[1:]))


def qty_from_bytes(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.int64)
    return (b[..., 0] << 24) | (b[..., 1] << 16) | (b[..., 2] << 8) | b[..., 3]


def unit_labels(
    verb: np.ndarray,
    crop: np.ndarray,
    uqty: np.ndarray,
    keep: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    seat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Counter]:
    """``[N, 29]`` 的單位動作 → ``[N, 100]`` 的每格任務與數量桶。"""
    n = verb.shape[0]
    task = np.zeros((n, rules.N_TILES), np.int8)
    qbucket = np.zeros((n, rules.N_TILES), np.int8)
    stats = Counter()
    op_to_task = {
        rules.OP_WATER: A.T_WATER,
        rules.OP_HARVEST: A.T_HARVEST,
        rules.OP_FERTILIZE: A.T_FERTILIZE,
        rules.OP_FEED: A.T_FEED,
        rules.OP_CARE: A.T_CARE,
        rules.OP_COLLECT_FERTILIZER: A.T_COLLECT,
        rules.OP_DIG: A.T_DIG,
        rules.OP_BUILD_COOP: A.T_BUILD_COOP,
        rules.OP_BUILD_PASTURE: A.T_BUILD_PASTURE,
        rules.OP_PICKUP: A.T_PICKUP,
        rules.OP_PLACE: A.T_PLACE,
    }
    moves = {rules.OP_NORTH, rules.OP_SOUTH, rules.OP_EAST, rules.OP_WEST}
    for r in range(n):
        s = int(seat[r])
        for k in range(verb.shape[1]):
            if not keep[r, k]:
                continue
            v = int(verb[r, k])
            if v == rules.OP_PASS:
                stats["unit_pass"] += 1
                continue
            if v in moves:
                stats["unit_move"] += 1
                continue
            if v == rules.OP_DROP:
                # DROP 在 Loop 4 由 PLACE 桶的「全部」覆蓋；單獨的 DROP 沒有對應任務類
                stats["unit_drop_unrepresentable"] += 1
                continue
            flat = int(uy[r, s, k]) * rules.BOARD_SIZE + int(ux[r, s, k])
            if v == rules.OP_PLANT:
                c = int(crop[r, k])
                if not (0 <= c < rules.N_CROPS):
                    stats["unit_plant_bad_crop"] += 1
                    continue
                t = A.T_PLANT0 + c
            elif v in op_to_task:
                t = op_to_task[v]
            else:
                stats["unit_unknown_op"] += 1
                continue
            if task[r, flat] != A.T_NOOP and task[r, flat] != t:
                stats["tile_conflict"] += 1
                continue
            task[r, flat] = t
            stats["unit_mapped"] += 1
            if v in (rules.OP_PICKUP, rules.OP_PLACE):
                q = int(uqty[r, k])
                # QTY_BUCKETS = (1, 2, 4, 全部)；對不上前三個就落到「全部」並記一筆
                b = {1: 0, 2: 1, 4: 2}.get(q)
                if b is None:
                    b = 3
                    stats["unit_qty_to_all_bucket"] += 1
                else:
                    stats["unit_qty_exact_bucket"] += 1
                qbucket[r, flat] = b
    return task, qbucket, stats


def market_labels(
    op: np.ndarray,
    item: np.ndarray,
    qty: np.ndarray,
    keep: np.ndarray,
    post: SimState,
    seat: np.ndarray,
) -> tuple[dict[str, np.ndarray], Counter]:
    """``[N, 10]`` 的訂單 → 各 head 的桶索引；對不上桶的記 ``unrepresentable``。"""
    n = op.shape[0]
    idx = np.arange(n)
    shed = np.asarray(post.shed)[idx, seat]  # 單位相位之後
    cnt = A.market_counts(post)
    n_empty = np.asarray(cnt["n_empty"])[idx, seat]
    feed_all = np.asarray(cnt["feed_all"])[idx, seat]
    fill = np.asarray(cnt["fill"])[idx, seat]
    out = {
        "sell": np.zeros((n, rules.N_PRODUCTS), np.int8),
        "buy_seed": np.zeros((n, rules.N_CROPS), np.int8),
        "buy_product": np.zeros((n, len(A.BUYABLE_PRODUCTS)), np.int8),
        "buy_animal": np.zeros((n, rules.N_ANIMALS), np.int8),
        "hire": np.zeros(n, np.int8),
        "buy_land": np.zeros(n, np.int8),
    }
    unrep = {k: np.zeros_like(v) for k, v in out.items()}
    stats = Counter()
    sell_abs = {1: 4, 2: 5, 3: 6, 4: 7, 6: 8, 8: 9}
    seed_abs = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 6}
    prod_abs = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 6, 15: 7}
    for r in range(n):
        if not keep[r]:
            continue
        seen = set()
        hires = 0
        for s in range(op.shape[1]):
            o = int(op[r, s])
            if o == rules.MO_NONE:
                continue
            stats["orders"] += 1
            if o == rules.MO_HIRE:
                hires += 1
                continue
            if o == rules.MO_BUY_LAND:
                out["buy_land"][r] = 1
                stats["mapped"] += 1
                continue
            it, q = int(item[r, s]), int(qty[r, s])
            if q <= 0:
                stats["zero_qty"] += 1
                continue
            if (o, it) in seen:
                stats["duplicate_op_item"] += 1
                continue
            seen.add((o, it))
            if o == rules.MO_SELL and 0 <= it < rules.N_PRODUCTS:
                held = int(shed[r, it])
                b = 0
                if held > 0 and q >= held:
                    b = 1
                elif held > 0 and q == held // 2:
                    b = 2
                elif held > 0 and q == held // 4:
                    b = 3
                elif q in sell_abs:
                    b = sell_abs[q]
                if b:
                    out["sell"][r, it] = b
                    stats["mapped"] += 1
                else:
                    unrep["sell"][r, it] = 1
                    stats["unrep_sell"] += 1
            elif o == rules.MO_BUY_SEED and 0 <= it < rules.N_CROPS:
                b = seed_abs.get(q, 7 if q == int(n_empty[r]) else 0)
                if b:
                    out["buy_seed"][r, it] = b
                    stats["mapped"] += 1
                else:
                    unrep["buy_seed"][r, it] = 1
                    stats["unrep_buy_seed"] += 1
            elif o == rules.MO_BUY_PRODUCT and it in A.BUYABLE_PRODUCTS:
                j = A.BUYABLE_PRODUCTS.index(it)
                b = prod_abs.get(q, 8 if q == int(feed_all[r]) else 0)
                if b:
                    out["buy_product"][r, j] = b
                    stats["mapped"] += 1
                else:
                    unrep["buy_product"][r, j] = 1
                    stats["unrep_buy_product"] += 1
            elif o == rules.MO_BUY_PRODUCT:
                stats["buy_product_dead_item"] += 1  # 引擎 no-op（非 WHEAT／FERTILIZER）
            elif o == rules.MO_BUY_ANIMAL and rules.ANIMAL_ITEM_BASE <= it < rules.N_ITEMS:
                a = it - rules.ANIMAL_ITEM_BASE
                b = 1 if q == 1 else 2 if q == 2 else (3 if q == int(fill[r, a]) else 0)
                if b:
                    out["buy_animal"][r, a] = b
                    stats["mapped"] += 1
                else:
                    unrep["buy_animal"][r, a] = 1
                    stats["unrep_buy_animal"] += 1
            else:
                stats["unknown_order"] += 1
        if hires:
            if hires <= A.N_HIRE - 1:
                out["hire"][r] = hires
                stats["mapped"] += hires
            else:
                unrep["hire"][r] = 1
                stats["unrep_hire"] += hires
    return {"labels": out, "unrepresentable": unrep}, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="schema-3 shard npz")
    ap.add_argument("--captures", type=int, default=0, help="非 0 時只做前 N 個 capture")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--labels-out",
        default="",
        help="非空時把標籤寫成一個 npz（給 scripts/loop4_il_train.py 用）",
    )
    a = ap.parse_args()

    path = ROOT / a.shard
    stats = Counter()
    out_labels: dict[str, list] | None = None
    if a.labels_out:
        out_labels = defaultdict(list)
    with zipfile.ZipFile(path) as zf:
        marked = _read(zf, "marked.npy")
        seat_all = np.argmax(marked, axis=1).astype(np.int64)
        verb = _read(zf, "d/unit_verb.npy")
        crop = _read(zf, "d/unit_crop.npy")
        uitem = _read(zf, "d/unit_item.npy")
        uqty = qty_from_bytes(_read(zf, "d/unit_qty_bytes.npy"))
        ukeep = _read(zf, "k/unit_keep.npy")
        mop = _read(zf, "d/market_op.npy")
        mitem = _read(zf, "d/market_item.npy")
        mqty = qty_from_bytes(_read(zf, "d/market_qty_bytes.npy"))
        mkeep = _read(zf, "k/market_keep.npy")
        n_rows = verb.shape[0]
        n_cap = n_rows // CALLS
        if a.captures:
            n_cap = min(n_cap, a.captures)
        small = {k: _read(zf, f"s/{k}.npy") for k in STATE_FIELDS if k not in BIG_FIELDS}
        streams = {k: open_chunks(zf, f"s/{k}.npy", CALLS) for k in BIG_FIELDS}
        for ci in range(n_cap):
            lo, hi = ci * CALLS, (ci + 1) * CALLS
            big = {k: next(streams[k]) for k in BIG_FIELDS}
            st = _state_from_parts(small, big, lo, hi)
            seat = seat_all[lo:hi]
            ux, uy = np.asarray(st.unit_x), np.asarray(st.unit_y)
            task, qb, s1 = unit_labels(
                verb[lo:hi], crop[lo:hi], uqty[lo:hi], ukeep[lo:hi], ux, uy, seat
            )
            stats.update(s1)
            # 單位相位之後的狀態（SELL 的桶要用它）
            act = SimAction(
                unit_op=jnp.asarray(verb[lo:hi])[:, None, :].repeat(2, axis=1),
                unit_arg=jnp.asarray(
                    np.where(verb[lo:hi] == rules.OP_PLANT, crop[lo:hi], uitem[lo:hi])
                )[:, None, :].repeat(2, axis=1),
                unit_qty=jnp.ones_like(jnp.asarray(verb[lo:hi]))[:, None, :].repeat(2, axis=1),
                order_op=jnp.zeros((hi - lo, 2, CAPS.max_orders), jnp.int32),
                order_item=jnp.full((hi - lo, 2, CAPS.max_orders), -1, jnp.int32),
                order_qty=jnp.zeros((hi - lo, 2, CAPS.max_orders), jnp.int32),
            )
            post = kernels.apply_unit_actions(st, act, CAPS)
            lab, s2 = market_labels(mop[lo:hi], mitem[lo:hi], mqty[lo:hi], mkeep[lo:hi], post, seat)
            stats.update(s2)
            if out_labels is not None:
                out_labels["tile_task"].append(task)
                out_labels["tile_qty"].append(qb)
                out_labels["seat"].append(seat.astype(np.int8))
                out_labels["step"].append(np.asarray(st.step))
                for k, v in lab["labels"].items():
                    out_labels[f"lab_{k}"].append(v)
                for k, v in lab["unrepresentable"].items():
                    out_labels[f"unrep_{k}"].append(v)
            del task, qb

    orders = stats["orders"]
    unrep = sum(v for k, v in stats.items() if k.startswith("unrep_"))
    unit_total = stats["unit_mapped"] + stats["unit_move"] + stats["unit_pass"]
    out = {
        "shard": a.shard,
        "captures": n_cap,
        "counts": dict(sorted(stats.items())),
        "market": {
            "orders": orders,
            "mapped": stats["mapped"],
            "share_mapped": stats["mapped"] / orders if orders else None,
            "unrepresentable": unrep,
            "share_unrepresentable": unrep / orders if orders else None,
        },
        "units": {
            "labelled_actions": unit_total,
            "mapped_to_tile_task": stats["unit_mapped"],
            "dropped_move": stats["unit_move"],
            "dropped_pass": stats["unit_pass"],
            "dropped_drop": stats["unit_drop_unrepresentable"],
            "tile_conflict": stats["tile_conflict"],
        },
    }
    if out_labels is not None:
        arrays = {k: np.concatenate(v, axis=0) for k, v in out_labels.items()}
        lp = ROOT / a.labels_out
        lp.parent.mkdir(parents=True, exist_ok=True)
        tmp = lp.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(lp)
        out["labels_npz"] = {
            "path": a.labels_out,
            "rows": int(next(iter(arrays.values())).shape[0]),
            "fields": {k: list(v.shape) for k, v in arrays.items()},
            "bytes": lp.stat().st_size,
        }
    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
