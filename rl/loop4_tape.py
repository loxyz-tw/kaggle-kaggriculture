"""Loop 4 的 tape 對手（任務 14 B3）：把錄下的對局動作在模擬器裡逐字重播。

依據 內部規格文件 v4 §3／v5：最終錦標賽的對手就是這些 tape，
所以 league 的固定對手直接用它們，實作零成本——不需要推論、不需要權重，
只是按回合索引把錄好的 `SimAction` 送出去。

**這是 tape 真正的行為**：公開 notebook 的 agent 在少數檢查點才分支，其餘時候照表操課，
所以「觀測不同但動作照舊」不是近似，是它本來就會做的事
（內部規格文件 v4 §6）。

**不得提交任何 tape 的重播或改寫版**（作者規定）。
本模組只能當**本地對手與評估尺**，`TapeOpponent` 不進 checkpoint、不進提交包。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import rules
from rl.actions import SimAction, encode_actions_np

I32 = jnp.int32


class TapeBank(NamedTuple):
    """``[n_tapes, 719, ...]`` 的動作序列；第 0 軸是不同的局。"""

    unit_op: jax.Array
    unit_arg: jax.Array
    unit_qty: jax.Array
    order_op: jax.Array
    order_item: jax.Array
    order_qty: jax.Array
    source: tuple[str, ...]  # 每條 tape 的來源（episode_id:seat）

    @property
    def n_tapes(self) -> int:
        return int(self.unit_op.shape[0])


def encode_tape(replay: dict, seat: int, caps: caps_mod.Caps) -> dict[str, np.ndarray]:
    """一局一座 → ``[719, ...]`` 的動作陣列。

    回合 t 的動作在 ``steps[t + 1][seat].action``（framework 的 action／observation 錯開一格，
    與 `scripts/loop4_action_stats.py` 同一個口徑）。
    """
    steps = replay["steps"]
    n = rules.N_TRANSITIONS
    out = {
        "unit_op": np.zeros((n, caps.max_units), np.int32),
        "unit_arg": np.full((n, caps.max_units), -1, np.int32),
        "unit_qty": np.ones((n, caps.max_units), np.int32),
        "order_op": np.zeros((n, caps.max_orders), np.int32),
        "order_item": np.full((n, caps.max_orders), -1, np.int32),
        "order_qty": np.zeros((n, caps.max_orders), np.int32),
    }
    for t in range(min(n, len(steps) - 1)):
        action = steps[t + 1][seat].get("action")
        pair = [action if seat == 0 else {}, action if seat == 1 else {}]
        arrays, _stats = encode_actions_np(pair, caps)
        for k, v in out.items():
            v[t] = arrays[k][seat]
    return out


def build_bank(entries: list[tuple[Path, int, str]], caps: caps_mod.Caps) -> TapeBank:
    """``entries`` 是 ``(replay JSON 路徑, seat, 標籤)``。"""
    cols: dict[str, list] = {
        k: [] for k in ("unit_op", "unit_arg", "unit_qty", "order_op", "order_item", "order_qty")
    }
    labels = []
    for path, seat, label in entries:
        with open(path) as f:
            replay = json.load(f)
        enc = encode_tape(replay, seat, caps)
        for k in cols:
            cols[k].append(enc[k])
        labels.append(label)
    return TapeBank(
        **{k: jnp.asarray(np.stack(v)) for k, v in cols.items()},
        source=tuple(labels),
    )


def save_bank(bank: TapeBank, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        source=np.asarray(bank.source),
        **{k: np.asarray(getattr(bank, k)) for k in bank._fields if k != "source"},
    )
    tmp.replace(path)


def load_bank(path: Path) -> TapeBank:
    z = np.load(path)
    return TapeBank(
        **{k: jnp.asarray(z[k]) for k in TapeBank._fields if k != "source"},
        source=tuple(str(x) for x in z["source"]),
    )


def tape_action(bank: TapeBank, tape_idx: jax.Array, turn: jax.Array, seat: int, caps):
    """取第 ``tape_idx`` 條 tape 在第 ``turn`` 回合的動作，放進一份兩座的 ``SimAction``。

    ``tape_idx``／``turn`` 都是 ``[B]``，所以一個 batch 裡每個 env 可以對上不同的 tape
    與不同的進度（resume 或從教師狀態開局時會用到）。另一座留空，由呼叫端合併。
    """
    t = jnp.clip(turn, 0, rules.N_TRANSITIONS - 1)
    idx = jnp.clip(tape_idx, 0, bank.n_tapes - 1)

    def take(arr):
        return arr[idx, t]  # [B, U] 或 [B, O]

    b = t.shape[0]
    u, o = caps.max_units, caps.max_orders
    zeros_u = jnp.zeros((b, 2, u), I32)
    arg_u = jnp.full((b, 2, u), -1, I32)
    qty_u = jnp.ones((b, 2, u), I32)
    zeros_o = jnp.zeros((b, 2, o), I32)
    item_o = jnp.full((b, 2, o), -1, I32)

    def put(base, val):
        return base.at[:, seat].set(val)

    return SimAction(
        unit_op=put(zeros_u, take(bank.unit_op)),
        unit_arg=put(arg_u, take(bank.unit_arg)),
        unit_qty=put(qty_u, take(bank.unit_qty)),
        order_op=put(zeros_o, take(bank.order_op)),
        order_item=put(item_o, take(bank.order_item)),
        order_qty=put(zeros_o, take(bank.order_qty)),
    )


def merge_seat_actions(a: SimAction, b: SimAction, seat_a: int) -> SimAction:
    """把兩份各自只填了一座的 SimAction 合成一份。"""
    return jax.tree.map(
        lambda x, y: x.at[:, 1 - seat_a].set(y[:, 1 - seat_a]),
        a,
        b,
    )


def entries_from_majkel30(
    stats_json: Path, replay_root: Path, team: str = "Majkel1337"
) -> list[tuple[Path, int, str]]:
    """從 內部量測紀錄 的選局清單造 entries。"""
    sel = json.loads(stats_json.read_text())["selection"]
    out = []
    for g in sel["games"]:
        out.append(
            (
                replay_root / f"{g['episode_id']}.json",
                int(g["seat"]),
                f"{team}:{g['episode_id']}:{g['seat']}",
            )
        )
    return out
