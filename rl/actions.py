"""Fixed-shape action tensors and the engine <-> tensor codec.

Phase 0/1 freezes only a *neutral* contract: raw per-unit verbs and the
engine's own 10 market order slots.  No macro-compiler, no factored-head
layout and no claim that order sequencing is useless -- those are Phase 2
decisions and deliberately not pre-empted here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import rules

I32 = jnp.int32


class SimAction(NamedTuple):
    unit_op: jax.Array  # [B, 2, U]   rules.OP_*
    unit_arg: jax.Array  # [B, 2, U]   item index (crop index == item index), -1 = none
    unit_qty: jax.Array  # [B, 2, U]   PICKUP / PLACE count
    order_op: jax.Array  # [B, 2, 10]  rules.MO_*
    order_item: jax.Array  # [B, 2, 10]
    order_qty: jax.Array  # [B, 2, 10]


def empty_action(batch: int, caps: caps_mod.Caps) -> SimAction:
    u, o = caps.max_units, caps.max_orders
    return SimAction(
        unit_op=jnp.zeros((batch, 2, u), I32),
        unit_arg=jnp.full((batch, 2, u), -1, I32),
        unit_qty=jnp.ones((batch, 2, u), I32),
        order_op=jnp.zeros((batch, 2, o), I32),
        order_item=jnp.full((batch, 2, o), -1, I32),
        order_qty=jnp.zeros((batch, 2, o), I32),
    )


class EncodeStats(NamedTuple):
    dropped_hand_actions: int  # hand slots past caps.max_units - 1
    dropped_orders: int  # order slots past caps.max_orders (engine truncates too)
    clamped_qty: int  # order quantities above caps.max_order_qty
    unknown_unit_ops: int
    unknown_items: int


def _item_index(name: Any) -> int:
    if isinstance(name, str):
        return rules.ITEM_INDEX.get(name, -1)
    return -1


def encode_actions_np(
    actions_by_player: list[Any], caps: caps_mod.Caps
) -> tuple[dict[str, np.ndarray], EncodeStats]:
    """Encode one turn of engine actions ``[p0_action, p1_action]``."""
    u, o = caps.max_units, caps.max_orders
    unit_op = np.zeros((2, u), dtype=np.int32)
    unit_arg = np.full((2, u), -1, dtype=np.int32)
    unit_qty = np.ones((2, u), dtype=np.int32)
    order_op = np.zeros((2, o), dtype=np.int32)
    order_item = np.full((2, o), -1, dtype=np.int32)
    order_qty = np.zeros((2, o), dtype=np.int32)
    dropped_hands = dropped_orders = clamped = unknown_ops = unknown_items = 0

    for p in range(2):
        action = actions_by_player[p] if isinstance(actions_by_player[p], dict) else {}
        farmer = action.get("farmer", ["PASS"])
        hands = action.get("hands", [])
        if not isinstance(hands, list):
            hands = []
        unit_actions = [farmer, *hands]
        if len(unit_actions) > u:
            dropped_hands += len(unit_actions) - u
            unit_actions = unit_actions[:u]
        for slot, a in enumerate(unit_actions):
            if not isinstance(a, list) or not a:
                continue
            op_name = a[0]
            op = rules.UNIT_OP_INDEX.get(op_name)
            if op is None:
                unknown_ops += 1
                continue  # engine: unknown op falls through every branch = no-op
            unit_op[p, slot] = op
            if len(a) >= 2:
                idx = _item_index(a[1])
                if idx < 0:
                    unknown_items += 1
                unit_arg[p, slot] = idx
            elif op in (rules.OP_PICKUP, rules.OP_PLANT, rules.OP_PLACE):
                # engine: `if len(action) < 2: return`
                unit_op[p, slot] = rules.OP_PASS
            if len(a) >= 3:
                try:
                    unit_qty[p, slot] = int(a[2])
                except (TypeError, ValueError):
                    unit_qty[p, slot] = 0

        market = action.get("market", [])
        if not isinstance(market, list):
            market = []
        if len(market) > o:
            dropped_orders += len(market) - o
            market = market[:o]
        for slot, entry in enumerate(market):
            if not isinstance(entry, list) or not entry:
                continue
            name = entry[0]
            if name == "HIRE":
                order_op[p, slot] = rules.MO_HIRE
                continue
            if name == "BUY_LAND":
                order_op[p, slot] = rules.MO_BUY_LAND
                continue
            op = rules.MARKET_OP_INDEX.get(name)
            if op is None or op in (rules.MO_NONE,):
                continue
            if len(entry) < 3:
                continue  # engine: _parse_order returns None
            try:
                n = int(entry[2])
            except (TypeError, ValueError):
                continue
            if n <= 0:
                continue
            if n > caps.max_order_qty:
                clamped += 1
                n = caps.max_order_qty
            order_op[p, slot] = op
            order_item[p, slot] = _item_index(entry[1])
            order_qty[p, slot] = n

    arrays = {
        "unit_op": unit_op,
        "unit_arg": unit_arg,
        "unit_qty": unit_qty,
        "order_op": order_op,
        "order_item": order_item,
        "order_qty": order_qty,
    }
    stats = EncodeStats(dropped_hands, dropped_orders, clamped, unknown_ops, unknown_items)
    return arrays, stats


def stack_actions(per_env: list[dict[str, np.ndarray]]) -> SimAction:
    return SimAction(**{k: jnp.asarray(np.stack([e[k] for e in per_env]), I32) for k in per_env[0]})
