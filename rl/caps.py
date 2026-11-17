"""Static shape caps for the fixed-shape simulator.

Two modes, per the approved architecture:

``EXACT_REFERENCE``
    Caps set above every value the official engine can reach under the default
    configuration given the resources actually obtainable in an episode.  Used
    for parity; overflow here would be a genuine bug.

``BOUNDED_TRAINING``
    Caps tightened for throughput.  Behaviour outside a cap is a documented
    divergence from the engine, so every rollout batch exposes overflow
    counters and any batch with a non-zero counter is void.

Nothing here is a guess: each number carries its derivation and safety factor
in ``SOURCES`` and is re-derived from the parity corpus by
``scripts/rl_derive_caps.py``.
"""

from __future__ import annotations

from typing import NamedTuple

from rl import rules

OVF_UNITS = 0
OVF_MARKET_STEPS = 1
OVF_ORDER_QTY = 2
OVF_PRICE_WINDOW = 3
OVF_WEED_DRAWS = 4
N_OVERFLOW = 5
OVERFLOW_NAMES = (
    "unit_slot_overflow",
    "market_step_overflow",
    "order_qty_overflow",
    "price_window_overflow",
    "weed_draw_overflow",
)


class Caps(NamedTuple):
    max_units: int  # farmer slot + hand slots
    max_orders: int  # hard engine cap (maxMarketOrdersPerTurn)
    market_steps: int  # flat lockstep-scan trip cap
    max_order_qty: int  # per-order quantity clamp
    name: str = "custom"


# --- derivations -----------------------------------------------------------
#
# max_units
#   Hard per-turn ceiling: 10 HIRE orders/turn x 24 turns.  The binding
#   constraint is money: the n-th hire of a day costs fib(n) with fib(0)=1, so
#   n hires cost fib(n+2)-1.  Reachable-money derivation:
#     * day 0 money is fixed at 3_000  -> at most 16 hires that day;
#     * the richest farm seen anywhere in the parity corpus / league is well
#       under 100_000 -> 23 hires;
#     * 4x that money (400_000) still only buys 27.
#   EXACT_REFERENCE uses 1 + 28 = 29 slots: 28 hires cost 832_039 and a 29th
#   would need 1_346_269, i.e. >13x the observed money ceiling (safety >13x).
#   BOUNDED_TRAINING also uses 29 -- i.e. no divergence from EXACT_REFERENCE
#   on this axis.  It previously used 1 + 16 = 17, derived from the fixed day-0
#   budget of $3,000 (17 hires would cost 4_180) and 1.4x the strongest hand
#   count seen in live top-10 play (12).  That derivation only bounds *day 0*:
#   a trained agent that grows its bank can fund more hires on later days, and
#   S3' run 2 did exactly that -- unit_slot_overflow first fired at update 91
#   and tripped the three-consecutive-batch stop at update 119, with every
#   other counter still zero.  Raising the training cap to the exact-reference
#   value removes the divergence instead of masking the behaviour; HIRE is
#   deliberately NOT blocked by an action mask, so the agent stays free to
#   choose the hire count and the counters stay meaningful.
#
# market_steps
#   The flat lockstep scan costs one trip per settled SELL / BUY_PRODUCT unit
#   plus one trip per order slot.  BUY_SEED / BUY_ANIMAL / HIRE / BUY_LAND
#   never touch market inventory, so they are settled in closed form and cost
#   no trips (see rl/market.py).  Per player per turn a SELL is bounded by shed
#   contents and a BUY_PRODUCT by shed room, both <= shedCapacity = 100; with
#   10 alternating slots the engine's own worst case is 10 x 100 = 1_000.
#   EXACT_REFERENCE therefore uses 1_000 + 10 = 1_010 (no divergence possible).
#   BOUNDED_TRAINING uses 210 = 2 x shedCapacity + 10, which covers any single
#   sell-everything / buy-to-full turn.
#
# max_order_qty
#   The engine has no quantity cap.  EXACT_REFERENCE clamps at 100_000 -- above
#   any quantity fundable at the $1 price floor from the richest reachable
#   farm.  BOUNDED_TRAINING clamps at 512.
SOURCES = {
    "max_units": (
        "engine: 10 HIRE orders/turn x 24 turns; money-bound n hires cost "
        "fib(n+2)-1 with startingMoney=3000 -> 16 on day 0; corpus money "
        "ceiling < 100k -> 23; exact mode uses 29 slots (>= 1.3M money). "
        "Training mode also uses 29: the day-0 budget bounds day 0 only, and a "
        "trained agent funds more hires later (S3' run 2 overflowed at u91)."
    ),
    "max_orders": "hard: configuration.maxMarketOrdersPerTurn default 10 (engine truncates).",
    "market_steps": (
        "shedCapacity=100 bounds SELL units by shed contents and BUY_PRODUCT "
        "units by shed room; 10 alternating order slots -> engine worst case "
        "1000 settled units + 10 slot-advance trips."
    ),
    "max_order_qty": "engine has no cap; exact mode clamps far above any fundable quantity.",
}

EXACT_REFERENCE = Caps(
    max_units=29,
    max_orders=rules.DEFAULT_CONFIG["maxMarketOrdersPerTurn"],
    market_steps=1010,
    max_order_qty=100_000,
    name="exact_reference",
)

BOUNDED_TRAINING = Caps(
    max_units=29,
    max_orders=rules.DEFAULT_CONFIG["maxMarketOrdersPerTurn"],
    market_steps=210,
    max_order_qty=512,
    name="bounded_training",
)

PRESETS = {c.name: c for c in (EXACT_REFERENCE, BOUNDED_TRAINING)}
