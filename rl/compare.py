"""Full-state comparison between a simulator slice and an engine observation.

Every field the engine records is compared; nothing is sampled and nothing is
tolerated.  Integers are compared bit-exactly, money included (it is always
integral in the engine).  ``inv_order`` compares the *dict insertion order* of
each unit inventory, because that order decides which items are discarded when
the shed overflows.
"""

from __future__ import annotations

import numpy as np

from rl import caps as caps_mod
from rl import rules
from rl.state import EncodedObservation

COMPARED_FIELDS = (
    "step",
    "money",
    "tiles",
    "unit_pos",
    "hires_today",
    "n_quads",
    "shed",
    "seeds",
    "inv",
    "inv_order",
    "market_inv",
    "shop_seq",
    "n_shops",
    "prices",
)


def compare(sim: dict, ref: EncodedObservation, caps: caps_mod.Caps) -> list[str]:
    """Return human-readable mismatch descriptions (empty list == identical)."""
    bad: list[str] = []
    if sim["step"] != ref.step:
        bad.append(f"step {sim['step']} != {ref.step}")
    if not np.array_equal(sim["money"], ref.money):
        bad.append(f"money {sim['money'].tolist()} != {ref.money.tolist()}")
    if not np.array_equal(sim["tiles"], ref.tiles):
        diff = np.argwhere(sim["tiles"] != ref.tiles)
        p, flat, ch = diff[0]
        bad.append(
            f"tiles[p{p}][x={flat % 10},y={flat // 10}].{rules.TILE_CHANNEL_NAMES[ch]} "
            f"{sim['tiles'][p, flat, ch]} != {ref.tiles[p, flat, ch]} "
            f"({len(diff)} channel diffs)"
        )
    for p in range(2):
        n_alive = min(int(ref.hires_today[p]) + 1, caps.max_units)
        sx, sy = sim["unit_x"][p, :n_alive], sim["unit_y"][p, :n_alive]
        rx, ry = ref.unit_x[p, :n_alive], ref.unit_y[p, :n_alive]
        if not (np.array_equal(sx, rx) and np.array_equal(sy, ry)):
            bad.append(
                f"unit_pos[p{p}] {list(zip(sx.tolist(), sy.tolist(), strict=True))} != "
                f"{list(zip(rx.tolist(), ry.tolist(), strict=True))}"
            )
    for key in ("hires_today", "n_quads", "shed", "seeds", "inv", "market_inv", "shop_seq"):
        got = sim[key]
        want = getattr(ref, key)
        if key == "inv":
            got = got[:, : want.shape[1]]
        if not np.array_equal(got, want):
            bad.append(f"{key} mismatch: {got.tolist()} != {want.tolist()}")
    if sim["n_shops"] != ref.n_shops:
        bad.append(f"n_shops {sim['n_shops']} != {ref.n_shops}")
    for p in range(2):
        want_p = [list(o) for o in ref.inv_order[p]]
        got_p = sim["inv_order"][p][: len(want_p)]
        if got_p != want_p:
            bad.append(f"inv_order[p{p}] {got_p} != {want_p}")
    return bad
