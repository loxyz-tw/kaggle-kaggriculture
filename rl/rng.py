"""RNG transplant.

Orbit Wars maps were deterministic once generated; Kaggriculture is not.  The
engine drives weed spawns and town-shop unlocks from
``random.Random((seed * 1_000_003) ^ day)``, a Mersenne Twister that JAX cannot
reproduce.  Worse, the *number* of floats consumed before the shop draw is
state-dependent: ``_spawn_weeds`` draws once per **empty** tile, player 0's
board then player 1's.

So exact-reference mode does not reimplement the RNG, it transplants it:

* ``weed_hit[day, k]`` -- the verdict of the ``k``-th weed draw of that day;
* ``shop_choice[day, k]`` -- the shop the engine draws *after* exactly ``k``
  float draws, tabulated for every reachable ``k``.

The simulator then consumes ``weed_hit`` with an exclusive cumsum over the
empty-tile mask (which is exactly the engine's draw index) and looks the shop
up at the resulting total.  Training mode ignores these tables and uses JAX's
own PRNG instead.
"""

from __future__ import annotations

import random

import jax.numpy as jnp
import numpy as np

from rl import kernels, rules

# 2 players x 100 tiles is the hard per-day ceiling on ``rng.random()`` calls:
# a draw happens only on an empty *unlocked* tile.
MAX_WEED_DRAWS = 2 * rules.N_TILES


def record_seed(seed: int, n_days: int = rules.N_DAYS, weed_chance: float | None = None):
    """Return ``(weed_hit[n_days, 200] uint8, shop_choice[n_days, 201] int32)``."""
    if weed_chance is None:
        weed_chance = rules.DEFAULT_CONFIG["weedSpawnChance"]
    weed = np.zeros((n_days, MAX_WEED_DRAWS), dtype=np.uint8)
    shop = np.zeros((n_days, MAX_WEED_DRAWS + 1), dtype=np.int32)
    shop_names = sorted(rules.SHOP_NAMES)
    for day in range(n_days):
        rng = random.Random((seed * 1_000_003) ^ day)
        for k in range(MAX_WEED_DRAWS + 1):
            snapshot = rng.getstate()
            shop[day, k] = rules.SHOP_NAMES.index(rng.choice(shop_names))
            rng.setstate(snapshot)
            if k < MAX_WEED_DRAWS:
                weed[day, k] = 1 if rng.random() < weed_chance else 0
    return weed, shop


def record_tables(seeds, n_days: int = rules.N_DAYS) -> kernels.RngTables:
    """Stack per-seed tables into a batched :class:`rl.kernels.RngTables`."""
    weeds, shops = [], []
    for seed in seeds:
        w, s = record_seed(int(seed), n_days)
        weeds.append(w)
        shops.append(s)
    return kernels.RngTables(
        weed_hit=jnp.asarray(np.stack(weeds), jnp.int32),
        shop_choice=jnp.asarray(np.stack(shops), jnp.int32),
    )
