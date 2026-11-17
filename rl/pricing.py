"""Exact market price lookup table.

``kaggriculture.market_price`` is ``max(1, int(round(<float64 expression>)))``.
Reproducing that expression in float32 on GPU risks flipping the
round-half-to-even boundary, so instead the table is built **by calling the
engine itself** over a static inventory window and then gathered on device.
That makes the price path exact by construction rather than by numerical luck,
and turns any inventory outside the window into an explicit overflow counter
rather than a silent wrong price.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from rl import rules

# Window derivation (documented in 內部規格文件):
#   * inventory starts at I0 = 10_000 for every product;
#   * it only falls through town demand -- bounded by 8 shop instances x 2 units
#     x 6 ticks/day x 30 days = 2_880 plus 30 town-centre pulls -- and through
#     BUY_PRODUCT, which is capped by shed capacity (100/turn/player) and money;
#   * it only rises through SELL, capped by shed capacity per turn per player.
#   A +-20k / +50k window is >6x the largest excursion ever observed in the
#   parity corpus, and excursions past it are counted, not tolerated.
INV_MIN = -20_000
INV_MAX = 60_000
INV_SPAN = INV_MAX - INV_MIN + 1

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_REPO_ROOT, "reports", "rl_cache")


def _cache_key() -> str:
    payload = f"{rules.ENGINE_VERSION}|{INV_MIN}|{INV_MAX}|{'.'.join(rules.PRODUCTS)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_price_table() -> np.ndarray:
    """``[N_PRODUCTS, INV_SPAN]`` int32 table; index by ``inventory - INV_MIN``."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as eng

    table = np.empty((rules.N_PRODUCTS, INV_SPAN), dtype=np.int32)
    for pi, item in enumerate(rules.PRODUCTS):
        row = table[pi]
        price = eng.market_price
        for k in range(INV_SPAN):
            row[k] = price(item, k + INV_MIN)
    return table


def load_price_table() -> np.ndarray:
    path = os.path.join(_CACHE_DIR, f"price_table_{_cache_key()}.npy")
    if os.path.exists(path):
        return np.load(path)
    table = build_price_table()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npy"
    np.save(tmp, table)
    os.replace(tmp, path)
    return table


_TABLE: np.ndarray | None = None


def price_table() -> np.ndarray:
    global _TABLE
    if _TABLE is None:
        _TABLE = load_price_table()
    return _TABLE
