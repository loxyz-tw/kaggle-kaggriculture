"""Fixed-shape simulator state, plus encoders to/from official engine state."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import rules

I32 = jnp.int32


class SimState(NamedTuple):
    """Struct-of-arrays episode state.  Leading axis is the batch."""

    step: jax.Array  # [B]        turns elapsed; day = step // 24
    money: jax.Array  # [B, 2]     always integral in the engine
    tiles: jax.Array  # [B, 2, 100, C]
    unit_x: jax.Array  # [B, 2, U]
    unit_y: jax.Array  # [B, 2, U]
    hires_today: jax.Array  # [B, 2]  == len(farm["hands"]) in the engine
    n_quads: jax.Array  # [B, 2]  extra unlocked quadrants (0..3)
    shed: jax.Array  # [B, 2, 12]
    seeds: jax.Array  # [B, 2, 5]
    inv: jax.Array  # [B, 2, U, 12]
    inv_seq: jax.Array  # [B, 2, U, 12]  dict insertion rank (0 = absent)
    inv_ctr: jax.Array  # [B, 2, U]      next insertion rank
    market_inv: jax.Array  # [B, 9]
    shop_seq: jax.Array  # [B, 8]  shop index per unlocked instance, -1 = empty
    n_shops: jax.Array  # [B]
    key: jax.Array  # [B, 2] uint32 PRNG key (training-mode RNG only)
    overflow: jax.Array  # [B, N_OVERFLOW] accumulated cap breaches
    market_trips: jax.Array  # [B] lockstep trips used by the last market phase
    market_units: jax.Array  # [B] SELL/BUY_PRODUCT units settled in that phase
    market_closed: jax.Array  # [B] BUY_SEED/BUY_ANIMAL units settled in closed form


def _empty_tile_row() -> np.ndarray:
    row = np.zeros(rules.N_TILE_CHANNELS, dtype=np.int32)
    row[rules.C_CROP] = -1
    row[rules.C_ANIMAL] = -1
    return row


CANONICAL_EMPTY = _empty_tile_row()


def initial_tiles_np() -> np.ndarray:
    """``[100, C]`` -- NW quadrant unlocked, the other three LOCKED."""
    tiles = np.tile(CANONICAL_EMPTY, (rules.N_TILES, 1))
    quad = np.asarray(rules.TILE_QUAD_RANK)
    tiles[quad > 0, rules.C_KIND] = rules.TK_LOCKED
    return tiles


def reset(batch: int, caps: caps_mod.Caps, seed: int = 0) -> SimState:
    u = caps.max_units
    tiles = np.broadcast_to(initial_tiles_np(), (batch, 2, rules.N_TILES, rules.N_TILE_CHANNELS))
    unit_x = np.zeros((batch, 2, u), dtype=np.int32)
    unit_y = np.zeros((batch, 2, u), dtype=np.int32)
    unit_x[:] = rules.DEFAULT_SPAWN[0]
    unit_y[:] = rules.DEFAULT_SPAWN[1]
    keys = jax.random.split(jax.random.PRNGKey(seed), batch)
    return SimState(
        step=jnp.zeros((batch,), I32),
        money=jnp.full((batch, 2), rules.DEFAULT_CONFIG["startingMoney"], I32),
        tiles=jnp.asarray(np.array(tiles), I32),
        unit_x=jnp.asarray(unit_x, I32),
        unit_y=jnp.asarray(unit_y, I32),
        hires_today=jnp.zeros((batch, 2), I32),
        n_quads=jnp.zeros((batch, 2), I32),
        shed=jnp.zeros((batch, 2, rules.N_ITEMS), I32),
        seeds=jnp.zeros((batch, 2, rules.N_CROPS), I32),
        inv=jnp.zeros((batch, 2, u, rules.N_ITEMS), I32),
        inv_seq=jnp.zeros((batch, 2, u, rules.N_ITEMS), I32),
        inv_ctr=jnp.ones((batch, 2, u), I32),
        market_inv=jnp.full((batch, rules.N_PRODUCTS), rules.MARKET_I0, I32),
        shop_seq=jnp.full((batch, rules.MAX_SHOP_INSTANCES), -1, I32),
        n_shops=jnp.zeros((batch,), I32),
        key=jnp.asarray(keys, jnp.uint32).reshape(batch, -1)[:, :2],
        overflow=jnp.zeros((batch, caps_mod.N_OVERFLOW), I32),
        market_trips=jnp.zeros((batch,), I32),
        market_units=jnp.zeros((batch,), I32),
        market_closed=jnp.zeros((batch,), I32),
    )


def unit_alive(state: SimState, caps: caps_mod.Caps) -> jax.Array:
    """``[B, 2, U]`` -- slot 0 is the farmer, slots 1..hires_today are hands."""
    idx = jnp.arange(caps.max_units, dtype=I32)
    live = jnp.minimum(state.hires_today, caps.max_units - 1)
    return idx[None, None, :] <= live[:, :, None]


# ---------------------------------------------------------------- encoders
def encode_tile(tile: Any) -> np.ndarray:
    """Engine tile (None / "LOCKED" / dict) -> canonical channel row."""
    row = CANONICAL_EMPTY.copy()
    if tile is None:
        return row
    if tile == "LOCKED":
        row[rules.C_KIND] = rules.TK_LOCKED
        return row
    kind = tile.get("kind")
    if kind == "WEED":
        row[rules.C_KIND] = rules.TK_WEED
        return row
    if kind == "PLANT":
        row[rules.C_KIND] = rules.TK_PLANT
        row[rules.C_CROP] = rules.ITEM_INDEX[tile["crop"]]
        row[rules.C_PLANTED_DAY] = tile["planted_day"]
        row[rules.C_WATERED] = int(bool(tile["watered_today"]))
        row[rules.C_UNWATERED] = tile["consecutive_unwatered"]
        row[rules.C_YIELD] = tile["yield_units"]
        row[rules.C_MLS] = tile["max_lifespan_step"]
        row[rules.C_FERT_UNTIL] = tile.get("fertilized_until_day", -1)
        return row
    row[rules.C_KIND] = rules.TK_COOP if kind == "COOP" else rules.TK_PASTURE
    if "animal" in tile:
        row[rules.C_ANIMAL] = rules.ITEM_INDEX[tile["animal"]] - rules.ANIMAL_ITEM_BASE
        row[rules.C_PLACED_DAY] = tile["placed_day"]
        row[rules.C_YIELD] = tile["yield_units"]
        row[rules.C_UNFED] = tile["consecutive_unfed"]
        row[rules.C_FED] = int(bool(tile["fed_today"]))
        row[rules.C_CARED] = int(bool(tile["cared_today"]))
        row[rules.C_FERT_AVAIL] = int(bool(tile["fertilizer_available"]))
        row[rules.C_CARE_BONUS] = tile.get("pending_care_bonus", 0)
    return row


class EncodedObservation(NamedTuple):
    """Numpy mirror of one engine observation, in simulator layout."""

    step: int
    money: np.ndarray  # [2]
    tiles: np.ndarray  # [2, 100, C]
    unit_x: np.ndarray  # [2, U]
    unit_y: np.ndarray  # [2, U]
    hires_today: np.ndarray  # [2]
    n_quads: np.ndarray  # [2]
    shed: np.ndarray  # [2, 12]
    seeds: np.ndarray  # [2, 5]
    inv: np.ndarray  # [2, U, 12]
    inv_order: list[list[list[int]]]  # per player, per unit: item ids in dict order
    market_inv: np.ndarray  # [9]
    prices: np.ndarray  # [9]
    shop_seq: np.ndarray  # [8]
    n_shops: int


def encode_observation(obs_by_player: list[Any], caps: caps_mod.Caps) -> EncodedObservation:
    """Encode ``[player0_observation, player1_observation]`` from a replay step."""
    obs0 = obs_by_player[0]
    farms = obs0["farms"]
    u = caps.max_units
    tiles = np.zeros((2, rules.N_TILES, rules.N_TILE_CHANNELS), dtype=np.int32)
    unit_x = np.zeros((2, u), dtype=np.int32)
    unit_y = np.zeros((2, u), dtype=np.int32)
    money = np.zeros(2, dtype=np.int64)
    hires = np.zeros(2, dtype=np.int32)
    n_quads = np.zeros(2, dtype=np.int32)
    shed = np.zeros((2, rules.N_ITEMS), dtype=np.int32)
    seeds = np.zeros((2, rules.N_CROPS), dtype=np.int32)
    inv = np.zeros((2, u, rules.N_ITEMS), dtype=np.int32)
    inv_order: list[list[list[int]]] = []
    for p in range(2):
        farm = farms[p]
        for flat in range(rules.N_TILES):
            y, x = divmod(flat, rules.BOARD_SIZE)
            tiles[p, flat] = encode_tile(farm["tiles"][y][x])
        money[p] = int(farm["money"])
        hires[p] = farm["hires_today"]
        n_quads[p] = len(farm["unlocked_quadrants"]) - 1
        unit_x[p, :] = rules.DEFAULT_SPAWN[0]
        unit_y[p, :] = rules.DEFAULT_SPAWN[1]
        unit_x[p, 0], unit_y[p, 0] = farm["farmer"][0], farm["farmer"][1]
        for h, pos in enumerate(farm["hands"]):
            if h + 1 < u:
                unit_x[p, h + 1], unit_y[p, h + 1] = pos[0], pos[1]
        private = obs_by_player[p]["private"]
        for item, n in private["shed"].items():
            shed[p, rules.ITEM_INDEX[item]] = n
        for crop, n in private["seeds"].items():
            seeds[p, rules.ITEM_INDEX[crop]] = n
        orders_p: list[list[int]] = []
        for ui, unit_inv in enumerate(private["inventories"]):
            order = []
            for item, n in unit_inv.items():
                if n <= 0:
                    continue
                order.append(rules.ITEM_INDEX[item])
                if ui < u:
                    inv[p, ui, rules.ITEM_INDEX[item]] = n
            orders_p.append(order)
        inv_order.append(orders_p)
    market = obs0["market"]
    market_inv = np.array([market["inventory"][i] for i in rules.PRODUCTS], dtype=np.int64)
    prices = np.array([market["prices"][i] for i in rules.PRODUCTS], dtype=np.int64)
    shops = obs0["town"]["unlocked_shops"]
    shop_seq = np.full(rules.MAX_SHOP_INSTANCES, -1, dtype=np.int32)
    for i, name in enumerate(shops):
        shop_seq[i] = rules.SHOP_NAMES.index(name)
    return EncodedObservation(
        step=int(obs0["step"]),
        money=money,
        tiles=tiles,
        unit_x=unit_x,
        unit_y=unit_y,
        hires_today=hires,
        n_quads=n_quads,
        shed=shed,
        seeds=seeds,
        inv=inv,
        inv_order=inv_order,
        market_inv=market_inv,
        prices=prices,
        shop_seq=shop_seq,
        n_shops=len(shops),
    )


def state_from_observation(
    obs_by_player: list[Any], caps: caps_mod.Caps, seed: int = 0
) -> SimState:
    """Build a batch-1 :class:`SimState` from an arbitrary engine observation.

    Used by the micro-fixture oracle so a fixture can start from any hand-built
    engine state rather than only from ``reset``.
    """
    enc = encode_observation(obs_by_player, caps)
    u = caps.max_units
    inv_seq = np.zeros((2, u, rules.N_ITEMS), dtype=np.int32)
    inv_ctr = np.ones((2, u), dtype=np.int32)
    for p in range(2):
        for ui, unit_inv in enumerate(obs_by_player[p]["private"]["inventories"]):
            if ui >= u:
                break
            rank = 1
            for item, n in unit_inv.items():
                if n <= 0:
                    continue
                inv_seq[p, ui, rules.ITEM_INDEX[item]] = rank
                rank += 1
            inv_ctr[p, ui] = rank
    base = reset(1, caps, seed=seed)
    return base._replace(
        step=jnp.asarray([enc.step], I32),
        money=jnp.asarray(enc.money[None], I32),
        tiles=jnp.asarray(enc.tiles[None], I32),
        unit_x=jnp.asarray(enc.unit_x[None], I32),
        unit_y=jnp.asarray(enc.unit_y[None], I32),
        hires_today=jnp.asarray(enc.hires_today[None], I32),
        n_quads=jnp.asarray(enc.n_quads[None], I32),
        shed=jnp.asarray(enc.shed[None], I32),
        seeds=jnp.asarray(enc.seeds[None], I32),
        inv=jnp.asarray(enc.inv[None], I32),
        inv_seq=jnp.asarray(inv_seq[None], I32),
        inv_ctr=jnp.asarray(inv_ctr[None], I32),
        market_inv=jnp.asarray(enc.market_inv[None], I32),
        shop_seq=jnp.asarray(enc.shop_seq[None], I32),
        n_shops=jnp.asarray([enc.n_shops], I32),
    )


def state_slice_np(state: SimState, b: int, caps: caps_mod.Caps) -> dict[str, np.ndarray]:
    """Pull one batch element off device as numpy, in comparison layout."""
    alive = np.asarray(unit_alive(state, caps))[b]
    inv = np.asarray(state.inv)[b]
    seq = np.asarray(state.inv_seq)[b]
    order: list[list[list[int]]] = []
    for p in range(2):
        per_unit = []
        for ui in range(caps.max_units):
            if not alive[p, ui]:
                per_unit.append([])
                continue
            present = [i for i in range(rules.N_ITEMS) if inv[p, ui, i] > 0]
            present.sort(key=lambda i: seq[p, ui, i])
            per_unit.append(present)
        order.append(per_unit)
    return {
        "step": int(np.asarray(state.step)[b]),
        "money": np.asarray(state.money)[b].astype(np.int64),
        "tiles": np.asarray(state.tiles)[b],
        "unit_x": np.asarray(state.unit_x)[b],
        "unit_y": np.asarray(state.unit_y)[b],
        "hires_today": np.asarray(state.hires_today)[b],
        "n_quads": np.asarray(state.n_quads)[b],
        "shed": np.asarray(state.shed)[b],
        "seeds": np.asarray(state.seeds)[b],
        "inv": inv,
        "inv_order": order,
        "market_inv": np.asarray(state.market_inv)[b].astype(np.int64),
        "shop_seq": np.asarray(state.shop_seq)[b],
        "n_shops": int(np.asarray(state.n_shops)[b]),
        "overflow": np.asarray(state.overflow)[b],
        "market_trips": int(np.asarray(state.market_trips)[b]),
        "market_units": int(np.asarray(state.market_units)[b]),
        "market_closed": int(np.asarray(state.market_closed)[b]),
    }
