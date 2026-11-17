"""Fixed-shape JAX transition kernels, transcribed one-to-one from the engine.

Slice order (each slice has its own micro-fixtures in ``tests/``):

1. :func:`apply_unit_actions` -- per-unit verbs, plant lifecycle, inventories.
2. :func:`process_market`     -- the per-unit lockstep market.
3. :func:`town_consume` / :func:`decay_plants` / :func:`end_of_day`.

Engine turn order (``kaggriculture.interpreter``)::

    unit actions (p0 then p1) -> _process_market -> _town_consume(step)
    -> _decay_plants(step) -> _end_of_day(day) if (step + 1) % 24 == 0

Two facts that shape every kernel here:

* an order list is settled **in list order and stops when cash runs out**, so
  order sequence is itself the cash allocator;
* ``BUY_SEED`` / ``BUY_ANIMAL`` / ``HIRE`` / ``BUY_LAND`` never touch market
  inventory, so they are provably independent of the opponent's interleaved
  units and are settled in closed form.  Only ``SELL`` / ``BUY_PRODUCT`` need
  the true per-unit lockstep loop.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import pricing, rules
from rl.actions import SimAction
from rl.state import CANONICAL_EMPTY, SimState

I32 = jnp.int32
_BIG = jnp.int32(1 << 30)


class RngTables(NamedTuple):
    """Recorded engine RNG for exact-reference replay (see ``rl/rng.py``).

    ``weed_hit[b, day, k]`` is the engine's ``rng.random() < weedSpawnChance``
    verdict for the ``k``-th weed draw of that day (player 0's empty tiles in
    (y, x) order, then player 1's).  ``shop_choice[b, day, k]`` is the shop the
    engine would draw *after* ``k`` float draws, which is what makes the draw
    count -- a state-dependent quantity -- reproducible without reimplementing
    the Mersenne Twister.
    """

    weed_hit: jax.Array  # [B, DAYS, 2 * 100] bool
    shop_choice: jax.Array  # [B, DAYS, 2 * 100 + 1] int32


# --------------------------------------------------------------- constants
_MOVE_DX = jnp.asarray(rules.MOVE_DX, I32)
_MOVE_DY = jnp.asarray(rules.MOVE_DY, I32)
_CROP_SEED_COST = jnp.asarray(rules.CROP_SEED_COST, I32)
_CROP_FIRST_YIELD = jnp.asarray(rules.CROP_FIRST_YIELD_DAY, I32)
_CROP_MAX_YIELD_DAY = jnp.asarray(rules.CROP_MAX_YIELD_DAY, I32)
_CROP_INTERVAL = jnp.asarray(rules.CROP_INTERVAL, I32)
_CROP_MAX_YIELD = jnp.asarray(rules.CROP_MAX_YIELD, I32)
_CROP_ONGOING = jnp.asarray(rules.CROP_ONGOING, I32)
_CROP_WINDOW_START = jnp.asarray(rules.CROP_WATER_WINDOW_START, I32)
_ANIMAL_COST = jnp.asarray(rules.ANIMAL_COST, I32)
_ANIMAL_FIRST_YIELD = jnp.asarray(rules.ANIMAL_FIRST_YIELD_DAY, I32)
_ANIMAL_INTERVAL = jnp.asarray(rules.ANIMAL_INTERVAL, I32)
_ANIMAL_MAX_HELD = jnp.asarray(rules.ANIMAL_MAX_HELD, I32)
_ANIMAL_PRODUCT = jnp.asarray(rules.ANIMAL_PRODUCT_ITEM, I32)
_ANIMAL_STRUCT = jnp.asarray(rules.ANIMAL_STRUCTURE_KIND, I32)
_LAND_PRICES = jnp.asarray(rules.LAND_PRICES, I32)
_TILE_QUAD_RANK = jnp.asarray(rules.TILE_QUAD_RANK, I32)
_SHED_ACCESS_FLAT = jnp.asarray(rules.SHED_ACCESS_FLAT, I32)
_SHOP_DEMAND = jnp.asarray(rules.SHOP_DEMAND, I32)

_FIB_LEN = 46
_FIB_TABLE = jnp.asarray([min(rules.fib(n), (1 << 31) - 1) for n in range(_FIB_LEN)], I32)

_EMPTY_ROW = jnp.asarray(CANONICAL_EMPTY, I32)


def _kind_row(kind: int) -> jax.Array:
    return _EMPTY_ROW.at[rules.C_KIND].set(kind)


_WEED_ROW = _kind_row(rules.TK_WEED)
_COOP_ROW = _kind_row(rules.TK_COOP)
_PASTURE_ROW = _kind_row(rules.TK_PASTURE)
_STRUCT_ROWS = jnp.stack([_COOP_ROW, _PASTURE_ROW])  # index by kind - TK_COOP

# Materialised eagerly at import: creating it lazily inside a traced function
# would capture a tracer in module state.
_PRICE_TABLE = jnp.asarray(np.ascontiguousarray(pricing.price_table().reshape(-1)), I32)


def price_table_device() -> jax.Array:
    return _PRICE_TABLE


def lookup_price(product: jax.Array, inventory: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Exact ``market_price`` via table gather.  Returns ``(price, in_window)``."""
    table = price_table_device()
    in_window = (inventory >= pricing.INV_MIN) & (inventory <= pricing.INV_MAX)
    off = jnp.clip(inventory - pricing.INV_MIN, 0, pricing.INV_SPAN - 1)
    lin = product * pricing.INV_SPAN + off
    return table[lin], in_window


def prices_of(market_inv: jax.Array) -> jax.Array:
    """``[B, 9]`` current quoted prices -- a pure function of inventory."""
    product = jnp.arange(rules.N_PRODUCTS, dtype=I32)[None, :]
    price, _ = lookup_price(jnp.broadcast_to(product, market_inv.shape), market_inv)
    return price


def _add_overflow(overflow: jax.Array, slot: int, count: jax.Array) -> jax.Array:
    return overflow.at[:, slot].add(count.astype(I32))


# ------------------------------------------------------------------ gathers
def _gather_tiles(tiles: jax.Array, flat: jax.Array) -> jax.Array:
    """``tiles[B,2,100,C]``, ``flat[B,2]`` -> ``[B,2,C]``."""
    return jnp.take_along_axis(tiles, flat[:, :, None, None], axis=2)[:, :, 0, :]


def _scatter_tiles(tiles: jax.Array, flat: jax.Array, row: jax.Array) -> jax.Array:
    b = jnp.arange(tiles.shape[0])[:, None]
    p = jnp.arange(2)[None, :]
    return tiles.at[b, p, flat].set(row)


def _dyn2(arr: jax.Array, k: jax.Array) -> jax.Array:
    """``arr[B,2,U]`` -> ``arr[:, :, k]``."""
    return jax.lax.dynamic_index_in_dim(arr, k, axis=2, keepdims=False)


def _dyn3(arr: jax.Array, k: jax.Array) -> jax.Array:
    """``arr[B,2,U,I]`` -> ``arr[:, :, k, :]``."""
    return jax.lax.dynamic_index_in_dim(arr, k, axis=2, keepdims=False)


def _set_dyn2(arr: jax.Array, k: jax.Array, value: jax.Array) -> jax.Array:
    return jax.lax.dynamic_update_index_in_dim(arr, value, k, axis=2)


def _set_dyn3(arr: jax.Array, k: jax.Array, value: jax.Array) -> jax.Array:
    return jax.lax.dynamic_update_index_in_dim(arr, value, k, axis=2)


def _take_item(table: jax.Array, idx: jax.Array) -> jax.Array:
    """``table[..., I]``, ``idx[...]`` -> ``table[..., idx]``."""
    return jnp.take_along_axis(table, idx[..., None], axis=-1)[..., 0]


def _one_hot(idx: jax.Array, n: int) -> jax.Array:
    return (jnp.arange(n, dtype=I32) == idx[..., None]).astype(I32)


def _update_inv_bookkeeping(
    old_inv: jax.Array, new_inv: jax.Array, seq: jax.Array, ctr: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Mirror CPython dict insertion order: a key re-added lands at the end."""
    newly = (old_inv == 0) & (new_inv > 0)
    gone = (old_inv > 0) & (new_inv == 0)
    seq = jnp.where(newly, ctr[..., None], jnp.where(gone, 0, seq))
    return seq, ctr + jnp.sum(newly.astype(I32), axis=-1)


# ---------------------------------------------------------------- slice 1
def apply_unit_actions(state: SimState, action: SimAction, caps: caps_mod.Caps) -> SimState:
    """``interpreter``'s per-player unit loop, including atomic PLANT validation."""
    u = caps.max_units
    day = (state.step // rules.TURNS_PER_DAY)[:, None]  # [B, 1]
    live = jnp.minimum(state.hires_today, u - 1)  # [B, 2]

    op = action.unit_op
    arg = action.unit_arg
    qty = action.unit_qty

    # Atomic PLANT validation: if total PLANT requests for a crop exceed the
    # seeds held, every PLANT request for that crop becomes PASS.
    crop_ok = (arg >= 0) & (arg < rules.N_CROPS)
    is_plant = op == rules.OP_PLANT
    crop_idx = jnp.clip(arg, 0, rules.N_CROPS - 1)
    demand = jnp.sum(
        jnp.where((is_plant & crop_ok)[..., None], _one_hot(crop_idx, rules.N_CROPS), 0), axis=2
    )
    blocked = demand > state.seeds  # [B, 2, 5]
    blocked_here = jnp.take_along_axis(blocked, crop_idx, axis=2)
    op = jnp.where(is_plant & crop_ok & blocked_here, rules.OP_PASS, op)

    n_units = jnp.minimum(jnp.max(state.hires_today) + 1, u)

    def body(k, carry):
        tiles, ux, uy, shed, seeds, inv, seq, ctr = carry
        opk = _dyn2(op, k)
        argk = _dyn2(arg, k)
        qtyk = _dyn2(qty, k)
        alive = k <= live
        x, y = _dyn2(ux, k), _dyn2(uy, k)
        inv_k, seq_k, ctr_k = _dyn3(inv, k), _dyn3(seq, k), _dyn2(ctr, k)

        flat = y * rules.BOARD_SIZE + x
        t = _gather_tiles(tiles, flat)
        kind = t[..., rules.C_KIND]
        crop = jnp.clip(t[..., rules.C_CROP], 0, rules.N_CROPS - 1)
        animal = t[..., rules.C_ANIMAL]
        anim_i = jnp.clip(animal, 0, rules.N_ANIMALS - 1)
        yield_u = t[..., rules.C_YIELD]

        # ---- movement (allowed onto LOCKED tiles; out-of-bounds is a no-op)
        mv = alive & (opk >= rules.OP_NORTH) & (opk <= rules.OP_WEST)
        nx = x + _MOVE_DX[opk]
        ny = y + _MOVE_DY[opk]
        mv = mv & (nx >= 0) & (nx < rules.BOARD_SIZE) & (ny >= 0) & (ny < rules.BOARD_SIZE)
        ux = _set_dyn2(ux, k, jnp.where(mv, nx, x))
        uy = _set_dyn2(uy, k, jnp.where(mv, ny, y))

        shed_adj = jnp.any(flat[..., None] == _SHED_ACCESS_FLAT, axis=-1)
        item_ok = (argk >= 0) & (argk < rules.N_ITEMS)
        item_i = jnp.clip(argk, 0, rules.N_ITEMS - 1)
        item_oh = _one_hot(item_i, rules.N_ITEMS)
        inv_at = _take_item(inv_k, item_i)
        shed_at = _take_item(shed, item_i)
        shed_total = jnp.sum(shed, axis=-1)
        room = jnp.maximum(0, rules.SHED_CAPACITY - shed_total)

        d_inv = jnp.zeros_like(inv_k)
        d_shed = jnp.zeros_like(shed)
        d_seeds = jnp.zeros_like(seeds)

        # ---- PICKUP (resolves before the LOCKED guard)
        n_pick = jnp.minimum(qtyk, shed_at)
        c_pick = alive & (opk == rules.OP_PICKUP) & shed_adj & item_ok & (qtyk > 0) & (n_pick > 0)
        n_pick = jnp.where(c_pick, n_pick, 0)
        d_inv = d_inv + item_oh * n_pick[..., None]
        d_shed = d_shed - item_oh * n_pick[..., None]

        # ---- PLACE: animal onto a matching empty structure, else shed drop
        is_anim_item = (argk >= rules.ANIMAL_ITEM_BASE) & (argk < rules.N_ITEMS)
        a_local = jnp.clip(argk - rules.ANIMAL_ITEM_BASE, 0, rules.N_ANIMALS - 1)
        c_place_a = (
            alive
            & (opk == rules.OP_PLACE)
            & is_anim_item
            & (kind == _ANIMAL_STRUCT[a_local])
            & (animal < 0)
        )
        c_place_a_ok = c_place_a & (inv_at >= 1)
        d_inv = d_inv - item_oh * c_place_a_ok[..., None].astype(I32)

        n_drop1 = jnp.minimum(jnp.minimum(qtyk, inv_at), room)
        c_place_b = (
            alive
            & (opk == rules.OP_PLACE)
            & (~c_place_a)
            & shed_adj
            & item_ok
            & (qtyk > 0)
            & (n_drop1 > 0)
        )
        n_drop1 = jnp.where(c_place_b, n_drop1, 0)
        d_inv = d_inv - item_oh * n_drop1[..., None]
        d_shed = d_shed + item_oh * n_drop1[..., None]

        # ---- DROP: whole inventory to shed, in dict insertion order
        c_drop = alive & (opk == rules.OP_DROP) & shed_adj
        sort_key = jnp.where(inv_k > 0, seq_k, _BIG)
        order = jnp.argsort(sort_key, axis=-1)
        n_sorted = jnp.where(
            jnp.take_along_axis(sort_key, order, axis=-1) >= _BIG,
            0,
            jnp.take_along_axis(inv_k, order, axis=-1),
        )
        excl = jnp.cumsum(n_sorted, axis=-1) - n_sorted
        take_sorted = jnp.clip(
            jnp.minimum(n_sorted, rules.SHED_CAPACITY - shed_total[..., None] - excl), 0
        )
        rank = jnp.argsort(order, axis=-1)
        take = jnp.take_along_axis(take_sorted, rank, axis=-1)
        d_shed = d_shed + jnp.where(c_drop[..., None], take, 0)
        d_inv = d_inv + jnp.where(c_drop[..., None], -inv_k, 0)

        # ---- tile operations (all require a non-LOCKED tile)
        not_locked = kind != rules.TK_LOCKED
        empty = kind == rules.TK_EMPTY
        is_plant_tile = kind == rules.TK_PLANT
        has_animal = animal >= 0

        # PLANT
        crop_idx_k = jnp.clip(argk, 0, rules.N_CROPS - 1)
        seeds_at = _take_item(seeds, crop_idx_k)
        c_plant = (
            alive
            & (opk == rules.OP_PLANT)
            & empty
            & (argk >= 0)
            & (argk < rules.N_CROPS)
            & (seeds_at > 0)
        )
        d_seeds = d_seeds - _one_hot(crop_idx_k, rules.N_CROPS) * c_plant[..., None].astype(I32)
        ongoing_new = _CROP_ONGOING[crop_idx_k]
        plant_row = jnp.broadcast_to(_EMPTY_ROW, t.shape)
        plant_row = plant_row.at[..., rules.C_KIND].set(rules.TK_PLANT)
        plant_row = plant_row.at[..., rules.C_CROP].set(crop_idx_k)
        plant_row = plant_row.at[..., rules.C_PLANTED_DAY].set(jnp.broadcast_to(day, t.shape[:2]))
        plant_row = plant_row.at[..., rules.C_UNWATERED].set(1)
        plant_row = plant_row.at[..., rules.C_YIELD].set(jnp.where(ongoing_new == 1, 0, 1))
        plant_row = plant_row.at[..., rules.C_MLS].set(
            jnp.where(
                ongoing_new == 1,
                -1,
                (day + _CROP_MAX_YIELD_DAY[crop_idx_k] + 1) * rules.TURNS_PER_DAY,
            )
        )
        plant_row = plant_row.at[..., rules.C_FERT_UNTIL].set(-1)

        # WATER
        c_water = alive & (opk == rules.OP_WATER) & is_plant_tile & (t[..., rules.C_WATERED] == 0)
        age = day - t[..., rules.C_PLANTED_DAY]
        in_window = (age >= _CROP_WINDOW_START[crop]) & (age <= _CROP_MAX_YIELD_DAY[crop])
        bonus = jnp.where(t[..., rules.C_FERT_UNTIL] >= day, 2, 1)
        watered_yield = jnp.minimum(_CROP_MAX_YIELD[crop], yield_u + bonus)
        c_water_bonus = c_water & (_CROP_ONGOING[crop] == 0) & in_window

        # HARVEST
        c_harv = alive & (opk == rules.OP_HARVEST) & (kind >= rules.TK_WEED) & (yield_u > 0)
        mature = (day - t[..., rules.C_PLANTED_DAY]) >= _CROP_FIRST_YIELD[crop]
        c_harv_p = c_harv & is_plant_tile & mature
        c_harv_a = c_harv & has_animal
        d_inv = d_inv + _one_hot(crop, rules.N_ITEMS) * jnp.where(c_harv_p, yield_u, 0)[..., None]
        d_inv = (
            d_inv
            + _one_hot(_ANIMAL_PRODUCT[anim_i], rules.N_ITEMS)
            * jnp.where(c_harv_a, yield_u, 0)[..., None]
        )
        c_harv_clear = c_harv_p & (_CROP_ONGOING[crop] == 0)

        # FERTILIZE
        fert_held = _take_item(inv_k, jnp.full_like(item_i, rules.I_FERTILIZER))
        c_fert = alive & (opk == rules.OP_FERTILIZE) & is_plant_tile & (fert_held >= 1)
        fert_oh = _one_hot(jnp.full_like(item_i, rules.I_FERTILIZER), rules.N_ITEMS)
        d_inv = d_inv - fert_oh * c_fert[..., None].astype(I32)

        # DIG / BUILD
        c_dig = alive & (opk == rules.OP_DIG) & not_locked & (~empty) & (~has_animal)
        c_coop = alive & (opk == rules.OP_BUILD_COOP) & empty
        c_past = alive & (opk == rules.OP_BUILD_PASTURE) & empty

        # FEED / COLLECT_FERTILIZER / CARE
        wheat_held = _take_item(inv_k, jnp.full_like(item_i, rules.I_WHEAT))
        c_feed = (
            alive
            & (opk == rules.OP_FEED)
            & has_animal
            & (t[..., rules.C_FED] == 0)
            & (wheat_held >= 1)
        )
        wheat_oh = _one_hot(jnp.full_like(item_i, rules.I_WHEAT), rules.N_ITEMS)
        d_inv = d_inv - wheat_oh * c_feed[..., None].astype(I32)
        c_coll = (
            alive
            & (opk == rules.OP_COLLECT_FERTILIZER)
            & has_animal
            & (t[..., rules.C_FERT_AVAIL] == 1)
        )
        d_inv = d_inv + fert_oh * c_coll[..., None].astype(I32)
        c_care = alive & (opk == rules.OP_CARE) & has_animal & (t[..., rules.C_CARED] == 0)

        # ---- assemble the new tile
        nt = t
        nt = nt.at[..., rules.C_WATERED].set(jnp.where(c_water, 1, nt[..., rules.C_WATERED]))
        nt = nt.at[..., rules.C_YIELD].set(
            jnp.where(c_water_bonus, watered_yield, nt[..., rules.C_YIELD])
        )
        nt = nt.at[..., rules.C_YIELD].set(
            jnp.where(c_harv_p | c_harv_a, 0, nt[..., rules.C_YIELD])
        )
        nt = nt.at[..., rules.C_FERT_UNTIL].set(
            jnp.where(
                c_fert,
                jnp.maximum(nt[..., rules.C_FERT_UNTIL], day + 2),
                nt[..., rules.C_FERT_UNTIL],
            )
        )
        nt = nt.at[..., rules.C_FED].set(jnp.where(c_feed, 1, nt[..., rules.C_FED]))
        nt = nt.at[..., rules.C_CARED].set(jnp.where(c_care, 1, nt[..., rules.C_CARED]))
        nt = nt.at[..., rules.C_FERT_AVAIL].set(jnp.where(c_coll, 0, nt[..., rules.C_FERT_AVAIL]))
        # PLACE animal keeps the structure kind and stamps a fresh animal.
        nt = nt.at[..., rules.C_ANIMAL].set(
            jnp.where(c_place_a_ok, a_local, nt[..., rules.C_ANIMAL])
        )
        nt = nt.at[..., rules.C_PLACED_DAY].set(
            jnp.where(c_place_a_ok, day, nt[..., rules.C_PLACED_DAY])
        )
        for ch in (
            rules.C_YIELD,
            rules.C_UNFED,
            rules.C_FED,
            rules.C_CARED,
            rules.C_FERT_AVAIL,
            rules.C_CARE_BONUS,
        ):
            nt = nt.at[..., ch].set(jnp.where(c_place_a_ok, 0, nt[..., ch]))

        nt = jnp.where(c_plant[..., None], plant_row, nt)
        nt = jnp.where((c_dig | c_harv_clear)[..., None], _EMPTY_ROW, nt)
        nt = jnp.where(c_coop[..., None], _COOP_ROW, nt)
        nt = jnp.where(c_past[..., None], _PASTURE_ROW, nt)
        tiles = _scatter_tiles(tiles, flat, nt)

        new_inv_k = inv_k + d_inv
        seq_k, ctr_k = _update_inv_bookkeeping(inv_k, new_inv_k, seq_k, ctr_k)
        inv = _set_dyn3(inv, k, new_inv_k)
        seq = _set_dyn3(seq, k, seq_k)
        ctr = _set_dyn2(ctr, k, ctr_k)
        return tiles, ux, uy, shed + d_shed, seeds + d_seeds, inv, seq, ctr

    carry = (
        state.tiles,
        state.unit_x,
        state.unit_y,
        state.shed,
        state.seeds,
        state.inv,
        state.inv_seq,
        state.inv_ctr,
    )
    tiles, ux, uy, shed, seeds, inv, seq, ctr = jax.lax.fori_loop(0, n_units, body, carry)
    return state._replace(
        tiles=tiles,
        unit_x=ux,
        unit_y=uy,
        shed=shed,
        seeds=seeds,
        inv=inv,
        inv_seq=seq,
        inv_ctr=ctr,
    )


# ---------------------------------------------------------------- slice 2
def _spawn_hand_pos(unit_x: jax.Array, unit_y: jax.Array, live: jax.Array) -> jax.Array:
    """``_spawn_hand``: least-occupied shed-access tile, NWSE tie-break."""
    u = unit_x.shape[2]
    flat = unit_y * rules.BOARD_SIZE + unit_x  # [B, 2, U]
    alive = jnp.arange(u, dtype=I32)[None, None, :] <= live[:, :, None]
    hit = (flat[..., None] == _SHED_ACCESS_FLAT) & alive[..., None]  # [B, 2, U, 4]
    counts = jnp.sum(hit.astype(I32), axis=2)  # [B, 2, 4]
    best = jnp.argmin(counts, axis=-1)  # first minimum == NWSE order
    return _SHED_ACCESS_FLAT[best]


def process_market(state: SimState, action: SimAction, caps: caps_mod.Caps) -> SimState:
    """``_process_market``: atomic ops once per slot, then per-unit lockstep."""
    b = state.step.shape[0]
    u = caps.max_units
    op_all = action.order_op
    item_all = action.order_item
    qty_all = action.order_qty
    n_slots = jnp.max(
        jnp.where(op_all != rules.MO_NONE, jnp.arange(caps.max_orders, dtype=I32) + 1, 0),
        axis=(1, 2),
    )

    def read(arr, slot):
        return jnp.take_along_axis(arr, slot[:, None, None], axis=2)[:, :, 0]

    def cond(carry):
        st, slot, _need, _rem, _alive, trips = carry
        del st
        return jnp.any(slot < n_slots) & (jnp.max(trips) < caps.market_steps)

    def body(carry):
        st, slot, need, rem, alive, trips = carry
        in_range = slot < n_slots  # [B]
        cur_op = read(op_all, slot)
        cur_item = read(item_all, slot)
        cur_qty = read(qty_all, slot)
        init = (need & in_range)[:, None]  # [B, 1] -> broadcasts over players

        money = st.money
        shed = st.shed
        seeds = st.seeds
        tiles = st.tiles
        hires = st.hires_today
        n_quads = st.n_quads
        ux, uy, inv, seq, ctr = st.unit_x, st.unit_y, st.inv, st.inv_seq, st.inv_ctr
        overflow = st.overflow

        item_i = jnp.clip(cur_item, 0, rules.N_ITEMS - 1)
        item_oh = _one_hot(item_i, rules.N_ITEMS)

        # ---- HIRE (atomic, once per slot, player order is irrelevant: farms
        #      are disjoint and hire cost depends only on that farm's counter).
        hire_cost = _FIB_TABLE[jnp.clip(hires, 0, _FIB_LEN - 1)]
        do_hire = init & (cur_op == rules.MO_HIRE) & (money >= hire_cost)
        money = money - jnp.where(do_hire, hire_cost, 0)
        new_slot_idx = hires + 1
        spawn_flat = _spawn_hand_pos(ux, uy, jnp.minimum(hires, u - 1))
        fits = do_hire & (new_slot_idx <= u - 1)
        overflow = _add_overflow(
            overflow, caps_mod.OVF_UNITS, jnp.sum((do_hire & ~fits).astype(I32), axis=1)
        )
        slot_oh = (jnp.arange(u, dtype=I32)[None, None, :] == new_slot_idx[:, :, None]) & fits[
            :, :, None
        ]
        ux = jnp.where(slot_oh, spawn_flat[:, :, None] % rules.BOARD_SIZE, ux)
        uy = jnp.where(slot_oh, spawn_flat[:, :, None] // rules.BOARD_SIZE, uy)
        inv = jnp.where(slot_oh[..., None], 0, inv)
        seq = jnp.where(slot_oh[..., None], 0, seq)
        ctr = jnp.where(slot_oh, 1, ctr)
        hires = hires + do_hire.astype(I32)

        # ---- BUY_LAND (atomic)
        land_cost = _LAND_PRICES[jnp.clip(n_quads, 0, len(rules.LAND_PRICES) - 1)]
        do_land = (
            init
            & (cur_op == rules.MO_BUY_LAND)
            & (n_quads < len(rules.LAND_PRICES))
            & (money >= land_cost)
        )
        money = money - jnp.where(do_land, land_cost, 0)
        unlock_rank = n_quads + 1
        unlock = (
            (_TILE_QUAD_RANK[None, None, :] == unlock_rank[:, :, None])
            & (tiles[..., rules.C_KIND] == rules.TK_LOCKED)
            & do_land[:, :, None]
        )
        tiles = jnp.where(unlock[..., None], _EMPTY_ROW, tiles)
        n_quads = n_quads + do_land.astype(I32)

        # ---- BUY_SEED / BUY_ANIMAL: constant price, no market-inventory
        #      coupling, therefore exactly settleable in closed form.
        seed_ok = (
            init & (cur_op == rules.MO_BUY_SEED) & (cur_item >= 0) & (cur_item < rules.N_CROPS)
        )
        crop_i = jnp.clip(cur_item, 0, rules.N_CROPS - 1)
        seed_cost = _CROP_SEED_COST[crop_i]
        n_seed = jnp.where(seed_ok, jnp.minimum(cur_qty, money // seed_cost), 0)
        money = money - n_seed * seed_cost
        seeds = seeds + _one_hot(crop_i, rules.N_CROPS) * n_seed[..., None]

        anim_ok = (
            init
            & (cur_op == rules.MO_BUY_ANIMAL)
            & (cur_item >= rules.ANIMAL_ITEM_BASE)
            & (cur_item < rules.N_ITEMS)
        )
        anim_i = jnp.clip(cur_item - rules.ANIMAL_ITEM_BASE, 0, rules.N_ANIMALS - 1)
        anim_cost = _ANIMAL_COST[anim_i]
        room = jnp.maximum(0, rules.SHED_CAPACITY - jnp.sum(shed, axis=-1))
        n_anim = jnp.where(anim_ok, jnp.minimum(jnp.minimum(cur_qty, money // anim_cost), room), 0)
        money = money - n_anim * anim_cost
        shed = shed + item_oh * n_anim[..., None]

        # ---- start-of-slot state for the lockstep orders
        sell_valid = (cur_op == rules.MO_SELL) & (cur_item >= 0) & (cur_item < rules.N_PRODUCTS)
        buy_valid = (cur_op == rules.MO_BUY_PRODUCT) & (
            (cur_item == rules.I_WHEAT) | (cur_item == rules.I_FERTILIZER)
        )
        lock_valid = sell_valid | buy_valid
        rem = jnp.where(init, jnp.where(lock_valid, cur_qty, 0), rem)
        alive = jnp.where(init, lock_valid, alive)

        # ---- one lockstep trip: quote both players at the same pre-commit
        #      inventory, then commit both.
        active = alive & (rem > 0) & in_range[:, None]
        is_sell = active & sell_valid
        is_buy = active & buy_valid
        quoted = is_sell | is_buy
        prod_i = jnp.clip(cur_item, 0, rules.N_PRODUCTS - 1)
        inv_at = jnp.take_along_axis(st.market_inv, prod_i, axis=1)
        sell_price, w1 = lookup_price(prod_i, inv_at)
        buy_price, w2 = lookup_price(prod_i, inv_at - 1)
        overflow = _add_overflow(
            overflow,
            caps_mod.OVF_PRICE_WINDOW,
            jnp.sum(((is_sell & ~w1) | (is_buy & ~w2)).astype(I32), axis=1),
        )

        shed_at = _take_item(shed, prod_i)
        shed_total = jnp.sum(shed, axis=-1)
        sell_ok = is_sell & (shed_at > 0)
        buy_ok = is_buy & (money >= buy_price) & (shed_total < rules.SHED_CAPACITY)

        money = money + jnp.where(sell_ok, sell_price, 0) - jnp.where(buy_ok, buy_price, 0)
        shed = shed + item_oh * (buy_ok.astype(I32) - sell_ok.astype(I32))[..., None]
        # Sales at the $1 floor do not add supply.
        mkt_delta = jnp.where(sell_ok & (sell_price > 1), 1, 0) - jnp.where(buy_ok, 1, 0)
        market_inv = st.market_inv + jnp.sum(
            _one_hot(prod_i, rules.N_PRODUCTS) * mkt_delta[..., None], axis=1
        )

        committed = sell_ok | buy_ok
        rem = rem - committed.astype(I32)
        alive = alive & ~(quoted & ~committed)
        advance = (~jnp.any(quoted, axis=1)) | (~jnp.any(committed, axis=1)) | (~in_range)

        st = st._replace(
            market_units=st.market_units + jnp.sum(committed.astype(I32), axis=1),
            market_closed=st.market_closed + jnp.sum(n_seed + n_anim, axis=1),
            money=money,
            shed=shed,
            seeds=seeds,
            tiles=tiles,
            hires_today=hires,
            n_quads=n_quads,
            unit_x=ux,
            unit_y=uy,
            inv=inv,
            inv_seq=seq,
            inv_ctr=ctr,
            market_inv=market_inv,
            overflow=overflow,
        )
        return st, slot + advance.astype(I32), advance, rem, alive, trips + in_range.astype(I32)

    init_carry = (
        state._replace(market_units=jnp.zeros((b,), I32), market_closed=jnp.zeros((b,), I32)),
        jnp.zeros((b,), I32),
        jnp.ones((b,), bool),
        jnp.zeros((b, 2), I32),
        jnp.zeros((b, 2), bool),
        jnp.zeros((b,), I32),
    )
    st, slot, _need, _rem, _alive, trips = jax.lax.while_loop(cond, body, init_carry)
    unfinished = (slot < n_slots).astype(I32)
    return st._replace(
        overflow=_add_overflow(st.overflow, caps_mod.OVF_MARKET_STEPS, unfinished),
        market_trips=trips,
    )


# ---------------------------------------------------------------- slice 3
def town_consume(state: SimState) -> SimState:
    """``_town_consume``: shop ticks every 4 steps, town centre every 24."""
    shop_ok = state.shop_seq >= 0
    counts = jnp.sum(
        jnp.where(
            shop_ok[..., None],
            _one_hot(jnp.clip(state.shop_seq, 0, rules.N_SHOPS - 1), rules.N_SHOPS),
            0,
        ),
        axis=1,
    )  # [B, 8]
    demand = counts @ _SHOP_DEMAND  # [B, 9]
    shop_tick = (state.step % rules.DEFAULT_CONFIG["townShopSellInterval"] == 0)[:, None]
    centre_tick = (state.step % rules.DEFAULT_CONFIG["townCenterSellInterval"] == 0)[:, None]
    centre = (jnp.arange(rules.N_PRODUCTS, dtype=I32) < len(rules.TOWN_CENTER_ITEMS)).astype(I32)
    market_inv = (
        state.market_inv - jnp.where(shop_tick, demand, 0) - jnp.where(centre_tick, centre, 0)
    )
    return state._replace(market_inv=market_inv)


def decay_plants(state: SimState) -> SimState:
    """``_decay_plants``: past-lifespan plants shed one yield every other step."""
    tiles = state.tiles
    kind = tiles[..., rules.C_KIND]
    mls = tiles[..., rules.C_MLS]
    step = state.step[:, None, None]
    hit = (kind == rules.TK_PLANT) & (mls >= 0) & (step >= mls) & (((step - mls) % 2) == 0)
    new_yield = tiles[..., rules.C_YIELD] - 1
    tiles = tiles.at[..., rules.C_YIELD].set(
        jnp.where(hit & (new_yield > 0), new_yield, tiles[..., rules.C_YIELD])
    )
    tiles = jnp.where((hit & (new_yield <= 0))[..., None], _WEED_ROW, tiles)
    return state._replace(tiles=tiles)


def _daily_refresh(state: SimState, day: jax.Array) -> SimState:
    tiles = state.tiles
    kind = tiles[..., rules.C_KIND]
    d = day[:, None, None]
    nd = d + 1

    # --- plants
    is_plant = kind == rules.TK_PLANT
    crop = jnp.clip(tiles[..., rules.C_CROP], 0, rules.N_CROPS - 1)
    was_watered = tiles[..., rules.C_WATERED] == 1
    unwatered = jnp.where(was_watered, 0, tiles[..., rules.C_UNWATERED] + 1)
    plant_dies = is_plant & (unwatered >= 2)
    ongoing = _CROP_ONGOING[crop] == 1
    dsf = nd - tiles[..., rules.C_PLANTED_DAY] - _CROP_FIRST_YIELD[crop]
    interval = jnp.maximum(_CROP_INTERVAL[crop], 1)
    pcount = dsf // interval + 1
    produces = (
        is_plant
        & (~plant_dies)
        & ongoing
        & (dsf >= 0)
        & ((dsf % interval) == 0)
        & (pcount <= _CROP_MAX_YIELD[crop])
    )
    fertilized = was_watered & (tiles[..., rules.C_FERT_UNTIL] >= d)
    grown = jnp.minimum(
        _CROP_MAX_YIELD[crop], tiles[..., rules.C_YIELD] + jnp.where(fertilized, 2, 1)
    )
    tiles = tiles.at[..., rules.C_UNWATERED].set(
        jnp.where(is_plant, unwatered, tiles[..., rules.C_UNWATERED])
    )
    tiles = tiles.at[..., rules.C_WATERED].set(jnp.where(is_plant, 0, tiles[..., rules.C_WATERED]))
    tiles = tiles.at[..., rules.C_YIELD].set(jnp.where(produces, grown, tiles[..., rules.C_YIELD]))
    tiles = tiles.at[..., rules.C_MLS].set(
        jnp.where(
            produces & (pcount == _CROP_MAX_YIELD[crop]),
            (nd + 1) * rules.TURNS_PER_DAY,
            tiles[..., rules.C_MLS],
        )
    )
    tiles = jnp.where(plant_dies[..., None], _WEED_ROW, tiles)

    # --- animals
    animal = tiles[..., rules.C_ANIMAL]
    ai = jnp.clip(animal, 0, rules.N_ANIMALS - 1)
    is_anim = animal >= 0
    fed = tiles[..., rules.C_FED] == 1
    unfed = jnp.where(fed, 0, tiles[..., rules.C_UNFED] + 1)
    escapes = is_anim & (unfed >= 2)
    a_dsf = nd - tiles[..., rules.C_PLACED_DAY] - _ANIMAL_FIRST_YIELD[ai]
    a_prod = is_anim & (~escapes) & (a_dsf >= 0) & ((a_dsf % _ANIMAL_INTERVAL[ai]) == 0)
    care_bonus = tiles[..., rules.C_CARE_BONUS]
    bonus = jnp.where(fed, care_bonus, 0)
    a_yield = jnp.minimum(_ANIMAL_MAX_HELD[ai], tiles[..., rules.C_YIELD] + 1 + bonus)
    new_bonus = jnp.where(a_prod, 0, care_bonus)
    cared = tiles[..., rules.C_CARED] == 1
    new_bonus = jnp.where(is_anim & (~escapes) & cared & fed, new_bonus + 1, new_bonus)

    alive_anim = is_anim & (~escapes)
    tiles = tiles.at[..., rules.C_UNFED].set(jnp.where(is_anim, unfed, tiles[..., rules.C_UNFED]))
    tiles = tiles.at[..., rules.C_YIELD].set(jnp.where(a_prod, a_yield, tiles[..., rules.C_YIELD]))
    tiles = tiles.at[..., rules.C_CARE_BONUS].set(
        jnp.where(alive_anim, new_bonus, tiles[..., rules.C_CARE_BONUS])
    )
    tiles = tiles.at[..., rules.C_FERT_AVAIL].set(
        jnp.where(alive_anim, 1, tiles[..., rules.C_FERT_AVAIL])
    )
    tiles = tiles.at[..., rules.C_FED].set(jnp.where(alive_anim, 0, tiles[..., rules.C_FED]))
    tiles = tiles.at[..., rules.C_CARED].set(jnp.where(alive_anim, 0, tiles[..., rules.C_CARED]))
    struct_row = _STRUCT_ROWS[jnp.clip(_ANIMAL_STRUCT[ai] - rules.TK_COOP, 0, 1)]
    tiles = jnp.where(escapes[..., None], struct_row, tiles)
    return state._replace(tiles=tiles)


def _drop_inventories(state: SimState, caps: caps_mod.Caps) -> SimState:
    """``_drop_inventories_to_shed``: unit order, then dict insertion order."""
    b = state.step.shape[0]
    u = caps.max_units
    n = u * rules.N_ITEMS
    unit_rank = jnp.arange(u, dtype=I32)[None, None, :, None] * 100_000
    key = jnp.where(state.inv > 0, unit_rank + state.inv_seq, _BIG).reshape(b, 2, n)
    counts = state.inv.reshape(b, 2, n)
    order = jnp.argsort(key, axis=-1)
    sorted_key = jnp.take_along_axis(key, order, axis=-1)
    n_sorted = jnp.where(sorted_key >= _BIG, 0, jnp.take_along_axis(counts, order, axis=-1))
    excl = jnp.cumsum(n_sorted, axis=-1) - n_sorted
    shed_total = jnp.sum(state.shed, axis=-1)[..., None]
    take_sorted = jnp.clip(jnp.minimum(n_sorted, rules.SHED_CAPACITY - shed_total - excl), 0)
    rank = jnp.argsort(order, axis=-1)
    take = jnp.take_along_axis(take_sorted, rank, axis=-1).reshape(b, 2, u, rules.N_ITEMS)
    return state._replace(shed=state.shed + jnp.sum(take, axis=2))


def end_of_day(
    state: SimState, caps: caps_mod.Caps, tables: RngTables | None, mask: jax.Array
) -> SimState:
    """``_end_of_day`` for the envs selected by ``mask``."""
    b = state.step.shape[0]
    day = state.step // rules.TURNS_PER_DAY
    before = state
    state = _daily_refresh(state, day)

    # --- weeds: the engine draws once per *empty* tile, player 0 then player 1
    empty = state.tiles[..., rules.C_KIND] == rules.TK_EMPTY  # [B, 2, 100]
    flat_empty = empty.reshape(b, 2 * rules.N_TILES)
    draw_idx = jnp.cumsum(flat_empty.astype(I32), axis=-1) - flat_empty.astype(I32)
    total_draws = jnp.sum(flat_empty.astype(I32), axis=-1)
    overflow = state.overflow
    key = state.key
    if tables is None:
        key, sub = _split_keys(key)
        rolls = jax.vmap(
            lambda k: jax.random.uniform(jax.random.wrap_key_data(k), (2 * rules.N_TILES,))
        )(sub)
        hit = rolls < rules.DEFAULT_CONFIG["weedSpawnChance"]
    else:
        width = tables.weed_hit.shape[2]
        overflow = _add_overflow(
            overflow, caps_mod.OVF_WEED_DRAWS, jnp.maximum(0, total_draws - width)
        )
        rows = tables.weed_hit[jnp.arange(b), jnp.clip(day, 0, tables.weed_hit.shape[1] - 1)]
        hit = jnp.take_along_axis(rows, jnp.clip(draw_idx, 0, width - 1), axis=-1) > 0
    weeded = (hit & flat_empty).reshape(b, 2, rules.N_TILES)
    tiles = jnp.where(weeded[..., None], _WEED_ROW, state.tiles)
    state = state._replace(tiles=tiles, overflow=overflow, key=key)

    state = _drop_inventories(state, caps)

    # --- daily reset of units
    u = caps.max_units
    spawn_x = jnp.full((b, 2, u), rules.DEFAULT_SPAWN[0], I32)
    spawn_y = jnp.full((b, 2, u), rules.DEFAULT_SPAWN[1], I32)
    state = state._replace(
        unit_x=spawn_x,
        unit_y=spawn_y,
        hires_today=jnp.zeros((b, 2), I32),
        inv=jnp.zeros_like(state.inv),
        inv_seq=jnp.zeros_like(state.inv_seq),
        inv_ctr=jnp.ones_like(state.inv_ctr),
    )

    # --- town shop unlock (drawn with replacement, capped at 8 instances)
    next_day = day + 1
    unlock = (
        (next_day > 0)
        & (next_day % rules.DEFAULT_CONFIG["townShopUnlockInterval"] == 0)
        & (state.n_shops < rules.MAX_SHOP_INSTANCES)
    )
    if tables is None:
        key, sub = _split_keys(state.key)
        choice = jax.vmap(
            lambda k: jax.random.randint(jax.random.wrap_key_data(k), (), 0, rules.N_SHOPS)
        )(sub).astype(I32)
        state = state._replace(key=key)
    else:
        width = tables.shop_choice.shape[2]
        choice = tables.shop_choice[
            jnp.arange(b),
            jnp.clip(day, 0, tables.shop_choice.shape[1] - 1),
            jnp.clip(total_draws, 0, width - 1),
        ]
    slot_oh = (
        jnp.arange(rules.MAX_SHOP_INSTANCES, dtype=I32)[None, :] == state.n_shops[:, None]
    ) & unlock[:, None]
    state = state._replace(
        shop_seq=jnp.where(slot_oh, choice[:, None], state.shop_seq),
        n_shops=state.n_shops + unlock.astype(I32),
    )
    return _select_state(mask, state, before)


def _select_state(mask: jax.Array, new: SimState, old: SimState) -> SimState:
    def pick(a, b):
        m = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
        return jnp.where(m, a, b)

    return jax.tree.map(pick, new, old)


def _split_keys(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a batch of raw ``[B, 2]`` uint32 keys without leaving the batch."""
    raw = jax.vmap(lambda k: jax.random.key_data(jax.random.split(jax.random.wrap_key_data(k))))(
        key.astype(jnp.uint32)
    )
    return raw[:, 0], raw[:, 1]
