"""Observation and legality contract (frozen for Phase 0/1).

Only the *neutral* parts of the interface are fixed here:

* a **masked per-seat observation** (own private state visible, opponent's
  private state zeroed with an explicit mask flag) for the deployed policy;
* a **full-state training view** for a centralised critic -- present from day
  one because adding it later is a refactor and adding it now is free;
* **legality masks** supplied cheaply by the simulator, factored per decision
  (verb, crop, item) rather than as one flat table, so no head layout is
  pre-judged;
* the engine's own **10 market order slots**, untouched.

Deliberately *not* decided here: market macro-compilation, factored-head
shapes, and whether order sequencing carries usable signal.  Those need the
replay action distribution and random-policy coverage evidence first.

Deployment constraint that already binds architecture search: inference must
run in numpy on 1.6 vCPU inside a 1 s ``actTimeout``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rl import caps as caps_mod
from rl import kernels, rules
from rl.state import SimState, unit_alive

OBS_SPEC_VERSION = "v0"
I32 = jnp.int32
F32 = jnp.float32


def _tile_planes(tiles: jax.Array) -> jax.Array:
    """``[B, 2, 100, P]`` float planes: one-hot kind + scaled scalar channels."""
    kind = tiles[..., rules.C_KIND]
    kind_oh = jax.nn.one_hot(kind, len(rules.TILE_KIND_NAMES), dtype=F32)
    crop_oh = jax.nn.one_hot(tiles[..., rules.C_CROP], rules.N_CROPS, dtype=F32)
    animal_oh = jax.nn.one_hot(tiles[..., rules.C_ANIMAL], rules.N_ANIMALS, dtype=F32)
    scalars = jnp.stack(
        [
            tiles[..., rules.C_WATERED],
            tiles[..., rules.C_UNWATERED],
            tiles[..., rules.C_YIELD],
            tiles[..., rules.C_FED],
            tiles[..., rules.C_CARED],
            tiles[..., rules.C_UNFED],
            tiles[..., rules.C_FERT_AVAIL],
            tiles[..., rules.C_CARE_BONUS],
        ],
        axis=-1,
    ).astype(F32)
    return jnp.concatenate([kind_oh, crop_oh, animal_oh, scalars / 6.0], axis=-1)


def full_state_view(state: SimState, caps: caps_mod.Caps) -> jax.Array:
    """Omniscient ``[B, D]`` view for a centralised critic."""
    b = state.step.shape[0]
    planes = _tile_planes(state.tiles).reshape(b, -1)
    scalars = jnp.concatenate(
        [
            (state.step[:, None] / rules.EPISODE_STEPS).astype(F32),
            ((state.step[:, None] % rules.TURNS_PER_DAY) / rules.TURNS_PER_DAY).astype(F32),
            state.money.astype(F32) / 10_000.0,
            state.hires_today.astype(F32) / 12.0,
            state.n_quads.astype(F32) / 3.0,
            state.shed.reshape(b, -1).astype(F32) / rules.SHED_CAPACITY,
            state.seeds.reshape(b, -1).astype(F32) / 20.0,
            state.inv.reshape(b, -1).astype(F32) / 10.0,
            (state.market_inv.astype(F32) - rules.MARKET_I0) / 1_000.0,
            kernels.prices_of(state.market_inv).astype(F32) / 100.0,
            _shop_counts(state).astype(F32) / rules.MAX_SHOP_INSTANCES,
        ],
        axis=-1,
    )
    del caps
    return jnp.concatenate([planes, scalars], axis=-1)


def _shop_counts(state: SimState) -> jax.Array:
    ok = state.shop_seq >= 0
    idx = jnp.clip(state.shop_seq, 0, rules.N_SHOPS - 1)
    return jnp.sum(jnp.where(ok[..., None], jax.nn.one_hot(idx, rules.N_SHOPS), 0.0), axis=1)


def seat_observation(state: SimState, seat: int, caps: caps_mod.Caps) -> jax.Array:
    """``[B, D]`` masked view for ``seat``: opponent private state is zeroed.

    ``farms`` (tiles, money, unit positions, quadrants, hires) is public in the
    engine; ``private`` (shed, seeds, per-unit inventories) is not.
    """
    b = state.step.shape[0]
    order = jnp.asarray([seat, 1 - seat])
    tiles = state.tiles[:, order]
    planes = _tile_planes(tiles).reshape(b, -1)
    private_mask = jnp.asarray([1.0, 0.0])[None, :, None]
    shed = (state.shed[:, order].astype(F32) / rules.SHED_CAPACITY) * private_mask
    seeds = (state.seeds[:, order].astype(F32) / 20.0) * private_mask
    inv = (state.inv[:, order].astype(F32) / 10.0) * private_mask[..., None]
    unit_x = state.unit_x[:, order].astype(F32) / rules.BOARD_SIZE
    unit_y = state.unit_y[:, order].astype(F32) / rules.BOARD_SIZE
    alive = unit_alive(state, caps)[:, order].astype(F32)
    scalars = jnp.concatenate(
        [
            (state.step[:, None] / rules.EPISODE_STEPS).astype(F32),
            ((state.step[:, None] % rules.TURNS_PER_DAY) / rules.TURNS_PER_DAY).astype(F32),
            state.money[:, order].astype(F32) / 10_000.0,
            state.hires_today[:, order].astype(F32) / 12.0,
            state.n_quads[:, order].astype(F32) / 3.0,
            shed.reshape(b, -1),
            seeds.reshape(b, -1),
            inv.reshape(b, -1),
            unit_x.reshape(b, -1),
            unit_y.reshape(b, -1),
            alive.reshape(b, -1),
            (state.market_inv.astype(F32) - rules.MARKET_I0) / 1_000.0,
            kernels.prices_of(state.market_inv).astype(F32) / 100.0,
            _shop_counts(state).astype(F32) / rules.MAX_SHOP_INSTANCES,
            jnp.broadcast_to(jnp.asarray([[1.0, 0.0]]), (b, 2)),  # private-visibility flags
        ],
        axis=-1,
    )
    return jnp.concatenate([planes, scalars], axis=-1)


def legality(state: SimState, caps: caps_mod.Caps) -> dict[str, jax.Array]:
    """Cheap "this would not be a guaranteed no-op" masks, factored per decision."""
    u = caps.max_units
    b = state.step.shape[0]
    alive = unit_alive(state, caps)  # [B, 2, U]
    flat = state.unit_y * rules.BOARD_SIZE + state.unit_x
    tiles = jnp.take_along_axis(state.tiles, flat[..., None], axis=2)  # [B, 2, U, C]
    kind = tiles[..., rules.C_KIND]
    crop = jnp.clip(tiles[..., rules.C_CROP], 0, rules.N_CROPS - 1)
    animal = tiles[..., rules.C_ANIMAL]
    has_animal = animal >= 0
    ai = jnp.clip(animal, 0, rules.N_ANIMALS - 1)
    day = (state.step // rules.TURNS_PER_DAY)[:, None, None]
    yield_u = tiles[..., rules.C_YIELD]

    shed = state.shed[:, :, None, :]  # [B, 2, 1, 12]
    shed_total = jnp.sum(state.shed, axis=-1)[:, :, None]
    shed_room = rules.SHED_CAPACITY - shed_total
    inv = state.inv  # [B, 2, U, 12]
    carrying = jnp.sum(inv, axis=-1) > 0
    shed_adj = jnp.any(flat[..., None] == kernels._SHED_ACCESS_FLAT, axis=-1)

    x, y = state.unit_x, state.unit_y
    move = jnp.stack(
        [y > 0, y < rules.BOARD_SIZE - 1, x < rules.BOARD_SIZE - 1, x > 0], axis=-1
    )  # NORTH, SOUTH, EAST, WEST

    empty = kind == rules.TK_EMPTY
    is_plant = kind == rules.TK_PLANT
    mature = (day - tiles[..., rules.C_PLANTED_DAY]) >= kernels._CROP_FIRST_YIELD[crop]
    seeds_here = state.seeds[:, :, None, :]  # [B, 2, 1, 5]

    plant_crop = empty[..., None] & (seeds_here > 0) & alive[..., None]
    pickup_item = shed_adj[..., None] & (shed > 0) & alive[..., None]
    struct_match = kind[..., None] == kernels._ANIMAL_STRUCT[None, None, None, :]
    place_animal = (
        struct_match
        & (~has_animal)[..., None]
        & (inv[..., rules.ANIMAL_ITEM_BASE :] > 0)
        & alive[..., None]
    )
    place_shed = shed_adj[..., None] & (inv > 0) & (shed_room[..., None] > 0) & alive[..., None]

    ops = jnp.zeros((b, 2, u, rules.N_UNIT_OPS), bool)
    ops = ops.at[..., rules.OP_PASS].set(alive)
    ops = ops.at[..., rules.OP_NORTH : rules.OP_WEST + 1].set(move & alive[..., None])
    ops = ops.at[..., rules.OP_PICKUP].set(jnp.any(pickup_item, axis=-1))
    ops = ops.at[..., rules.OP_PLANT].set(jnp.any(plant_crop, axis=-1))
    ops = ops.at[..., rules.OP_WATER].set(alive & is_plant & (tiles[..., rules.C_WATERED] == 0))
    ops = ops.at[..., rules.OP_HARVEST].set(
        alive & (yield_u > 0) & ((is_plant & mature) | has_animal)
    )
    ops = ops.at[..., rules.OP_FERTILIZE].set(alive & is_plant & (inv[..., rules.I_FERTILIZER] > 0))
    ops = ops.at[..., rules.OP_BUILD_COOP].set(alive & empty)
    ops = ops.at[..., rules.OP_BUILD_PASTURE].set(alive & empty)
    ops = ops.at[..., rules.OP_DIG].set(
        alive & (kind != rules.TK_EMPTY) & (kind != rules.TK_LOCKED) & (~has_animal)
    )
    ops = ops.at[..., rules.OP_PLACE].set(
        jnp.any(place_animal, axis=-1) | jnp.any(place_shed, axis=-1)
    )
    ops = ops.at[..., rules.OP_FEED].set(
        alive & has_animal & (tiles[..., rules.C_FED] == 0) & (inv[..., rules.I_WHEAT] > 0)
    )
    ops = ops.at[..., rules.OP_COLLECT_FERTILIZER].set(
        alive & has_animal & (tiles[..., rules.C_FERT_AVAIL] == 1)
    )
    ops = ops.at[..., rules.OP_CARE].set(alive & has_animal & (tiles[..., rules.C_CARED] == 0))
    ops = ops.at[..., rules.OP_DROP].set(alive & shed_adj & carrying)
    del ai

    money = state.money
    prices = kernels.prices_of(state.market_inv)
    hire_cost = kernels._FIB_TABLE[jnp.clip(state.hires_today, 0, len(kernels._FIB_TABLE) - 1)]
    land_cost = kernels._LAND_PRICES[jnp.clip(state.n_quads, 0, len(rules.LAND_PRICES) - 1)]
    buyable = jnp.asarray([rules.I_WHEAT, rules.I_FERTILIZER])
    return {
        "unit_ops": ops,
        "plant_crop": plant_crop,
        "pickup_item": pickup_item,
        "place_item": place_shed.at[..., rules.ANIMAL_ITEM_BASE :].set(
            place_shed[..., rules.ANIMAL_ITEM_BASE :] | place_animal
        ),
        "buy_seed": money[..., None] >= kernels._CROP_SEED_COST[None, None, :],
        "buy_product": (money[..., None] >= prices[:, None, buyable])
        & (jnp.sum(state.shed, axis=-1)[..., None] < rules.SHED_CAPACITY),
        "buy_animal": (money[..., None] >= kernels._ANIMAL_COST[None, None, :])
        & (jnp.sum(state.shed, axis=-1)[..., None] < rules.SHED_CAPACITY),
        "sell": state.shed[..., : rules.N_PRODUCTS] > 0,
        "hire": money >= hire_cost,
        "buy_land": (state.n_quads < len(rules.LAND_PRICES)) & (money >= land_cost),
    }
