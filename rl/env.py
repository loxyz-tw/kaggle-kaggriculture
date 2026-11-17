"""Episode assembly: reset / step / rollout for the fixed-shape simulator.

``step`` reproduces one call of ``kaggriculture.interpreter``: one *global
env-turn*, i.e. both seats acting for one in-game hour.  That is the single
throughput unit used by every benchmark in this package; decisions/s is
exactly twice it.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from rl import caps as caps_mod
from rl import kernels, rules
from rl.actions import SimAction
from rl.state import SimState, reset

__all__ = ["reset", "step", "rollout", "rewards", "is_done", "ENV_TURNS_PER_EPISODE"]

# 720 recorded observations, 719 interpreter transitions carrying an action.
ENV_TURNS_PER_EPISODE = rules.N_TRANSITIONS


def step(
    state: SimState,
    action: SimAction,
    caps: caps_mod.Caps,
    tables: kernels.RngTables | None = None,
) -> SimState:
    """One global env-turn, in the engine's own phase order."""
    state = kernels.apply_unit_actions(state, action, caps)
    state = kernels.process_market(state, action, caps)
    state = kernels.town_consume(state)
    state = kernels.decay_plants(state)

    eod = ((state.step + 1) % rules.TURNS_PER_DAY) == 0
    state = jax.lax.cond(
        jnp.any(eod),
        lambda s: kernels.end_of_day(s, caps, tables, eod),
        lambda s: s,
        state,
    )
    return state._replace(step=state.step + 1)


@functools.partial(jax.jit, static_argnums=(2,))
def jit_step(
    state: SimState,
    action: SimAction,
    caps: caps_mod.Caps,
    tables: kernels.RngTables | None = None,
) -> SimState:
    """``step`` compiled once per (caps, shape) -- what harnesses should call."""
    return step(state, action, caps, tables)


def rewards(state: SimState) -> jax.Array:
    """Engine reward: end-of-game money, per seat."""
    return state.money.astype(jnp.float32)


def is_done(state: SimState) -> jax.Array:
    return state.step >= rules.EPISODE_STEPS - 1


def prices(state: SimState) -> jax.Array:
    return kernels.prices_of(state.market_inv)


def rollout(
    state: SimState,
    policy,
    caps: caps_mod.Caps,
    n_turns: int,
    tables: kernels.RngTables | None = None,
):
    """Scan ``policy`` for ``n_turns`` env-turns.

    ``policy(state, key) -> (SimAction, key)``.  Returns the final state and a
    per-turn stack of whatever the policy chose to emit as its second output.
    """

    def body(carry, _):
        st, key = carry
        act, key = policy(st, key)
        return (step(st, act, caps, tables), key), None

    (final, _), _ = jax.lax.scan(body, (state, state.key), None, length=n_turns)
    return final
