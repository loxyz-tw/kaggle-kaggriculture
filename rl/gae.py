"""GAE(λ) advantage 估計，含 done mask 與 terminal bootstrap 0。

Binding review 第 2 條：``γ=0.999``、``λ=0.95``、terminal bootstrap 0（``done=1``）、
episode 固定 719 transitions 無 early termination、**rollout window 跨 episode 邊界，
以 done mask 截斷**。

遞推式（時間軸為前導軸）::

    δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)
    A_t = δ_t + γ·λ·(1 − done_t)·A_{t+1}

``done_t`` 指的是「``s_{t+1}`` 是終局」，因此它同時關掉 bootstrap 與跨界傳遞——
這正是讓固定長度 rollout window 可以安全跨 episode 邊界的條件。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rl.reward import GAE_LAMBDA, GAMMA


def gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> tuple[jax.Array, jax.Array]:
    """回傳 ``(advantages, returns)``，形狀與 ``rewards`` 相同。

    ``rewards``/``values``/``dones`` 皆為 ``[T, ...]``；``last_value`` 是 ``[...]``
    的 ``V(s_T)``（window 尾端的 bootstrap，若該處 ``done`` 為真會被 mask 掉）。
    """
    dones = dones.astype(rewards.dtype)

    def step(carry, xs):
        adv, next_value = carry
        r, v, d = xs
        not_done = 1.0 - d
        delta = r + gamma * next_value * not_done - v
        adv = delta + gamma * lam * not_done * adv
        return (adv, v), adv

    (_, _), advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    return advantages, advantages + values
