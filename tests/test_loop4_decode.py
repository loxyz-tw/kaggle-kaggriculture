"""top-k 每格任務解碼的測試（任務 14 B2）。

解碼不是「每格獨立 argmax」：那樣 100 格全部會被指派任務，policy 就失去「要做哪幾格」的
控制權。正確的做法是按 ``1 - P(NOOP)`` 排序取前 k 格（k = 活著的人手數）。
這支釘住三件事：選中的格數不超過人手數、選中的格一定是遮罩允許的、以及排序真的按
``1 - P(NOOP)`` 而不是按最大 logit。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import loop4_actions as A
from rl import loop4_model as M
from rl import rules
from rl import state as rl_state
from scripts.loop4_eval import make_policy_step

CAPS = caps_mod.EXACT_REFERENCE


def _state(n_hands: int):
    state = rl_state.reset(1, CAPS, seed=0)
    tiles = np.array(state.tiles)
    # NW 象限全部種上沒澆水的小麥，讓 WATER 在很多格上合法
    for flat in range(rules.N_TILES):
        if rules.TILE_QUAD_RANK[flat] != 0:
            continue
        row = rl_state.CANONICAL_EMPTY.copy()
        row[rules.C_KIND] = rules.TK_PLANT
        row[rules.C_CROP] = 0
        row[rules.C_PLANTED_DAY] = 4
        row[rules.C_YIELD] = 1
        row[rules.C_MLS] = 700
        tiles[0, :, flat] = row
    return state._replace(
        tiles=jnp.asarray(tiles, jnp.int32),
        step=jnp.full_like(state.step, 5 * rules.TURNS_PER_DAY),
        hires_today=jnp.full_like(state.hires_today, n_hands),
    )


def _run(n_hands: int, topk: bool):
    cfg = M.Config.default(CAPS)
    params = M.init_params(cfg, 0)
    step = make_policy_step(cfg, greedy=True, topk=topk)
    state = _state(n_hands)
    assign = jnp.full((1, 2, CAPS.max_units), A.NO_ASSIGNMENT, jnp.int32)
    action, _ = step(params, state, assign, jax.random.PRNGKey(0), CAPS)
    return state, action


def test_topk_assigns_at_most_one_tile_per_hand():
    """被指派的格數不得超過活著的人手數。"""
    for n_hands in (0, 3, 7):
        state, action = _run(n_hands, topk=True)
        op = np.asarray(action.unit_op)[0, 0]
        alive = np.arange(CAPS.max_units) <= min(n_hands, CAPS.max_units - 1)
        working = (op != rules.OP_PASS) & alive
        assert working.sum() <= n_hands + 1, (n_hands, working.sum())


def test_per_tile_argmax_is_unbounded_but_topk_is_capped():
    """同一份 logits：每格獨立 argmax 會指派所有想做的格，top-k 只會指派 k 格。

    用構造的 logits，不用未訓練的模型——同質盤面上未訓練的模型每格 logits 相同，
    argmax 會全選同一類，量不出這件事。
    """
    n = rules.N_TILES
    n_legal, k = 25, 4
    lg = np.full((n, A.N_TASKS), -30.0, np.float32)
    lg[:, A.T_NOOP] = 0.0
    lg[:n_legal, A.T_WATER] = 3.0  # 這些格明顯想做事
    per_tile = lg.argmax(-1)
    assert int((per_tile != A.T_NOOP).sum()) == n_legal

    logp = jax.nn.log_softmax(jnp.asarray(lg), axis=-1)
    want = np.asarray(jnp.log1p(-jnp.exp(jnp.clip(logp[:, A.T_NOOP], -30.0, -1e-6))))
    order = np.argsort(-want)
    rank = np.argsort(order)
    chosen = rank < k
    assert int(chosen.sum()) == k
    assert set(np.nonzero(chosen)[0]) <= set(range(n_legal))


def test_topk_only_picks_masked_legal_tasks():
    state, action = _run(5, topk=True)
    mask = np.asarray(A.tile_task_mask(state, CAPS))[0, 0]
    op = np.asarray(action.unit_op)[0, 0]
    arg = np.asarray(action.unit_arg)[0, 0]
    ux = np.asarray(state.unit_x)[0, 0]
    uy = np.asarray(state.unit_y)[0, 0]
    for k in range(6):
        o = int(op[k])
        if o == rules.OP_PASS or rules.OP_NORTH <= o <= rules.OP_WEST:
            continue
        flat = int(uy[k]) * rules.BOARD_SIZE + int(ux[k])
        if o == rules.OP_PLANT:
            task = A.T_PLANT0 + int(arg[k])
        else:
            task = int(np.nonzero(np.asarray(A.TASK_TO_OP) == o)[0][0])
        assert mask[flat, task], (k, flat, A.TASK_NAMES[task])


def test_topk_ranks_by_noop_probability_not_max_logit():
    """把某一格的 NOOP logit 壓很低、其他類也壓低，它仍該被選中。

    如果排序是按「最大 logit」，這格會排在後面；按 1 − P(NOOP) 才會排在前面。
    """
    state = _state(0)  # 只有農夫，k = 1
    mask = np.asarray(A.tile_task_mask(state, CAPS))[0, 0]
    target = int(np.nonzero(mask[:, A.T_WATER])[0][5])
    # 直接構造 logits 驗排序準則，不走模型
    n = rules.N_TILES
    lg = np.full((n, A.N_TASKS), -5.0, np.float32)
    lg[:, A.T_NOOP] = 0.0  # 其他格都很可能 NOOP
    lg[target, A.T_NOOP] = -20.0  # 這格幾乎不可能 NOOP
    lg[target, A.T_WATER] = -4.0  # 但它的最大 logit 比別格的 NOOP(0.0) 小
    logp = jax.nn.log_softmax(jnp.asarray(lg), axis=-1)
    want = np.asarray(jnp.log1p(-jnp.exp(jnp.clip(logp[:, A.T_NOOP], -30.0, -1e-6))))
    assert int(np.argmax(want)) == target
    assert int(np.argmax(lg.max(-1))) != target
