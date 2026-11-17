"""完整動作 PPO **bridge 首段**：actor → 兩座 SimAction → 單次 `env.step` → 逐 token 重算。

規格：內部規格文件（審查者 §32 必要正確性 PASS 的六條落點）。

**本段不做 updater、不做 PPO 配方、不動任何 IL 凍結依賴。** 只證一件事：
actor 的動作與逐 token 機率**和環境語義一致、重算精確**。

四條釘死的契約
--------------
1. **重算不是新 mode**：`decode` 的 `MODES` 只有 sample／greedy／teacher。重算＝
   `decode("teacher")` 拿 rollout 實際抽到的 decision 當 labels，重用同一份 mask／AR parent／
   byte 順序。斷言 `teacher_drop` 全 0（自己抽的 token 必在 mask 內）、`active` 逐位相同。
   **不把 teacher keep 當 policy active。**
2. **CAPS 是完整動作的**（U31 請求／1010 market steps）。`BOUNDED_TRAINING` 的 512／210 會截
   大單與市場 lockstep，那是把舊限制搬回來。`H.config` 只看 `caps.max_units`，兩版都是 29，
   所以 **IL 的 checkpoint 原樣可載**，差別只在 `kernels.process_market` 的 trips 上限。
3. **橋接固定走** `D.build_action(..., market=True)` → `merge_by_seat` → `rl.env.step` **恰一次**。
   不經 `rl/actions.py` 的 JSON 512 夾取；`decode` 回的 `projected` 只供決策觀測，
   **不能拿去再 step**（那會把單位相位做兩次）。
4. **座位**：一次 rollout 收固定 Python int seat，兩座**各一支 static jit**、每步都從
   **同一個原 state** 各呼叫一次再合併。異質 actor 的分組留給 updater，
   不在 scan 裡把 `sv[B]` 塞進 static decoder。

P1 的局末**純函式**復用：`last_valid_step`／`scatter_to_last_valid`／`gae`／`seat_vec`／
`pick_seat`／`merge_by_seat`——它們只吃 `[T, B]` 陣列，與動作空間無關。

**但 `scripts/loop4_ppo_train.py::flatten_traj` 不能原樣用**：它吃的是 Loop 4 的
`Transition`（有 `assignment`、`logp_h` 是 218 component）。本線自己有
`flatten_bridge()`（見下），欄位是 263 槽的 `logp_h`／`active`，沒有 `assignment`。
**不複製 Loop 4 的 218 component 欄位，也不假裝有任務格指派。**
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from rl import caps as caps_mod
from rl import env as env_mod
from rl import full_actions_decode as D
from rl import full_actions_head as H
from rl import loop4_ppo as P1
from rl import loop4_tape as T
from rl import rules

#: 完整動作的 caps：**檔內明列**，不 import 舊 PPO 的整條 runner。
#: 數值與 `rl/ordered_v3_ppo.py::CAPS` 相同（測試逐欄比對），只有 `name` 不同。
CAPS = caps_mod.Caps(
    max_units=29,
    max_orders=10,
    market_steps=1010,
    max_order_qty=2**31 - 1,
    name="full-actions-ppo-u31",
)

UNIT_FIELDS = ("unit_verb", "unit_crop", "unit_item", "unit_qty_bytes")
MARKET_FIELDS = ("market_op", "market_item", "market_qty_bytes")
LABEL_FIELDS = UNIT_FIELDS + MARKET_FIELDS


class Transition(NamedTuple):
    """每一步留下的東西。**不存 obs**（重算時用 state 再算一次）。

    欄位名與 P1 的 `Transition` 對齊，`scripts/loop4_ppo_train.py::flatten_traj` 的
    valid 過濾與 `env_id`／`turn` 補欄才能原樣用；`assignment` 這一欄完整動作不需要。
    """

    state: Any  # SimState（原 state，重算 obs／重建 prefix 用）
    decision: dict  # DecisionV3 八欄（自己這座）
    logp: jax.Array  # [B]      joint ＝ Σ active·logp（**只作報告**，不是 update 的 ratio）
    logp_h: jax.Array  # [B, 263] 逐 token
    active: jax.Array  # [B, 263] policy 真的做了選擇的位置
    value: jax.Array  # [B]      V(s)，取動作前的 h1
    done: jax.Array  # [B]      這一步之後是否已到真局末
    seat: jax.Array  # [B]      這一列是坐哪一座出的決策
    valid: jax.Array  # [B]      pre-state 還在局內（False ＝ padding，不進訓練）
    env_id: jax.Array  # [B]     過濾後列→env 的對應會斷，明帶一欄
    step: jax.Array  # [B]      真 step（turn）
    # T3 effect-mask：[B, 263] 值確定不影響引擎結果的 qty 槽（⊆ active）；PPO v4 才讀
    noeffect: jax.Array


def joint_logp(logp_h: jax.Array, active: jax.Array) -> jax.Array:
    """``Σ active·logp``。**只用於報告**——update 的 ratio 是逐 component clip（既有裁定）。"""
    return jnp.sum(jnp.where(active, logp_h, 0.0), axis=-1)


def make_decoders(cfg: dict[str, Any], caps: caps_mod.Caps = CAPS):
    """兩座各一支 **static jit**（`seat` 是 Python int，關進 closure）。

    回 `{"sample": {0: fn, 1: fn}, "teacher": {0: fn, 1: fn}}`。
    每支都只吃**原 state**；不把對手的動作或 projected 餵進我方觀測。
    """
    out: dict[str, dict[int, Any]] = {"sample": {}, "teacher": {}}
    for mode in ("sample", "teacher"):
        run = D.decode(mode)
        for seat in (0, 1):

            def fn(params, state, key=None, labels=None, _run=run, _seat=seat):
                return _run(params, cfg, state, _seat, caps, key=key, labels=labels)

            out[mode][seat] = jax.jit(fn)
    return out


def own_fields(decision: dict) -> dict:
    """decision → `build_action` 要的欄（含市場，保序、保 EOS、保重複訂單）。"""
    return {k: decision[k] for k in LABEL_FIELDS}


def joint_action(state, dec0: dict, dec1: dict, caps: caps_mod.Caps = CAPS):
    """兩座 decision → **同一個** `SimAction`。

    各自從**原 state** 建（`build_action` 只填自己那一座），再用 `merge_by_seat` 以
    `sv=0` 取 `a0` 的第 0 座、`a1` 的第 1 座。qty 走既有 U31 codec，市場長度／EOS／
    重複訂單原樣保留；**不經 `rl/actions.py` 的 512 JSON 夾取**。
    """
    b = state.step.shape[0]
    a0 = D.build_action(state, own_fields(dec0), 0, caps, market=True)
    a1 = D.build_action(state, own_fields(dec1), 1, caps, market=True)
    return P1.merge_by_seat(a0, a1, P1.seat_vec(0, b))


def step_once(state, dec0: dict, dec1: dict, caps: caps_mod.Caps = CAPS):
    """從**原 state** 呼叫 `rl.env.step` **恰一次**。

    不先 `apply_unit_actions` 再 step（單位相位會做兩次），也不用 decoder 的 `projected`。
    """
    return env_mod.step(state, joint_action(state, dec0, dec1, caps), caps)


def rollout(
    params,
    cfg: dict[str, Any],
    init_state,
    key,
    *,
    seat: int,
    t: int,
    decoders=None,
    caps: caps_mod.Caps = CAPS,
    opp_params=None,
    opp_tape=None,
):
    """自走 `t` 步。`seat` 是**固定的 Python int**（本段不做逐 env 座位輪換）。

    `opp_params` 是 None 時對手座位用同一組 `params`（自我對戰，舊行為逐位不變）；
    給了就用它當**不更新的對手**（計畫 內部規格文件）。

    `opp_tape` 給 `(bank, tape_idx[B])` 時，對手座改成照 `rl.loop4_tape.tape_action` 按
    `state.step` 送 tape 的動作（計畫 內部規格文件）；
    我方座照舊 `build_action`，兩份用 `merge_by_seat` 合成後 `env.step` 一次。

    每步：兩座各從同一個原 state 呼叫自己那支 static decode 一次 → 合併 → `env.step` 一次。
    **P1 的 freeze-in-carry 照抄**：越過真局末的 env 凍結 state，`valid=False` 的列不再
    改變任何東西（reward／GAE／loss 都以 `valid` 為準）。
    """
    dec = decoders or make_decoders(cfg, caps)
    b = init_state.step.shape[0]
    env_id = jnp.arange(b, dtype=jnp.int32)
    sv = P1.seat_vec(seat, b)
    opp = 1 - seat

    def body(carry, _):
        state, k = carry
        k, k0, k1 = jax.random.split(k, 3)
        valid = state.step < rules.N_TRANSITIONS  # [B]
        out_own = dec["sample"][seat](params, state, key=k0)
        if opp_tape is None:
            out_opp = dec["sample"][opp](
                params if opp_params is None else opp_params, state, key=k1
            )
            d0, d1 = (out_own, out_opp) if seat == 0 else (out_opp, out_own)
            nxt = step_once(state, d0["decision"], d1["decision"], caps)
        else:
            bank, tape_idx = opp_tape
            a_own = D.build_action(state, own_fields(out_own["decision"]), seat, caps, market=True)
            a_opp = T.tape_action(bank, tape_idx, state.step, opp, caps)
            nxt = env_mod.step(state, P1.merge_by_seat(a_own, a_opp, sv), caps)

        # 凍結：無效列的 next state 原封不動（`jnp.where` 在 carry 上做，不動 `rl/env.py`）
        nxt = jax.tree.map(
            lambda a, b_: jnp.where(valid.reshape((-1,) + (1,) * (a.ndim - 1)), a, b_), nxt, state
        )
        tr = Transition(
            state=state,
            decision=out_own["decision"],
            logp=joint_logp(out_own["logp"], out_own["active"]),
            logp_h=out_own["logp"],
            active=out_own["active"],
            value=out_own["value"],
            done=nxt.step >= rules.N_TRANSITIONS,
            seat=sv,
            valid=valid,
            env_id=env_id,
            step=state.step,
            noeffect=out_own["noeffect"],
        )
        return (nxt, k), tr

    (final, _), traj = jax.lax.scan(body, (init_state, key), None, length=t)
    return final, traj


def recompute(params, cfg, state, decision: dict, *, seat: int, decoders=None, caps=CAPS):
    """**重算不是新 mode**：`decode("teacher")` 餵 rollout 實際抽到的 decision。

    回 `(out, report)`；`report` 是 sample→recompute 的對帳，呼叫端自己斷言。
    """
    dec = decoders or make_decoders(cfg, caps)
    labels = {k: decision[k] for k in LABEL_FIELDS}
    return dec["teacher"][seat](params, state, labels=labels)


def recompute_report(sampled: dict, recomputed: dict) -> dict[str, Any]:
    """sample 與 teacher 重算的差異報告：**逐 token 與 joint 都報**，drop 必須全 0。"""
    lp_a, lp_b = sampled["logp"], recomputed["logp"]
    act_a, act_b = sampled["active"], recomputed["active"]
    drop = recomputed["teacher_drop"]
    n_drop = sum(int(jnp.sum(v)) for grp in ("unit", "market") for v in drop[grp].values())
    return {
        "active_bitwise_equal": bool(jnp.all(act_a == act_b)),
        "teacher_drop_total": n_drop,
        "max_abs_token_logp_diff": float(jnp.max(jnp.abs(lp_a - lp_b))),
        "max_abs_joint_logp_diff": float(
            jnp.max(jnp.abs(joint_logp(lp_a, act_a) - joint_logp(lp_b, act_b)))
        ),
        "inactive_logp_all_zero": bool(jnp.all(jnp.where(act_b, 0.0, jnp.abs(lp_b)) == 0.0)),
        "n_active": int(jnp.sum(act_a)),
    }


def flatten_bridge(traj: Transition, *, valid_filter: bool = True) -> dict[str, Any]:
    """`[T, B, ...]` → `[T*B, ...]` 並搬回 host。**本線自己的 adapter**。

    為什麼不用 `scripts/loop4_ppo_train.py::flatten_traj`：那支吃 Loop 4 的 `Transition`
    （`assignment` ＋ 218 component 的 `logp_h`）。這裡的欄位是 263 槽，也沒有指派。
    共用的只有**行為契約**：valid 過濾、以及過濾後列→env 的對應會斷，所以**明帶
    `env_id` 與 `turn` 兩欄**（不靠 `r % n_envs` 推）。
    """
    import numpy as np  # noqa: PLC0415

    tt, bb = traj.logp.shape
    keep = np.asarray(traj.valid).reshape(-1) if valid_filter else np.ones(tt * bb, bool)

    def f(x):
        a = np.asarray(x)
        return a.reshape((-1,) + a.shape[2:])[keep]

    env_id = np.broadcast_to(np.arange(bb, dtype=np.int32), (tt, bb)).reshape(-1)[keep]
    turn = np.broadcast_to(np.arange(tt, dtype=np.int32)[:, None], (tt, bb)).reshape(-1)[keep]
    return {
        "state": jax.tree.map(f, traj.state),
        "decision": {k: f(v) for k, v in traj.decision.items()},
        "logp": f(traj.logp),  # joint，只作報告
        "logp_h": f(traj.logp_h),  # [N, 263]
        "active": f(traj.active),  # [N, 263]
        "noeffect": f(traj.noeffect),  # [N, 263]，⊆ active（T3 effect-mask）
        "value": f(traj.value),
        "done": f(traj.done),
        "seat": f(traj.seat),
        "valid": f(traj.valid),
        "env_id": env_id,
        "turn": turn,
        "game_step": f(traj.step),  # 真 step（重開局的 env 與 turn 不同）
        "n_rows_total": int(tt * bb),
        "n_rows_kept": int(keep.sum()),
    }


def padding_matches_final(traj: Transition, final) -> bool:
    """**padding 的每一個 state 葉**都要逐位等於該 env 的真局末 state。

    定義：對每個 env，`turn > last_valid_t` 的所有列，`traj.state` 的**全部葉**
    （不只 `step`）都等於 `final` 的那一列——freeze-in-carry 之後 state 不再變動。
    """
    import numpy as np  # noqa: PLC0415

    valid = np.asarray(traj.valid)
    t, b = valid.shape
    last = np.asarray(P1.last_valid_step(traj.valid))
    pad = np.arange(t)[:, None] > last[None, :]
    if not pad.any():
        return True
    for leaf_t, leaf_f in zip(jax.tree.leaves(traj.state), jax.tree.leaves(final), strict=True):
        a = np.asarray(leaf_t)  # [T, B, ...]
        f_ = np.asarray(leaf_f)  # [B, ...]
        want = np.broadcast_to(f_[None], a.shape)
        if not np.array_equal(a[pad], want[pad]):
            return False
    return True


def trace_summary(traj: Transition, final, *, seat: int) -> dict[str, Any]:
    """小段 trace 的局末帳：valid／padding 行數、done 位置、真局末錢——沿 P1 的口徑。"""
    valid = traj.valid
    last_t = P1.last_valid_step(valid)  # [B]
    money = final.money[:, seat]
    return {
        "T": int(valid.shape[0]),
        "B": int(valid.shape[1]),
        "n_valid_rows": int(jnp.sum(valid)),
        "n_padding_rows": int(jnp.sum(~valid)),
        "last_valid_t": [int(x) for x in last_t],
        "done_first_t": [
            int(jnp.argmax(traj.done[:, e])) if bool(jnp.any(traj.done[:, e])) else -1
            for e in range(valid.shape[1])
        ],
        "final_step": [int(x) for x in final.step],
        "final_money_own": [int(x) for x in money],
        # freeze 的定義與 `padding_matches_final` 一致：padding 列的**全部 state 葉**
        # 逐位等於該 env 的真局末 state，不只 `step` 單調。
        "padding_state_equals_final": padding_matches_final(traj, final),
    }


def config_for(caps: caps_mod.Caps = CAPS) -> dict[str, Any]:
    """actor 的 cfg。`H.config` 只看 `caps.max_units`，所以與 IL 的 cfg 逐欄相同。"""
    return H.config(caps)


__all__ = [
    "CAPS",
    "flatten_bridge",
    "padding_matches_final",
    "LABEL_FIELDS",
    "Transition",
    "config_for",
    "joint_action",
    "joint_logp",
    "make_decoders",
    "own_fields",
    "recompute",
    "recompute_report",
    "rollout",
    "step_once",
    "trace_summary",
]
