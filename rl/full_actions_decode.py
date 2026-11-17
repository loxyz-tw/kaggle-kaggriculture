"""完整動作表示的條件解碼核心：sample／greedy／teacher 同一支。

規格：內部規格文件 §第一段；審查者 §14 三項落點。

契約（**已釘，不得退回近似**）：

1. **qty 不因持有量被遮或截**。正整數 request 可以超過持有，引擎允許部分執行；
   qty 只受 U31 的域限制：``b0 ≤ 127``、四 byte 不得全零
   （`rl/ordered_v3_contract.py:644-657`）。
2. **不做 tile_claim 這種手寫近似**。同一格「先 DIG 再 PLANT」「PLANT 再 WATER」是合法序列，
   要由**真 prefix state** 決定。每個單位的合法遮罩都從
   ``kernels.apply_unit_actions(原 state, 已選 prefix ＋ 其餘 PASS)`` 重建的 state 算，
   **每次都從原 state 重建**，不在投影過的 state 上反覆重放。
3. **qty 是條件 decoder**：hidden 由 (unit token, 已選 verb, 已選 item) 起始，逐 byte 更新
   （`byte_embed` ＋ `prefix_step`，與 `rl/ordered_v3_policy.py::_qty_scan` 同結構）；
   市場的 item logits 吃**已選的 op**，qty hidden 吃 (slot hidden, op, item)。
   sample／greedy／teacher 走同一條 prefix。

引擎的 **atomic PLANT 預檢**（`rl/kernels.py:181-191`）：某作物的 PLANT 需求總量超過持有時，
該作物的**每一個** PLANT 都變 PASS（不是先到先得）。我們**不另外預留種子**——遮罩來自
prefix state 的 `plant_crop`（`empty & seeds > 0`），種子在 prefix 重建時已經被前面的單位扣掉，
所以逐單位看到的是真實剩餘；raw codec 仍保留 overdraw 的表示能力。

兩趟 encoder：第一趟 trunk → 單位 token；單位相選完後，用**真 item／真 qty** 投影
（同一顆 `kernels.apply_unit_actions`），第二趟 trunk → 市場相。只用自家可見資訊，
對手在投影用的 action 裡全 PASS。

逐 token 的 `logp`／`active` 排成固定 263 格：前 29×(verb, crop, item, b0..b3)，
後 10×(op, item, b0..b3)。inactive 位置 logp ＝ 0、active ＝ False。
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from rl import caps as caps_mod
from rl import full_actions_head as H
from rl import kernels, rules
from rl import obs as rl_obs
from rl import ordered_v3_contract as C
from rl.actions import SimAction

F32 = jnp.float32
I32 = jnp.int32
NEG = -1e30
#: KL 錨的完整分布類別維（最大 head 是 qty byte 的 256 類）
DIST_CLASSES = 256

MODES = ("sample", "greedy", "teacher")
N_SLOTS = H.N_LOGP_SLOTS  # 263
UNIT_SLOTS = 3 + C.N_QTY_BYTES  # 7
MARKET_SLOTS = 2 + C.N_QTY_BYTES  # 6

_ITEM_SUPPORT = jnp.asarray(
    [
        [it in C.MARKET_ITEM_SUPPORT.get(op, frozenset()) for it in range(H.N_ITEMS)]
        for op in range(H.N_MARKET_OPS)
    ]
)
_NEEDS_QTY = jnp.asarray(
    [op not in (C.MO_NONE, C.MO_HIRE, C.MO_BUY_LAND) for op in range(H.N_MARKET_OPS)]
)


def _entropy(lp, mask):
    """masked 分布的條件 entropy ``-Σ p·log p``。

    被遮掉的類別 `lp` 是 `NEG`，`exp(lp)·lp` 會是極小的負數而不是 0，所以**明確用
    `where` 歸零**，不靠數值僥倖；這也讓 `NEG` 換成 `-inf` 時不會冒出 nan。
    """
    return -jnp.sum(jnp.where(mask, jnp.exp(lp) * lp, 0.0), axis=-1)


def _pick(logits, mask, key, mode, label):
    """一個 categorical 位置 → `(token, logp, eligible, entropy)`。

    `entropy` 是**這個位置在已給 prefix 下的精確條件 entropy**（同一份 `mask`、
    同一份 `log_softmax`），三個 mode 共用；不是 `-logp`，也不另跑一次 forward。

    `mask` 為 True 的類別才可選。**teacher 模式**照樣吃 raw label（不改標、不換 PASS），
    但回 `eligible = mask[label]`：落在共用遮罩外的 teacher token **不進 loss 分母**
    （審查者 §14.1），原始 token 與真 prefix 投影都保留。
    """
    lg = jnp.where(mask, logits, NEG)
    lp = jax.nn.log_softmax(lg, axis=-1)
    if mode == "teacher":
        tok = label.astype(I32)
        ok = jnp.take_along_axis(mask, tok[..., None], axis=-1)[..., 0]
    else:
        tok = (
            jnp.argmax(lg, axis=-1) if mode == "greedy" else jax.random.categorical(key, lg, -1)
        ).astype(I32)
        ok = jnp.ones(tok.shape, bool)
    lpv = jnp.take_along_axis(lp, tok[..., None], axis=-1)[..., 0]
    return tok, jnp.where(ok, lpv, 0.0), ok, _entropy(lp, mask), _dist(lp, mask)


def _dist(lp, mask):
    """完整條件分布（T3 KL 錨用）：非法類別 logp 置 0、mask 置 False，類別維補到 256。

    補齊只為了把不同 head 疊成同一張 ``[..., 263, 256]``；補出來的類別 mask 為 False，
    KL 以 `where(mask, …)` 歸零，不會用到其 logp。
    """
    pad = DIST_CLASSES - lp.shape[-1]
    widths = [(0, 0)] * (lp.ndim - 1) + [(0, pad)]
    return (
        jnp.pad(jnp.where(mask, lp, 0.0), widths),
        jnp.pad(mask, widths, constant_values=False),
    )


def _u31_byte_mask(i: int, prefix_zero: jax.Array) -> jax.Array:
    """第 `i` 個 byte 的合法類別 ``[..., 256]``。**只有 U31 的域**，與持有量無關。"""
    cls = jnp.arange(C.BYTE_CLASSES)
    if i == 0:
        return jnp.broadcast_to(cls <= C.QTY_HIGH_BYTE_MAX, prefix_zero.shape + (C.BYTE_CLASSES,))
    if i == C.N_QTY_BYTES - 1:
        return (cls > 0)[None, :] | (~prefix_zero)[..., None]
    return jnp.broadcast_to(cls >= 0, prefix_zero.shape + (C.BYTE_CLASSES,))


def qty_scan(params, h0, key, mode, labels, active, dist: bool = False):
    """逐 byte 的**條件** scan：`h` 每步吃上一個 byte。回 `(bytes[..., 4], logp[..., 4])`。

    `dist=True`（靜態）時另回各 byte 的完整條件分布 ``(lp[..., 4, 256], mask[..., 4, 256])``；
    否則最後一項是 `None`（rollout／部署路徑不產生、不保留）。
    """
    keys = jax.random.split(key, C.N_QTY_BYTES) if key is not None else [None] * C.N_QTY_BYTES
    prefix_zero = jnp.ones(active.shape, bool)
    h = h0
    toks, lps, oks, ents, dl, dm = [], [], [], [], [], []
    for i in range(C.N_QTY_BYTES):
        lg = H.byte_logits(params, h)
        lab = labels[..., i] if labels is not None else None
        tok, lp, ok, ent, (d_lp, d_m) = _pick(
            lg, _u31_byte_mask(i, prefix_zero), keys[i], mode, lab
        )
        tok = jnp.where(active, tok, 0)
        toks.append(tok)
        lps.append(jnp.where(active, lp, 0.0))
        oks.append(ok)
        ents.append(jnp.where(active, ent, 0.0))
        dl.append(d_lp)
        dm.append(d_m)
        prefix_zero = prefix_zero & (tok == 0)
        h = jnp.where(active[..., None], H.byte_advance(params, h, tok), h)
    d = (jnp.stack(dl, axis=-2), jnp.stack(dm, axis=-2)) if dist else None
    return (
        jnp.stack(toks, axis=-1),
        jnp.stack(lps, axis=-1),
        jnp.stack(oks, axis=-1),
        jnp.stack(ents, axis=-1),
        d,
    )


def _empty_action(state, caps: caps_mod.Caps) -> SimAction:
    b = state.step.shape[0]
    u, o = caps.max_units, caps.max_orders
    return SimAction(
        unit_op=jnp.zeros((b, 2, u), I32),
        unit_arg=jnp.zeros((b, 2, u), I32),
        unit_qty=jnp.zeros((b, 2, u), I32),
        order_op=jnp.zeros((b, 2, o), I32),
        order_item=jnp.full((b, 2, o), -1, I32),
        order_qty=jnp.zeros((b, 2, o), I32),
    )


def unit_arg_of(verb, crop, item) -> jax.Array:
    """引擎的 `unit_arg`：PLANT 帶作物、PICKUP／PLACE 帶品項，其餘 0。"""
    return jnp.where(
        verb == C.OP_PLANT,
        crop,
        jnp.where((verb == C.OP_PICKUP) | (verb == C.OP_PLACE), item, jnp.zeros_like(verb)),
    )


def u31_from_bytes(qty_bytes: jax.Array) -> jax.Array:
    """big-endian 四 byte → int32（與 `rl/ordered_v3_codec` 同一定義）。"""
    w = jnp.asarray([1 << 24, 1 << 16, 1 << 8, 1], I32)
    return jnp.sum(qty_bytes.astype(I32) * w, axis=-1)


def build_action(
    state,
    fields: dict,
    seat: int,
    caps: caps_mod.Caps,
    upto: int | None = None,
    *,
    market: bool = False,
):
    """**從原 state** 組 `SimAction`：只放自己這一座、只放前 `upto` 個單位，其餘 PASS。

    `market=True` 時連訂單一起放（`fields` 要有 `market_*`）——給 `rl.env.step` 驗**成交**用；
    單位相的 prefix 重建一律 `market=False`，市場不在那一相發生。
    """
    act = _empty_action(state, caps)
    u = caps.max_units
    verb, crop, item = fields["unit_verb"], fields["unit_crop"], fields["unit_item"]
    qty = u31_from_bytes(fields["unit_qty_bytes"])
    if upto is not None:
        live = jnp.arange(u)[None, :] < upto
        verb = jnp.where(live, verb, C.OP_PASS)
        crop = jnp.where(live, crop, 0)
        item = jnp.where(live, item, 0)
        qty = jnp.where(live, qty, 0)
    act = act._replace(
        unit_op=act.unit_op.at[:, seat].set(verb),
        unit_arg=act.unit_arg.at[:, seat].set(unit_arg_of(verb, crop, item)),
        unit_qty=act.unit_qty.at[:, seat].set(qty),
    )
    if not market:
        return act
    op = fields["market_op"]
    # 沒有品項的 op（HIRE／BUY_LAND／EOS）維持 -1，與 `_empty_action` 的 sentinel 一致
    m_item = jnp.where(jnp.any(_ITEM_SUPPORT[op], axis=-1), fields["market_item"], -1)
    return act._replace(
        order_op=act.order_op.at[:, seat].set(op),
        order_item=act.order_item.at[:, seat].set(m_item),
        order_qty=act.order_qty.at[:, seat].set(u31_from_bytes(fields["market_qty_bytes"])),
    )


def prefix_state(state, fields: dict, seat: int, caps: caps_mod.Caps, upto: int):
    """前 `upto` 個單位做完之後的 state——**從原 state 重建**，不鏈式投影。"""
    return kernels.apply_unit_actions(state, build_action(state, fields, seat, caps, upto), caps)


#: A5：成功放上結構的動物，qty 被引擎忽略；policy 固定填這個代表值（U31 的 1），不抽 qty
A5_QTY_BYTES = jnp.asarray([0, 0, 0, 1], I32)


def _unit_tile_and_inv(st, seat, k):
    """單位 k 在 prefix state 的所在格 row ``[B, C]``、背包 ``[B, 12]``、是否在 shed 取用格。"""
    x = st.unit_x[:, seat, k]
    y = st.unit_y[:, seat, k]
    flat = y * rules.BOARD_SIZE + x
    tile = jnp.take_along_axis(st.tiles[:, seat], flat[:, None, None], axis=1)[:, 0]
    shed_adj = jnp.any(flat[:, None] == kernels._SHED_ACCESS_FLAT[None, :], axis=-1)
    return tile, st.inv[:, seat, k, :], shed_adj


def a5_place_animal_success(st, seat, k, verb, item) -> jax.Array:
    """T3 A5：真 prefix 下 PLACE 會走「放上對應空結構」成功分支（`kernels.py` 的 c_place_a_ok）。"""
    tile, inv_k, _adj = _unit_tile_and_inv(st, seat, k)
    is_anim = (item >= rules.ANIMAL_ITEM_BASE) & (item < rules.N_ITEMS)
    a_local = jnp.clip(item - rules.ANIMAL_ITEM_BASE, 0, rules.N_ANIMALS - 1)
    held = jnp.take_along_axis(inv_k, jnp.clip(item, 0, rules.N_ITEMS - 1)[:, None], 1)[:, 0]
    return (
        (verb == C.OP_PLACE)
        & is_anim
        & (tile[:, rules.C_KIND] == kernels._ANIMAL_STRUCT[a_local])
        & (tile[:, rules.C_ANIMAL] < 0)
        & (held >= 1)
    )


def pickup_certainly_empty(st, seat, k, verb, item) -> jax.Array:
    """T3 空 PICKUP：在 shed 取用格、輪到它時（真 prefix）該品 shed 存量確定為 0。"""
    _tile, _inv, adj = _unit_tile_and_inv(st, seat, k)
    stock = jnp.take_along_axis(
        st.shed[:, seat, :], jnp.clip(item, 0, rules.N_ITEMS - 1)[:, None], 1
    )[:, 0]
    return (verb == C.OP_PICKUP) & adj & (stock == 0)


def _m2_step(upper, lower, real, op, item, qty_bytes):
    """T3 M2：一筆訂單推進自家 shed 保守上下界（decoder 與 census 共用這一支）。

    起點＝單位相投影後的自家 shed。SELL 確定只賣 ``min(req, 存量)``；BUY_PRODUCT／BUY_ANIMAL
    可能失敗（錢不足、對手交錯）→ 只推高上界（``+min(req,100)``、上界 ≤ 100），下界不動。
    上界＝0 ⇒ 該 SELL 確定是空單（M2）；下界＝0＜上界 ⇒ unknown（保留 loss、計數）。
    回 ``(m2, unknown, upper, lower)``。
    """
    q_req = jnp.minimum(u31_from_bytes(qty_bytes), rules.SHED_CAPACITY)
    it_c = jnp.clip(item, 0, rules.N_ITEMS - 1)
    up_i = jnp.take_along_axis(upper, it_c[:, None], 1)[:, 0]
    lo_i = jnp.take_along_axis(lower, it_c[:, None], 1)[:, 0]
    is_sell = real & (op == C.MO_SELL)
    m2 = is_sell & (up_i == 0)
    unknown = is_sell & (up_i > 0) & (lo_i == 0)
    is_buy = real & ((op == C.MO_BUY_PRODUCT) | (op == C.MO_BUY_ANIMAL))
    oh = jax.nn.one_hot(it_c, rules.N_ITEMS, dtype=I32).astype(bool)
    new_up = jnp.where(is_sell, jnp.maximum(up_i - q_req, 0), jnp.where(is_buy, up_i + q_req, up_i))
    new_up = jnp.minimum(new_up, rules.SHED_CAPACITY)
    new_lo = jnp.where(is_sell, jnp.maximum(lo_i - q_req, 0), lo_i)
    upper = jnp.where(oh, new_up[:, None], upper)
    lower = jnp.where(oh, new_lo[:, None], lower)
    return m2, unknown, upper, lower


#: T3 effect-mask（`t3_stabilization_v4`）規則版本；改規則必須換版本名
EFFECT_RULES_VERSION = "t3-effect-v1"
#: E4：品項最低單價。BUY_ANIMAL 用固定動物價；BUY_PRODUCT 用 `rules.PRICE_FLOOR`
_ANIMAL_COST_BY_ITEM = jnp.asarray([0] * rules.ANIMAL_ITEM_BASE + list(rules.ANIMAL_COST), I32)


def effect_pre_qty(upper, real, op, item, cash_up, cash_unknown):
    """T3 effect-mask：**在該訂單 qty 開始之前**（op／item 已定）判定 E3／E4（四 byte 共用）。

    - E3：SELL 且自家 shed 上界＝0（與 `_m2_step` 的 m2 同一條件）→ 必零成交。
    - E4：現金上界已知（`~cash_unknown`）且 BUY_ANIMAL 的 ``cash_up < 動物固定價``，或
      BUY_PRODUCT 的 ``cash_up < PRICE_FLOOR`` → 第一單位即失敗、整單零成交。
    - 本單若是**可能有收入的 SELL**（上界 >0），其後現金上界變 unknown（不造價格模型）。
      BUY／HIRE／BUY_LAND 不減上界（可能失敗）。上界為負視為 unknown。
    回 ``(e3, e4, e4_unknown_buy, cash_unknown_next)``。
    """
    it_c = jnp.clip(item, 0, rules.N_ITEMS - 1)
    up_i = jnp.take_along_axis(upper, it_c[:, None], 1)[:, 0]
    is_sell = real & (op == C.MO_SELL)
    e3 = is_sell & (up_i == 0)
    known = ~cash_unknown & (cash_up >= 0)
    is_ba = real & (op == C.MO_BUY_ANIMAL) & (item >= rules.ANIMAL_ITEM_BASE)
    is_bp = real & (op == C.MO_BUY_PRODUCT)
    price = jnp.where(is_ba, _ANIMAL_COST_BY_ITEM[it_c], rules.PRICE_FLOOR)
    is_buy = is_ba | is_bp
    e4 = is_buy & known & (cash_up < price)
    e4_unknown = is_buy & ~known
    cash_unknown_next = cash_unknown | (cash_up < 0) | (is_sell & (up_i > 0))
    return e3, e4, e4_unknown, cash_unknown_next


def teacher_unit_fields(labels: dict, n_alive) -> dict:
    """teacher 模式下 decoder 實際放進 prefix 的單位欄位（raw label 依 verb 清成語法欄位）。

    與 `_unit_step` teacher 分支逐欄相同：死單位 PASS；crop 只在 PLANT；item／qty 只在
    PICKUP／PLACE（A5 也保留原 qty）。
    """
    u = labels["unit_verb"].shape[1]
    alive = jnp.arange(u)[None, :] < jnp.asarray(n_alive, I32)[:, None]
    verb = jnp.where(alive, labels["unit_verb"], C.OP_PASS).astype(I32)
    iq = alive & ((verb == C.OP_PICKUP) | (verb == C.OP_PLACE))
    return {
        "unit_verb": verb,
        "unit_crop": jnp.where(alive & (verb == C.OP_PLANT), labels["unit_crop"], 0).astype(I32),
        "unit_item": jnp.where(iq, labels["unit_item"], 0).astype(I32),
        "unit_qty_bytes": jnp.where(iq[..., None], labels["unit_qty_bytes"], 0).astype(I32),
    }


def il_mask_info_from_labels(state, labels: dict, seat: int, caps: caps_mod.Caps, n_alive):
    """teacher 遮罩（A5／空 PICKUP／M2／unknown）**不跑神經網路**的版本，給 I2 全量計數用。

    與 `decode("teacher")` 用**同一套** prefix 定義：每個單位都以
    ``prefix_state(原 state, 已選 prefix, upto=k)`` 從原 state 重建（含 HARVEST 清格、
    atomic PLANT 只看已選 prefix 的需求），同一組 `a5_place_animal_success`／
    `pickup_certainly_empty`，以及同一個 `_m2_step`。逐項相等由測試對 teacher decode 核。
    """
    fields = teacher_unit_fields(labels, n_alive)
    u = fields["unit_verb"].shape[1]
    alive_all = jnp.arange(u)[None, :] < jnp.asarray(n_alive, I32)[:, None]

    def body(carry, k):
        st_k = prefix_state(state, fields, seat, caps, k)
        verb = fields["unit_verb"][:, k]
        item = fields["unit_item"][:, k]
        alive = alive_all[:, k]
        need_iq = alive & ((verb == C.OP_PICKUP) | (verb == C.OP_PLACE))
        a5 = need_iq & a5_place_animal_success(st_k, seat, k, verb, item)
        pk = alive & pickup_certainly_empty(st_k, seat, k, verb, item)
        return carry, (a5, pk)

    _c, (a5_s, pk_s) = jax.lax.scan(body, 0, jnp.arange(u))
    nxt = kernels.apply_unit_actions(state, build_action(state, fields, seat, caps), caps)

    def mbody(carry, j):
        open_, upper, lower = carry
        op = jnp.where(open_, labels["market_op"][:, j], C.MO_NONE)
        real = open_ & (op != C.MO_NONE)
        has_item = jnp.any(_ITEM_SUPPORT[op], axis=-1) & real
        item = jnp.where(has_item, labels["market_item"][:, j], 0)
        need_q = _NEEDS_QTY[op] & real
        qty = jnp.where(need_q[:, None], labels["market_qty_bytes"][:, j], 0)
        m2, unk, upper, lower = _m2_step(upper, lower, real, op, item, qty)
        return (real, upper, lower), (m2, unk)

    shed0 = nxt.shed[:, seat, :].astype(I32)
    b = shed0.shape[0]
    _c2, (m2_s, unk_s) = jax.lax.scan(
        mbody, (jnp.ones((b,), bool), shed0, shed0), jnp.arange(C.MAX_ORDERS)
    )
    mv = lambda x: jnp.moveaxis(x, 0, 1)  # noqa: E731
    return {"a5": mv(a5_s), "pickup_empty": mv(pk_s), "m2": mv(m2_s), "m2_unknown": mv(unk_s)}


def _unit_step(params, h_k, st_k, seat, k, alive, key, mode, lab, caps, dist=False):
    """單一單位（引擎 slot 序）：遮罩來自**真 prefix state** `st_k`。

    `caps` 由呼叫端串下來——**不得寫死 `BOUNDED_TRAINING`**，否則部署／評估用別的 caps
    時遮罩會跟實際 slot 數對不上。
    """
    leg = rl_obs.legality(st_k, caps)
    kk = jax.random.split(key, 4) if key is not None else (None,) * 4
    lg = H.unit_logits(params, h_k[:, None])
    verb, l_verb, ok_verb, h_verb, d_verb = _pick(
        lg["verb"][:, 0], leg["unit_ops"][:, seat, k], kk[0], mode, lab["unit_verb"]
    )
    verb = jnp.where(alive, verb, C.OP_PASS)
    l_verb = jnp.where(alive, l_verb, 0.0)
    h_verb = jnp.where(alive, h_verb, 0.0)
    ok_verb = ok_verb | ~alive

    need_crop = alive & (verb == C.OP_PLANT)
    crop, l_crop, ok_crop, h_crop, d_crop = _pick(
        lg["crop"][:, 0], leg["plant_crop"][:, seat, k], kk[1], mode, lab["unit_crop"]
    )
    crop = jnp.where(need_crop, crop, 0)
    l_crop = jnp.where(need_crop, l_crop, 0.0)
    h_crop = jnp.where(need_crop, h_crop, 0.0)

    # item 的 logits **吃已選 verb**（PICKUP 與 PLACE 不共用同一組 logits）
    need_iq = alive & ((verb == C.OP_PICKUP) | (verb == C.OP_PLACE))
    item_mask = jnp.where(
        (verb == C.OP_PICKUP)[:, None],
        leg["pickup_item"][:, seat, k],
        leg["place_item"][:, seat, k],
    )
    item, l_item, ok_item, h_item, d_item = _pick(
        H.unit_item_logits(params, h_k, verb),
        jnp.where(need_iq[:, None], item_mask, jnp.ones_like(item_mask)),
        kk[2],
        mode,
        lab["unit_item"],
    )
    item = jnp.where(need_iq, item, 0)
    l_item = jnp.where(need_iq, l_item, 0.0)
    h_item = jnp.where(need_iq, h_item, 0.0)

    # T3 A5：成功放動物 → qty 不是 policy 選擇（全部 qty 槽 inactive、logp／entropy 為 0）
    a5 = need_iq & a5_place_animal_success(st_k, seat, k, verb, item)
    need_q = need_iq & ~a5
    qty, l_qty, ok_qty, h_qty, d_qty = qty_scan(
        params,
        H.qty_init_unit(params, h_k, verb, item),
        kk[3],
        mode,
        lab["unit_qty_bytes"],
        need_q,
        dist,
    )
    # teacher 保留原 qty（核原動作效果）；sample／greedy 填固定代表值 1
    a5_fill = lab["unit_qty_bytes"] if mode == "teacher" else A5_QTY_BYTES[None, :]
    qty = jnp.where(a5[:, None], a5_fill, qty)
    # T3 空 PICKUP：只影響 IL loss（verb/item/qty），不改 policy_active
    pk_empty = alive & pickup_certainly_empty(st_k, seat, k, verb, item)
    ok_crop = ok_crop & ok_verb
    ok_item = ok_item & ok_verb
    ok_qty = ok_qty & (ok_verb & ok_item)[:, None]
    fields = {"unit_verb": verb, "unit_crop": crop, "unit_item": item, "unit_qty_bytes": qty}
    lp = {"verb": l_verb, "crop": l_crop, "item": l_item, "qty": l_qty}
    ent = {"verb": h_verb, "crop": h_crop, "item": h_item, "qty": h_qty}
    need_q4 = jnp.broadcast_to(need_q[:, None], qty.shape)
    act = {
        "verb": alive & ok_verb,
        "crop": need_crop & ok_crop,
        "item": need_iq & ok_item,
        "qty": need_q4 & ok_qty,
    }
    drop = {
        "verb": alive & ~ok_verb,
        "crop": need_crop & ~ok_crop,
        "item": need_iq & ~ok_item,
        "qty": need_q4 & ~ok_qty,
        "verb_by_op": jax.nn.one_hot(verb, H.N_UNIT_OPS, dtype=I32) * (alive & ~ok_verb)[:, None],
    }
    info = {
        "a5": a5,
        "pickup_empty": pk_empty,
        # T3 effect-mask E1：空 PICKUP 的 qty（只遮已 active 的 qty 槽）
        "noeffect_qty": pk_empty[:, None] & act["qty"],
    }
    if dist:
        info["dist"] = {"verb": d_verb, "crop": d_crop, "item": d_item, "qty": d_qty}
    return fields, lp, act, drop, ent, info


def _unit_phase(params, h_unit, state, seat, caps, n_alive, key, mode, labels, dist=False):
    """單位相：`lax.scan` 跑 29 個 slot（**不 Python 展開**，否則 29 份 kernel 會炸編譯）。

    每一步的遮罩都從 ``prefix_state(原 state, 已選 prefix, upto=k)`` 重建。
    """
    b, u, _ = h_unit.shape
    keys = jax.random.split(key, u) if key is not None else jnp.zeros((u, 2), jnp.uint32)
    alive_all = jnp.arange(u)[None, :] < jnp.asarray(n_alive, I32)[:, None]
    zeros_u = jnp.zeros((b, u), I32)
    init_fields = {
        "unit_verb": jnp.full((b, u), C.OP_PASS, I32),
        "unit_crop": zeros_u,
        "unit_item": zeros_u,
        "unit_qty_bytes": jnp.zeros((b, u, C.N_QTY_BYTES), I32),
    }

    def body(fields, k):
        st_k = prefix_state(state, fields, seat, caps, k)
        lab = (
            {
                "unit_verb": labels["unit_verb"][:, k],
                "unit_crop": labels["unit_crop"][:, k],
                "unit_item": labels["unit_item"][:, k],
                "unit_qty_bytes": labels["unit_qty_bytes"][:, k],
            }
            if labels
            else dict.fromkeys(("unit_verb", "unit_crop", "unit_item", "unit_qty_bytes"), None)
        )
        f, lp, act, drop, ent, info = _unit_step(
            params,
            h_unit[:, k],
            st_k,
            seat,
            k,
            alive_all[:, k],
            None if key is None else keys[k],
            mode,
            lab,
            caps,
            dist,
        )
        nxt = {
            "unit_verb": fields["unit_verb"].at[:, k].set(f["unit_verb"]),
            "unit_crop": fields["unit_crop"].at[:, k].set(f["unit_crop"]),
            "unit_item": fields["unit_item"].at[:, k].set(f["unit_item"]),
            "unit_qty_bytes": fields["unit_qty_bytes"].at[:, k].set(f["unit_qty_bytes"]),
        }
        return nxt, (lp, act, drop, ent, info)

    fields, (lp_s, act_s, drop_s, ent_s, info_s) = jax.lax.scan(body, init_fields, jnp.arange(u))
    mv = lambda x: jnp.moveaxis(x, 0, 1)  # noqa: E731  # [u, b, ...] → [b, u, ...]
    lp = {k: mv(v) for k, v in lp_s.items()}
    act = {k: mv(v) for k, v in act_s.items()}
    drop = {k: mv(v) for k, v in drop_s.items() if k != "verb_by_op"}
    drop["verb_by_op"] = jnp.sum(drop_s["verb_by_op"], axis=0)
    info = {k: mv(v) for k, v in info_s.items() if k != "dist"}  # [b, u]
    if dist:
        info["dist"] = jax.tree.map(mv, info_s["dist"])
    return fields, lp, act, drop, {k: mv(v) for k, v in ent_s.items()}, info


def _market_phase(params, h2, nxt, seat, key, mode, labels, dist=False):
    """市場相：`lax.scan` 跑 10 槽。

    **遮罩只有語法域**：`MARKET_ITEM_SUPPORT`（op → 可用品項）、EOS、U31。
    **不以資金／庫存遮 intent**——「先 SELL 再 BUY」「同品項重複下單」都是合法意圖，
    拿 `rl/obs.legality` 的 money／shed 條件去遮就是把舊限制搬回來。三個 mode 同一份遮罩。
    """
    h_glob = h2["glob"]
    b = h_glob.shape[0]
    # T3 A1：pointer candidate keys 一回合算一次
    # （第二趟 product hidden／global ＋ ID ＋ 投影後可見量）
    ptr = H.pointer_keys(params, h2["prod"], h_glob, H.pointer_aux(nxt, seat))
    keys = (
        jax.random.split(key, C.MAX_ORDERS)
        if key is not None
        else jnp.zeros((C.MAX_ORDERS, 2), jnp.uint32)
    )
    op_mask = jnp.ones((b, H.N_MARKET_OPS), bool)  # 七類都是合法語法（0 ＝ EOS）

    def body(carry, k):
        h, open_, upper, lower, cash_up, cash_unk = carry
        kk = jax.random.split(keys[k], 3) if key is not None else (None,) * 3
        lab_op = labels["market_op"][:, k] if labels else None
        lab_item = labels["market_item"][:, k] if labels else None
        lab_q = labels["market_qty_bytes"][:, k] if labels else None

        op, l_op, ok_op, h_op, d_op = _pick(
            H.market_op_logits(params, h), op_mask, kk[0], mode, lab_op
        )
        op = jnp.where(open_, op, C.MO_NONE)
        l_op = jnp.where(open_, l_op, 0.0)
        h_op = jnp.where(open_, h_op, 0.0)
        ok_op = ok_op | ~open_
        real = open_ & (op != C.MO_NONE)

        sup = _ITEM_SUPPORT[op]  # 只有語法域
        has_item = jnp.any(sup, axis=-1) & real
        item, l_item, ok_item, h_item, d_item = _pick(
            H.market_item_logits(params, h, op, ptr),
            jnp.where(has_item[:, None], sup, jnp.ones_like(sup)),
            kk[1],
            mode,
            lab_item,
        )
        item = jnp.where(has_item, item, 0)
        l_item = jnp.where(has_item, l_item, 0.0)
        h_item = jnp.where(has_item, h_item, 0.0)

        need_q = _NEEDS_QTY[op] & real
        # T3 effect-mask：qty 開始**之前**固定判定（四 byte 共用；不看採到的 q）
        e3, e4, e4_unk, cash_unk_next = effect_pre_qty(upper, real, op, item, cash_up, cash_unk)
        qty, l_q, ok_q, h_q, d_q = qty_scan(
            params, H.qty_init_market(params, h, op, item, ptr), kk[2], mode, lab_q, need_q, dist
        )
        ok_item = ok_item & ok_op
        ok_q = ok_q & (ok_op & ok_item)[:, None]
        need_q4 = jnp.broadcast_to(need_q[:, None], qty.shape)

        m2, m2_unknown, upper, lower = _m2_step(upper, lower, real, op, item, qty)

        out = (
            {"op": op, "item": item, "qty": qty},
            {"op": l_op, "item": l_item, "qty": l_q},
            {"op": open_ & ok_op, "item": has_item & ok_item, "qty": need_q4 & ok_q},
            {
                "op": open_ & ~ok_op,
                "item": has_item & ~ok_item,
                "qty": need_q4 & ~ok_q,
                "op_by_code": jax.nn.one_hot(op, H.N_MARKET_OPS, dtype=I32)
                * (open_ & ~ok_op)[:, None],
            },
            {"op": h_op, "item": h_item, "qty": h_q},
            {
                "m2": m2,
                "m2_unknown": m2_unknown,
                "noeffect_qty": (e3 | e4)[:, None] & need_q4 & ok_q,
                "e3": e3 & need_q,
                "e4": e4 & need_q,
                "e4_unknown": e4_unk & need_q,
                "e5": real & ((op == C.MO_HIRE) | (op == C.MO_BUY_LAND)),
            }
            | ({"dist": {"op": d_op, "item": d_item, "qty": d_q}} if dist else {}),
        )
        nxt_carry = (H.market_advance(params, h, op, item, qty), real, upper, lower)
        return nxt_carry + (cash_up, cash_unk_next), out

    shed0 = nxt.shed[:, seat, :].astype(I32)
    # 現金上界起點＝單位相完成後、市場相開始時的該座 money（單位相不改 money）
    cash0 = nxt.money[:, seat].astype(I32)
    (_h, _open, _up, _lo, _cu, _ck), (f_s, lp_s, act_s, drop_s, ent_s, info_s) = jax.lax.scan(
        body,
        (
            H.market_init(params, h_glob),
            jnp.ones((b,), bool),
            shed0,
            shed0,
            cash0,
            jnp.zeros((b,), bool),
        ),
        jnp.arange(C.MAX_ORDERS),
    )
    mv = lambda x: jnp.moveaxis(x, 0, 1)  # noqa: E731
    fields = {
        "market_op": mv(f_s["op"]),
        "market_item": mv(f_s["item"]),
        "market_qty_bytes": mv(f_s["qty"]),
    }
    # length 用 raw token 算（EOS 之前的真訂單數），不受 teacher 資格影響
    fields["market_length"] = jnp.sum(fields["market_op"] != C.MO_NONE, axis=1).astype(I32)
    drop = {k: mv(v) for k, v in drop_s.items() if k != "op_by_code"}
    drop["op_by_code"] = jnp.sum(drop_s["op_by_code"], axis=0)
    return (
        fields,
        {k: mv(v) for k, v in lp_s.items()},
        {k: mv(v) for k, v in act_s.items()},
        drop,
        {k: mv(v) for k, v in ent_s.items()},
        {k: (jax.tree.map(mv, v) if k == "dist" else mv(v)) for k, v in info_s.items()},
    )


def il_loss_mask(unit_info: dict, market_info: dict) -> jax.Array:
    """T3 §5.1.3：IL 直接 loss 的額外遮罩，固定 263 槽（與 `pack_one` 同序）。

    只**額外去除**直接 loss；不改 raw、合法 support、teacher prefix、後續 hidden、policy_active。
    - 空 PICKUP：該單位 verb／crop／item／qty 七槽為 0。
    - A5：該單位 qty 四槽為 0（policy 本來就 inactive，這裡再明示）。
    - M2：該筆訂單 op／item／qty 六槽為 0；其他訂單與 EOS 不受影響。
    """
    pk = unit_info["pickup_empty"]  # [B, U]
    a5 = unit_info["a5"]
    keep_u = ~pk
    q_keep = keep_u & ~a5
    unit = {
        "verb": keep_u,
        "crop": keep_u,
        "item": keep_u,
        "qty": jnp.broadcast_to(q_keep[..., None], q_keep.shape + (C.N_QTY_BYTES,)),
    }
    keep_m = ~market_info["m2"]  # [B, 10]
    market = {
        "op": keep_m,
        "item": keep_m,
        "qty": jnp.broadcast_to(keep_m[..., None], keep_m.shape + (C.N_QTY_BYTES,)),
    }
    return pack_one(unit, market)


def pack_one(unit: dict, market: dict) -> jax.Array:
    """單位（verb／crop／item／4 qty byte）＋ 市場（op／item／4 qty byte）→ 固定 263 格。

    `logp`／`active`／`entropy` 共用這一支，槽序只定義在這裡一次。
    """
    u = jnp.concatenate(
        [unit["verb"][..., None], unit["crop"][..., None], unit["item"][..., None], unit["qty"]],
        axis=-1,
    )
    m = jnp.concatenate(
        [market["op"][..., None], market["item"][..., None], market["qty"]], axis=-1
    )
    b = u.shape[0]
    return jnp.concatenate([u.reshape(b, -1), m.reshape(b, -1)], axis=-1)


def pack_slots(ulp, uact, mlp, mact) -> tuple[jax.Array, jax.Array]:
    """逐 head 的 logp／active → 固定 263 格（PPO 用）。"""
    return pack_one(ulp, mlp), pack_one(uact, mact)


def pack_dist(unit: dict, market: dict) -> tuple[jax.Array, jax.Array]:
    """各 head 的 `(lp, mask)`（類別維 256）→ ``[B, 263, 256]`` 兩張，槽序同 `pack_one`。"""

    def one(i):
        u = jnp.concatenate(
            [
                unit["verb"][i][..., None, :],
                unit["crop"][i][..., None, :],
                unit["item"][i][..., None, :],
                unit["qty"][i],
            ],
            axis=-2,
        )
        m = jnp.concatenate(
            [market["op"][i][..., None, :], market["item"][i][..., None, :], market["qty"][i]],
            axis=-2,
        )
        b = u.shape[0]
        return jnp.concatenate(
            [u.reshape(b, -1, DIST_CLASSES), m.reshape(b, -1, DIST_CLASSES)], axis=1
        )

    return one(0), one(1)


def _dist_out(udist: dict, mdist: dict) -> dict:
    lp, mask = pack_dist(udist, mdist)
    return {"lp": lp, "mask": mask}


def decode(mode: str, dist: bool = False):
    """回一支解碼函式；`mode` 關進 closure（不得當 traced 值）。

    `dist=True`（T3 KL 錨，只給 teacher 模式）另回 ``"dist": {"lp", "mask"}``：各槽在同一
    prefix／同一合法 mask 下的完整條件分布 ``[B, 263, 256]``。預設 False，輸出與既有完全相同。
    """
    if dist and mode != "teacher":
        raise ValueError("dist 只給 teacher 模式（KL 錨沿 rollout 原 prefix 重算）")
    if mode not in MODES:
        raise ValueError(f"mode 必須是 {MODES}，得到 {mode!r}")

    def run(
        params,
        cfg: dict[str, Any],
        state,
        seat: int,
        caps: caps_mod.Caps,
        *,
        key=None,
        labels: dict | None = None,
        obs_fn=None,
    ):
        from rl import full_actions_obs as FO  # noqa: PLC0415

        obs_fn = obs_fn or FO.observation
        if mode == "teacher" and labels is None:
            raise ValueError("teacher 模式要給 labels")
        if mode == "sample" and key is None:
            raise ValueError("sample 模式要給 key")
        k1, k2 = jax.random.split(key, 2) if key is not None else (None, None)

        h1 = H.encode_tokens(params, obs_fn(state, seat, caps), cfg)
        n_alive = jnp.minimum(state.hires_today[:, seat] + 1, cfg["max_units"])
        ufields, ulp, uact, udrop, uent, uinfo = _unit_phase(
            params, h1["unit"], state, seat, caps, n_alive, k1, mode, labels, dist
        )

        # 第二趟：**真 item／真 qty** 投影後的 state；這一趟只供 actor 的市場頭
        nxt = kernels.apply_unit_actions(state, build_action(state, ufields, seat, caps), caps)
        h2 = H.encode_tokens(params, obs_fn(nxt, seat, caps), cfg)
        mfields, mlp, mact, mdrop, ment, minfo = _market_phase(
            params, h2, nxt, seat, k2, mode, labels, dist
        )
        udist, mdist = uinfo.pop("dist", None), minfo.pop("dist", None)
        # T3 effect-mask（新增欄，不改既有輸出）：只遮 qty 四槽
        u_noeff = uinfo.pop("noeffect_qty")
        m_noeff = minfo.pop("noeffect_qty")
        eff = {k: minfo.pop(k) for k in ("e3", "e4", "e4_unknown", "e5")}
        zu, zm = jnp.zeros(u_noeff.shape[:2], bool), jnp.zeros(m_noeff.shape[:2], bool)
        noeffect = pack_one(
            {"verb": zu, "crop": zu, "item": zu, "qty": u_noeff},
            {"op": zm, "item": zm, "qty": m_noeff},
        )

        logp, active = pack_slots(ulp, uact, mlp, mact)
        # 條件 entropy：**同一份 mask、同一次 forward**，槽序與 `logp`／`active` 相同。
        # 只在 policy 真的做了選擇的位置有值（其餘已在各相位歸 0）。
        entropy = pack_one(uent, ment)
        return {
            "decision": {**ufields, **mfields},
            "logp": logp,
            "active": active,
            "entropy": entropy,
            # **V(s) 吃動作前的 state**（h1）；h2 是投影後的，只給市場頭用
            "value": H.value(params, h1["glob"]),
            "n_alive": n_alive,
            "projected": nxt,
            "teacher_drop": {
                "unit": {k: v for k, v in udrop.items() if k != "verb_by_op"},
                "unit_verb_by_op": udrop["verb_by_op"],
                "market": {k: v for k, v in mdrop.items() if k != "op_by_code"},
                "market_op_by_code": mdrop["op_by_code"],
            },
            # T3 §5.1.3：IL 專用。PPO／sample 只看 `active`（policy_active），不得讀這兩欄
            "il_loss_mask": il_loss_mask(uinfo, minfo),
            "il_mask_info": {**uinfo, **minfo},
            "noeffect": noeffect,
            "noeffect_info": {
                "e1_orders": jnp.sum(jnp.any(u_noeff, axis=-1), axis=-1),
                "e2_a5": jnp.sum(uinfo["a5"], axis=-1),
                "e3_orders": jnp.sum(eff["e3"], axis=-1),
                "e4_orders": jnp.sum(eff["e4"], axis=-1),
                "e4_unknown_buy": jnp.sum(eff["e4_unknown"], axis=-1),
                "e5_orders": jnp.sum(eff["e5"], axis=-1),
                "m2_unknown": jnp.sum(minfo["m2_unknown"], axis=-1),
                "n_noeffect_slots": jnp.sum(noeffect, axis=-1),
                # 逐訂單旗標 [B, 10]（census／引擎交叉驗證用）
                "order_e3": eff["e3"],
                "order_e4": eff["e4"],
                "order_e4_unknown": eff["e4_unknown"],
                "order_m2_unknown": minfo["m2_unknown"],
                "unit_e1": jnp.any(u_noeff, axis=-1),
            },
            **({"dist": _dist_out(udist, mdist)} if dist else {}),
        }

    run.mode = mode
    return run


__all__ = [
    "EFFECT_RULES_VERSION",
    "effect_pre_qty",
    "MARKET_SLOTS",
    "MODES",
    "N_SLOTS",
    "UNIT_SLOTS",
    "build_action",
    "a5_place_animal_success",
    "decode",
    "il_loss_mask",
    "il_mask_info_from_labels",
    "teacher_unit_fields",
    "pack_one",
    "pack_slots",
    "pickup_certainly_empty",
    "prefix_state",
    "qty_scan",
    "u31_from_bytes",
    "unit_arg_of",
]
