"""Loop 4 動作表示（任務 12 A2）：每格任務 head ＋ 走路執行器 ＋ 市場桶。

依 內部規格文件 v4 §5 與任務 12 檔頭的裁定。

三層，彼此不混：

1. :func:`tile_task_mask` ``[B, 2, 100, 17]``——**以格為主體**的合法遮罩。判準沿
   `rl/obs.py::legality` 的語意（「不是保證 no-op」，不是「保證成功」），但主體從
   「這個單位站的那格」換成「這格」，因為 v4 的 policy 是對格出任務、不是對單位出動作。
2. :func:`assign_and_walk`——走路執行器。把被指派的格依**曼哈頓距離**貪婪分給還沒事做的
   人手；站上去的就做該格動作，到不了的就往該格移動一步。這是動作空間定義的一部分
   （Jake Will「角度由引擎算」、felix 語意動作），不是推論期補丁。
3. :func:`market_orders`——市場桶 → 整數 → 依優先序截斷到 10 筆。
   **整數用單位相位之後的 shed 算**（任務 11 量到教師 48.2% 的 SELL 數量超過回合開始時的庫存，
   因為 HARVEST→DROP 先補倉），所以本模組提供 :func:`loop4_step`，它重組
   `rl/env.py::step` 的相位順序、在 `apply_unit_actions` 之後才算市場整數。
   **不改 `rl/kernels.py` 與 `rl/env.py`**：相位函式逐字沿用。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rl import caps as caps_mod
from rl import kernels, rules
from rl.actions import SimAction
from rl.state import SimState, unit_alive

I32 = jnp.int32

# ------------------------------------------------------------- 每格任務類
T_NOOP = 0
T_PLANT0 = 1  # .. T_PLANT0 + 4
T_WATER = 6
T_HARVEST = 7
T_FERTILIZE = 8
T_FEED = 9
T_CARE = 10
T_COLLECT = 11
T_DIG = 12
T_BUILD_COOP = 13
T_BUILD_PASTURE = 14
T_PICKUP = 15
T_PLACE = 16
N_TASKS = 17

TASK_NAMES = (
    "NOOP",
    *(f"PLANT_{c}" for c in rules.CROPS),
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "FEED",
    "CARE",
    "COLLECT",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "PICKUP",
    "PLACE",
)
assert len(TASK_NAMES) == N_TASKS

#: 任務類 → 引擎 unit op。PLANT 的作物由任務類本身帶。
TASK_TO_OP = jnp.asarray(
    [
        rules.OP_PASS,
        *([rules.OP_PLANT] * rules.N_CROPS),
        rules.OP_WATER,
        rules.OP_HARVEST,
        rules.OP_FERTILIZE,
        rules.OP_FEED,
        rules.OP_CARE,
        rules.OP_COLLECT_FERTILIZER,
        rules.OP_DIG,
        rules.OP_BUILD_COOP,
        rules.OP_BUILD_PASTURE,
        rules.OP_PICKUP,
        rules.OP_PLACE,
    ],
    I32,
)

#: PICKUP／PLACE 的數量桶（掃描：教師七成不是 1）。
QTY_BUCKETS = (1, 2, 4, -1)  # -1 = 全部
N_QTY_BUCKETS = len(QTY_BUCKETS)

# ---------------------------------------------------------------- 市場桶
SELL_BUCKETS = ("all", "half", "quarter", 1, 2, 3, 4, 6, 8)
BUY_SEED_BUCKETS = (1, 2, 3, 4, 5, 10, "all_empty")
BUY_PRODUCT_BUCKETS = (1, 2, 3, 4, 5, 10, 15, "feed_all")
BUY_ANIMAL_BUCKETS = (1, 2, "fill")
N_SELL = len(SELL_BUCKETS) + 1  # +1 = 不賣
N_BUY_SEED = len(BUY_SEED_BUCKETS) + 1
N_BUY_PRODUCT = len(BUY_PRODUCT_BUCKETS) + 1
N_BUY_ANIMAL = len(BUY_ANIMAL_BUCKETS) + 1
N_HIRE = 11  # 0..10
N_BUY_LAND = 2
#: BUY_PRODUCT 只有 WHEAT 與 FERTILIZER 是引擎接受的品項。
BUYABLE_PRODUCTS = (rules.I_WHEAT, rules.I_FERTILIZER)

#: 發單順序＝截斷順序（任務 12 裁定第 2 點；引擎按 list 順序結算、錢盡即停）。
EMIT_ORDER = (
    rules.MO_SELL,
    rules.MO_BUY_LAND,
    rules.MO_HIRE,
    rules.MO_BUY_SEED,
    rules.MO_BUY_PRODUCT,
    rules.MO_BUY_ANIMAL,
)


@jax.tree_util.register_pytree_node_class
class Loop4Decision:
    """policy 的輸出（兩座一起）。註冊成 pytree，所以可以直接進 ``jax.jit``／``scan``。"""

    __slots__ = (
        "tile_task",
        "tile_qty",
        "sell",
        "buy_seed",
        "buy_product",
        "buy_animal",
        "hire",
        "buy_land",
    )

    def __init__(
        self, tile_task, tile_qty, sell, buy_seed, buy_product, buy_animal, hire, buy_land
    ):
        self.tile_task = tile_task  # [B, 2, 100] int32, 0..16
        self.tile_qty = tile_qty  # [B, 2, 100] int32, 0..3（QTY_BUCKETS 索引）
        self.sell = sell  # [B, 2, 9] int32, 0 = 不賣
        self.buy_seed = buy_seed  # [B, 2, 5] int32, 0 = 不買
        self.buy_product = buy_product  # [B, 2, 2] int32
        self.buy_animal = buy_animal  # [B, 2, 3] int32
        self.hire = hire  # [B, 2] int32, 0..10
        self.buy_land = buy_land  # [B, 2] int32, 0/1

    def tree_flatten(self):
        return tuple(getattr(self, f) for f in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


# ------------------------------------------------------------ 每格遮罩
def tile_task_mask(state: SimState, caps: caps_mod.Caps) -> jax.Array:
    """``[B, 2, 100, 17]``：這格上**有可能**做成的任務。

    **一個引擎 quirk 要照抄不要「修正」**：PICKUP 在引擎裡**在 LOCKED 守衛之前結算**
    （`rl/kernels.py` 的 "PICKUP (resolves before the LOCKED guard)"），PLACE-to-shed 的
    ``c_place_b`` 也沒有 ``not_locked`` 條件。所以這兩個任務在 LOCKED 格上是真的做得成的，
    遮罩照引擎放行。實際盤面上 shed 相鄰格永遠在 NW（一開始就解鎖），碰不到這條。

    與 `rl/obs.py::legality` 同一個口徑：「不是保證 no-op」。差別在主體是格不是單位，
    所以「需要攜帶物」的任務（FERTILIZE／FEED／PLACE 動物）只檢查**這一座身上或倉庫裡有沒有**，
    真正能不能做要看執行器派到的那個人手有沒有帶——那由 :func:`assign_and_walk` 負責，
    遮罩不冒充它。
    """
    tiles = state.tiles
    b = tiles.shape[0]
    kind = tiles[..., rules.C_KIND]
    crop = jnp.clip(tiles[..., rules.C_CROP], 0, rules.N_CROPS - 1)
    animal = tiles[..., rules.C_ANIMAL]
    has_animal = animal >= 0
    yld = tiles[..., rules.C_YIELD]
    day = (state.step // rules.TURNS_PER_DAY)[:, None, None]
    empty = kind == rules.TK_EMPTY
    is_plant = kind == rules.TK_PLANT
    mature = (day - tiles[..., rules.C_PLANTED_DAY]) >= kernels._CROP_FIRST_YIELD[crop]

    seeds = state.seeds[:, :, None, :]  # [B, 2, 1, 5]
    shed = state.shed[:, :, None, :]
    inv_total = jnp.sum(state.inv, axis=2)[:, :, None, :]  # [B, 2, 1, 12]
    held = shed + inv_total
    shed_total = jnp.sum(state.shed, axis=-1)[:, :, None]
    flat = jnp.arange(rules.N_TILES, dtype=I32)[None, None, :]
    shed_adj = jnp.any(flat[..., None] == kernels._SHED_ACCESS_FLAT, axis=-1)
    shed_adj = jnp.broadcast_to(shed_adj, kind.shape)

    struct_match = kind[..., None] == kernels._ANIMAL_STRUCT[None, None, None, :]
    animal_held = inv_total[..., rules.ANIMAL_ITEM_BASE :] > 0
    place_animal = jnp.any(struct_match & (~has_animal)[..., None] & animal_held, axis=-1)
    place_shed = shed_adj & (jnp.sum(inv_total, axis=-1) > 0) & (shed_total < rules.SHED_CAPACITY)

    m = jnp.zeros((b, 2, rules.N_TILES, N_TASKS), bool)
    m = m.at[..., T_NOOP].set(True)
    m = m.at[..., T_PLANT0 : T_PLANT0 + rules.N_CROPS].set(empty[..., None] & (seeds > 0))
    m = m.at[..., T_WATER].set(is_plant & (tiles[..., rules.C_WATERED] == 0))
    m = m.at[..., T_HARVEST].set((yld > 0) & ((is_plant & mature) | has_animal))
    m = m.at[..., T_FERTILIZE].set(is_plant & (held[..., rules.I_FERTILIZER] > 0))
    m = m.at[..., T_FEED].set(
        has_animal & (tiles[..., rules.C_FED] == 0) & (held[..., rules.I_WHEAT] > 0)
    )
    m = m.at[..., T_CARE].set(has_animal & (tiles[..., rules.C_CARED] == 0))
    m = m.at[..., T_COLLECT].set(has_animal & (tiles[..., rules.C_FERT_AVAIL] == 1))
    m = m.at[..., T_DIG].set((kind != rules.TK_EMPTY) & (kind != rules.TK_LOCKED) & (~has_animal))
    m = m.at[..., T_BUILD_COOP].set(empty)
    m = m.at[..., T_BUILD_PASTURE].set(empty)
    m = m.at[..., T_PICKUP].set(shed_adj & (jnp.sum(state.shed, axis=-1)[:, :, None] > 0))
    m = m.at[..., T_PLACE].set(place_animal | place_shed)
    del caps
    return m


# --------------------------------------------------------- 走路執行器
#: 換目標的距離代價：只有當另一個格比「上一回合正在做的那格」近超過這麼多步，才准換。
#: 0 等於沒有記憶。預設 4 是由 `scripts/loop4_walk_multistep.py` 的掉頭率量出來的。
SWITCH_COST = 4

#: ``prev_assignment`` 的「沒有指派」哨兵。
NO_ASSIGNMENT = -1


def assign_and_walk(
    state: SimState,
    tile_task: jax.Array,
    tile_qty: jax.Array,
    caps: caps_mod.Caps,
    prev_assignment: jax.Array | None = None,
    switch_cost: int = SWITCH_COST,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """把被指派的格分給人手，回傳 ``(unit_op, unit_arg, unit_qty, assignment)``。

    ``prev_assignment`` ``[B, 2, U]`` 是上一回合每個人手在做的格（``NO_ASSIGNMENT`` = 沒有）。
    給了它之後，**不是上一回合那一對**的配對距離會加上 ``switch_cost``，所以人手會黏住原本的
    目標，除非新目標近超過 ``switch_cost`` 步。這是為了修 policy 每回合重新指派造成的掉頭
    （實測無記憶時 35.3% 的連續移動是直接掉頭，內部量測紀錄）。
    回傳的 ``assignment`` 要餵給下一回合。

    規則（確定性，無隨機）：

    * **全域最小距離優先**的貪婪配對：每一輪在所有「還沒事做的人手 × 還沒被認領的任務格」裡
      取曼哈頓距離最小的一對，平手時先比人手槽號、再比格的 flat 索引。最多配 ``U`` 輪。
      **不是**按人手槽序各自挑最近的——那樣會讓遠處的人手先搶走別人正站著的格
      （實測教師語料：槽序版到達率只有 43.6%，因為站在格上的人手排在後面就被跳過了）。
    * 配到之後：站在該格上就出該格的任務動作；否則往該格走一步——**先對齊 x 再對齊 y**
      （固定順序才有確定性）。
    * 沒配到任務的人手出 PASS；任務格多於人手時，多出來的格這回合沒人去。
    """
    if prev_assignment is None:
        prev_assignment = jnp.full(state.unit_x.shape, NO_ASSIGNMENT, I32)
    u = caps.max_units
    b = state.step.shape[0]
    alive = unit_alive(state, caps)  # [B, 2, U]
    want = tile_task != T_NOOP  # [B, 2, 100]
    ys, xs = jnp.divmod(jnp.arange(rules.N_TILES, dtype=I32), rules.BOARD_SIZE)
    big = jnp.int32(1 << 29)
    units = jnp.arange(u, dtype=I32)
    flats = jnp.arange(rules.N_TILES, dtype=I32)
    # [B, 2, U, 100] 曼哈頓距離；U ≤ 29、100 格，形狀固定
    dist = jnp.abs(xs[None, None, None, :] - state.unit_x[..., None]) + jnp.abs(
        ys[None, None, None, :] - state.unit_y[..., None]
    )
    # 黏性：不是上一回合那一對的，距離加上 switch_cost
    stay = flats[None, None, None, :] == prev_assignment[..., None]
    eff = dist + jnp.where(stay, 0, switch_cost)
    # 排序鍵：有效距離 > 人手槽號 > 格索引，全部塞進一個 int32
    order_key = eff * (u * rules.N_TILES) + units[None, None, :, None] * rules.N_TILES + flats

    def body(_, carry):
        free_unit, free_tile, op, arg, qty, assign = carry
        ok = free_unit[..., None] & (free_tile & want)[:, :, None, :]
        key = jnp.where(ok, order_key, big).reshape(b, 2, u * rules.N_TILES)
        best = jnp.argmin(key, axis=-1).astype(I32)  # [B, 2]
        found = jnp.min(key, axis=-1) < big
        ui = best // rules.N_TILES
        ti = best % rules.N_TILES
        take_u = (units[None, None, :] == ui[..., None]) & found[..., None]
        take_t = (flats[None, None, :] == ti[..., None]) & found[..., None]
        free_unit = free_unit & ~take_u
        free_tile = free_tile & ~take_t

        ux = jnp.take_along_axis(state.unit_x, ui[..., None], axis=2)[..., 0]
        uy = jnp.take_along_axis(state.unit_y, ui[..., None], axis=2)[..., 0]
        tgt_x, tgt_y = ti % rules.BOARD_SIZE, ti // rules.BOARD_SIZE
        task = jnp.take_along_axis(tile_task, ti[..., None], axis=2)[..., 0]
        qbucket = jnp.take_along_axis(tile_qty, ti[..., None], axis=2)[..., 0]
        on_tile = (ux == tgt_x) & (uy == tgt_y)
        move_op = jnp.where(
            tgt_x > ux,
            rules.OP_EAST,
            jnp.where(
                tgt_x < ux,
                rules.OP_WEST,
                jnp.where(tgt_y > uy, rules.OP_SOUTH, rules.OP_NORTH),
            ),
        )
        do_op = jnp.where(on_tile, TASK_TO_OP[task], move_op)
        is_plant = (task >= T_PLANT0) & (task < T_PLANT0 + rules.N_CROPS)
        item, n = _carry_item(state, ui, task, qbucket)
        a = jnp.where(is_plant, task - T_PLANT0, jnp.where(task >= T_PICKUP, item, -1))
        a = jnp.where(on_tile, a, -1)
        n = jnp.where(on_tile & (task >= T_PICKUP), n, 1)

        sel = take_u  # [B, 2, U]
        op = jnp.where(sel, do_op[..., None], op)
        arg = jnp.where(sel, a[..., None], arg)
        qty = jnp.where(sel, n[..., None], qty)
        assign = jnp.where(sel, ti[..., None], assign)
        return free_unit, free_tile, op, arg, qty, assign

    init = (
        alive,
        jnp.ones((b, 2, rules.N_TILES), bool),
        jnp.zeros((b, 2, u), I32),
        jnp.full((b, 2, u), -1, I32),
        jnp.ones((b, 2, u), I32),
        jnp.full((b, 2, u), NO_ASSIGNMENT, I32),
    )
    _, _, op, arg, qty, assign = jax.lax.fori_loop(0, u, body, init)
    return op, arg, qty, assign


def _carry_item(
    state: SimState, ui: jax.Array, task: jax.Array, qbucket: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """PICKUP／PLACE 要帶的品項與數量。

    PICKUP：倉庫裡數量最多的品項（確定性，平手取索引小的）。
    PLACE：這個單位身上 **dict 插入序最早**的品項——與引擎 DROP 丟棄順序同一個排序鍵，
    所以放不下時被丟掉的東西可預測。
    數量桶 ``QTY_BUCKETS``：1／2／4／全部；``全部`` 以對應來源的持有量算。
    """
    n_items = state.inv.shape[-1]
    idx = jnp.broadcast_to(ui[..., None, None], ui.shape + (1, n_items))
    inv_k = jnp.take_along_axis(state.inv, idx, axis=2)[:, :, 0]  # [B, 2, 12]
    seq_k = jnp.take_along_axis(state.inv_seq, idx, axis=2)[:, :, 0]
    pick_item = jnp.argmax(state.shed, axis=-1).astype(I32)
    place_item = jnp.argmin(jnp.where(inv_k > 0, seq_k, 1 << 29), axis=-1).astype(I32)
    is_pickup = task == T_PICKUP
    item = jnp.where(is_pickup, pick_item, place_item)
    have = jnp.where(
        is_pickup,
        jnp.take_along_axis(state.shed, item[..., None], axis=-1)[..., 0],
        jnp.take_along_axis(inv_k, item[..., None], axis=-1)[..., 0],
    )
    table = jnp.asarray([1, 2, 4, -1], I32)
    raw = table[jnp.clip(qbucket, 0, N_QTY_BUCKETS - 1)]
    n = jnp.where(raw < 0, have, jnp.minimum(raw, jnp.maximum(have, 1)))
    return item, jnp.maximum(n, 1)


# --------------------------------------------------------------- 市場桶
def _sell_quantity(bucket: jax.Array, held: jax.Array) -> jax.Array:
    """``[.., 9]`` 桶索引（1..9）→ 整數。0 代表不賣。``held`` 必須是**單位相位之後**的 shed。"""
    table_abs = jnp.asarray([0, 0, 0, 0, 1, 2, 3, 4, 6, 8], I32)  # 索引 0..9
    rel = jnp.where(
        bucket == 1, held, jnp.where(bucket == 2, held // 2, jnp.where(bucket == 3, held // 4, 0))
    )
    n = jnp.where(bucket <= 3, rel, table_abs[jnp.clip(bucket, 0, 9)])
    return jnp.where(bucket == 0, 0, jnp.minimum(jnp.maximum(n, 0), held))


def _buy_seed_quantity(bucket: jax.Array, n_empty: jax.Array) -> jax.Array:
    table = jnp.asarray([0, 1, 2, 3, 4, 5, 10, 0], I32)
    n = jnp.where(bucket == 7, n_empty, table[jnp.clip(bucket, 0, 7)])
    return jnp.where(bucket == 0, 0, jnp.maximum(n, 0))


def _buy_product_quantity(bucket: jax.Array, feed_all: jax.Array) -> jax.Array:
    table = jnp.asarray([0, 1, 2, 3, 4, 5, 10, 15, 0], I32)
    n = jnp.where(bucket == 8, feed_all, table[jnp.clip(bucket, 0, 8)])
    return jnp.where(bucket == 0, 0, jnp.maximum(n, 0))


def _buy_animal_quantity(bucket: jax.Array, fill: jax.Array) -> jax.Array:
    table = jnp.asarray([0, 1, 2, 0], I32)
    n = jnp.where(bucket == 3, fill, table[jnp.clip(bucket, 0, 3)])
    return jnp.where(bucket == 0, 0, jnp.maximum(n, 0))


def market_counts(state: SimState) -> dict[str, jax.Array]:
    """桶要用到的三個狀態計數（空地、可餵動物、空結構），全部 ``[B, 2]`` 或 ``[B, 2, 3]``。"""
    kind = state.tiles[..., rules.C_KIND]
    animal = state.tiles[..., rules.C_ANIMAL]
    n_empty = jnp.sum(kind == rules.TK_EMPTY, axis=2)
    feed_all = jnp.sum((animal >= 0) & (state.tiles[..., rules.C_FED] == 0), axis=2)
    free_coop = jnp.sum((kind == rules.TK_COOP) & (animal < 0), axis=2)
    free_past = jnp.sum((kind == rules.TK_PASTURE) & (animal < 0), axis=2)
    fill = jnp.stack(
        [jnp.where(k == rules.TK_COOP, free_coop, free_past) for k in rules.ANIMAL_STRUCTURE_KIND],
        axis=-1,
    )
    return {"n_empty": n_empty, "feed_all": feed_all, "fill": fill}


def market_orders(
    state: SimState, decision: Loop4Decision, caps: caps_mod.Caps
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """桶 → ``(order_op, order_item, order_qty)``，形狀 ``[B, 2, 10]``，已按優先序截斷。

    ``state`` 必須是**單位相位之後**的狀態，SELL 的「全部／一半／1/4」才對得上引擎結算時
    真正在倉庫裡的量（任務 11：教師 48.2% 的 SELL 數量超過回合開始時的庫存）。

    優先序 ``SELL > BUY_LAND > HIRE > BUY_SEED > BUY_PRODUCT > BUY_ANIMAL``：引擎按 list 順序
    結算、錢用完就停（`rl/kernels.py` 檔頭），所以賺錢的排在花錢的前面；任務 11 量到教師在
    同時賣與買的回合有 94.8% 把 SELL 排最前。
    """
    b = state.step.shape[0]
    cnt = market_counts(state)
    rows_op, rows_item, rows_qty = [], [], []

    def push(op, item, qty):
        rows_op.append(jnp.where(qty > 0, op, rules.MO_NONE).astype(I32))
        rows_item.append(jnp.where(qty > 0, item, -1).astype(I32))
        rows_qty.append(jnp.maximum(qty, 0).astype(I32))

    # SELL（每產品一筆，按產品索引）
    for p in range(rules.N_PRODUCTS):
        n = _sell_quantity(decision.sell[:, :, p], state.shed[:, :, p])
        push(rules.MO_SELL, jnp.full((b, 2), p, I32), n)
    # BUY_LAND（二元，每回合最多一筆）
    push(
        rules.MO_BUY_LAND,
        jnp.full((b, 2), -1, I32),
        jnp.where(decision.buy_land > 0, 1, 0),
    )
    # HIRE：0..10 計數，展開成同樣多筆訂單
    for i in range(N_HIRE - 1):
        push(rules.MO_HIRE, jnp.full((b, 2), -1, I32), (decision.hire > i).astype(I32))
    # BUY_SEED
    for c in range(rules.N_CROPS):
        n = _buy_seed_quantity(decision.buy_seed[:, :, c], cnt["n_empty"])
        push(rules.MO_BUY_SEED, jnp.full((b, 2), c, I32), n)
    # BUY_PRODUCT（只有 WHEAT／FERTILIZER）
    for j, item in enumerate(BUYABLE_PRODUCTS):
        n = _buy_product_quantity(decision.buy_product[:, :, j], cnt["feed_all"])
        push(rules.MO_BUY_PRODUCT, jnp.full((b, 2), item, I32), n)
    # BUY_ANIMAL
    for a in range(rules.N_ANIMALS):
        n = _buy_animal_quantity(decision.buy_animal[:, :, a], cnt["fill"][:, :, a])
        push(rules.MO_BUY_ANIMAL, jnp.full((b, 2), rules.ANIMAL_ITEM_BASE + a, I32), n)

    op = jnp.stack(rows_op, axis=-1)  # [B, 2, N_CAND]
    item = jnp.stack(rows_item, axis=-1)
    qty = jnp.stack(rows_qty, axis=-1)
    return _truncate(op, item, qty, caps.max_orders)


def _truncate(
    op: jax.Array, item: jax.Array, qty: jax.Array, max_orders: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """把候選訂單壓成前 ``max_orders`` 筆，**保持候選順序**（候選已按優先序排好）。

    HIRE 是唯一沒有數量的訂單，所以它以「展開成 n 筆」的形式進來；截斷對它的語意就是
    「錢／槽不夠時少雇幾個」，與引擎一致（引擎也是逐筆結算、付不起就跳過）。
    """
    live = op != rules.MO_NONE
    rank = jnp.cumsum(live.astype(I32), axis=-1) - 1  # 第幾筆有效訂單
    keep = live & (rank < max_orders)
    n_cand = op.shape[-1]
    slot = jnp.where(keep, rank, n_cand)  # 不留的推到界外
    b, s = op.shape[0], op.shape[1]
    out_op = jnp.full((b, s, max_orders + 1), rules.MO_NONE, I32)
    out_item = jnp.full((b, s, max_orders + 1), -1, I32)
    out_qty = jnp.zeros((b, s, max_orders + 1), I32)
    bi = jnp.arange(b)[:, None, None]
    si = jnp.arange(s)[None, :, None]
    slot = jnp.clip(slot, 0, max_orders)
    out_op = out_op.at[bi, si, slot].set(op, mode="drop")
    out_item = out_item.at[bi, si, slot].set(item, mode="drop")
    out_qty = out_qty.at[bi, si, slot].set(qty, mode="drop")
    return out_op[..., :max_orders], out_item[..., :max_orders], out_qty[..., :max_orders]


# ------------------------------------------------------- 相位重組的 step
def loop4_step(
    state: SimState,
    decision: Loop4Decision,
    caps: caps_mod.Caps,
    tables: kernels.RngTables | None = None,
    prev_assignment: jax.Array | None = None,
) -> tuple[SimState, SimAction, jax.Array]:
    """一個 env-turn，相位順序與 `rl/env.py::step` 逐字相同，只是市場整數在中間算。

    回傳 ``(下一個 state, 實際送進引擎的 SimAction, 這回合的指派)``。
    第三項餵給下一回合的 ``prev_assignment``；**日末會被清掉**，因為 ``end_of_day`` 把人手
    收回 spawn、``hires_today`` 歸零，上一天的配對對新的一天沒有意義。
    """
    unit_op, unit_arg, unit_qty, assignment = assign_and_walk(
        state, decision.tile_task, decision.tile_qty, caps, prev_assignment
    )
    placeholder = SimAction(
        unit_op=unit_op,
        unit_arg=unit_arg,
        unit_qty=unit_qty,
        order_op=jnp.zeros((state.step.shape[0], 2, caps.max_orders), I32),
        order_item=jnp.full((state.step.shape[0], 2, caps.max_orders), -1, I32),
        order_qty=jnp.zeros((state.step.shape[0], 2, caps.max_orders), I32),
    )
    s1 = kernels.apply_unit_actions(state, placeholder, caps)
    order_op, order_item, order_qty = market_orders(s1, decision, caps)
    action = placeholder._replace(order_op=order_op, order_item=order_item, order_qty=order_qty)
    s2 = kernels.process_market(s1, action, caps)
    s3 = kernels.town_consume(s2)
    s4 = kernels.decay_plants(s3)
    eod = ((s4.step + 1) % rules.TURNS_PER_DAY) == 0
    s5 = jax.lax.cond(
        jnp.any(eod),
        lambda s: kernels.end_of_day(s, caps, tables, eod),
        lambda s: s,
        s4,
    )
    assignment = jnp.where(eod[:, None, None], NO_ASSIGNMENT, assignment)
    return s5._replace(step=s5.step + 1), action, assignment
