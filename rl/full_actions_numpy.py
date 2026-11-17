"""完整動作表示的 **NumPy 部署路徑**（不 import jax、不 import 引擎）。

規格：`docs/full-actions-restoration-2026-09-21.md` §第一段；Fable §14／§14.1 落點。

與 JAX 側（`rl/full_actions_obs.py`／`_head.py`／`_decode.py`）逐項對應：
同觀測（35/30/10/32）、同 `unit_slot_emb`、同 trunk、**同一條遮罩規則**、
同條件 decoder（qty 逐 byte、item 吃已選 verb／op）、同兩趟（真 item／真 qty 投影）。

投影用 `rl/loop4_numpy.py::apply_unit_actions`（**不用 definitions_v2 那條死棧**）。
`sample` 與 `greedy` 分開，推論期**不做 top-k、不改數量**。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from rl import caps as caps_mod
from rl import loop4_numpy as NP
from rl import ordered_v3_contract as C
from rl import rules

F32 = np.float32
I32 = np.int32
NEG = np.float32(-1e30)

N_UNIT_OPS = len(C.UNIT_OP_NAMES)
N_MARKET_OPS = len(C.MARKET_OP_NAMES)
N_ITEMS = rules.N_ITEMS
SEED_SCALE = float(np.log1p(np.float32(2**31 - 1)))
SHED_CAP = 100.0

#: 三個模式共用同一顆 `decode`：sample／greedy 產生動作，teacher 給定動作評分。
MODES = ("sample", "greedy", "teacher")
UNIT_SLOTS = 3 + C.N_QTY_BYTES  # verb, crop, item, b0..b3
MARKET_SLOTS = 2 + C.N_QTY_BYTES  # op, item, b0..b3
#: PPO 的固定 token 位置數 ＝ 29×7 ＋ 10×6 ＝ 263。
N_LOGP_SLOTS = C.MAX_UNITS * UNIT_SLOTS + C.MAX_ORDERS * MARKET_SLOTS

#: 觀測維度。來源定義在 `rl/full_actions_obs.py`（它 import jax，部署路徑不能載），
#: 這裡照抄並由 `tests/test_full_actions.py::test_obs_dims_single_source` 逐項對帳防漂移。
#: `rl/loop4_model.py::N_SEGMENTS`（該模組 import jax）；由 dims 對帳測試防漂移。
N_SEGMENTS = 5
TILE_DIM = 35
#: T3 F5：6 基本 ＋ 12 log 量 ＋ 12 線性量 ＋ 12 規範化順位（與 `full_actions_obs` 同 layout）
UNIT_BASE_DIM = 6
UNIT_DIM = UNIT_BASE_DIM + 3 * rules.N_ITEMS  # 42
QTY_LOG_SCALE = float(np.log1p(1000.0))
QTY_LIN_SCALE = 100.0
#: T3 A1：pointer candidate 的可見數量維度（shed/100、作物種子 log 尺度）
POINTER_AUX = 2
PRODUCT_DIM = 10
GLOBAL_DIM = 24 + rules.N_CROPS + rules.N_ANIMALS  # 32

#: 權重／checkpoint 的 schema。**定義在這裡**（這支是 jax-free 的部署路徑），
#: `rl/full_actions_head.py` 直接 import 這一份，避免兩邊各寫一個字串後漂移。
SCHEMA = "full-actions-v1"
OBS_SCHEMA = "full-actions-obs-v2"
#: T3 第二批的動作 codec 標記（§5.1.6 fallback：原正 U31 四 byte、263 policy 槽）。
#: 短碼 `u31-short255-escape-v1` 若之後定版會是不同標記；兩者**不可互載**（loader 嚴格比對）。
ACTION_CODEC = "u31-fourbyte-t3-v1"
#: T3 §5.1.3 的 IL loss 遮罩契約（A5／M2／空 PICKUP；只影響 IL，不影響 policy_active 以外）
IL_MASK_CONTRACT = "t3-a5-m2-pickup-v1"

_ITEM_SUPPORT = np.asarray(
    [
        [it in C.MARKET_ITEM_SUPPORT.get(op, frozenset()) for it in range(N_ITEMS)]
        for op in range(N_MARKET_OPS)
    ]
)
_NEEDS_QTY = np.asarray(
    [op not in (C.MO_NONE, C.MO_HIRE, C.MO_BUY_LAND) for op in range(N_MARKET_OPS)]
)


# ------------------------------------------------------------------ 官方 obs → 狀態
#: 觀測執行契約；與 `rl/full_actions_obs.py::EXEC_OBS_CONTRACT` 同值（dims 對帳測試防漂移）
EXEC_OBS_CONTRACT = "t3-batch1"
#: `loop4_obs.global_features` 的 market_trips／market_units（T3 F3 恆 0）
MARKET_COUNTER_IDX = (22, 23)


class ObsSchemaError(ValueError):
    """full-actions 部署 adapter 讀到不合官方 schema 的 observation（不默默 fallback）。"""


def _strict_nonneg_int(v: Any, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise ObsSchemaError(f"{where} 必須是非負整數，得到 {v!r}")
    return int(v)


def state_from_obs(obs: dict, seat: int, caps: caps_mod.Caps) -> NP.NpState:
    """full-actions 專用的官方 obs adapter（T3 F1／F2）。

    底層沿用 `loop4_numpy.state_from_obs`（Loop 4 的共用 helper **不改**，其歷史行為可重現），
    再用官方欄位名覆寫它讀錯的兩處：

    * **F1**：商店在 ``town.unlocked_shops``（官方 `kaggriculture.py.txt:189,739,890`；
      IL schema `rl/ordered_v3_runtime.py:427-436`），舊 helper 讀不存在的 ``town.shops``，
      永遠得到 0 間。保留名稱與重複次數；缺欄、非 list、超過上限或未知商店名一律報錯。
    * **F2**：動物的累積照料在 ``pending_care_bonus``（官方 `:240,826-830`；IL schema
      `ordered_v3_runtime.py:291`），舊 helper 讀 ``care_bonus``，永遠得到 0。
      有動物的格缺此欄就報錯；`cared_today` 是另一個欄位，不混用。
    """
    town = obs.get("town")
    if not isinstance(town, dict) or "unlocked_shops" not in town:
        raise ObsSchemaError(f"obs.town 缺 unlocked_shops：{town!r}")
    shops = town["unlocked_shops"]
    if not isinstance(shops, list) or len(shops) > rules.MAX_SHOP_INSTANCES:
        raise ObsSchemaError(f"town.unlocked_shops 不合法：{shops!r}")
    shop_seq = np.full(rules.MAX_SHOP_INSTANCES, -1, I32)
    for i, name in enumerate(shops):
        if name not in rules.SHOP_NAMES:
            raise ObsSchemaError(f"town.unlocked_shops[{i}]={name!r} 未知")
        shop_seq[i] = rules.SHOP_NAMES.index(name)

    state = NP.state_from_obs(obs, seat, caps)
    state.shop_seq = shop_seq
    state.n_shops = len(shops)
    # T3 I3：不逐格掃 200 格。底層 helper 已把「有 `animal` 鍵」的格編成 C_ANIMAL ≥ 0
    # （`loop4_numpy.encode_tile`），只回頭讀這些格的 `pending_care_bonus`。
    tiles = state.tiles
    ps, flats = np.nonzero(tiles[:, :, rules.C_ANIMAL] >= 0)
    if ps.size:
        tiles = tiles.copy()
        srcs = (seat, 1 - seat)
        for p, flat in zip(ps.tolist(), flats.tolist(), strict=True):
            src = srcs[p]
            y, x = divmod(flat, rules.BOARD_SIZE)
            tile = obs["farms"][src]["tiles"][y][x]
            if "pending_care_bonus" not in tile:
                raise ObsSchemaError(f"farms[{src}].tiles[{y}][{x}] 有動物但缺 pending_care_bonus")
            tiles[p, flat, rules.C_CARE_BONUS] = _strict_nonneg_int(
                tile["pending_care_bonus"], f"farms[{src}].tiles[{y}][{x}].pending_care_bonus"
            )
        state.tiles = tiles
    return state


def inv_rank_canonical(inv: np.ndarray, seq: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """T3 F4：``rank_i = 1 + #{present_j, seq_j < seq_i}``，``feature = rank / (N_present + 1)``。

    與 `rl/full_actions_obs.py::inv_rank_features` 同式。`inv`／`seq` ``[2, U, 12]``、
    `alive` ``[2, U]``。存在品項 seq 非正或重複 → :class:`ObsSchemaError`（不以 item ID 排序補）。
    """
    present = (inv > 0) & alive[..., None]
    seq = seq.astype(np.int64)
    if np.any(present & (seq <= 0)):
        raise ObsSchemaError("存在品項的 inv_seq 非正")
    same = present[..., None, :] & present[..., :, None] & (seq[..., None, :] == seq[..., :, None])
    if np.any(same.sum(-1) > 1):
        raise ObsSchemaError("存在品項的 inv_seq 重複")
    earlier = present[..., None, :] & (seq[..., None, :] < seq[..., :, None])
    rank = 1 + earlier.sum(-1)
    n_present = present.sum(-1, keepdims=True)
    return np.where(present, rank.astype(F32) / (n_present + 1).astype(F32), 0.0).astype(F32)


# ------------------------------------------------------------------ 觀測
def observation(state: NP.NpState, caps: caps_mod.Caps) -> dict[str, np.ndarray]:
    """Loop 4 的 NumPy 觀測 ＋ 三處補齊（與 `rl/full_actions_obs.py` 同定義）。"""
    base = NP.observation(state, caps, in_progress=None)  # in_progress 恆 0
    seeds = state.seeds[0, : rules.N_CROPS].astype(F32)
    seed_f = np.clip(np.log1p(np.maximum(seeds, 0.0)) / SEED_SCALE, 0.0, 1.0)
    animals = state.shed[0, rules.ANIMAL_ITEM_BASE :].astype(F32)
    animal_f = np.clip(animals / SHED_CAP, 0.0, 1.0)
    base_glob = base["global"].copy()
    base_glob[:, list(MARKET_COUNTER_IDX)] = 0.0  # T3 F3（loop4_numpy 本來就填 0，這裡明示）
    glob = np.concatenate([base_glob, seed_f[None], animal_f[None]], axis=-1)

    alive = NP.unit_alive(state, caps)
    rank = inv_rank_canonical(state.inv, state.inv_seq, alive)  # [2, U, 12]
    n = np.maximum(state.inv.astype(F32), 0.0)
    log_q = np.where(alive[..., None], np.log1p(n) / F32(QTY_LOG_SCALE), 0.0).astype(F32)
    lin_q = np.where(alive[..., None], np.clip(n / F32(QTY_LIN_SCALE), 0.0, 1.0), 0.0).astype(F32)
    priv = np.concatenate([log_q, lin_q, rank.astype(F32)], axis=-1)[None].copy()  # [1, 2, U, 36]
    priv[:, 1] = 0.0  # 對手持有量與順位是 private
    units = np.concatenate([base["units"][..., :UNIT_BASE_DIM], priv], axis=-1).astype(F32)
    return {"tiles": base["tiles"], "units": units, "products": base["products"], "global": glob}


# ------------------------------------------------------------------ trunk
def build_tokens(params, obs, cfg):
    d = cfg["d_model"]
    tiles = obs["tiles"]
    b = tiles.shape[0]
    tile_tok = tiles @ params["in_tile_w"] + params["in_tile_b"]
    tile_tok = tile_tok + params["tile_pos_emb"][None, None]
    tile_tok = tile_tok + np.stack([params["seg_emb"][0], params["seg_emb"][1]])[None, :, None, :]
    tile_tok = tile_tok.reshape(b, 2 * rules.N_TILES, d)
    units = obs["units"][:, 0]
    unit_tok = units @ params["in_unit_w"] + params["in_unit_b"]
    unit_tok = unit_tok + params["seg_emb"][2] + params["unit_slot_emb"][None]
    prod_tok = obs["products"] @ params["in_product_w"] + params["in_product_b"]
    prod_tok = prod_tok + params["seg_emb"][3] + params["item_id_emb"][: rules.N_PRODUCTS]  # F6
    glob_tok = (obs["global"] @ params["in_global_w"] + params["in_global_b"])[:, None, :]
    glob_tok = glob_tok + params["seg_emb"][4]
    x = np.concatenate([tile_tok, unit_tok, prod_tok, glob_tok], axis=1).astype(F32)
    alive = obs["units"][:, 0, :, 2] > 0.5
    mask = np.concatenate(
        [
            np.ones((b, 2 * rules.N_TILES), bool),
            alive,
            np.ones((b, rules.N_PRODUCTS + 1), bool),
        ],
        axis=1,
    )
    return x, mask


def encode_tokens(params, obs, cfg) -> dict[str, np.ndarray]:
    x, mask = build_tokens(params, obs, cfg)
    for i in range(cfg["n_layers"]):
        x = NP._block(x, params, i, cfg["n_heads"], mask)
    x = NP._ln(x, params["ln_f_g"], params["ln_f_b"])
    u0 = 2 * rules.N_TILES
    p0 = u0 + cfg["max_units"]
    return {"unit": x[:, u0:p0], "prod": x[:, p0 : p0 + rules.N_PRODUCTS], "glob": x[:, -1]}


def pointer_aux(state: NP.NpState) -> np.ndarray:
    """``[12, 2]``：與 `full_actions_head.pointer_aux` 同義（自家＝索引 0、投影後 state）。"""
    shed = state.shed[0].astype(F32) / F32(rules.SHED_CAPACITY)
    seeds = state.seeds[0, : rules.N_CROPS].astype(F32)
    seed_f = np.clip(np.log1p(np.maximum(seeds, 0.0)) / SEED_SCALE, 0.0, 1.0).astype(F32)
    seed12 = np.concatenate([seed_f, np.zeros((N_ITEMS - rules.N_CROPS,), F32)])
    return np.stack([shed, seed12], axis=-1).astype(F32)


def pointer_keys(params, h_prod: np.ndarray, h_glob: np.ndarray, aux: np.ndarray) -> np.ndarray:
    """``[12, d]``：商品用第二趟 product hidden，動物用 global hidden；加 ID 與 aux。"""
    n_anim = N_ITEMS - rules.N_PRODUCTS
    cand = np.concatenate([h_prod, np.broadcast_to(h_glob[None], (n_anim, h_glob.shape[-1]))], 0)
    x = np.concatenate([cand, params["item_id_emb"], aux], axis=-1)
    return np.tanh(x @ params["pointer_key_w"] + params["pointer_key_b"]).astype(F32)


# ------------------------------------------------------------------ 合法遮罩（單位相）
def unit_legality(state: NP.NpState, caps: caps_mod.Caps, seat: int = 0) -> dict[str, np.ndarray]:
    """`rl/obs.py::legality` 的單座 NumPy 等價（只做單位相；市場只有語法域）。

    部署狀態把自己放索引 0（`loop4_numpy.state_from_obs(canonical=True)`），所以 `seat`
    預設 0；仍然串傳，**不寫死**——與 `_unit_step` 必須吃呼叫端 caps 是同一個理由。
    """
    u = caps.max_units
    alive = NP.unit_alive(state, caps)[seat]  # [U]
    x, y = state.unit_x[seat], state.unit_y[seat]
    flat = y * rules.BOARD_SIZE + x
    tiles = state.tiles[seat][flat]  # [U, C]
    kind = tiles[:, rules.C_KIND]
    crop = np.clip(tiles[:, rules.C_CROP], 0, rules.N_CROPS - 1)
    animal = tiles[:, rules.C_ANIMAL]
    has_animal = animal >= 0
    day = state.step // rules.TURNS_PER_DAY
    yield_u = tiles[:, rules.C_YIELD]
    inv = state.inv[seat]  # [U, 12]
    shed = state.shed[seat]  # [12]
    shed_room = rules.SHED_CAPACITY - int(shed.sum())
    shed_adj = np.any(flat[:, None] == NP._SHED_ACCESS_FLAT, axis=-1)
    empty = kind == rules.TK_EMPTY
    is_plant = kind == rules.TK_PLANT
    mature = (day - tiles[:, rules.C_PLANTED_DAY]) >= NP._CROP_FIRST_YIELD[crop]
    seeds = state.seeds[seat]

    plant_crop = empty[:, None] & (seeds[None, :] > 0) & alive[:, None]
    pickup_item = shed_adj[:, None] & (shed[None, :] > 0) & alive[:, None]
    struct_match = kind[:, None] == NP._ANIMAL_STRUCT[None, :]
    place_animal = (
        struct_match
        & (~has_animal)[:, None]
        & (inv[:, rules.ANIMAL_ITEM_BASE :] > 0)
        & alive[:, None]
    )
    place_shed = shed_adj[:, None] & (inv > 0) & (shed_room > 0) & alive[:, None]
    place_item = place_shed.copy()
    place_item[:, rules.ANIMAL_ITEM_BASE :] |= place_animal

    ops = np.zeros((u, N_UNIT_OPS), bool)
    ops[:, rules.OP_PASS] = alive
    move = np.stack([y > 0, y < rules.BOARD_SIZE - 1, x < rules.BOARD_SIZE - 1, x > 0], axis=-1)
    ops[:, rules.OP_NORTH : rules.OP_WEST + 1] = move & alive[:, None]
    ops[:, rules.OP_PICKUP] = np.any(pickup_item, axis=-1)
    ops[:, rules.OP_PLANT] = np.any(plant_crop, axis=-1)
    ops[:, rules.OP_WATER] = alive & is_plant & (tiles[:, rules.C_WATERED] == 0)
    ops[:, rules.OP_HARVEST] = alive & (yield_u > 0) & ((is_plant & mature) | has_animal)
    ops[:, rules.OP_FERTILIZE] = alive & is_plant & (inv[:, rules.I_FERTILIZER] > 0)
    ops[:, rules.OP_BUILD_COOP] = alive & empty
    ops[:, rules.OP_BUILD_PASTURE] = alive & empty
    ops[:, rules.OP_DIG] = (
        alive & (kind != rules.TK_EMPTY) & (kind != rules.TK_LOCKED) & (~has_animal)
    )
    ops[:, rules.OP_PLACE] = np.any(place_animal, axis=-1) | np.any(place_shed, axis=-1)
    ops[:, rules.OP_FEED] = (
        alive & has_animal & (tiles[:, rules.C_FED] == 0) & (inv[:, rules.I_WHEAT] > 0)
    )
    ops[:, rules.OP_COLLECT_FERTILIZER] = alive & has_animal & (tiles[:, rules.C_FERT_AVAIL] == 1)
    ops[:, rules.OP_CARE] = alive & has_animal & (tiles[:, rules.C_CARED] == 0)
    ops[:, rules.OP_DROP] = alive & shed_adj & (np.sum(inv, axis=-1) > 0)
    return {
        "unit_ops": ops,
        "plant_crop": plant_crop,
        "pickup_item": pickup_item,
        "place_item": place_item,
    }


# ------------------------------------------------------------------ 逐 token 取樣／評分
def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def _pick(logits, mask, rng, mode: str, label=None) -> tuple[int, float, bool]:
    """一個 categorical 位置 → `(token, logp, eligible)`。與 `full_actions_decode._pick` 同義。

    teacher 模式照樣吃 **raw label**（不改標、不換 PASS）；落在共用遮罩外時
    `eligible=False`、logp 記 0，該 token 與依賴 children 不進 loss 分母（Fable §14.1）。
    """
    lg = np.where(mask, logits, NEG).astype(F32)
    lp = _log_softmax(lg)
    if mode == "teacher":
        tok = int(label)
        ok = bool(mask[tok])
    elif mode == "greedy":
        tok = int(np.argmax(lg))
        ok = True
    else:
        p = np.exp(lp)
        p = p / p.sum()
        tok = int(rng.choice(p.shape[0], p=p))
        ok = True
    return tok, (float(lp[tok]) if ok else 0.0), ok


def _u31_byte_mask(i: int, prefix_zero: bool) -> np.ndarray:
    """第 `i` 個 byte 的合法類別。**只有 U31 的域**（`b0≤127`、四 byte 不得全零），與持有量無關。"""
    cls = np.arange(C.BYTE_CLASSES)
    if i == 0:
        return cls <= C.QTY_HIGH_BYTE_MAX
    if i == C.N_QTY_BYTES - 1 and prefix_zero:
        return cls > 0
    return np.ones(C.BYTE_CLASSES, bool)


def _qty_scan(params, h0, rng, mode: str, labels=None):
    """逐 byte 的**條件** scan：`h` 每步吃上一個 byte（與 JAX `qty_scan` 同結構）。"""
    h = h0
    toks = np.zeros((C.N_QTY_BYTES,), I32)
    lps = np.zeros((C.N_QTY_BYTES,), np.float64)
    oks = np.ones((C.N_QTY_BYTES,), bool)
    prefix_zero = True
    for i in range(C.N_QTY_BYTES):
        lg = h @ params["byte_out_w"] + params["byte_out_b"]
        lab = None if labels is None else labels[i]
        tok, lp, ok = _pick(lg, _u31_byte_mask(i, prefix_zero), rng, mode, lab)
        toks[i], lps[i], oks[i] = tok, lp, ok
        prefix_zero = prefix_zero and tok == 0
        h = np.tanh(
            np.concatenate([h, params["byte_embed"][tok]]) @ params["prefix_w"] + params["prefix_b"]
        )
    return toks, lps, oks


def _onehot(i: int, n: int) -> np.ndarray:
    v = np.zeros((n,), F32)
    v[i] = 1.0
    return v


def _unit_arg(verb, crop, item) -> np.ndarray:
    return np.where(
        verb == C.OP_PLANT,
        crop,
        np.where((verb == C.OP_PICKUP) | (verb == C.OP_PLACE), item, np.zeros_like(verb)),
    ).astype(I32)


def _u31(qty_bytes: np.ndarray) -> np.ndarray:
    w = np.asarray([1 << 24, 1 << 16, 1 << 8, 1], I32)
    return (qty_bytes.astype(I32) * w).sum(-1).astype(I32)


def prefix_state(state: NP.NpState, fields: dict, caps: caps_mod.Caps, upto: int, seat: int = 0):
    """前 `upto` 個單位做完之後的 state——**從原 state 重建**，不鏈式投影。

    與 `rl/full_actions_decode.py::prefix_state` 同定義（那邊用 `kernels.apply_unit_actions`，
    這邊用 `rl/loop4_numpy.py::apply_unit_actions`；引擎 PLANT atomic 預檢兩邊都在裡面）。
    """
    live = np.arange(caps.max_units) < upto
    verb = np.where(live, fields["unit_verb"], C.OP_PASS).astype(I32)
    crop = np.where(live, fields["unit_crop"], 0).astype(I32)
    item = np.where(live, fields["unit_item"], 0).astype(I32)
    qty = np.where(live, _u31(fields["unit_qty_bytes"]), 0).astype(I32)
    return NP.apply_unit_actions(state, verb, _unit_arg(verb, crop, item), qty, caps, seat=seat)


#: A5 成功放動物時 policy 固定填的 qty（U31 的 1），與 JAX `A5_QTY_BYTES` 同值
A5_QTY_BYTES = np.asarray([0, 0, 0, 1], I32)


def a5_place_animal_success(state: NP.NpState, k: int, verb: int, item: int, seat: int = 0) -> bool:
    """T3 A5：真 prefix 下 PLACE 走「放上對應空結構」成功分支（與 JAX 同條件）。"""
    if verb != C.OP_PLACE or not (rules.ANIMAL_ITEM_BASE <= item < rules.N_ITEMS):
        return False
    flat = int(state.unit_y[seat, k]) * rules.BOARD_SIZE + int(state.unit_x[seat, k])
    tile = state.tiles[seat, flat]
    struct = rules.ANIMAL_STRUCTURE_KIND[item - rules.ANIMAL_ITEM_BASE]
    return bool(
        tile[rules.C_KIND] == struct and tile[rules.C_ANIMAL] < 0 and state.inv[seat, k, item] >= 1
    )


def value(params, h_glob: np.ndarray) -> float:
    """critic ``V(s)``。`value_w` 是 ``[d, 1]``，乘出來是 shape ``[1]``——
    NumPy 2 不讓 size-1 陣列隱式轉 float，所以明確取 `[0]`（不改 head 形狀）。"""
    return float((h_glob @ params["value_w"] + params["value_b"])[0])


def _empty_slots():
    return (
        np.zeros((N_LOGP_SLOTS,), np.float64),
        np.zeros((N_LOGP_SLOTS,), bool),
    )


def decode(
    params,
    cfg: dict[str, Any],
    state: NP.NpState,
    caps: caps_mod.Caps,
    *,
    mode: str,
    rng=None,
    labels: dict | None = None,
    seat: int = 0,
) -> dict[str, Any]:
    """`act` 與 `evaluate` **共用的唯一核心**（不得另寫簡化評分）。

    `mode` ∈ {"sample", "greedy", "teacher"}；三者共用同一份遮罩與同一條真 prefix。
    回傳的 `logp`／`active` 是固定 263 格，排法與 `rl/full_actions_decode.py::pack_slots`
    完全一樣：前 29×(verb, crop, item, b0..b3)、後 10×(op, item, b0..b3)。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必須是 {MODES}，得到 {mode!r}")
    if mode == "teacher" and labels is None:
        raise ValueError("teacher 模式要給 labels")
    if mode == "sample" and rng is None:
        raise ValueError("sample 模式要給 rng")
    if seat != 0:
        # `observation()` 永遠讀索引 0（`state.seeds[0]`、`shed[0]`、對手 rank 清 `[:, 1]`），
        # 部署狀態是 canonical（自己＝0，`loop4_numpy.state_from_obs`）。傳 1 進來會變成
        # 「觀測看座位 0、遮罩看座位 1」，靜默算錯——所以明拒，不默默算。
        raise ValueError(f"NumPy 路徑只吃 canonical 狀態（自己＝索引 0），得到 seat={seat}")

    u = caps.max_units
    h1 = encode_tokens(params, observation(state, caps), cfg)
    hu = h1["unit"][0]
    n_alive = min(int(state.hires_today[seat]) + 1, u)

    fields = {
        "unit_verb": np.full((u,), C.OP_PASS, I32),
        "unit_crop": np.zeros((u,), I32),
        "unit_item": np.zeros((u,), I32),
        "unit_qty_bytes": np.zeros((u, C.N_QTY_BYTES), I32),
    }
    logp, active = _empty_slots()
    drop = np.zeros((N_LOGP_SLOTS,), bool)
    drop_verb_by_op = np.zeros((N_UNIT_OPS,), I32)
    drop_op_by_code = np.zeros((N_MARKET_OPS,), I32)

    for k in range(u):
        base = k * UNIT_SLOTS
        if k >= n_alive:
            continue  # 死槽：verb 恆 PASS、全部 inactive（與 JAX 的 `alive` 分支同義）
        # 遮罩來自**真 prefix state**（原 state ＋ 已選前綴 ＋ 其餘 PASS）
        st_k = prefix_state(state, fields, caps, k, seat)
        leg = unit_legality(st_k, caps, seat=seat)

        verb, lp_v, ok_v = _pick(
            hu[k] @ params["verb_w"] + params["verb_b"],
            leg["unit_ops"][k],
            rng,
            mode,
            None if labels is None else labels["unit_verb"][k],
        )
        fields["unit_verb"][k] = verb
        logp[base], active[base], drop[base] = lp_v, ok_v, not ok_v
        if not ok_v:
            drop_verb_by_op[verb] += 1

        if verb == C.OP_PLANT:
            crop, lp_c, ok_c = _pick(
                hu[k] @ params["ucrop_w"] + params["ucrop_b"],
                leg["plant_crop"][k],
                rng,
                mode,
                None if labels is None else labels["unit_crop"][k],
            )
            ok_c = ok_c and ok_v
            fields["unit_crop"][k] = crop
            logp[base + 1] = lp_c if ok_c else 0.0
            active[base + 1], drop[base + 1] = ok_c, not ok_c

        if verb in (C.OP_PICKUP, C.OP_PLACE):
            ctx = np.tanh(
                np.concatenate([hu[k], _onehot(verb, N_UNIT_OPS)]) @ params["uitem_ctx_w"]
                + params["uitem_ctx_b"]
            )
            m = leg["pickup_item"][k] if verb == C.OP_PICKUP else leg["place_item"][k]
            item, lp_i, ok_i = _pick(
                ctx @ params["uitem_w"] + params["uitem_b"],
                m,
                rng,
                mode,
                None if labels is None else labels["unit_item"][k],
            )
            ok_i = ok_i and ok_v
            fields["unit_item"][k] = item
            logp[base + 2] = lp_i if ok_i else 0.0
            active[base + 2], drop[base + 2] = ok_i, not ok_i

            if a5_place_animal_success(st_k, k, verb, item, seat):
                # T3 A5：qty 被引擎忽略 → 不抽 qty、四槽 inactive；teacher 保留原 qty
                fields["unit_qty_bytes"][k] = (
                    np.asarray(labels["unit_qty_bytes"][k], I32)
                    if mode == "teacher"
                    else A5_QTY_BYTES
                )
                continue
            hq = np.tanh(
                np.concatenate([hu[k], _onehot(verb, N_UNIT_OPS), _onehot(item, N_ITEMS)])
                @ params["uqty_init_w"]
                + params["uqty_init_b"]
            )
            qb, lp_q, ok_q = _qty_scan(
                params, hq, rng, mode, None if labels is None else labels["unit_qty_bytes"][k]
            )
            ok_q = ok_q & (ok_v and ok_i)
            fields["unit_qty_bytes"][k] = qb
            sl = slice(base + 3, base + 3 + C.N_QTY_BYTES)
            logp[sl] = np.where(ok_q, lp_q, 0.0)
            active[sl], drop[sl] = ok_q, ~ok_q

    # 兩趟：**真 item／真 qty** 投影後才算市場（`value` 仍取動作前的 h1）
    nxt = prefix_state(state, fields, caps, u, seat)
    h2 = encode_tokens(params, observation(nxt, caps), cfg)
    hm = np.tanh(h2["glob"][0] @ params["market_init_w"] + params["market_init_b"])
    ptr = pointer_keys(params, h2["prod"][0], h2["glob"][0], pointer_aux(nxt))  # [12, d]
    sqrt_d = F32(np.sqrt(ptr.shape[-1]))

    m_op = np.zeros((C.MAX_ORDERS,), I32)
    m_item = np.zeros((C.MAX_ORDERS,), I32)
    m_qty = np.zeros((C.MAX_ORDERS, C.N_QTY_BYTES), I32)
    op_mask = np.ones((N_MARKET_OPS,), bool)  # 七類都是合法語法（0 ＝ EOS）
    base0 = u * UNIT_SLOTS
    for k in range(C.MAX_ORDERS):
        base = base0 + k * MARKET_SLOTS
        op, lp_o, ok_o = _pick(
            hm @ params["mop_w"] + params["mop_b"],
            op_mask,
            rng,
            mode,
            None if labels is None else labels["market_op"][k],
        )
        m_op[k] = op
        logp[base], active[base], drop[base] = lp_o, ok_o, not ok_o
        if not ok_o:
            drop_op_by_code[op] += 1
        if op == C.MO_NONE:
            break  # EOS：之後的槽在 JAX 側也全是 MO_NONE／inactive

        sup = _ITEM_SUPPORT[op]
        if sup.any():
            q = np.tanh(
                np.concatenate([hm, _onehot(op, N_MARKET_OPS)]) @ params["pointer_query_w"]
                + params["pointer_query_b"]
            )
            item, lp_i, ok_i = _pick(
                (ptr @ q) / sqrt_d,
                sup,
                rng,
                mode,
                None if labels is None else labels["market_item"][k],
            )
            ok_i = ok_i and ok_o
            m_item[k] = item
            logp[base + 1] = lp_i if ok_i else 0.0
            active[base + 1], drop[base + 1] = ok_i, not ok_i
        else:
            ok_i = ok_o

        if _NEEDS_QTY[op]:
            it_k = int(m_item[k])
            hq = np.tanh(
                np.concatenate([hm, _onehot(op, N_MARKET_OPS), _onehot(it_k, N_ITEMS), ptr[it_k]])
                @ params["mqty_init_w"]
                + params["mqty_init_b"]
            )
            qb, lp_q, ok_q = _qty_scan(
                params, hq, rng, mode, None if labels is None else labels["market_qty_bytes"][k]
            )
            ok_q = ok_q & (ok_o and ok_i)
            m_qty[k] = qb
            sl = slice(base + 2, base + 2 + C.N_QTY_BYTES)
            logp[sl] = np.where(ok_q, lp_q, 0.0)
            active[sl], drop[sl] = ok_q, ~ok_q

        msg = (
            params["mop_emb"][op]
            + params["mitem_emb"][int(m_item[k])]
            + (m_qty[k].astype(F32) / float(C.BYTE_CLASSES - 1)) @ params["mqty_in_w"]
        )
        hm = np.tanh((hm + msg) @ params["mstep_w"] + params["mstep_b"])

    decision = {
        **fields,
        "market_op": m_op,
        "market_item": m_item,
        "market_qty_bytes": m_qty,
        "market_length": int(np.sum(m_op != C.MO_NONE)),
    }
    return {
        "decision": decision,
        "logp": logp.astype(np.float32),
        "active": active,
        "logp_total": float(logp.sum()),
        "value": value(params, h1["glob"][0]),  # **動作前**的 state
        "n_alive": n_alive,
        "projected": nxt,
        "teacher_drop": {
            "slots": drop,
            "unit_verb_by_op": drop_verb_by_op,
            "market_op_by_code": drop_op_by_code,
        },
    }


def act(
    params, cfg, state: NP.NpState, caps: caps_mod.Caps, *, rng=None, greedy: bool = True
) -> dict[str, Any]:
    """一步決策：`DecisionV3` 八欄 ＋ **逐 263 token 的 logp／active**。`rng=None` ⇒ greedy。

    推論期**不做 top-k、不改數量**；`sample` 與 `greedy` 只差在 `_pick` 取樣方式。
    """
    mode = "greedy" if (greedy or rng is None) else "sample"
    return decode(params, cfg, state, caps, mode=mode, rng=rng)


def evaluate(
    params, cfg, state: NP.NpState, caps: caps_mod.Caps, decision: dict, *, seat: int = 0
) -> dict[str, Any]:
    """給定 decision 的**逐 token** logp／active，走的是 `act` 的同一顆 `decode` 核心。

    用於雙向耦合對帳：NumPy 抽出的 decision 交給 JAX teacher 評分、JAX 抽出的 decision
    交給這支評分。**不要求跨 PRNG 產生同一個 decision**，要求的是同一個 decision 的分數相同。
    """
    return decode(params, cfg, state, caps, mode="teacher", labels=decision, seat=seat)


# ------------------------------------------------------------------ 權重身分與 checkpoint I/O
def params_signature(params: dict[str, np.ndarray]) -> dict[str, Any]:
    """flat 參數樹的身分：逐葉 shape ＋ 總數。載入時逐項比對，不符就拒。"""
    return {
        "layout": "flat",
        "count": int(sum(int(np.prod(np.shape(v))) for v in params.values())),
        "n_leaves": len(params),
        "shapes": {k: list(np.shape(v)) for k, v in sorted(params.items())},
    }


def config_from_params(params: dict[str, np.ndarray], caps: caps_mod.Caps) -> dict[str, Any]:
    """從權重形狀反推 cfg——**只給明確的非正式診斷用**（例如手上只有一包裸權重時看架構）。

    正式 checkpoint 一律走 `config_from_meta`：`n_heads` 不在任何 shape 裡，這裡的
    ``6 if d % 6 == 0 else 4`` 是**猜的**，micro cfg 上會猜錯並讓部署與訓練切成不同模型。
    """
    d = int(np.shape(params["in_tile_w"])[1])
    layers = [int(m.group(1)) for m in (re.match(r"l(\d+)_", k) for k in params) if m]
    return {
        "d_model": d,
        "n_layers": 1 + max(layers),
        "n_heads": 6 if d % 6 == 0 else 4,
        "d_ff": int(np.shape(params["l0_ff1_w"])[1]),
        "max_units": int(np.shape(params["unit_slot_emb"])[0]),
        "n_tokens": 2 * rules.N_TILES + caps.max_units + rules.N_PRODUCTS + 1,
    }


def expected_param_shapes(cfg: dict[str, Any]) -> dict[str, list[int]]:
    """由 **cfg 導出**的參數鍵與 shape。

    載入時拿它當真值核 checkpoint，**不只信 meta 自報的簽章**——自報簽章只能證明
    「這個檔自己前後一致」，證不了「它就是 full-actions-v1 這個架構」。
    這份表與 `rl/full_actions_head.py::init_params` 必須逐項相同，由
    `tests/test_full_actions.py::test_expected_param_shapes_matches_init_params` 對帳。
    """
    d, ff, u = int(cfg["d_model"]), int(cfg["d_ff"]), int(cfg["max_units"])
    out: dict[str, list[int]] = {}
    for name, dim in (
        ("tile", TILE_DIM),
        ("unit", UNIT_DIM),
        ("product", PRODUCT_DIM),
        ("global", GLOBAL_DIM),
    ):
        out[f"in_{name}_w"] = [dim, d]
        out[f"in_{name}_b"] = [d]
    out["seg_emb"] = [N_SEGMENTS, d]
    out["tile_pos_emb"] = [rules.N_TILES, d]
    out["unit_slot_emb"] = [u, d]
    out["item_id_emb"] = [N_ITEMS, d]
    for i in range(int(cfg["n_layers"])):
        out[f"l{i}_ln1_g"] = [d]
        out[f"l{i}_ln1_b"] = [d]
        out[f"l{i}_qkv_w"] = [d, 3 * d]
        out[f"l{i}_proj_w"] = [d, d]
        out[f"l{i}_ln2_g"] = [d]
        out[f"l{i}_ln2_b"] = [d]
        out[f"l{i}_ff1_w"] = [d, ff]
        out[f"l{i}_ff1_b"] = [ff]
        out[f"l{i}_ff2_w"] = [ff, d]
        out[f"l{i}_ff2_b"] = [d]
    out["ln_f_g"] = [d]
    out["ln_f_b"] = [d]
    out["verb_w"] = [d, N_UNIT_OPS]
    out["verb_b"] = [N_UNIT_OPS]
    out["ucrop_w"] = [d, rules.N_CROPS]
    out["ucrop_b"] = [rules.N_CROPS]
    out["uitem_w"] = [d, N_ITEMS]
    out["uitem_b"] = [N_ITEMS]
    out["uitem_ctx_w"] = [d + N_UNIT_OPS, d]
    out["uitem_ctx_b"] = [d]
    out["uqty_init_w"] = [d + N_UNIT_OPS + N_ITEMS, d]
    out["uqty_init_b"] = [d]
    out["market_init_w"] = [d, d]
    out["market_init_b"] = [d]
    out["mop_emb"] = [N_MARKET_OPS, d]
    out["mitem_emb"] = [N_ITEMS, d]
    out["mqty_in_w"] = [C.N_QTY_BYTES, d]
    out["mstep_w"] = [d, d]
    out["mstep_b"] = [d]
    out["mop_w"] = [d, N_MARKET_OPS]
    out["mop_b"] = [N_MARKET_OPS]
    out["pointer_key_w"] = [2 * d + POINTER_AUX, d]
    out["pointer_key_b"] = [d]
    out["pointer_query_w"] = [d + N_MARKET_OPS, d]
    out["pointer_query_b"] = [d]
    out["mqty_init_w"] = [d + N_MARKET_OPS + N_ITEMS + d, d]
    out["mqty_init_b"] = [d]
    out["byte_embed"] = [C.BYTE_CLASSES, d]
    out["byte_out_w"] = [d, C.BYTE_CLASSES]
    out["byte_out_b"] = [C.BYTE_CLASSES]
    out["prefix_w"] = [2 * d, d]
    out["prefix_b"] = [d]
    out["value_w"] = [d, 1]
    out["value_b"] = [1]
    return out


def config_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """**正式路徑**的 cfg 來源：checkpoint 的 metadata。

    `n_heads` 不寫在任何權重的 shape 裡（qkv 是 ``[d, 3d]``，怎麼切頭看不出來），
    所以只能讀 metadata；用 `d % 6` 猜會在 micro cfg 上靜默切錯頭，訓練與部署就不同模型。
    """
    for field in ("trunk", "unit_slot_emb"):
        if field not in meta:
            raise ValueError(f"meta 缺 {field}：無法取得 cfg")
    t = meta["trunk"]
    missing = [k for k in ("d_model", "n_layers", "n_heads", "d_ff", "n_tokens") if k not in t]
    if missing:
        raise ValueError(f"meta.trunk 缺 {missing}")
    cfg = {k: int(t[k]) for k in ("d_model", "n_layers", "n_heads", "d_ff", "n_tokens")}
    slot = list(meta["unit_slot_emb"])
    if len(slot) != 2 or int(slot[1]) != cfg["d_model"]:
        raise ValueError(f"meta.unit_slot_emb {slot} 與 d_model {cfg['d_model']} 不符")
    cfg["max_units"] = int(slot[0])
    if cfg["d_model"] % cfg["n_heads"]:
        raise ValueError(f"d_model {cfg['d_model']} 不能被 n_heads {cfg['n_heads']} 整除")
    want_tokens = 2 * rules.N_TILES + cfg["max_units"] + rules.N_PRODUCTS + 1
    if cfg["n_tokens"] != want_tokens:
        raise ValueError(f"meta.trunk.n_tokens {cfg['n_tokens']} 應為 {want_tokens}")
    return cfg


def check_caps(cfg: dict[str, Any], meta: dict[str, Any], caps: caps_mod.Caps) -> None:
    """caps 與 checkpoint 相容性：槽數對不上就拒，不靜默截斷或補零。"""
    if caps.max_units != cfg["max_units"]:
        raise ValueError(f"caps.max_units {caps.max_units} ≠ checkpoint {cfg['max_units']}")
    want = int(meta["action_space"]["max_orders"])
    if caps.max_orders != want:
        raise ValueError(f"caps.max_orders {caps.max_orders} ≠ checkpoint {want}")


def _check_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """schema／obs／action_space／cfg／簽章**全部必填**，缺一就拒。"""
    for field in ("schema", "obs", "trunk", "unit_slot_emb", "action_space", "params"):
        if field not in meta:
            raise ValueError(f"meta 缺必填欄位 {field}")
    if meta["schema"] != SCHEMA:
        raise ValueError(f"schema 是 {meta['schema']!r}，不是 {SCHEMA!r}")
    o = meta["obs"]
    if o.get("obs_version") != OBS_SCHEMA:
        raise ValueError(f"obs_version 是 {o.get('obs_version')!r}，不是 {OBS_SCHEMA!r}")
    for key, want in (
        ("tile_dim", TILE_DIM),
        ("unit_dim", UNIT_DIM),
        ("product_dim", PRODUCT_DIM),
        ("global_dim", GLOBAL_DIM),
    ):
        if int(o.get(key, -1)) != want:
            raise ValueError(f"obs.{key} 是 {o.get(key)}，本版是 {want}")
    if o.get("in_progress") != "always_zero":
        raise ValueError(f"obs.in_progress 必須是 always_zero，得到 {o.get('in_progress')!r}")
    a = meta["action_space"]
    for key, want in (
        ("codec", ACTION_CODEC),
        ("max_units", C.MAX_UNITS),
        ("max_orders", C.MAX_ORDERS),
        ("n_unit_ops", N_UNIT_OPS),
        ("n_market_ops", N_MARKET_OPS),
        ("n_items", N_ITEMS),
        ("n_crops", rules.N_CROPS),
        ("qty_bytes", C.N_QTY_BYTES),
        ("byte_classes", C.BYTE_CLASSES),
        ("n_logp_slots", N_LOGP_SLOTS),
    ):
        if a.get(key) != want:
            raise ValueError(f"action_space.{key} 是 {a.get(key)!r}，本版是 {want!r}")
    return config_from_meta(meta)


def save_params(path: str | Path, params: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    """存成 ``p/<key>`` ＋ ``meta``（JSON 字串）。參數樹是 flat 的，沒有 trunk／head 兩枝。

    寫之前就把 meta 與參數核過一次（`_check_meta` ＋ cfg 導出的 shape），
    不讓不完整的 metadata 進磁碟；訓練用的 optimizer 狀態由訓練腳本另外以 ``o/<i>``
    寫進同一個 npz，這支只寫 params，載入時會忽略 ``o/*``。
    """
    meta = {**meta, "params": params_signature(params)}
    cfg = _check_meta(meta)
    _check_params(params, cfg)
    path = Path(path)
    d: dict[str, Any] = {f"p/{k}": np.asarray(v) for k, v in params.items()}
    d["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(path)


def _check_params(params: dict[str, np.ndarray], cfg: dict[str, Any]) -> None:
    want = expected_param_shapes(cfg)
    got = {k: list(np.shape(v)) for k, v in params.items()}
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing or extra:
        raise ValueError(f"參數鍵與 cfg 導出的不符：缺 {missing[:8]}、多 {extra[:8]}")
    bad = sorted(k for k in want if got[k] != want[k])
    if bad:
        raise ValueError(f"參數 shape 與 cfg 導出的不符：{[(k, got[k], want[k]) for k in bad[:4]]}")


def load_params(
    path: str | Path, *, caps: caps_mod.Caps | None = None, expect_sha256: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """讀回 `save_params` 的 checkpoint。**schema／obs／action_space／cfg／簽章全部必填**，
    參數鍵與 shape 以 **cfg 導出的預期值**核（不只比對 meta 自報的簽章），舊權重明拒。

    拒載的情形：沒有 `meta`、schema／obs_version／action_space 不是本版、meta 缺 cfg 或簽章、
    自報簽章與實際參數不一致、參數鍵／shape 與 cfg 導出的不符、給了 `caps` 但槽數不相容、
    給了 `expect_sha256` 但檔案 SHA 不符。一律 `ValueError`，不猜形狀、不補零。
    """
    path = Path(path)
    if expect_sha256 is not None:
        got_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if got_sha != expect_sha256:
            raise ValueError(f"checkpoint SHA256 不符：expect {expect_sha256}、實得 {got_sha}")
    z = np.load(path, allow_pickle=False)
    if "meta" not in z.files:
        raise ValueError(f"{path} 沒有 meta：不是 {SCHEMA} 的 checkpoint（舊權重明拒）")
    meta = json.loads(str(z["meta"]))
    cfg = _check_meta(meta)
    params = {k[2:]: np.asarray(z[k], F32) for k in z.files if k.startswith("p/")}
    if not params:
        raise ValueError(f"{path} 沒有 p/* 參數")
    _check_params(params, cfg)  # 以 cfg 導出的預期鍵／shape 為真值
    want, got = meta["params"], params_signature(params)
    for field in ("layout", "count", "n_leaves", "shapes"):
        if got[field] != want.get(field):
            raise ValueError(f"meta 自報的 params.{field} 與實際參數不符")
    if caps is not None:
        check_caps(cfg, meta, caps)
    return params, meta


# ------------------------------------------------------------------ 部署 agent
class FullActionsNumpyAgent:
    """部署用 agent：``agent(obs)`` → 引擎動作 dict。全程 NumPy，**不 import jax**。

    決策走的是 `act`／`decode` 這顆核心，與訓練／評分同一條遮罩與同一條真 prefix；
    推論期沒有 top-k、沒有改數量。`rng=None` 是 greedy（部署預設）。
    """

    def __init__(
        self,
        params: dict[str, np.ndarray],
        seat: int,
        caps: caps_mod.Caps | None = None,
        cfg: dict[str, Any] | None = None,
        rng=None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.caps = caps or caps_mod.EXACT_REFERENCE
        self.params = params
        if cfg is None:
            if not meta:
                raise ValueError("cfg 與 meta 至少要有一個：n_heads 不能從權重 shape 猜")
            cfg = config_from_meta(meta)
        self.cfg = cfg
        self.seat = seat
        self.rng = rng
        self.meta = meta or {}
        self.last: dict[str, Any] = {}
        # 舊 checkpoint 經相容 adapter 使用：記下它訓練時的觀測契約與現在執行的契約，不冒稱相同
        self.obs_contract = {
            "execution": EXEC_OBS_CONTRACT,
            "trained": (self.meta.get("obs") or {}).get("execution_obs_contract", "pre-t3"),
        }

    def __call__(self, obs: dict, _config: Any = None) -> dict:
        state = state_from_obs(obs, self.seat, self.caps)
        return self.decide(state)

    def decide(self, state: NP.NpState) -> dict:
        """部署路徑的完整一步：**觀測 → 解碼 → 引擎 JSON**（延遲量測量的就是這一段）。"""
        out = act(self.params, self.cfg, state, self.caps, rng=self.rng, greedy=self.rng is None)
        js = decision_to_engine_json(out["decision"], int(out["n_alive"]))
        # runtime capture 直接取當次真輸入／真輸出，不必回推
        self.last = {
            "decision": out["decision"],
            "logp": out["logp"],
            "active": out["active"],
            "value": out["value"],
            "n_alive": out["n_alive"],
            "action": js,
        }
        return js


def decision_to_engine_json(decision: dict, n_alive: int) -> dict:
    """`DecisionV3` 八欄 → 引擎動作 dict（`loop4_numpy.action_to_engine_json` 的包裝）。"""
    m_qty = _u31(decision["market_qty_bytes"])
    orders = [
        (int(decision["market_op"][k]), int(decision["market_item"][k]), int(m_qty[k]))
        for k in range(int(decision["market_length"]))
    ]
    return NP.action_to_engine_json(
        decision["unit_verb"],
        _unit_arg(decision["unit_verb"], decision["unit_crop"], decision["unit_item"]),
        _u31(decision["unit_qty_bytes"]),
        orders,
        n_alive,
    )


def make_agent(
    path: str | Path,
    seat: int,
    caps: caps_mod.Caps | None = None,
    *,
    rng=None,
    expect_sha256: str | None = None,
) -> FullActionsNumpyAgent:
    """從 checkpoint 造 agent。**cfg 一律讀 metadata**，不由權重 shape 猜。

    `n_heads` 猜不出來（qkv 是 ``[d, 3d]``），猜錯就是訓練與部署切成不同模型；
    schema／obs／action_space／cfg／簽章與 caps 相容性任一不符都拒載。
    """
    caps = caps or caps_mod.EXACT_REFERENCE
    params, meta = load_params(path, caps=caps, expect_sha256=expect_sha256)
    return FullActionsNumpyAgent(params, seat, caps, cfg=config_from_meta(meta), rng=rng, meta=meta)


__all__ = [
    "ACTION_CODEC",
    "EXEC_OBS_CONTRACT",
    "IL_MASK_CONTRACT",
    "MARKET_COUNTER_IDX",
    "MODES",
    "ObsSchemaError",
    "inv_rank_canonical",
    "state_from_obs",
    "GLOBAL_DIM",
    "PRODUCT_DIM",
    "TILE_DIM",
    "UNIT_DIM",
    "N_LOGP_SLOTS",
    "SCHEMA",
    "FullActionsNumpyAgent",
    "A5_QTY_BYTES",
    "a5_place_animal_success",
    "act",
    "build_tokens",
    "check_caps",
    "config_from_meta",
    "config_from_params",
    "expected_param_shapes",
    "decode",
    "decision_to_engine_json",
    "encode_tokens",
    "pointer_aux",
    "pointer_keys",
    "evaluate",
    "load_params",
    "make_agent",
    "observation",
    "params_signature",
    "prefix_state",
    "save_params",
    "unit_legality",
    "value",
]
