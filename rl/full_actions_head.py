"""完整動作表示的 trunk ＋ head（full-actions v1）。

規格：內部規格文件 §第一段精確實作規格。

與 `rl/loop4_model.py` 的關係：**沿用它的 block／LayerNorm 當純 trunk helper**
（`_block`／`_ln`／`N_TILE_TOKENS`／段別常數），但**自己建 token 與自己的 head**：

* 輸入投影用 full-actions 的維度（tile 35／unit 42（obs v2，T3 F5）／product 10／global 32）。
* **T3 F6**：`item_id_emb [12, 192]`，0..8 加到兩趟 encoder 的 product token（不靠 row 位置猜 ID）。
* **T3 A1**：市場 item 用 pointer 頭（candidate：商品＝第二趟 product hidden、動物＝global hidden，
  各帶 ID 與 2 維可見量）；qty 起始 hidden 另吃選定 candidate 的 key。舊 `mitem_w/b`、
  `mitem_ctx_w/b` 刪除。
* **新增 `unit_slot_emb [29, 192]`**：Loop 4 只有 segment embedding，單位 token 的陣列位置
  本身不編碼身分；沒有它的話「同一格、同樣背包」的兩個雇工會拿到完全相同的 logits。
* 舊 head（`task/tqty/sell/farm`）**完全不建**，不留沒用到的可訓練葉。
* 初始化 `seed=0` 全新，不搬舊權重、不暖啟動。

head 維度（全部取自 `rl/ordered_v3_contract`／`rl/rules`，不是估計）：

| head | 形狀 | 類別 |
|---|---|---|
| `unit_verb` | ``[B, 29, 18]`` | `UNIT_OP_NAMES` |
| `unit_crop` | ``[B, 29, 5]`` | `N_CROPS` |
| `unit_item` | ``[B, 29, 12]`` | `N_ITEMS` |
| `unit_qty` | ``[B, 29, 4, 256]`` | U31 big-endian 四 byte |
| `market_op` | ``[B, 10, 7]`` | `MARKET_OP_NAMES`（`NONE` ＝ EOS） |
| `market_item` | ``[B, 10, 12]`` | `N_ITEMS`；T3 A1 起為 pointer：``dot(q, k_i)/sqrt(d)`` |
| `market_qty` | ``[B, 10, 4, 256]`` | 同 U31 |

市場是**逐槽 scan**：第 k 槽的隱藏狀態吃前一槽的 (op, item, qty) ——「訂單意圖 prefix」，
不假冒已成交 state。單位相的 head 從各自的 unit token 出（每個單位就是一個 token，
所以條件在 token 上，不需要再對單位做 scan）。
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import full_actions_numpy as FN
from rl import full_actions_obs as FO
from rl import loop4_model as M
from rl import ordered_v3_contract as C
from rl import rules

F32 = jnp.float32

#: schema／參數簽章的**單一定義**在 jax-free 的部署模組裡（`rl/full_actions_numpy.py`），
#: 這裡只 re-export。兩邊各寫一份字串遲早漂移，而部署路徑不能 import 這個檔（它吃 jax）。
SCHEMA = FN.SCHEMA
OBS_SCHEMA = FN.OBS_SCHEMA
params_signature = FN.params_signature
#: 首版固定：4 層 × d192 × 6 head（與 Loop 4 的 A 架構同尺寸，但權重全新）
DEFAULT_ARCH = {"d_model": 192, "n_layers": 4, "n_heads": 6, "d_ff": 768}

N_UNIT_OPS = len(C.UNIT_OP_NAMES)  # 18
N_MARKET_OPS = len(C.MARKET_OP_NAMES)  # 7
N_ITEMS = rules.N_ITEMS  # 12
N_CROPS = rules.N_CROPS  # 5
N_QTY_BYTES = C.N_QTY_BYTES  # 4
BYTE_CLASSES = C.BYTE_CLASSES  # 256
MAX_UNITS = C.MAX_UNITS  # 29
MAX_ORDERS = C.MAX_ORDERS  # 10

#: PPO 的固定 token 位置數：29×(verb+crop+item+4 bytes) ＋ 10×(op+item+4 bytes) ＝ 263
N_LOGP_SLOTS = FN.N_LOGP_SLOTS
#: pointer candidate 的可見數量維度（shed/100、種子 log 尺度）
POINTER_AUX = FN.POINTER_AUX


def config(caps: caps_mod.Caps | None = None, **over: int) -> dict[str, Any]:
    caps = caps or caps_mod.BOUNDED_TRAINING
    cfg = dict(DEFAULT_ARCH)
    cfg.update(over)
    cfg["max_units"] = caps.max_units
    cfg["n_tokens"] = M.N_TILE_TOKENS + caps.max_units + rules.N_PRODUCTS + 1
    return cfg


def _glorot(key, shape):
    lim = float(np.sqrt(6.0 / (shape[0] + shape[-1])))
    return jax.random.uniform(key, shape, F32, -lim, lim)


def init_params(cfg: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """全新初始化（規格：seed 0，不搬舊權重）。

    **參數樹是 flat 的**（審查者 2026-09-21 裁定，不為命名改樹）：trunk 與 head 的葉全部在
    同一層 dict，因此同一個 optimizer 樹、checkpoint 以 ``p/<key>`` 存，不會有 head 漏進
    optimizer 或 resume 掉葉的問題。身分靠 `meta()` 的 `params.shapes`／`count` 與 schema
    校驗，不靠命名前綴。
    """
    d = cfg["d_model"]
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 64 + 4 * cfg["n_layers"]))
    p: dict[str, Any] = {}
    for name, dim in (
        ("tile", FO.TILE_DIM),
        ("unit", FO.UNIT_DIM),
        ("product", FO.PRODUCT_DIM),
        ("global", FO.GLOBAL_DIM),
    ):
        p[f"in_{name}_w"] = _glorot(next(keys), (dim, d))
        p[f"in_{name}_b"] = jnp.zeros((d,), F32)
    p["seg_emb"] = 0.02 * jax.random.normal(next(keys), (M.N_SEGMENTS, d), F32)
    p["tile_pos_emb"] = 0.02 * jax.random.normal(next(keys), (rules.N_TILES, d), F32)
    # 單位槽身分（Loop 4 沒有；同位置同背包的雇工必須可區分）
    p["unit_slot_emb"] = 0.02 * jax.random.normal(next(keys), (cfg["max_units"], d), F32)
    # T3 F6：品項身分。0..8 加到兩趟 encoder 的 product token；9..11（動物）只給 pointer 用
    p["item_id_emb"] = 0.02 * jax.random.normal(next(keys), (N_ITEMS, d), F32)
    for i in range(cfg["n_layers"]):
        p[f"l{i}_ln1_g"] = jnp.ones((d,), F32)
        p[f"l{i}_ln1_b"] = jnp.zeros((d,), F32)
        p[f"l{i}_qkv_w"] = _glorot(next(keys), (d, 3 * d))
        p[f"l{i}_proj_w"] = _glorot(next(keys), (d, d))
        p[f"l{i}_ln2_g"] = jnp.ones((d,), F32)
        p[f"l{i}_ln2_b"] = jnp.zeros((d,), F32)
        p[f"l{i}_ff1_w"] = _glorot(next(keys), (d, cfg["d_ff"]))
        p[f"l{i}_ff1_b"] = jnp.zeros((cfg["d_ff"],), F32)
        p[f"l{i}_ff2_w"] = _glorot(next(keys), (cfg["d_ff"], d))
        p[f"l{i}_ff2_b"] = jnp.zeros((d,), F32)
    p["ln_f_g"] = jnp.ones((d,), F32)
    p["ln_f_b"] = jnp.zeros((d,), F32)

    # --- 單位 head（每個單位一組，從自己的 token 出） ---
    p["verb_w"] = _glorot(next(keys), (d, N_UNIT_OPS))
    p["verb_b"] = jnp.zeros((N_UNIT_OPS,), F32)
    p["ucrop_w"] = _glorot(next(keys), (d, N_CROPS))
    p["ucrop_b"] = jnp.zeros((N_CROPS,), F32)
    p["uitem_w"] = _glorot(next(keys), (d, N_ITEMS))
    p["uitem_b"] = jnp.zeros((N_ITEMS,), F32)
    # qty 是**條件** decoder：hidden 由 (unit token, verb, item) 起始，逐 byte 更新
    p["uitem_ctx_w"] = _glorot(next(keys), (d + N_UNIT_OPS, d))
    p["uitem_ctx_b"] = jnp.zeros((d,), F32)
    p["uqty_init_w"] = _glorot(next(keys), (d + N_UNIT_OPS + N_ITEMS, d))
    p["uqty_init_b"] = jnp.zeros((d,), F32)

    # --- 市場 head（逐槽 scan，prefix ＝ 前一槽的訂單意圖） ---
    p["market_init_w"] = _glorot(next(keys), (d, d))
    p["market_init_b"] = jnp.zeros((d,), F32)
    p["mop_emb"] = 0.02 * jax.random.normal(next(keys), (N_MARKET_OPS, d), F32)
    p["mitem_emb"] = 0.02 * jax.random.normal(next(keys), (N_ITEMS, d), F32)
    p["mqty_in_w"] = _glorot(next(keys), (N_QTY_BYTES, d))
    p["mstep_w"] = _glorot(next(keys), (d, d))
    p["mstep_b"] = jnp.zeros((d,), F32)
    p["mop_w"] = _glorot(next(keys), (d, N_MARKET_OPS))
    p["mop_b"] = jnp.zeros((N_MARKET_OPS,), F32)
    # T3 A1：市場 item 改成 pointer 頭（舊 `mitem_w/b`、`mitem_ctx_w/b` 刪除，不留死參數）
    p["pointer_key_w"] = _glorot(next(keys), (2 * d + POINTER_AUX, d))
    p["pointer_key_b"] = jnp.zeros((d,), F32)
    p["pointer_query_w"] = _glorot(next(keys), (d + N_MARKET_OPS, d))
    p["pointer_query_b"] = jnp.zeros((d,), F32)
    # qty 起始 hidden 另吃**選定 candidate 的 key**：[h_slot, op, item, k_sel] = 192+7+12+192
    p["mqty_init_w"] = _glorot(next(keys), (d + N_MARKET_OPS + N_ITEMS + d, d))
    p["mqty_init_b"] = jnp.zeros((d,), F32)
    # 逐 byte 的共用機件（單位相與市場相共用，與 v3 的 byte_embed／prefix_step 同結構）
    p["byte_embed"] = 0.02 * jax.random.normal(next(keys), (BYTE_CLASSES, d), F32)
    p["byte_out_w"] = _glorot(next(keys), (d, BYTE_CLASSES))
    p["byte_out_b"] = jnp.zeros((BYTE_CLASSES,), F32)
    p["prefix_w"] = _glorot(next(keys), (2 * d, d))
    p["prefix_b"] = jnp.zeros((d,), F32)

    # --- critic（PPO 第二段要，IL 不用；留在同一棵樹避免 resume 漏葉） ---
    p["value_w"] = _glorot(next(keys), (d, 1))
    p["value_b"] = jnp.zeros((1,), F32)
    return p


def param_count(params: dict[str, Any]) -> int:
    return int(sum(np.prod(np.shape(v)) for v in params.values()))


def build_tokens(params, obs: dict[str, jax.Array], cfg: dict[str, Any]):
    """四段觀測 → ``[B, T, d]`` 與 ``[B, T]`` 遮罩。與 Loop 4 同結構，**多 `unit_slot_emb`**。"""
    tiles = obs["tiles"]
    b = tiles.shape[0]
    d = cfg["d_model"]
    tile_tok = tiles @ params["in_tile_w"] + params["in_tile_b"]
    tile_tok = tile_tok + params["tile_pos_emb"][None, None, :, :]
    tile_tok = (
        tile_tok
        + jnp.stack([params["seg_emb"][M.SEG_OWN_TILES], params["seg_emb"][M.SEG_OPP_TILES]])[
            None, :, None, :
        ]
    )
    tile_tok = tile_tok.reshape(b, M.N_TILE_TOKENS, d)

    units = obs["units"][:, 0]  # 只有自己的單位會行動
    unit_tok = units @ params["in_unit_w"] + params["in_unit_b"]
    unit_tok = unit_tok + params["seg_emb"][M.SEG_UNITS] + params["unit_slot_emb"][None, :, :]
    prod_tok = obs["products"] @ params["in_product_w"] + params["in_product_b"]
    # T3 F6：product token 固定依 PRODUCTS 順序（0..8），加上對應品項 ID
    prod_tok = (
        prod_tok + params["seg_emb"][M.SEG_PRODUCTS] + params["item_id_emb"][: rules.N_PRODUCTS]
    )
    glob_tok = obs["global"] @ params["in_global_w"] + params["in_global_b"]
    glob_tok = (glob_tok + params["seg_emb"][M.SEG_GLOBAL])[:, None, :]

    x = jnp.concatenate([tile_tok, unit_tok, prod_tok, glob_tok], axis=1)
    alive = obs["units"][:, 0, :, 2] > 0.5
    mask = jnp.concatenate(
        [
            jnp.ones((b, M.N_TILE_TOKENS), bool),
            alive,
            jnp.ones((b, rules.N_PRODUCTS + 1), bool),
        ],
        axis=1,
    )
    return x, mask


def encode_tokens(params, obs: dict[str, jax.Array], cfg: dict[str, Any], remat: bool = False):
    """trunk 前向 → ``{"unit": [B, U, d], "glob": [B, d], "hidden": [B, T, d]}``。

    `_block`／`_ln` 直接用 `rl/loop4_model` 的（純函式，**沒有改它**，舊數值不受影響）。
    """
    x, mask = build_tokens(params, obs, cfg)
    block = jax.checkpoint(M._block, static_argnums=(2, 3, 5)) if remat else M._block
    for i in range(cfg["n_layers"]):
        x = block(x, params, i, cfg["n_heads"], mask, None)
    x = M._ln(x, params["ln_f_g"], params["ln_f_b"])
    u0 = M.N_TILE_TOKENS
    p0 = u0 + cfg["max_units"]
    return {
        "hidden": x,
        "unit": x[:, u0:p0],
        "prod": x[:, p0 : p0 + rules.N_PRODUCTS],
        "glob": x[:, -1],
        "token_mask": mask,
    }


def unit_logits(params, h_unit: jax.Array) -> dict[str, jax.Array]:
    """單位相的三組 categorical logits。``h_unit`` ``[B, U, d]``。

    qty **不在這裡**：它是逐 byte 的條件 scan（`qty_init`／`byte_logits`／`byte_advance`），
    hidden 吃 (unit token, 已選 verb, 已選 item)，不是一次並排四個 byte 的 logits。
    """
    return {
        "verb": h_unit @ params["verb_w"] + params["verb_b"],  # [B, U, 18]
        "crop": h_unit @ params["ucrop_w"] + params["ucrop_b"],  # [B, U, 5]
        "item": h_unit @ params["uitem_w"] + params["uitem_b"],  # [B, U, 12]
    }


def unit_item_logits(params, h_unit, verb) -> jax.Array:
    """單位 item 的 logits **吃已選 verb**（PICKUP 與 PLACE 不共用同一組 logits）。"""
    v = jax.nn.one_hot(verb, N_UNIT_OPS, dtype=F32)
    ctx = jnp.tanh(
        _lin(params["uitem_ctx_w"], params["uitem_ctx_b"], jnp.concatenate([h_unit, v], axis=-1))
    )
    return ctx @ params["uitem_w"] + params["uitem_b"]


def _lin(w, b, x):
    return x @ w + b


def qty_init_unit(params, h_unit, verb, item) -> jax.Array:
    """單位 qty 的起始 hidden：``tanh(W [h_unit, onehot(verb), onehot(item)])``。"""
    v = jax.nn.one_hot(verb, N_UNIT_OPS, dtype=F32)
    it = jax.nn.one_hot(item, N_ITEMS, dtype=F32)
    x = jnp.concatenate([h_unit, v, it], axis=-1)
    return jnp.tanh(_lin(params["uqty_init_w"], params["uqty_init_b"], x))


def qty_init_market(params, h_slot, op, item, keys) -> jax.Array:
    """市場 qty 的起始 hidden：``tanh([h_slot, onehot(op), onehot(item), k_item] @ W + b)``。

    `keys` ``[B, 12, d]`` 是本回合的 pointer candidate keys（`pointer_keys`）；qty 也讀得到
    選定品項的價格／持有量，不只 item 頭用 pointer（T3 §5.1.2）。
    """
    o = jax.nn.one_hot(op, N_MARKET_OPS, dtype=F32)
    it = jax.nn.one_hot(item, N_ITEMS, dtype=F32)
    k_sel = jnp.take_along_axis(keys, item[:, None, None].astype(jnp.int32), axis=1)[:, 0]
    x = jnp.concatenate([h_slot, o, it, k_sel], axis=-1)
    return jnp.tanh(_lin(params["mqty_init_w"], params["mqty_init_b"], x))


def byte_logits(params, h) -> jax.Array:
    """目前 hidden → 這一個 byte 的 256 類 logits。"""
    return _lin(params["byte_out_w"], params["byte_out_b"], h)


def byte_advance(params, h, byt) -> jax.Array:
    """把選到的 byte 餵回 hidden（與 v3 `prefix_step` 同結構）。"""
    x = jnp.concatenate([h, params["byte_embed"][byt]], axis=-1)
    return jnp.tanh(_lin(params["prefix_w"], params["prefix_b"], x))


def pointer_aux(state, seat: int) -> jax.Array:
    """每個 candidate 的 2 個可見數量 ``[B, 12, 2]``（自家、單位相投影後的 state）。

    商品 0..8：``shed/100`` 與（作物 0..4）種子的既有 log 尺度、其餘 0；
    動物 9..11：``shed/100`` 與 0。
    不讀對手 private，也不假造市場成交。
    """
    shed = state.shed[:, seat, :].astype(F32) / float(rules.SHED_CAPACITY)  # [B, 12]
    seed = FO.seed_features(state, seat)  # [B, 5]
    seed12 = jnp.concatenate([seed, jnp.zeros((seed.shape[0], N_ITEMS - N_CROPS), F32)], -1)
    return jnp.stack([shed, seed12], axis=-1)


def pointer_keys(params, h_prod: jax.Array, h_glob: jax.Array, aux: jax.Array) -> jax.Array:
    """12 個 candidate 的 key ``[B, 12, d]``：商品用第二趟 product hidden，動物用 global hidden。

    ``k_i = tanh([cand_i, ID_i, aux_i] @ pointer_key_w + b)``；一回合算一次，各槽共用。
    """
    b = h_glob.shape[0]
    n_anim = N_ITEMS - rules.N_PRODUCTS
    cand = jnp.concatenate(
        [h_prod, jnp.broadcast_to(h_glob[:, None], (b, n_anim, h_glob.shape[-1]))], 1
    )
    ids = jnp.broadcast_to(params["item_id_emb"][None], cand.shape)
    x = jnp.concatenate([cand, ids, aux.astype(F32)], axis=-1)
    return jnp.tanh(_lin(params["pointer_key_w"], params["pointer_key_b"], x))


def market_item_logits(params, h_slot, op, keys) -> jax.Array:
    """市場 item 的 pointer logits ``[B, 12]``：``dot(q, k_i) / sqrt(d)``。

    q 吃槽 hidden 與已選 op；support 遮罩由呼叫端依 op 套用。
    """
    o = jax.nn.one_hot(op, N_MARKET_OPS, dtype=F32)
    q = jnp.tanh(
        _lin(params["pointer_query_w"], params["pointer_query_b"], jnp.concatenate([h_slot, o], -1))
    )
    return jnp.einsum("bd,bid->bi", q, keys) / jnp.sqrt(jnp.asarray(q.shape[-1], F32))


def market_init(params, h_glob: jax.Array) -> jax.Array:
    """第二趟 trunk 的 global hidden → 市場 scan 的起始隱藏狀態 ``[B, d]``。"""
    return jnp.tanh(h_glob @ params["market_init_w"] + params["market_init_b"])


def market_op_logits(params, h: jax.Array) -> jax.Array:
    """單一槽的 op logits ``[B, 7]``（item／qty 各自吃已選的前綴，見上）。"""
    return h @ params["mop_w"] + params["mop_b"]


def market_advance(params, h: jax.Array, op: jax.Array, item: jax.Array, qty_bytes: jax.Array):
    """把「這一槽下了什麼」餵回隱藏狀態（訂單**意圖** prefix，不假冒成交）。

    `op` ``[B]``、`item` ``[B]``、`qty_bytes` ``[B, 4]``（0..255）。
    """
    qn = qty_bytes.astype(F32) / float(BYTE_CLASSES - 1)
    msg = params["mop_emb"][op] + params["mitem_emb"][item] + qn @ params["mqty_in_w"]
    return jnp.tanh((h + msg) @ params["mstep_w"] + params["mstep_b"])


def value(params, h_glob: jax.Array) -> jax.Array:
    return (h_glob @ params["value_w"] + params["value_b"])[:, 0]


def meta(
    cfg: dict[str, Any], *, seed: int, two_pass: bool = True, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """寫進 checkpoint 的 meta（規格：config／動作空間／obs／參數 shape 與 count／seed）。

    參數樹是 **flat**：checkpoint 以 ``p/<key>`` 存所有 trunk＋head 葉、``o/<i>`` 存
    optimizer 狀態；載入時用 `params_signature` 逐葉比對 shape 與總數。
    """
    out = {
        "schema": SCHEMA,
        "trunk": {k: cfg[k] for k in ("d_model", "n_layers", "n_heads", "d_ff", "n_tokens")},
        "unit_slot_emb": [cfg["max_units"], cfg["d_model"]],
        "obs": FO.dims(),
        "action_space": {
            "codec": FN.ACTION_CODEC,
            "il_mask_contract": FN.IL_MASK_CONTRACT,
            "max_units": MAX_UNITS,
            "max_orders": MAX_ORDERS,
            "n_unit_ops": N_UNIT_OPS,
            "n_market_ops": N_MARKET_OPS,
            "n_items": N_ITEMS,
            "n_crops": N_CROPS,
            "qty_bytes": N_QTY_BYTES,
            "byte_classes": BYTE_CLASSES,
            "n_logp_slots": N_LOGP_SLOTS,
        },
        "two_pass": two_pass,
        "seed": seed,
        "params_layout": "flat (trunk+head 同一層；checkpoint p/<key>、optimizer o/<i>)",
    }
    if params is not None:
        out["params"] = params_signature(params)
    return out
