"""Loop 4 的模型（任務 14 B1）：Transformer encoder ＋ 每格任務／市場桶／value 的 head。

token 佈局（內部規格文件 §3、內部量測紀錄 §1f）：

===============  ====  =====================================================
段               數量  維
自己的格         100   `loop4_obs.TILE_DIM`
對手的格         100   同上（公開視圖，私有欄位已在觀測端遮掉）
單位             ≤29   `loop4_obs.UNIT_DIM`
產品             9     `loop4_obs.PRODUCT_DIM`
全域             1     `loop4_obs.GLOBAL_DIM`
===============  ====  =====================================================

head（全部由對應的 token 讀出，不共用一個 flatten）：

* **每格任務**：自己那 100 個格 token → 17 類（`rl/loop4_actions.py` 的 `N_TASKS`）。
* **PICKUP／PLACE 數量**：同樣 100 個格 token → 4 桶。
* **市場**：產品 token → SELL 10；種子／動物／可買產品由全域 token 投影
  （它們是「整座農場一筆」的決定，不屬於任何一格）；HIRE 11、BUY_LAND 2。
* **value**：全域 token。集中式——觀測端給的是自己那一座的視角，但 critic 在訓練時
  可以吃 `loop4_obs.full_view`（尚未實作，B3 才需要）。

寫法上刻意**不依賴任何訓練框架**：純 JAX 的 dict-of-arrays 參數，與 Loop 3 的
`rl/ordered_v3_policy.py` 同一個風格，所以 checkpoint 就是一個 npz。
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rl import caps as caps_mod
from rl import loop4_actions as A
from rl import loop4_obs as O
from rl import rules

F32 = jnp.float32

N_TILE_TOKENS = 2 * rules.N_TILES
SEG_OWN_TILES = 0
SEG_OPP_TILES = 1
SEG_UNITS = 2
SEG_PRODUCTS = 3
SEG_GLOBAL = 4
N_SEGMENTS = 5

#: 市場 head 的 logit 數，逐項對 `rl/loop4_actions.py` 的桶。
N_MARKET_LOGITS = (
    rules.N_PRODUCTS * A.N_SELL
    + rules.N_CROPS * A.N_BUY_SEED
    + len(A.BUYABLE_PRODUCTS) * A.N_BUY_PRODUCT
    + rules.N_ANIMALS * A.N_BUY_ANIMAL
    + A.N_HIRE
    + A.N_BUY_LAND
)


class Config(dict):
    """模型形狀。用 dict 是為了能直接寫進 result.json。"""

    @staticmethod
    def default(
        caps: caps_mod.Caps, d_model: int = 192, n_layers: int = 4, n_heads: int = 0
    ) -> Config:
        # d 要被 head 數整除。192 用 6（Ender），128 不能，所以用 4（任務 14b 的 B／D 變體）。
        n_heads = n_heads or (6 if d_model % 6 == 0 else 4)
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} 不能被 n_heads {n_heads} 整除")
        return Config(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=4 * d_model,
            max_units=caps.max_units,
            n_tokens=N_TILE_TOKENS + caps.max_units + rules.N_PRODUCTS + 1,
        )


def _glorot(key, shape):
    fan_in, fan_out = shape[-2], shape[-1]
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, F32, -lim, lim)


def init_params(cfg: Config, seed: int = 0) -> dict[str, Any]:
    d = cfg["d_model"]
    # 每層吃 4 把（qkv／proj／ff1／ff2），前綴 6 把、head 5 把。**下限留 64**：
    # 16 層才會超過，而改動 split 的數量會改掉所有既有權重的初始化，
    # 任務 14b 的變體 A 就對不回 il_all 了。
    n_keys = max(64, 6 + 4 * cfg["n_layers"] + 5)
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), n_keys))
    p: dict[str, Any] = {}
    # 各段自己的輸入投影：維度不同，不能共用
    for name, dim in (
        ("tile", O.TILE_DIM),
        ("unit", O.UNIT_DIM),
        ("product", O.PRODUCT_DIM),
        ("global", O.GLOBAL_DIM),
    ):
        p[f"in_{name}_w"] = _glorot(next(keys), (dim, d))
        p[f"in_{name}_b"] = jnp.zeros((d,), F32)
    # 段別 embedding ＋ 格的位置 embedding（100 格是固定的棋盤）
    p["seg_emb"] = 0.02 * jax.random.normal(next(keys), (N_SEGMENTS, d), F32)
    p["tile_pos_emb"] = 0.02 * jax.random.normal(next(keys), (rules.N_TILES, d), F32)
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
    p["task_w"] = _glorot(next(keys), (d, A.N_TASKS))
    p["task_b"] = jnp.zeros((A.N_TASKS,), F32)
    p["tqty_w"] = _glorot(next(keys), (d, A.N_QTY_BUCKETS))
    p["tqty_b"] = jnp.zeros((A.N_QTY_BUCKETS,), F32)
    p["sell_w"] = _glorot(next(keys), (d, A.N_SELL))
    p["sell_b"] = jnp.zeros((A.N_SELL,), F32)
    n_farm = (
        rules.N_CROPS * A.N_BUY_SEED
        + len(A.BUYABLE_PRODUCTS) * A.N_BUY_PRODUCT
        + rules.N_ANIMALS * A.N_BUY_ANIMAL
        + A.N_HIRE
        + A.N_BUY_LAND
    )
    p["farm_w"] = _glorot(next(keys), (d, n_farm))
    p["farm_b"] = jnp.zeros((n_farm,), F32)
    p["value_w"] = _glorot(next(keys), (d, 1))
    p["value_b"] = jnp.zeros((1,), F32)
    return p


def param_count(params: dict[str, Any]) -> int:
    return int(sum(np.prod(np.shape(v)) for v in params.values()))


def _ln(x, g, b, eps=1e-5):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps) * g + b


def _block(x, p, i, n_heads, mask, fast=None):
    """一個 transformer block。``fast`` 是訓練期的快速前向路徑（任務 25 第 4 點）：

    * ``None``（預設）＝原本的 einsum 注意力、全 f32，**逐位不變**。IL、eval、NumPy parity 走這條。
    * ``"f32"`` ＝殘差流維持 f32，注意力改 cuDNN 融合核（q/k/v 轉 bf16）。
    * ``"bf16"`` ＝整個 block 的激活與權重轉 bf16，出口轉回 f32。

    cuDNN 核要求序列長被支援（239 的反向會 `NotImplementedError`），所以 ``fast`` 時
    `forward` 會先把 token 補到 256；本函式只吃補好的 x／mask。
    """
    d = x.shape[-1]
    bf = fast == "bf16"

    def w(name):
        v = p[f"l{i}_{name}"]
        return v.astype(jnp.bfloat16) if bf else v

    x = x.astype(jnp.bfloat16) if bf else x
    h = _ln(x, w("ln1_g"), w("ln1_b"))
    qkv = h @ w("qkv_w")
    q, k, v = jnp.split(qkv, 3, axis=-1)
    b, t, _ = q.shape
    dh = d // n_heads
    shape = (b, t, n_heads, dh)
    if fast:
        # jax.nn.dot_product_attention 吃 [B, T, H, D]；cuDNN 的 mask **不做 query 維廣播**
        # （`Bias must have same seq length as QKV, got 1 and 256`），所以要攤成 [B, 1, Tq, Tk]。
        # head 維留 1 可以廣播。四層共用同一份，XLA 只會算一次。
        out = jax.nn.dot_product_attention(
            q.reshape(shape).astype(jnp.bfloat16),
            k.reshape(shape).astype(jnp.bfloat16),
            v.reshape(shape).astype(jnp.bfloat16),
            mask=jnp.broadcast_to(mask[:, None, None, :], (b, 1, t, t)),
            implementation="cudnn",
        )
        out = out.reshape(b, t, d).astype(x.dtype)
    else:
        q = q.reshape(shape).transpose(0, 2, 1, 3)
        k = k.reshape(shape).transpose(0, 2, 1, 3)
        v = v.reshape(shape).transpose(0, 2, 1, 3)
        att = jnp.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(dh)
        att = jnp.where(mask[:, None, None, :], att, -1e30)
        att = jax.nn.softmax(att, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", att, v).transpose(0, 2, 1, 3).reshape(b, t, d)
    x = x + out @ w("proj_w")
    h = _ln(x, w("ln2_g"), w("ln2_b"))
    h = jax.nn.gelu(h @ w("ff1_w") + w("ff1_b"))
    x = x + h @ w("ff2_w") + w("ff2_b")
    return x.astype(F32) if bf else x


def build_tokens(params, obs: dict[str, jax.Array], cfg: Config):
    """把四段觀測投影成 ``[B, T, d]``，並回傳 ``[B, T]`` 的有效 token 遮罩。"""
    tiles = obs["tiles"]  # [B, 2, 100, TILE_DIM]
    b = tiles.shape[0]
    d = cfg["d_model"]
    tile_tok = tiles @ params["in_tile_w"] + params["in_tile_b"]  # [B, 2, 100, d]
    tile_tok = tile_tok + params["tile_pos_emb"][None, None, :, :]
    tile_tok = (
        tile_tok
        + jnp.stack([params["seg_emb"][SEG_OWN_TILES], params["seg_emb"][SEG_OPP_TILES]])[
            None, :, None, :
        ]
    )
    tile_tok = tile_tok.reshape(b, N_TILE_TOKENS, d)

    units = obs["units"][:, 0]  # 只有自己的單位會被指派工作
    unit_tok = units @ params["in_unit_w"] + params["in_unit_b"]
    unit_tok = unit_tok + params["seg_emb"][SEG_UNITS]
    prod_tok = obs["products"] @ params["in_product_w"] + params["in_product_b"]
    prod_tok = prod_tok + params["seg_emb"][SEG_PRODUCTS]
    glob_tok = obs["global"] @ params["in_global_w"] + params["in_global_b"]
    glob_tok = (glob_tok + params["seg_emb"][SEG_GLOBAL])[:, None, :]

    x = jnp.concatenate([tile_tok, unit_tok, prod_tok, glob_tok], axis=1)
    alive = obs["units"][:, 0, :, 2] > 0.5  # UNIT_DIM 的第 3 維是 alive
    mask = jnp.concatenate(
        [
            jnp.ones((b, N_TILE_TOKENS), bool),
            alive,
            jnp.ones((b, rules.N_PRODUCTS + 1), bool),
        ],
        axis=1,
    )
    return x, mask


#: `fast` 路徑把 token 補到這個數的倍數：cuDNN 融合注意力的反向不支援序列長 239
#: （`NotImplementedError: Unsupported sequence length Q 239, KV 239`），補到 256 就能跑。
FAST_PAD_MULTIPLE = 256


def pad_tokens_for_fast(x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """把 ``[B, T, d]`` 補到 `FAST_PAD_MULTIPLE` 的倍數，pad token **全遮罩**且插在 glob 之前。

    插在倒數第二個位置，所以 `x[:, :N_TILES]`（自己的格）、產品段的切片位置、`x[:, -1]`（glob）
    三個讀出點的索引完全不變——head 不必跟著改。
    """
    t = x.shape[1]
    n = -(-t // FAST_PAD_MULTIPLE) * FAST_PAD_MULTIPLE
    if n == t:
        return x, mask
    pad = n - t
    zx = jnp.zeros((x.shape[0], pad, x.shape[2]), x.dtype)
    zm = jnp.zeros((mask.shape[0], pad), bool)
    return (
        jnp.concatenate([x[:, :-1], zx, x[:, -1:]], axis=1),
        jnp.concatenate([mask[:, :-1], zm, mask[:, -1:]], axis=1),
    )


def forward(
    params,
    obs: dict[str, jax.Array],
    cfg: Config,
    remat: bool = False,
    fast: str | None = None,
) -> dict[str, jax.Array]:
    """``remat=True`` 時每個 block 包 `jax.checkpoint`：反向重算激活，換掉 attention 的常駐。

    只有 PPO 的 update 用得到（任務 23：`[1024, 6, 239, 239]` float32 ＝ 1.40 GiB／層，
    4 層存給 backward 就 5.6 GiB，實測 grad 峰值 12.13 GiB，13.53 GiB 的預算塞不下其他東西）。
    推論與 IL 路徑維持 `remat=False`，數值完全不變——`jax.checkpoint` 只改反向怎麼算，不改前向。

    ``fast``（任務 25 第 4 點）＝ ``None``／``"f32"``／``"bf16"``，見 `_block`。**預設 `None` 時
    這個函式與改動前逐位相同**：IL 訓練、`scripts/loop4_eval.py`、NumPy 部署 parity 都走預設路徑。
    """
    x, mask = build_tokens(params, obs, cfg)
    if fast:
        x, mask = pad_tokens_for_fast(x, mask)
    block = jax.checkpoint(_block, static_argnums=(2, 3, 5)) if remat else _block
    for i in range(cfg["n_layers"]):
        x = block(x, params, i, cfg["n_heads"], mask, fast)
    x = _ln(x, params["ln_f_g"], params["ln_f_b"])
    own = x[:, : rules.N_TILES]
    prod = x[:, N_TILE_TOKENS + cfg["max_units"] : N_TILE_TOKENS + cfg["max_units"] + 9]
    glob = x[:, -1]
    farm = glob @ params["farm_w"] + params["farm_b"]
    o = 0

    def take(n):
        nonlocal o
        out = farm[:, o : o + n]
        o += n
        return out

    return {
        "task": own @ params["task_w"] + params["task_b"],  # [B, 100, 17]
        "tile_qty": own @ params["tqty_w"] + params["tqty_b"],  # [B, 100, 4]
        "sell": prod @ params["sell_w"] + params["sell_b"],  # [B, 9, N_SELL]
        "buy_seed": take(rules.N_CROPS * A.N_BUY_SEED).reshape(-1, rules.N_CROPS, A.N_BUY_SEED),
        "buy_product": take(len(A.BUYABLE_PRODUCTS) * A.N_BUY_PRODUCT).reshape(
            -1, len(A.BUYABLE_PRODUCTS), A.N_BUY_PRODUCT
        ),
        "buy_animal": take(rules.N_ANIMALS * A.N_BUY_ANIMAL).reshape(
            -1, rules.N_ANIMALS, A.N_BUY_ANIMAL
        ),
        "hire": take(A.N_HIRE),
        "buy_land": take(A.N_BUY_LAND),
        "value": (glob @ params["value_w"] + params["value_b"])[:, 0],
    }
