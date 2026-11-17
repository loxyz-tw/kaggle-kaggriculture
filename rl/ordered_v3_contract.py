"""ordered_actions_v3 的 stdlib 契約：版本、shape、leaf 338406、數值域與錯誤碼。

本檔**只**依賴 Python 標準庫與 `rl.rules` 常數（op／item 名稱與官方預設
``maxMarketOrdersPerTurn``）。不 import JAX、NumPy、舊 bins、舊 encoder。
決策陣列以 duck-typed ``.shape``／``.dtype`` 驗收，供 codec／policy／data／
NumPy 共用同一份規格。
"""

from __future__ import annotations

from typing import Any, NamedTuple

from rl import rules

# ==============================================================================
# 版本／metadata（dataset 與 checkpoint；舊 loader 已拒的保留 prefix）
# ==============================================================================
DATA_SCHEMA_VERSION = 3
POLICY_VERSION = "ordered_actions_v3"
OBS_VERSION = "v2"
PACKAGE_KIND_DATA = "local-action-v3-test"
PACKAGE_KIND_CHECKPOINT = "untrained-test"
META_FORMAT_VERSION = 1
WIDTH_CONFIG = "W1-ordered-v3"
QTY_CODEC = "u31-be4"
RUNTIME_READY = False

CHECKPOINT_META_KEY = "__definitions_v2_ordered_v3_meta__"
DATASET_META_KEY = "__definitions_v2_ordered_v3_data__"

OBS_DIM_TILE = 15
OBS_DIM_SCALAR = 594
OBS_DIM_UNIT = 49
N_TILES = 200

MAX_UNITS = 29
MAX_ORDERS = 10
MAX_MARKET_ORDERS_PER_TURN_REQUIRED = 10
N_UNIT_OPS = rules.N_UNIT_OPS  # 18
N_CROPS = rules.N_CROPS  # 5
N_ITEMS = rules.N_ITEMS  # 12
N_MARKET_OPS = rules.N_MARKET_OPS  # 7；0=NONE 在本模型為 EOS
N_QTY_BYTES = 4
BYTE_CLASSES = 256
QTY_HIGH_BYTE_MAX = 127
U31_MIN = 1
U31_MAX = 2147483647  # 2**31 - 1；int32 正域，不是官方數量帽

UNIT_OP_NAMES = rules.UNIT_OP_NAMES
UNIT_OP_INDEX = rules.UNIT_OP_INDEX
MARKET_OP_NAMES = rules.MARKET_OP_NAMES
MARKET_OP_INDEX = rules.MARKET_OP_INDEX
ITEMS = rules.ITEMS
ITEM_INDEX = rules.ITEM_INDEX
CROPS = rules.CROPS
CROP_INDEX = {name: i for i, name in enumerate(CROPS)}
ANIMALS = rules.ANIMALS
PRODUCTS = rules.PRODUCTS
if tuple(ITEM_INDEX[c] for c in CROPS) != tuple(range(N_CROPS)):
    raise RuntimeError("CROP_INDEX 必須與 ITEM_INDEX 作物前綴同值；不得依賴隱含 CROPS==ITEMS[:5]")

OP_PASS = rules.OP_PASS
OP_PICKUP = rules.OP_PICKUP
OP_PLANT = rules.OP_PLANT
OP_PLACE = rules.OP_PLACE
MO_NONE = rules.MO_NONE  # 新語義：EOS
MO_BUY_SEED = rules.MO_BUY_SEED
MO_BUY_PRODUCT = rules.MO_BUY_PRODUCT
MO_BUY_ANIMAL = rules.MO_BUY_ANIMAL
MO_SELL = rules.MO_SELL
MO_HIRE = rules.MO_HIRE
MO_BUY_LAND = rules.MO_BUY_LAND
EOS_OP = MO_NONE

# 市場 item 語法域（只限官方 _quote_order；不以 money／shed 再裁）。
MARKET_ITEM_SUPPORT: dict[int, frozenset[int]] = {
    MO_SELL: frozenset(range(9)),  # PRODUCTS 0..8
    MO_BUY_PRODUCT: frozenset({rules.I_WHEAT, rules.I_FERTILIZER}),
    MO_BUY_SEED: frozenset(range(5)),
    MO_BUY_ANIMAL: frozenset({9, 10, 11}),
    MO_HIRE: frozenset(),
    MO_BUY_LAND: frozenset(),
    MO_NONE: frozenset(),
}

# unit C 保留的參數；其餘 op 只留 [op]。
UNIT_NEEDS_CROP = frozenset({OP_PLANT})
UNIT_NEEDS_ITEM_QTY = frozenset({OP_PICKUP, OP_PLACE})

# ==============================================================================
# 錯誤碼（host 明拒；不得溢位截量後冒稱遊戲非法）
# ==============================================================================
E_VERSION_CONFLICT = "VERSION_CONFLICT"
E_NONCANONICAL_INPUT = "NONCANONICAL_INPUT"
E_NUMERIC_DOMAIN_GAP = "NUMERIC_DOMAIN_GAP"
E_ILLEGAL_ACTION = "ILLEGAL_ACTION"
E_UNSUPPORTED = "UNSUPPORTED"
E_INVALID_SHAPE = "INVALID_SHAPE"
E_INVALID_DTYPE = "INVALID_DTYPE"
E_INVALID_SEAT = "INVALID_SEAT"
E_INVALID_N_ALIVE = "INVALID_N_ALIVE"
E_CONFIG_ORDERS = "CONFIG_ORDERS_NOT_DEFAULT"
E_MASK_EMPTY = "MASK_EMPTY"
E_NO_SUPPORT = "NO_SUPPORT"
E_LEAF_COUNT = "LEAF_PARAM_TOTAL_MISMATCH"
E_UNKNOWN_PACKAGE = "UNKNOWN_PACKAGE"
E_UNKNOWN_WIDTH = "UNKNOWN_WIDTH"
E_UNKNOWN_ACTOR_ARCH = "UNKNOWN_ACTOR_ARCH"
E_RUNTIME_UNREADY = "RUNTIME_UNREADY"
E_LEGACY_VERSION = "LEGACY_VERSION_REJECTED"
E_POLICY_COMPILE_BUDGET_STOP = "POLICY_COMPILE_BUDGET_STOP"
E_POLICY_TEST_BUDGET_STOP = "POLICY_TEST_BUDGET_STOP"
E_NETWORK_ISOLATION_UNAVAILABLE = "NETWORK_ISOLATION_UNAVAILABLE"
E_JAX_CACHE_STALE = "JAX_CACHE_STALE"
E_DUPLICATE_NPZ_KEY = "DUPLICATE_NPZ_KEY"
E_NONFINITE = "NONFINITE_WEIGHT"
E_SHA_MISMATCH = "WEIGHTS_SHA256_MISMATCH"
E_TOKEN_SEMANTICS = "TOKEN_SEMANTICS"
E_MARKET_TRUNCATION = "MARKET_SILENT_TRUNCATION_FORBIDDEN"
E_EXPLICIT_NONE = "EXPLICIT_NONE_NOT_MISSING"
E_PROJECTION_UNUSABLE = "PROJECTION_UNUSABLE"

OFFICIAL_ENGINE_DIR = "docs/official/kaggriculture-1.32.7"
ENGINE_VERSION = rules.ENGINE_VERSION


class OrderedV3Error(ValueError):
    """契約／codec 的具名失敗。``code`` 是上列常數之一。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.detail = message


# ==============================================================================
# 五桶（component 分母；主桶唯一順序）
# ==============================================================================
BUCKET_ILLEGAL = "illegal"
BUCKET_NONCANONICAL = "noncanonical"
BUCKET_NUMERIC = "numeric"
BUCKET_C_DIFF = "c_diff"
BUCKET_EXACT = "exact"
PRIMARY_BUCKET_ORDER = (
    BUCKET_ILLEGAL,
    BUCKET_NONCANONICAL,
    BUCKET_NUMERIC,
    BUCKET_C_DIFF,
    BUCKET_EXACT,
)

# 原因碼 → 桶。多原因時取 PRIMARY_BUCKET_ORDER 中最先出現者。
REASON_TO_BUCKET: dict[str, str] = {
    "UNKNOWN_OP": BUCKET_ILLEGAL,
    "UNKNOWN_ITEM": BUCKET_ILLEGAL,
    "UNKNOWN_CROP": BUCKET_ILLEGAL,
    "ILLEGAL_ITEM_FOR_OP": BUCKET_ILLEGAL,
    "NON_LIST": BUCKET_ILLEGAL,
    "EMPTY_ACTION": BUCKET_ILLEGAL,
    "MISSING_ITEM": BUCKET_ILLEGAL,
    "MISSING_CROP": BUCKET_ILLEGAL,
    "MISSING_QTY": BUCKET_ILLEGAL,
    "UNITS_EXCEED_N_ALIVE": BUCKET_ILLEGAL,
    "ORDERS_EXCEED_MAX": BUCKET_ILLEGAL,
    "DEAD_SLOT_IN_JSON": BUCKET_ILLEGAL,
    "EOS_IN_JSON": BUCKET_ILLEGAL,
    "NONCANONICAL_INPUT": BUCKET_NONCANONICAL,
    "QTY_NOT_PYTHON_INT": BUCKET_NONCANONICAL,
    "NUMERIC_DOMAIN_GAP": BUCKET_NUMERIC,
    "QTY_OMITTED_DEFAULT_1": BUCKET_C_DIFF,
    "IGNORED_SUFFIX": BUCKET_C_DIFF,
    "MISSING_FARMER_DEFAULT_PASS": BUCKET_C_DIFF,
    "MISSING_HANDS_DEFAULT_EMPTY": BUCKET_C_DIFF,
    "MISSING_MARKET_DEFAULT_EMPTY": BUCKET_C_DIFF,
    "PAD_PASS_TO_N_ALIVE": BUCKET_C_DIFF,
    "EXPLICIT_NONE": BUCKET_ILLEGAL,
    "DEAD_SLOT_SENTINEL": BUCKET_ILLEGAL,
    "INACTIVE_SENTINEL": BUCKET_ILLEGAL,
    "PADDING_NOT_ZERO": BUCKET_ILLEGAL,
    "L_MISMATCH": BUCKET_ILLEGAL,
    "TRUE_ORDER_IS_EOS": BUCKET_ILLEGAL,
    "MARKET_COMPACTED": BUCKET_ILLEGAL,
    # 安全 PASS 占位是**表示層改寫**（為安全進 tensor），
    # 不代表原遊戲動作非法：numeric／noncanonical 原因因此保留各自主桶，
    # 真正非法原因仍使主桶為 illegal。data 層另以 reason 含
    # SAFETY_PLACEHOLDER 拒監督，不靠主桶把占位擋在 loss 外。
    "SAFETY_PLACEHOLDER": BUCKET_C_DIFF,
    "IGNORED_KEY": BUCKET_C_DIFF,
    "EXACT": BUCKET_EXACT,
}

# normalization_diffs 每筆必填；缺欄不得用 raw 代替這三個具名欄位。
NORMALIZATION_DIFF_REQUIRED_KEYS = (
    "path",
    "reason",
    "qty_omitted",
    "original_length",
    "ignored_suffix",
)


def primary_bucket(reasons: tuple[str, ...] | list[str]) -> str:
    """「非法→非正典→numeric→C差異→exact」唯一主桶。"""
    present = {REASON_TO_BUCKET.get(r, BUCKET_ILLEGAL) for r in reasons}
    if not present:
        return BUCKET_EXACT
    for bucket in PRIMARY_BUCKET_ORDER:
        if bucket in present:
            return bucket
    return BUCKET_EXACT


# ==============================================================================
# DecisionV3 形狀
# ==============================================================================
DECISION_V3_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("unit_verb", (MAX_UNITS,)),
    ("unit_crop", (MAX_UNITS,)),
    ("unit_item", (MAX_UNITS,)),
    ("unit_qty_bytes", (MAX_UNITS, N_QTY_BYTES)),
    ("market_op", (MAX_ORDERS,)),
    ("market_item", (MAX_ORDERS,)),
    ("market_qty_bytes", (MAX_ORDERS, N_QTY_BYTES)),
    ("market_length", ()),
)


class DecisionV3(NamedTuple):
    """單座決策（batch 在第 0 軸）。dtype 一律 int32。

    unit 不需參數時 crop／item／qty bytes 為 0；死 slot 強制 PASS。
    市場 L<10 時第 L 格為第一次 EOS，其後全 0；L=10 沒有 EOS 決策。
    """

    unit_verb: Any
    unit_crop: Any
    unit_item: Any
    unit_qty_bytes: Any
    market_op: Any
    market_item: Any
    market_qty_bytes: Any
    market_length: Any


# SimAction 欄位名（codec 以 NumPy 填；不在本檔 import jax.SimAction）。
SIMACTION_FIELD_NAMES = (
    "unit_op",
    "unit_arg",
    "unit_qty",
    "order_op",
    "order_item",
    "order_qty",
)


# ==============================================================================
# W1 leaf 338406（初始化 key id 0..17；實測總數不同即 STOP）
# ==============================================================================
class LeafSpec(NamedTuple):
    key: str
    key_id: int
    kind: str  # "dense" | "embedding" | "projection"
    w_shape: tuple[int, ...]
    has_bias: bool
    n_params: int
    is_output_head: bool  # Dense 輸出 head：w 再乘 0.01


LEAF_KINDS = ("dense", "embedding", "projection")
KIND_DENSE = "dense"
KIND_EMBEDDING = "embedding"
KIND_PROJECTION = "projection"


def _n_params(w_shape: tuple[int, ...], has_bias: bool) -> int:
    n = 1
    for d in w_shape:
        n *= d
    if has_bias:
        n += w_shape[-1]
    return n


def _leaf(
    key: str,
    key_id: int,
    kind: str,
    w_shape: tuple[int, ...],
    *,
    has_bias: bool,
    is_output_head: bool = False,
) -> LeafSpec:
    if kind not in LEAF_KINDS:
        raise RuntimeError(f"{E_INVALID_SHAPE}: 未知 leaf kind {kind!r}")
    if kind == KIND_PROJECTION and has_bias:
        raise RuntimeError(f"{E_INVALID_SHAPE}: projection {key!r} 必須是裸 array、不得有 bias")
    return LeafSpec(
        key=key,
        key_id=key_id,
        kind=kind,
        w_shape=w_shape,
        has_bias=has_bias,
        n_params=_n_params(w_shape, has_bias),
        is_output_head=is_output_head,
    )


# tile_embed 表列為 15×48＝768，含 b[48]，故契約以 Dense 計（四個無 bias embedding
# 是 pos／op／item／byte，對齊「四個 embedding（含 pos）normal×0.02」）。
LEAF_SPECS: tuple[LeafSpec, ...] = (
    _leaf("tile_embed", 0, "dense", (OBS_DIM_TILE, 48), has_bias=True),
    _leaf("pos_embed", 1, "embedding", (N_TILES, 48), has_bias=False),
    _leaf("scalar", 2, "dense", (OBS_DIM_SCALAR, 128), has_bias=True),
    _leaf("trunk1", 3, "dense", (320, 256), has_bias=True),
    _leaf("trunk2", 4, "dense", (256, 256), has_bias=True),
    _leaf("unit_shared", 5, "dense", (353, 128), has_bias=True),
    _leaf("unit_head_verb", 6, "dense", (128, N_UNIT_OPS), has_bias=True, is_output_head=True),
    _leaf("unit_head_crop", 7, "dense", (128, N_CROPS), has_bias=True, is_output_head=True),
    _leaf("unit_head_item", 8, "dense", (128, N_ITEMS), has_bias=True, is_output_head=True),
    _leaf("market_init", 9, "dense", (256, 64), has_bias=True),
    _leaf("unit_qty_init", 10, "dense", (158, 64), has_bias=True),
    _leaf("prefix_step", 11, "dense", (80, 64), has_bias=True),
    _leaf("op_embed", 12, "embedding", (N_MARKET_OPS, 16), has_bias=False),
    _leaf("item_embed", 13, "embedding", (N_ITEMS, 16), has_bias=False),
    _leaf("byte_embed", 14, "embedding", (BYTE_CLASSES, 16), has_bias=False),
    _leaf("market_op_out", 15, "dense", (64, N_MARKET_OPS), has_bias=True, is_output_head=True),
    _leaf("market_item_out", 16, "dense", (64, N_ITEMS), has_bias=True, is_output_head=True),
    _leaf("byte_out", 17, "dense", (64, BYTE_CLASSES), has_bias=True, is_output_head=True),
)

PARAM_TOTAL = 338406
LEAF_BY_KEY: dict[str, LeafSpec] = {spec.key: spec for spec in LEAF_SPECS}
OUTPUT_HEAD_KEYS = frozenset(spec.key for spec in LEAF_SPECS if spec.is_output_head)
EMBEDDING_KEYS = frozenset(spec.key for spec in LEAF_SPECS if spec.kind == "embedding")

if sum(spec.n_params for spec in LEAF_SPECS) != PARAM_TOTAL:
    raise RuntimeError(
        f"{E_LEAF_COUNT}: formula {sum(spec.n_params for spec in LEAF_SPECS)} != {PARAM_TOTAL}"
    )
if tuple(spec.key_id for spec in LEAF_SPECS) != tuple(range(18)):
    raise RuntimeError(f"{E_LEAF_COUNT}: leaf key_id 必須是固定 0..17")

ACTOR_ARCH_BASELINE = "baseline"
ACTOR_ARCH_UNIT_SPATIAL_V1 = "unit_spatial_v1"
ACTOR_ARCHS = (ACTOR_ARCH_BASELINE, ACTOR_ARCH_UNIT_SPATIAL_V1)
SPATIAL_PARAM_TOTAL = 356870
SPATIAL_WIDTH_CONFIG = "W1-ordered-v3-unit-spatial-v1"
UNIT_IN_DIM = 353
SPATIAL_D = 32
UNIT_SHARED_OUT = 128
SPATIAL_LEAF_SPECS: tuple[LeafSpec, ...] = LEAF_SPECS + (
    _leaf("spatial_q", 18, KIND_PROJECTION, (UNIT_IN_DIM, SPATIAL_D), has_bias=False),
    _leaf("spatial_k", 19, KIND_PROJECTION, (48, SPATIAL_D), has_bias=False),
    _leaf("spatial_v", 20, KIND_PROJECTION, (48, SPATIAL_D), has_bias=False),
    _leaf("spatial_out", 21, KIND_PROJECTION, (SPATIAL_D, UNIT_SHARED_OUT), has_bias=False),
)
SPATIAL_LEAF_BY_KEY: dict[str, LeafSpec] = {spec.key: spec for spec in SPATIAL_LEAF_SPECS}
PROJECTION_KEYS = frozenset(spec.key for spec in SPATIAL_LEAF_SPECS if spec.kind == KIND_PROJECTION)

if sum(spec.n_params for spec in SPATIAL_LEAF_SPECS) != SPATIAL_PARAM_TOTAL:
    raise RuntimeError(
        f"{E_LEAF_COUNT}: spatial formula "
        f"{sum(spec.n_params for spec in SPATIAL_LEAF_SPECS)} != {SPATIAL_PARAM_TOTAL}"
    )
if tuple(spec.key_id for spec in SPATIAL_LEAF_SPECS) != tuple(range(22)):
    raise RuntimeError(f"{E_LEAF_COUNT}: spatial leaf key_id 必須是固定 0..21")


def require_actor_arch(value: object) -> str:
    if type(value) is not str or value not in ACTOR_ARCHS:
        raise OrderedV3Error(
            E_UNKNOWN_ACTOR_ARCH,
            f"actor_arch {value!r} 不是 {ACTOR_ARCH_BASELINE!r} 或 {ACTOR_ARCH_UNIT_SPATIAL_V1!r}",
        )
    return value


def parse_checkpoint_actor_arch(meta: object) -> str:
    """先解析 metadata actor_arch。缺欄＝baseline；其餘具名拒。"""
    if not isinstance(meta, dict):
        raise OrderedV3Error(E_INVALID_SHAPE, "metadata 必須是物件")
    if "actor_arch" not in meta:
        return ACTOR_ARCH_BASELINE
    return require_actor_arch(meta["actor_arch"])


def leaf_specs_for(actor_arch: object = ACTOR_ARCH_BASELINE) -> tuple[LeafSpec, ...]:
    arch = require_actor_arch(actor_arch)
    if arch == ACTOR_ARCH_UNIT_SPATIAL_V1:
        return SPATIAL_LEAF_SPECS
    return LEAF_SPECS


def param_total_for(actor_arch: object = ACTOR_ARCH_BASELINE) -> int:
    arch = require_actor_arch(actor_arch)
    if arch == ACTOR_ARCH_UNIT_SPATIAL_V1:
        return SPATIAL_PARAM_TOTAL
    return PARAM_TOTAL


def width_config_for(actor_arch: object = ACTOR_ARCH_BASELINE) -> str:
    arch = require_actor_arch(actor_arch)
    if arch == ACTOR_ARCH_UNIT_SPATIAL_V1:
        return SPATIAL_WIDTH_CONFIG
    return WIDTH_CONFIG


def classify_actor_arch_keys(keys: object) -> str:
    if not isinstance(keys, (set, frozenset)):
        keys = set(keys)
    base = {spec.key for spec in LEAF_SPECS}
    cand = {spec.key for spec in SPATIAL_LEAF_SPECS}
    extra_spatial = cand - base
    if keys == base:
        return ACTOR_ARCH_BASELINE
    if keys == cand:
        return ACTOR_ARCH_UNIT_SPATIAL_V1
    present_spatial = keys & extra_spatial
    missing_base = base - keys
    extra = keys - cand
    raise OrderedV3Error(
        E_LEAF_COUNT,
        f"參數樹不是完整 {ACTOR_ARCH_BASELINE} 或 {ACTOR_ARCH_UNIT_SPATIAL_V1}："
        f"缺舊葉 {sorted(missing_base)}、新葉 {sorted(present_spatial)}、多餘 {sorted(extra)}",
    )


def identify_params_arch(params: object) -> str:
    """以完整鍵集合辨識兩種固定樹，再核 shape；不是靠總數猜。"""
    if not isinstance(params, dict):
        raise OrderedV3Error(E_INVALID_SHAPE, f"params 必須是 dict，得到 {type(params).__name__}")
    arch = classify_actor_arch_keys(set(params))
    require_params_match_arch(params, arch)
    return arch


def require_params_match_arch(params: object, actor_arch: object) -> str:
    arch = require_actor_arch(actor_arch)
    if not isinstance(params, dict):
        raise OrderedV3Error(
            E_INVALID_SHAPE,
            f"actor_arch={arch!r} 的 params 必須是 dict，得到 {type(params).__name__}",
        )
    specs = leaf_specs_for(arch)
    want = {spec.key for spec in specs}
    got = set(params)
    if got != want:
        raise OrderedV3Error(
            E_LEAF_COUNT,
            f"actor_arch={arch!r} leaf 集合不符：缺 {sorted(want - got)}、多 {sorted(got - want)}",
        )
    for spec in specs:
        node = params[spec.key]
        if spec.kind == KIND_DENSE:
            if not isinstance(node, dict) or "w" not in node or "b" not in node:
                raise OrderedV3Error(
                    E_INVALID_SHAPE,
                    f"actor_arch={arch!r} 的 {spec.key!r} 必須是含 w/b 的 dense 節點",
                )
            if _as_shape(node["w"]) != spec.w_shape:
                raise OrderedV3Error(
                    E_INVALID_SHAPE,
                    f"actor_arch={arch!r} 的 {spec.key!r}.w shape "
                    f"{_as_shape(node['w'])} != {spec.w_shape}",
                )
            if _as_shape(node["b"]) != (spec.w_shape[-1],):
                raise OrderedV3Error(
                    E_INVALID_SHAPE,
                    f"actor_arch={arch!r} 的 {spec.key!r}.b shape "
                    f"{_as_shape(node['b'])} != {(spec.w_shape[-1],)}",
                )
        elif spec.kind in (KIND_EMBEDDING, KIND_PROJECTION):
            if isinstance(node, dict):
                raise OrderedV3Error(
                    E_INVALID_SHAPE,
                    f"actor_arch={arch!r} 的 {spec.key!r} kind={spec.kind} 必須是裸 array",
                )
            if _as_shape(node) != spec.w_shape:
                raise OrderedV3Error(
                    E_INVALID_SHAPE,
                    f"actor_arch={arch!r} 的 {spec.key!r} shape "
                    f"{_as_shape(node)} != {spec.w_shape}",
                )
        else:
            raise OrderedV3Error(
                E_INVALID_SHAPE,
                f"actor_arch={arch!r} 的 {spec.key!r} 未知 kind {spec.kind!r}",
            )
    return arch


# decode_stream mode、巢狀 scan 與 JAX cache 路徑（policy／guard）
DECODE_STREAM_MODES = ("sample", "greedy", "teacher")
NESTED_SCAN_MARKET_SLOTS = MAX_ORDERS
NESTED_SCAN_BYTE_STEPS = N_QTY_BYTES
JAX_CACHE_DIR_TEMPLATE = "tmp/{attempt_id}/jax-cache"  # 相對 OUTDIR
CHECKPOINT_META_REQUIRED_KEYS = (
    "meta_format_version",
    "policy_version",
    "obs_version",
    "package_kind",
    "width_config",
    "dims",
    "max_units",
    "max_orders",
    "qty_codec",
    "param_total",
    "runtime_ready",
    "source_head",
    "source_closure_sha256",
    "leaf_manifest",
    "weights_sha256",
)
SOURCE_HEAD_HEX_LEN = 40
SOURCE_CLOSURE_HEX_LEN = 64
RNG_STREAM_VERB_CROP = 0
RNG_STREAM_UNIT_ITEM = 1
RNG_STREAM_UNIT_QTY = 2
RNG_STREAM_MARKET = 3
MARKET_FIELD_OP = 0
MARKET_FIELD_ITEM = 1
MARKET_FIELD_BYTE0 = 2  # byte j 的 field = 2+j


def unit_qty_fold_id(unit_slot: int, byte_j: int) -> int:
    """``fold_in(R[2], 4*u+j)`` 的整數座標。"""
    if not (isinstance(unit_slot, int) and type(unit_slot) is int and 0 <= unit_slot < MAX_UNITS):
        raise OrderedV3Error(E_INVALID_N_ALIVE, f"unit_slot {unit_slot!r} 不在 0..{MAX_UNITS - 1}")
    if not (isinstance(byte_j, int) and type(byte_j) is int and 0 <= byte_j < N_QTY_BYTES):
        raise OrderedV3Error(E_INVALID_SHAPE, f"byte_j {byte_j!r} 不在 0..{N_QTY_BYTES - 1}")
    return N_QTY_BYTES * unit_slot + byte_j


def market_fold_id(slot: int, field: int) -> int:
    """``fold_in(R[3], 6*s+f)``；f=0 op、1 item、2..5 四 byte。"""
    if not (isinstance(slot, int) and type(slot) is int and 0 <= slot < MAX_ORDERS):
        raise OrderedV3Error(E_INVALID_SHAPE, f"market slot {slot!r} 不在 0..{MAX_ORDERS - 1}")
    if not (isinstance(field, int) and type(field) is int and 0 <= field < 6):
        raise OrderedV3Error(E_INVALID_SHAPE, f"market field {field!r} 不在 0..5")
    return 6 * slot + field


def rng_coordinate_table() -> tuple[tuple[str, int, int], ...]:
    """固定 seed 展開用的具名座標表（不含 JAX key）。

    每列 (name, stream, fold_id)。
    """
    rows: list[tuple[str, int, int]] = []
    # verb/crop：既有 split(R[0], 58).reshape(29, 2, -1)；座標以 (u, head) 記錄。
    for u in range(MAX_UNITS):
        rows.append((f"unit{u}.verb", RNG_STREAM_VERB_CROP, 2 * u))
        rows.append((f"unit{u}.crop", RNG_STREAM_VERB_CROP, 2 * u + 1))
        rows.append((f"unit{u}.item", RNG_STREAM_UNIT_ITEM, u))
        for j in range(N_QTY_BYTES):
            rows.append((f"unit{u}.qty{j}", RNG_STREAM_UNIT_QTY, unit_qty_fold_id(u, j)))
    for s in range(MAX_ORDERS):
        rows.append((f"market{s}.op", RNG_STREAM_MARKET, market_fold_id(s, MARKET_FIELD_OP)))
        rows.append((f"market{s}.item", RNG_STREAM_MARKET, market_fold_id(s, MARKET_FIELD_ITEM)))
        for j in range(N_QTY_BYTES):
            rows.append(
                (f"market{s}.qty{j}", RNG_STREAM_MARKET, market_fold_id(s, MARKET_FIELD_BYTE0 + j))
            )
    return tuple(rows)


# ==============================================================================
# 嚴格驗證
# ==============================================================================
def is_strict_int(value: object) -> bool:
    """Python ``int`` 且**不是** ``bool``（bool 是 int 子類，必須排除）。"""
    return type(value) is int


def require_seat(seat: object) -> int:
    if not is_strict_int(seat) or seat not in (0, 1):
        raise OrderedV3Error(E_INVALID_SEAT, f"seat 必須是嚴格 int 0 或 1，得到 {seat!r}")
    return seat


def require_n_alive(n_alive: object) -> int:
    if not is_strict_int(n_alive) or not 1 <= n_alive <= MAX_UNITS:
        raise OrderedV3Error(
            E_INVALID_N_ALIVE,
            f"n_alive 必須是嚴格 int 1..{MAX_UNITS}（實際在場，含 farmer），得到 {n_alive!r}",
        )
    return n_alive


def require_hires_capacity(hires_seat0: object, hires_seat1: object) -> tuple[int, int]:
    """兩座 ``hires_today`` 都必須可表示為 ``n_alive = hires+1 ∈ 1..MAX_UNITS``。

    超容量具名拒收，不以 ``min`` 裁切。不改 legacy ``unit_alive``。
    """
    out: list[int] = []
    for name, hires in (("seat0", hires_seat0), ("seat1", hires_seat1)):
        if not is_strict_int(hires):
            raise OrderedV3Error(
                E_INVALID_N_ALIVE,
                f"{name} hires_today 必須是嚴格 int，得到 {hires!r}",
            )
        n_alive = hires + 1
        if n_alive < 1 or n_alive > MAX_UNITS:
            raise OrderedV3Error(
                E_INVALID_N_ALIVE,
                f"{name} hires_today={hires} → n_alive={n_alive} 超過容量 {MAX_UNITS}；不截斷",
            )
        out.append(hires)
    return out[0], out[1]


def require_default_max_orders(max_market_orders_per_turn: object) -> int:
    """只接受官方預設 10；其他 configuration 明拒，不截斷。"""
    default = int(rules.DEFAULT_CONFIG["maxMarketOrdersPerTurn"])
    if default != MAX_MARKET_ORDERS_PER_TURN_REQUIRED:
        raise OrderedV3Error(
            E_VERSION_CONFLICT,
            f"rules.DEFAULT_CONFIG maxMarketOrdersPerTurn={default} != "
            f"{MAX_MARKET_ORDERS_PER_TURN_REQUIRED}",
        )
    if not is_strict_int(max_market_orders_per_turn) or (
        max_market_orders_per_turn != MAX_MARKET_ORDERS_PER_TURN_REQUIRED
    ):
        raise OrderedV3Error(
            E_CONFIG_ORDERS,
            f"maxMarketOrdersPerTurn 必須是 {MAX_MARKET_ORDERS_PER_TURN_REQUIRED}，"
            f"得到 {max_market_orders_per_turn!r}；"
            f"其他 configuration 不截斷、不算可表示",
        )
    return MAX_MARKET_ORDERS_PER_TURN_REQUIRED


def require_u31_qty(q: object) -> int:
    """正典數量：嚴格 Python int、1..2^31-1。否則具名分桶錯誤。"""
    if type(q) is bool or not is_strict_int(q):
        raise OrderedV3Error(E_NONCANONICAL_INPUT, f"數量不是正典 Python int：{type(q).__name__}")
    if q < U31_MIN or q > U31_MAX:
        raise OrderedV3Error(
            E_NUMERIC_DOMAIN_GAP,
            f"數量 {q} 不在 {U31_MIN}..{U31_MAX}（int32 正域；不得截量）",
        )
    return q


def u31_byte_ok(bytes_4: object) -> bool:
    """只接受長度 4 的嚴格 Python int；浮點／字串／bool 在轉型前即失敗。"""
    if not isinstance(bytes_4, (tuple, list)) or len(bytes_4) != N_QTY_BYTES:
        return False
    if not all(is_strict_int(x) for x in bytes_4):
        return False
    b0, b1, b2, b3 = bytes_4[0], bytes_4[1], bytes_4[2], bytes_4[3]
    if not 0 <= b0 <= QTY_HIGH_BYTE_MAX:
        return False
    if not all(0 <= b <= 255 for b in (b1, b2, b3)):
        return False
    if b0 == 0 and b1 == 0 and b2 == 0 and b3 == 0:
        return False
    return True


def u31_bytes_are_sentinel_zero(bytes_4: object) -> bool:
    """未使用的 qty：四 byte 皆為嚴格 int 0。型別不對則不是 sentinel。"""
    if not isinstance(bytes_4, (tuple, list)) or len(bytes_4) != N_QTY_BYTES:
        return False
    if not all(is_strict_int(x) for x in bytes_4):
        return False
    return bytes_4[0] == 0 and bytes_4[1] == 0 and bytes_4[2] == 0 and bytes_4[3] == 0


def qty_to_u31_byte_tuple(q: int) -> tuple[int, int, int, int]:
    """``b[j] = (q >> (8*(3-j))) & 255``；呼叫端已 ``require_u31_qty``。"""
    q = require_u31_qty(q)
    return (
        (q >> 24) & 255,
        (q >> 16) & 255,
        (q >> 8) & 255,
        q & 255,
    )


def u31_byte_tuple_to_qty(bytes_4: object) -> int:
    """先驗嚴格 Python int，再位移 OR。不得先 ``int()`` 接納浮點／字串。"""
    if not isinstance(bytes_4, (tuple, list)):
        raise OrderedV3Error(
            E_NONCANONICAL_INPUT,
            f"U31 bytes 必須是 tuple/list，得到 {type(bytes_4).__name__}",
        )
    if len(bytes_4) != N_QTY_BYTES:
        raise OrderedV3Error(E_INVALID_SHAPE, f"U31 bytes 長度 {len(bytes_4)} != {N_QTY_BYTES}")
    if not all(is_strict_int(x) for x in bytes_4):
        kinds = tuple(type(x).__name__ for x in bytes_4)
        raise OrderedV3Error(
            E_NONCANONICAL_INPUT,
            f"U31 bytes 必須先是嚴格 Python int（非 bool），得到 {kinds}",
        )
    if not u31_byte_ok(bytes_4):
        raise OrderedV3Error(
            E_NUMERIC_DOMAIN_GAP,
            f"U31 bytes {tuple(bytes_4)!r} 非法（首位 0..127、末三 0..255、全 0 排除）",
        )
    b0, b1, b2, b3 = bytes_4[0], bytes_4[1], bytes_4[2], bytes_4[3]
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3


def inferred_market_length_ops(ops: object) -> int:
    """第一次 EOS 的 index，或 10。先驗 10 個嚴格 int，再比較，不先 cast。"""
    if not isinstance(ops, (tuple, list)) or len(ops) != MAX_ORDERS:
        raise OrderedV3Error(
            E_INVALID_SHAPE,
            f"market_op 必須是長度 {MAX_ORDERS} 的 tuple/list，得到 {type(ops).__name__}",
        )
    if not all(is_strict_int(x) for x in ops):
        kinds = tuple(type(x).__name__ for x in ops)
        raise OrderedV3Error(
            E_NONCANONICAL_INPUT,
            f"market_op 必須先是嚴格 Python int，得到 {kinds}",
        )
    for i, op in enumerate(ops):
        if op == EOS_OP:
            return i
    return MAX_ORDERS


def unit_slot_token_reasons(
    *,
    op: object,
    crop: object,
    item: object,
    qty_bytes: object,
    alive: bool,
) -> tuple[str, ...]:
    """單一 unit slot 的共用語意：死 slot／inactive sentinel／qty 域。"""
    if not is_strict_int(op) or not is_strict_int(crop) or not is_strict_int(item):
        return ("NONCANONICAL_INPUT",)
    if not isinstance(qty_bytes, (tuple, list)) or len(qty_bytes) != N_QTY_BYTES:
        return ("NONCANONICAL_INPUT",)
    if not all(is_strict_int(x) for x in qty_bytes):
        return ("NONCANONICAL_INPUT",)
    if type(alive) is not bool:
        return ("NONCANONICAL_INPUT",)
    reasons: list[str] = []
    if not alive:
        if op != OP_PASS or crop != 0 or item != 0 or not u31_bytes_are_sentinel_zero(qty_bytes):
            return ("DEAD_SLOT_SENTINEL",)
        return ()
    if not 0 <= op < N_UNIT_OPS:
        return ("UNKNOWN_OP",)
    if op == OP_PLANT:
        if not 0 <= crop < N_CROPS:
            reasons.append("UNKNOWN_CROP")
        if item != 0 or not u31_bytes_are_sentinel_zero(qty_bytes):
            reasons.append("INACTIVE_SENTINEL")
        return tuple(reasons)
    if op in UNIT_NEEDS_ITEM_QTY:
        if crop != 0:
            reasons.append("INACTIVE_SENTINEL")
        if not 0 <= item < N_ITEMS:
            reasons.append("UNKNOWN_ITEM")
            return tuple(reasons)
        if not u31_byte_ok(qty_bytes):
            reasons.append("NUMERIC_DOMAIN_GAP")
        return tuple(reasons)
    if crop != 0 or item != 0 or not u31_bytes_are_sentinel_zero(qty_bytes):
        reasons.append("INACTIVE_SENTINEL")
    return tuple(reasons)


def market_slot_token_reasons(
    *,
    op: object,
    item: object,
    qty_bytes: object,
    role: str,
) -> tuple[str, ...]:
    """``role`` 為 ``true_order``／``eos``／``padding``。

    N6：原因碼維持既有名稱。``eos`` 槽非 sentinel 用 ``INACTIVE_SENTINEL``，
    ``padding`` 槽非全 0 用 ``PADDING_NOT_ZERO``。兩者都進 illegal 桶；
    EOS 是活躍終止決策，padding 不是決策，不得把兩個角色混成同一語意。
    """
    if role not in ("true_order", "eos", "padding"):
        return ("UNKNOWN_OP",)
    if not is_strict_int(op) or not is_strict_int(item):
        return ("NONCANONICAL_INPUT",)
    if not isinstance(qty_bytes, (tuple, list)) or not all(is_strict_int(x) for x in qty_bytes):
        return ("NONCANONICAL_INPUT",)
    if role in ("eos", "padding"):
        if op != EOS_OP or item != 0 or not u31_bytes_are_sentinel_zero(qty_bytes):
            return ("PADDING_NOT_ZERO",) if role == "padding" else ("INACTIVE_SENTINEL",)
        return ()
    if op == EOS_OP:
        return ("TRUE_ORDER_IS_EOS",)
    if not 0 <= op < N_MARKET_OPS:
        return ("UNKNOWN_OP",)
    if op in (MO_HIRE, MO_BUY_LAND):
        if item != 0 or not u31_bytes_are_sentinel_zero(qty_bytes):
            return ("INACTIVE_SENTINEL",)
        return ()
    support = MARKET_ITEM_SUPPORT.get(op, frozenset())
    reasons: list[str] = []
    if item not in support:
        reasons.append("ILLEGAL_ITEM_FOR_OP")
    if not u31_byte_ok(qty_bytes):
        reasons.append("NUMERIC_DOMAIN_GAP")
    return tuple(reasons)


def market_row_token_reasons(
    ops: object,
    items: object,
    qty_rows: object,
    stored_length: object,
) -> tuple[str, ...]:
    """一整列 10 槽：L、第一次 EOS、padding、真訂單 sentinel。"""
    if not is_strict_int(stored_length):
        return ("NONCANONICAL_INPUT",)
    try:
        inferred = inferred_market_length_ops(ops)
    except OrderedV3Error as exc:
        if exc.code == E_NONCANONICAL_INPUT:
            return ("NONCANONICAL_INPUT",)
        return ("L_MISMATCH",)
    if not isinstance(items, (tuple, list)) or len(items) != MAX_ORDERS:
        return ("NONCANONICAL_INPUT",)
    if not isinstance(qty_rows, (tuple, list)) or len(qty_rows) != MAX_ORDERS:
        return ("NONCANONICAL_INPUT",)
    reasons: list[str] = []
    if stored_length != inferred:
        reasons.append("L_MISMATCH")
    if inferred < 0 or inferred > MAX_ORDERS:
        reasons.append("L_MISMATCH")
        return tuple(reasons)
    for s in range(inferred):
        reasons.extend(
            market_slot_token_reasons(
                op=ops[s],
                item=items[s],
                qty_bytes=qty_rows[s],
                role="true_order",
            )
        )
    if inferred < MAX_ORDERS:
        reasons.extend(
            market_slot_token_reasons(
                op=ops[inferred],
                item=items[inferred],
                qty_bytes=qty_rows[inferred],
                role="eos",
            )
        )
        for s in range(inferred + 1, MAX_ORDERS):
            reasons.extend(
                market_slot_token_reasons(
                    op=ops[s],
                    item=items[s],
                    qty_bytes=qty_rows[s],
                    role="padding",
                )
            )
    return tuple(reasons)


def require_n_alive_rows(n_alive: object, batch: int) -> tuple[int, ...]:
    """每列各自的實際在場數。

    B>1 必須是長度 B 的嚴格 int 列；不得猜全部 29。
    """
    if not is_strict_int(batch) or batch < 1:
        raise OrderedV3Error(E_INVALID_SHAPE, f"batch 必須是嚴格正 int，得到 {batch!r}")
    if is_strict_int(n_alive):
        if batch != 1:
            raise OrderedV3Error(
                E_INVALID_N_ALIVE,
                f"batch={batch} 時 n_alive 必須是每列一值的長度 {batch} 序列，"
                f"不得用單一值或猜 {MAX_UNITS}",
            )
        return (require_n_alive(n_alive),)
    if not isinstance(n_alive, (tuple, list)) or len(n_alive) != batch:
        raise OrderedV3Error(
            E_INVALID_N_ALIVE,
            f"n_alive 必須是長度 {batch} 的嚴格 int 列（每列可不同），"
            f"得到 {type(n_alive).__name__}",
        )
    return tuple(require_n_alive(x) for x in n_alive)


def expected_flat_leaf_spec(
    actor_arch: object = ACTOR_ARCH_BASELINE,
) -> dict[str, dict[str, object]]:
    """獨立於實際權重的期望扁平 leaf；metadata 與 weights 都對這份驗。"""
    out: dict[str, dict[str, object]] = {}
    for spec in leaf_specs_for(actor_arch):
        if spec.kind == KIND_DENSE:
            out[f"['{spec.key}']['w']"] = {"shape": list(spec.w_shape), "dtype": "float32"}
            out[f"['{spec.key}']['b']"] = {
                "shape": [spec.w_shape[-1]],
                "dtype": "float32",
            }
        elif spec.kind in (KIND_EMBEDDING, KIND_PROJECTION):
            out[f"['{spec.key}']"] = {"shape": list(spec.w_shape), "dtype": "float32"}
        else:
            raise OrderedV3Error(
                E_INVALID_SHAPE,
                f"actor_arch={actor_arch!r} 的 {spec.key!r} 未知 kind {spec.kind!r}",
            )
    return out


def jax_compilation_cache_dir(outdir: str, attempt_id: str) -> str:
    """``OUTDIR/tmp/<attempt_id>/jax-cache``；每 attempt 唯一。"""
    if not isinstance(outdir, str) or not outdir:
        raise OrderedV3Error(E_INVALID_SHAPE, "outdir 必須是非空 str")
    if not isinstance(attempt_id, str) or not attempt_id or "/" in attempt_id or "\\" in attempt_id:
        raise OrderedV3Error(E_JAX_CACHE_STALE, f"attempt_id 非法：{attempt_id!r}")
    return outdir.rstrip("/") + "/" + JAX_CACHE_DIR_TEMPLATE.format(attempt_id=attempt_id)


def reject_stale_jax_cache(attempt_id: str, failed_attempt_ids: object) -> None:
    if not isinstance(attempt_id, str) or not attempt_id:
        raise OrderedV3Error(E_JAX_CACHE_STALE, f"attempt_id 非法：{attempt_id!r}")
    if failed_attempt_ids is None:
        return
    if not isinstance(failed_attempt_ids, (set, frozenset, tuple, list)):
        raise OrderedV3Error(E_JAX_CACHE_STALE, "failed_attempt_ids 必須是字串集合")
    if attempt_id in failed_attempt_ids:
        raise OrderedV3Error(
            E_JAX_CACHE_STALE,
            f"attempt {attempt_id} 的 JAX cache 已因 STOP／中斷失效，不得再讀用",
        )


def _dtype_name(arr: Any) -> str:
    dt = getattr(arr, "dtype", None)
    if dt is None:
        raise OrderedV3Error(E_INVALID_DTYPE, f"缺少 dtype：{type(arr).__name__}")
    name = getattr(dt, "name", None)
    if name:
        return str(name)
    return str(dt)


def _as_shape(arr: Any) -> tuple[int, ...]:
    sh = getattr(arr, "shape", None)
    if sh is None:
        raise OrderedV3Error(E_INVALID_SHAPE, f"缺少 shape：{type(arr).__name__}")
    return tuple(int(x) for x in sh)


def validate_decision_v3(decision: DecisionV3, *, batch: int | None = None) -> int:
    """嚴格核 DecisionV3 的 batch／尾維／int32。回傳 B。"""
    if not isinstance(decision, DecisionV3):
        raise OrderedV3Error(E_INVALID_SHAPE, f"需要 DecisionV3，得到 {type(decision).__name__}")
    b_seen: int | None = None
    for name, tail in DECISION_V3_FIELDS:
        arr = getattr(decision, name)
        shape = _as_shape(arr)
        if name == "market_length":
            if len(shape) < 1:
                raise OrderedV3Error(E_INVALID_SHAPE, "market_length 必須含 batch 軸")
            want = (shape[0],)
            if shape != want:
                raise OrderedV3Error(
                    E_INVALID_SHAPE, f"market_length shape {shape} 不是 (B,) 得到 {shape}"
                )
        else:
            if len(shape) < 1 + len(tail):
                raise OrderedV3Error(E_INVALID_SHAPE, f"{name} shape {shape} 缺少 batch 軸")
            want = (shape[0],) + tail
            if shape != want:
                raise OrderedV3Error(E_INVALID_SHAPE, f"{name} shape {shape} != {want}")
        dtype_name = _dtype_name(arr)
        if dtype_name != "int32":
            raise OrderedV3Error(
                E_INVALID_DTYPE, f"{name} dtype {dtype_name!r} 必須是 int32（拒 bool）"
            )
        b_here = int(shape[0])
        if b_seen is None:
            b_seen = b_here
        elif b_here != b_seen:
            raise OrderedV3Error(E_INVALID_SHAPE, f"{name} batch {b_here} != {b_seen}")
    assert b_seen is not None
    if batch is not None and b_seen != batch:
        raise OrderedV3Error(E_INVALID_SHAPE, f"batch {b_seen} != 期望 {batch}")
    if b_seen < 1:
        raise OrderedV3Error(E_INVALID_SHAPE, f"batch {b_seen} < 1")
    return b_seen


def validate_leaf_param_total(n_params: object, actor_arch: object = ACTOR_ARCH_BASELINE) -> None:
    want = param_total_for(actor_arch)
    if not is_strict_int(n_params) or n_params != want:
        raise OrderedV3Error(
            E_LEAF_COUNT,
            f"actor_arch={require_actor_arch(actor_arch)!r} param_total {n_params!r} "
            f"!= 公式 {want}（不得調表湊數）",
        )


def leaf_manifest_template(
    actor_arch: object = ACTOR_ARCH_BASELINE,
) -> tuple[dict[str, object], ...]:
    """供 checkpoint metadata 對帳；不含權重。"""
    rows = []
    for spec in leaf_specs_for(actor_arch):
        if spec.kind not in LEAF_KINDS:
            raise OrderedV3Error(
                E_INVALID_SHAPE,
                f"actor_arch={actor_arch!r} 的 {spec.key!r} 未知 kind {spec.kind!r}",
            )
        rows.append(
            {
                "key": spec.key,
                "key_id": spec.key_id,
                "kind": spec.kind,
                "w_shape": spec.w_shape,
                "bias_shape": (spec.w_shape[-1],) if spec.has_bias else None,
                "n_params": spec.n_params,
                "is_output_head": spec.is_output_head,
                "dtype": "float32",
            }
        )
    return tuple(rows)


def dataset_metadata_template() -> dict[str, object]:
    return {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "obs_version": OBS_VERSION,
        "package_kind": PACKAGE_KIND_DATA,
        "max_units": MAX_UNITS,
        "max_orders": MAX_ORDERS,
        "qty_codec": QTY_CODEC,
    }


def checkpoint_metadata_template(
    actor_arch: object = ACTOR_ARCH_BASELINE,
) -> dict[str, object]:
    arch = require_actor_arch(actor_arch)
    meta: dict[str, object] = {
        "meta_format_version": META_FORMAT_VERSION,
        "policy_version": POLICY_VERSION,
        "obs_version": OBS_VERSION,
        "package_kind": PACKAGE_KIND_CHECKPOINT,
        "width_config": width_config_for(arch),
        "dims": {"tile": OBS_DIM_TILE, "scalar": OBS_DIM_SCALAR, "unit": OBS_DIM_UNIT},
        "max_units": MAX_UNITS,
        "max_orders": MAX_ORDERS,
        "qty_codec": QTY_CODEC,
        "param_total": param_total_for(arch),
        "runtime_ready": RUNTIME_READY,
        "leaf_manifest": list(leaf_manifest_template(arch)),
    }
    if arch == ACTOR_ARCH_UNIT_SPATIAL_V1:
        meta["actor_arch"] = ACTOR_ARCH_UNIT_SPATIAL_V1
    return meta


def reject_legacy_policy_version(policy_version: object) -> None:
    if policy_version != POLICY_VERSION:
        raise OrderedV3Error(
            E_LEGACY_VERSION,
            f"policy_version {policy_version!r} 不是 {POLICY_VERSION}（拒 legacy／v2／P1）",
        )


def reject_if_runtime_anchor(anchor: object) -> None:
    if anchor is not None:
        raise OrderedV3Error(E_RUNTIME_UNREADY, "本案只允許 untrained-test；anchor 必須是 None")


def strict_json_equal(got: object, want: object, *, path: str = "meta") -> None:
    """JSON 欄位嚴格型別＋值。``True != 1``、``1.0 != 1``、bool 不得充 int。"""
    if want is None:
        if got is not None:
            raise OrderedV3Error(E_INVALID_DTYPE, f"{path} 必須是 None，得到 {got!r}")
        return
    if type(want) is bool:
        if type(got) is not bool or got != want:
            raise OrderedV3Error(E_INVALID_DTYPE, f"{path} 必須是 bool {want!r}，得到 {got!r}")
        return
    if type(want) is int:
        if type(got) is not int or got != want:
            raise OrderedV3Error(E_INVALID_DTYPE, f"{path} 必須是嚴格 int {want!r}，得到 {got!r}")
        return
    if type(want) is str:
        if type(got) is not str or got != want:
            raise OrderedV3Error(E_INVALID_DTYPE, f"{path} 必須是 str {want!r}，得到 {got!r}")
        return
    if type(want) is list:
        if type(got) is not list or len(got) != len(want):
            raise OrderedV3Error(
                E_INVALID_SHAPE, f"{path} 必須是長度 {len(want)} 的 list，得到 {got!r}"
            )
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            strict_json_equal(g, w, path=f"{path}[{i}]")
        return
    if type(want) is dict:
        if type(got) is not dict:
            raise OrderedV3Error(E_INVALID_SHAPE, f"{path} 必須是物件，得到 {type(got).__name__}")
        extra = sorted(set(got) - set(want))
        missing = sorted(set(want) - set(got))
        if extra or missing:
            raise OrderedV3Error(E_INVALID_SHAPE, f"{path} keys 缺 {missing}、多 {extra}")
        for key in sorted(want):
            strict_json_equal(got[key], want[key], path=f"{path}.{key}")
        return
    raise OrderedV3Error(E_INVALID_DTYPE, f"{path} 不支援的期望型別 {type(want).__name__}")


__all__ = (
    "ACTOR_ARCHS",
    "ACTOR_ARCH_BASELINE",
    "ACTOR_ARCH_UNIT_SPATIAL_V1",
    "ANIMALS",
    "BUCKET_C_DIFF",
    "BUCKET_EXACT",
    "BUCKET_ILLEGAL",
    "BUCKET_NONCANONICAL",
    "BUCKET_NUMERIC",
    "BYTE_CLASSES",
    "CHECKPOINT_META_KEY",
    "CROP_INDEX",
    "CROPS",
    "DATASET_META_KEY",
    "DATA_SCHEMA_VERSION",
    "DECISION_V3_FIELDS",
    "DECODE_STREAM_MODES",
    "DecisionV3",
    "EMBEDDING_KEYS",
    "ENGINE_VERSION",
    "EOS_OP",
    "E_CONFIG_ORDERS",
    "E_DUPLICATE_NPZ_KEY",
    "E_EXPLICIT_NONE",
    "E_ILLEGAL_ACTION",
    "E_INVALID_DTYPE",
    "E_INVALID_N_ALIVE",
    "E_INVALID_SEAT",
    "E_INVALID_SHAPE",
    "E_JAX_CACHE_STALE",
    "E_LEAF_COUNT",
    "E_LEGACY_VERSION",
    "E_MARKET_TRUNCATION",
    "E_MASK_EMPTY",
    "E_NETWORK_ISOLATION_UNAVAILABLE",
    "E_NONCANONICAL_INPUT",
    "E_NONFINITE",
    "E_NO_SUPPORT",
    "E_NUMERIC_DOMAIN_GAP",
    "E_PROJECTION_UNUSABLE",
    "E_POLICY_COMPILE_BUDGET_STOP",
    "E_POLICY_TEST_BUDGET_STOP",
    "E_RUNTIME_UNREADY",
    "E_SHA_MISMATCH",
    "E_TOKEN_SEMANTICS",
    "E_UNKNOWN_ACTOR_ARCH",
    "E_UNKNOWN_PACKAGE",
    "E_UNKNOWN_WIDTH",
    "E_UNSUPPORTED",
    "E_VERSION_CONFLICT",
    "ITEMS",
    "ITEM_INDEX",
    "KIND_DENSE",
    "KIND_EMBEDDING",
    "KIND_PROJECTION",
    "LEAF_KINDS",
    "JAX_CACHE_DIR_TEMPLATE",
    "LEAF_BY_KEY",
    "LEAF_SPECS",
    "LeafSpec",
    "MARKET_FIELD_BYTE0",
    "MARKET_FIELD_ITEM",
    "MARKET_FIELD_OP",
    "MARKET_ITEM_SUPPORT",
    "MARKET_OP_INDEX",
    "MARKET_OP_NAMES",
    "MAX_MARKET_ORDERS_PER_TURN_REQUIRED",
    "MAX_ORDERS",
    "MAX_UNITS",
    "META_FORMAT_VERSION",
    "MO_BUY_ANIMAL",
    "MO_BUY_LAND",
    "MO_BUY_PRODUCT",
    "MO_BUY_SEED",
    "MO_HIRE",
    "MO_NONE",
    "MO_SELL",
    "NESTED_SCAN_BYTE_STEPS",
    "NESTED_SCAN_MARKET_SLOTS",
    "N_CROPS",
    "N_ITEMS",
    "N_MARKET_OPS",
    "N_QTY_BYTES",
    "N_TILES",
    "N_UNIT_OPS",
    "NORMALIZATION_DIFF_REQUIRED_KEYS",
    "OBS_DIM_SCALAR",
    "OBS_DIM_TILE",
    "OBS_DIM_UNIT",
    "OBS_VERSION",
    "OFFICIAL_ENGINE_DIR",
    "OP_PASS",
    "OP_PICKUP",
    "OP_PLACE",
    "OP_PLANT",
    "OUTPUT_HEAD_KEYS",
    "PROJECTION_KEYS",
    "OrderedV3Error",
    "PACKAGE_KIND_CHECKPOINT",
    "PACKAGE_KIND_DATA",
    "PARAM_TOTAL",
    "SPATIAL_D",
    "SPATIAL_LEAF_BY_KEY",
    "SPATIAL_LEAF_SPECS",
    "SPATIAL_PARAM_TOTAL",
    "SPATIAL_WIDTH_CONFIG",
    "POLICY_VERSION",
    "PRIMARY_BUCKET_ORDER",
    "PRODUCTS",
    "QTY_CODEC",
    "QTY_HIGH_BYTE_MAX",
    "REASON_TO_BUCKET",
    "RNG_STREAM_MARKET",
    "RNG_STREAM_UNIT_ITEM",
    "RNG_STREAM_UNIT_QTY",
    "RNG_STREAM_VERB_CROP",
    "RUNTIME_READY",
    "SIMACTION_FIELD_NAMES",
    "SOURCE_CLOSURE_HEX_LEN",
    "SOURCE_HEAD_HEX_LEN",
    "CHECKPOINT_META_REQUIRED_KEYS",
    "U31_MAX",
    "U31_MIN",
    "UNIT_IN_DIM",
    "UNIT_NEEDS_CROP",
    "UNIT_NEEDS_ITEM_QTY",
    "UNIT_SHARED_OUT",
    "UNIT_OP_INDEX",
    "UNIT_OP_NAMES",
    "WIDTH_CONFIG",
    "checkpoint_metadata_template",
    "classify_actor_arch_keys",
    "dataset_metadata_template",
    "expected_flat_leaf_spec",
    "identify_params_arch",
    "inferred_market_length_ops",
    "is_strict_int",
    "leaf_specs_for",
    "param_total_for",
    "parse_checkpoint_actor_arch",
    "require_actor_arch",
    "require_params_match_arch",
    "width_config_for",
    "jax_compilation_cache_dir",
    "leaf_manifest_template",
    "market_fold_id",
    "market_row_token_reasons",
    "market_slot_token_reasons",
    "primary_bucket",
    "qty_to_u31_byte_tuple",
    "reject_if_runtime_anchor",
    "reject_legacy_policy_version",
    "reject_stale_jax_cache",
    "require_default_max_orders",
    "require_hires_capacity",
    "require_n_alive",
    "require_n_alive_rows",
    "require_seat",
    "require_u31_qty",
    "rng_coordinate_table",
    "strict_json_equal",
    "u31_byte_ok",
    "u31_byte_tuple_to_qty",
    "u31_bytes_are_sentinel_zero",
    "unit_qty_fold_id",
    "unit_slot_token_reasons",
    "validate_decision_v3",
    "validate_leaf_param_total",
)
