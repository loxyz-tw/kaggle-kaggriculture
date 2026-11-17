"""ordered_actions_v3 codec：raw JSON ↔ DecisionV3 tokens ↔ C／SimAction 欄位。

stdlib + NumPy。**不** import JAX、``rl.actions``、舊 ``encode_actions``／bins／
``RoleDecisions``。數量只走 U31 big-endian 四 byte，不經 float32 或 legacy clamp。

公開入口：
``canonicalize_action``、``encode_action``、``decode_action``、U31 雙射、
逐 component 六桶（五桶＋`engine_noop`）、``engine_json_from_decision``、
``simaction_numpy_from_decision``、``validate_decision_semantics``（JSON 與 SimAction 共用）。
"""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

import numpy as np

from rl.ordered_v3_contract import (
    ANIMALS,
    BUCKET_C_DIFF,
    BUCKET_EXACT,
    CROP_INDEX,
    CROPS,
    DECISION_V3_FIELDS,
    E_ILLEGAL_ACTION,
    E_INVALID_DTYPE,
    E_INVALID_N_ALIVE,
    E_INVALID_SHAPE,
    E_MARKET_TRUNCATION,
    E_NONCANONICAL_INPUT,
    E_TOKEN_SEMANTICS,
    EOS_OP,
    ITEM_INDEX,
    ITEMS,
    MARKET_ITEM_SUPPORT,
    MARKET_OP_INDEX,
    MARKET_OP_NAMES,
    MAX_ORDERS,
    MAX_UNITS,
    MO_BUY_LAND,
    MO_HIRE,
    MO_NONE,
    N_QTY_BYTES,
    NORMALIZATION_DIFF_REQUIRED_KEYS,
    OP_PASS,
    OP_PLANT,
    PRIMARY_BUCKET_ORDER,
    PRODUCTS,
    SIMACTION_FIELD_NAMES,
    U31_MAX,
    U31_MIN,
    UNIT_NEEDS_CROP,
    UNIT_NEEDS_ITEM_QTY,
    UNIT_OP_INDEX,
    UNIT_OP_NAMES,
    DecisionV3,
    OrderedV3Error,
    inferred_market_length_ops,
    is_strict_int,
    market_row_token_reasons,
    primary_bucket,
    qty_to_u31_byte_tuple,
    require_default_max_orders,
    require_n_alive,
    require_n_alive_rows,
    require_seat,
    require_u31_qty,
    u31_byte_tuple_to_qty,
    unit_slot_token_reasons,
    validate_decision_v3,
)

_KNOWN_ACTION_KEYS = frozenset({"farmer", "hands", "market"})

I32 = np.int32
_ZERO_BYTES = (0, 0, 0, 0)

# ==============================================================================
# 引擎 no-op 市場訂單（內部規格文件 v1.1）
# ==============================================================================
#: 第三類桶：「已證明無效、可接受刪除」。**不在** SUPPORT_BUCKETS，也不消耗 canonical slot。
#: 這是 codec 層、**只限 market component** 的分類；`rl.ordered_v3_contract` 一個 byte 都不改。
BUCKET_ENGINE_NOOP = "engine_noop"

#: 引擎（內部規格文件）lockstep 允許的品項。
#: SELL→PRODUCTS、BUY_PRODUCT→{WHEAT,FERTILIZER}、BUY_SEED→CROPS、BUY_ANIMAL→ANIMALS。
#: 測試以引擎快照的 PRODUCTS／CROPS／ANIMALS 逐一比對相等，並斷言
#: `MARKET_ITEM_SUPPORT` ⊆ 引擎允許集（否則本分類會刪掉引擎真的會執行的訂單）。
ENGINE_ALLOWED_ITEMS: dict[str, frozenset[str]] = {
    "SELL": frozenset(PRODUCTS),
    "BUY_PRODUCT": frozenset({"WHEAT", "FERTILIZER"}),
    "BUY_SEED": frozenset(CROPS),
    "BUY_ANIMAL": frozenset(ANIMALS),
}
#: 會走 `_parse_order` 交易分支的 op（`:639–649`）；HIRE／BUY_LAND 是 atomic，不在此列。
ENGINE_TRADE_OPS = tuple(ENGINE_ALLOWED_ITEMS)


def is_engine_noop_market(raw: object) -> bool:
    """§2.1 的**封閉**定義：這筆 market 訂單已證明在引擎是單筆 no-op。

    以 **raw 值**判，不以 codec reason 反推（reason 只說我們不支援，不代表引擎不執行）：

    * **N1**：非空 list、``raw[0]`` 是交易 op、``len(raw) >= 3``、``raw[1]`` 是 **str**
      且不在 :data:`ENGINE_ALLOWED_ITEMS` 的允許集、``raw[2]`` 是嚴格 int（拒 bool）且 ``>= 1``。
      這保證引擎 `_parse_order` 會成功（`int(raw[2])` 不拋、``n > 0``），只在 lockstep
      因品項不合法把該 index 設 None——**只作廢這一筆**，下一 index 照常。
    * **N2**：空 list ``[]``；`_parse_order` 直接回 None。

    其他一切（數字字串／float／bool 數量、**合法品項**的正數超 u31、item 為
    list／dict、未知 op、缺 item／qty、非 list）都**不是** no-op：引擎可能照常
    執行或拋例外，維持原桶與原拒收。
    先核 list 長度與 op 型別再索引或查集合（審查者 r2 N2），不讓短 list 拋例外。
    """
    if not isinstance(raw, list):
        return False
    if not raw:
        return True
    op = raw[0]
    if not isinstance(op, str) or op not in ENGINE_ALLOWED_ITEMS:
        return False
    if len(raw) < 3:
        return False
    item = raw[1]
    if not isinstance(item, str) or item in ENGINE_ALLOWED_ITEMS[op]:
        return False
    qty = raw[2]
    return is_strict_int(qty) and qty >= 1


def market_component_bucket(raw: object, reasons: tuple[str, ...] | list[str]) -> str:
    """market component 的主桶：先判 §2.1 的 no-op，否則回 contract 既有 `primary_bucket`。

    只在 market component 建構處呼叫；unit／`market_overflow`／`unit_overflow` 走原路徑。
    """
    if is_engine_noop_market(raw):
        return BUCKET_ENGINE_NOOP
    return primary_bucket(reasons)


class ComponentRecord(NamedTuple):
    """單一 unit／market／overflow component 的分桶列。"""

    component_id: str
    kind: str
    index: int
    raw: Any
    canonical: Any
    reasons: tuple[str, ...]
    primary_bucket: str


class CanonicalResult(NamedTuple):
    canonical: dict[str, Any]
    diffs: tuple[dict[str, Any], ...]
    n_alive: int
    components: tuple[ComponentRecord, ...]
    market_compacted: bool
    unit_safety_placeholder: bool
    #: 被判 :data:`BUCKET_ENGINE_NOOP` 的 market component 的**原始** index 與原樣 raw。
    engine_noop_indices: tuple[int, ...] = ()
    engine_noop_raw: tuple[Any, ...] = ()


class EncodeResult(NamedTuple):
    decision: DecisionV3
    canonical: dict[str, Any]
    raw_action: dict[str, Any]
    normalization_diffs: tuple[dict[str, Any], ...]
    components: tuple[ComponentRecord, ...]
    roundtrip_ok: bool
    n_alive: int
    inferred_market_length: int
    market_compacted: bool
    unit_safety_placeholder: bool
    market_keep_eligible: bool
    representation_success: bool
    engine_noop_indices: tuple[int, ...] = ()
    engine_noop_raw: tuple[Any, ...] = ()


def _require_int32_ndarray(arr: Any, shape: tuple[int, ...], name: str) -> Any:
    """先驗 ndarray＋int32，再允許讀值。拒 bool／float／先 cast。"""
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        raise OrderedV3Error(E_INVALID_SHAPE, f"{name} 缺少 shape/dtype：{type(arr).__name__}")
    got = tuple(int(x) for x in arr.shape)
    if got != shape:
        raise OrderedV3Error(E_INVALID_SHAPE, f"{name} shape {got} != {shape}")
    dt = arr.dtype
    kind = getattr(dt, "kind", None)
    dt_name = getattr(dt, "name", str(dt))
    if kind == "b" or dt_name == "bool":
        raise OrderedV3Error(E_NONCANONICAL_INPUT, f"{name} 拒 bool")
    if kind != "i" or dt_name != "int32":
        raise OrderedV3Error(
            E_INVALID_DTYPE,
            f"{name} dtype 必須是 int32，得到 {dt_name!r}（不得先 cast）",
        )
    return arr


def _py_ints_1d(arr: Any, length: int, name: str) -> tuple[int, ...]:
    if isinstance(arr, (tuple, list)):
        if len(arr) != length:
            raise OrderedV3Error(E_INVALID_SHAPE, f"{name} 長度 {len(arr)} != {length}")
        if not all(is_strict_int(x) for x in arr):
            kinds = tuple(type(x).__name__ for x in arr)
            raise OrderedV3Error(
                E_NONCANONICAL_INPUT, f"{name} 必須先是嚴格 Python int，得到 {kinds}"
            )
        return tuple(arr)
    a = _require_int32_ndarray(arr, (length,), name)
    return tuple(int(a[i]) for i in range(length))


def require_marked_bool(marked: object, batch: int) -> np.ndarray:
    """``marked[B, 2]`` 必須嚴格 bool，且每列恰一位 True。padding 保留真 seat。"""
    if not is_strict_int(batch) or batch < 1:
        raise OrderedV3Error(E_INVALID_SHAPE, f"batch 必須是嚴格正 int，得到 {batch!r}")
    if not hasattr(marked, "shape") or not hasattr(marked, "dtype"):
        raise OrderedV3Error(
            E_INVALID_SHAPE, f"marked 必須是 ndarray，得到 {type(marked).__name__}"
        )
    dt = marked.dtype
    dt_name = getattr(dt, "name", str(dt))
    kind = getattr(dt, "kind", None)
    if kind != "b" and dt_name not in ("bool", "bool_"):
        raise OrderedV3Error(
            E_INVALID_DTYPE,
            f"marked dtype 必須是 bool，得到 {dt_name!r}（拒 int／float 0/1）",
        )
    arr = np.asarray(marked)
    if arr.dtype != np.bool_:
        raise OrderedV3Error(E_INVALID_DTYPE, f"marked 必須是 bool ndarray，得到 {arr.dtype}")
    got = tuple(int(x) for x in arr.shape)
    if got != (batch, 2):
        raise OrderedV3Error(E_INVALID_SHAPE, f"marked shape {got} != ({batch}, 2)")
    if not bool(np.all(np.sum(arr.astype(np.int32), axis=-1) == 1)):
        raise OrderedV3Error(
            E_TOKEN_SEMANTICS,
            "marked 每列必須恰一位 True（padding 列保留真實 teacher seat，靠 keep=False 排除）",
        )
    return arr


def require_bool_ndarray(arr: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """keep／flag：嚴格 bool ndarray，拒 float／int。"""
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        raise OrderedV3Error(E_INVALID_SHAPE, f"{name} 缺少 shape/dtype：{type(arr).__name__}")
    dt = arr.dtype
    dt_name = getattr(dt, "name", str(dt))
    kind = getattr(dt, "kind", None)
    if kind != "b" and dt_name not in ("bool", "bool_"):
        raise OrderedV3Error(E_INVALID_DTYPE, f"{name} dtype 必須是 bool，得到 {dt_name!r}")
    out = np.asarray(arr)
    if out.dtype != np.bool_:
        raise OrderedV3Error(E_INVALID_DTYPE, f"{name} 必須是 bool ndarray，得到 {out.dtype}")
    got = tuple(int(x) for x in out.shape)
    if got != shape:
        raise OrderedV3Error(E_INVALID_SHAPE, f"{name} shape {got} != {shape}")
    return out


def require_n_alive_host(n_alive: object, batch: int) -> np.ndarray:
    """host：n_alive 先驗嚴格 int／int32，再成 int32[B]。拒 float／bool。"""
    if hasattr(n_alive, "shape") and hasattr(n_alive, "dtype"):
        arr = _require_int32_ndarray(n_alive, (batch,), "n_alive")
        rows = tuple(require_n_alive(int(arr[i])) for i in range(batch))
        return np.asarray(rows, dtype=I32)
    rows = require_n_alive_rows(n_alive, batch)
    return np.asarray(rows, dtype=I32)


def _py_int4(arr: Any, name: str) -> tuple[int, int, int, int]:
    vals = _py_ints_1d(arr, N_QTY_BYTES, name)
    return (vals[0], vals[1], vals[2], vals[3])


def _py_qty_rows(arr: Any, n_slots: int, name: str) -> tuple[tuple[int, int, int, int], ...]:
    if isinstance(arr, (tuple, list)):
        if len(arr) != n_slots:
            raise OrderedV3Error(E_INVALID_SHAPE, f"{name} 列數 {len(arr)} != {n_slots}")
        return tuple(_py_int4(row, f"{name}[{i}]") for i, row in enumerate(arr))
    a = _require_int32_ndarray(arr, (n_slots, N_QTY_BYTES), name)
    return tuple(_py_int4(a[i], f"{name}[{i}]") for i in range(n_slots))


def qty_to_u31_bytes(q: int) -> np.ndarray:
    """``b[j]=(q>>(8*(3-j)))&255`` → int32[4]。"""
    return np.asarray(qty_to_u31_byte_tuple(require_u31_qty(q)), dtype=I32)


def u31_bytes_to_qty(byte_arr: Any) -> int:
    """先驗嚴格 int／int32，再位移 OR。

    不得 ``asarray(..., dtype=int32)`` 接納浮點。
    """
    vals = _py_int4(byte_arr, "U31 bytes")
    return u31_byte_tuple_to_qty(vals)


def _qty_from_raw_token(raw_q: object) -> tuple[int | None, tuple[str, ...]]:
    """解析原件數量。成功回 (q, extra_reasons)；失敗 q=None。"""
    if type(raw_q) is bool or not is_strict_int(raw_q):
        return None, ("QTY_NOT_PYTHON_INT", "NONCANONICAL_INPUT")
    if raw_q < U31_MIN or raw_q > U31_MAX:
        return None, ("NUMERIC_DOMAIN_GAP",)
    return raw_q, ()


def _record(
    component_id: str,
    kind: str,
    index: int,
    raw: Any,
    canonical: Any,
    reasons: tuple[str, ...],
    bucket: str | None = None,
) -> ComponentRecord:
    """``bucket`` 只給 market component 用（:func:`market_component_bucket`）；

    其餘一律 ``None``，走 contract 既有 `primary_bucket`（行為逐字不變）。
    """
    if not reasons:
        reasons = ("EXACT",)
    return ComponentRecord(
        component_id=component_id,
        kind=kind,
        index=index,
        raw=copy.deepcopy(raw),
        canonical=copy.deepcopy(canonical),
        reasons=reasons,
        primary_bucket=primary_bucket(reasons) if bucket is None else bucket,
    )


def _norm_diff(
    *,
    path: str,
    reason: str,
    qty_omitted: bool,
    original_length: int | None,
    ignored_suffix: list | None,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path,
        "reason": reason,
        "qty_omitted": bool(qty_omitted),
        "original_length": original_length,
        "ignored_suffix": copy.deepcopy(ignored_suffix) if ignored_suffix is not None else None,
    }
    row.update(extra)
    missing = [k for k in NORMALIZATION_DIFF_REQUIRED_KEYS if k not in row]
    if missing:
        raise OrderedV3Error(E_INVALID_SHAPE, f"normalization_diff 缺必填欄 {missing}")
    return row


def _suffix_info(raw: list, keep: int) -> tuple[tuple[str, ...], list, int, list | None]:
    original_length = len(raw)
    if original_length > keep:
        return ("IGNORED_SUFFIX",), list(raw[:keep]), original_length, list(raw[keep:])
    kept = list(raw[:keep]) if original_length >= keep else list(raw)
    return (), kept, original_length, None


def canonical_unit_entry(raw: object) -> tuple[list | None, tuple[str, ...], list[dict[str, Any]]]:
    """單支 farmer／hand → C list。無法表示則 canonical=None。"""
    diffs: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return None, ("NON_LIST",) if raw is not None else ("EXPLICIT_NONE", "NON_LIST"), diffs
    if not raw:
        return None, ("EMPTY_ACTION",), diffs
    op_name = raw[0]
    if not isinstance(op_name, str) or op_name not in UNIT_OP_INDEX:
        return None, ("UNKNOWN_OP",), diffs
    op = UNIT_OP_INDEX[op_name]
    reasons: list[str] = []

    def _emit_suffix(
        path: str, extra: tuple[str, ...], original_length: int, suffix: list | None
    ) -> None:
        if "IGNORED_SUFFIX" in extra:
            diffs.append(
                _norm_diff(
                    path=path,
                    reason="IGNORED_SUFFIX",
                    qty_omitted=False,
                    original_length=original_length,
                    ignored_suffix=suffix,
                )
            )

    if op in UNIT_NEEDS_CROP:
        if len(raw) < 2:
            return None, ("MISSING_CROP",), diffs
        crop = raw[1]
        if not isinstance(crop, str) or crop not in CROPS:
            return None, ("UNKNOWN_CROP",), diffs
        extra, _head, orig, suffix = _suffix_info(raw, 2)
        reasons.extend(extra)
        _emit_suffix("unit", extra, orig, suffix)
        return [op_name, crop], tuple(reasons) or ("EXACT",), diffs

    if op in UNIT_NEEDS_ITEM_QTY:
        if len(raw) < 2:
            return None, ("MISSING_ITEM",), diffs
        item = raw[1]
        if not isinstance(item, str) or item not in ITEM_INDEX:
            return None, ("UNKNOWN_ITEM",), diffs
        if len(raw) < 3:
            reasons.append("QTY_OMITTED_DEFAULT_1")
            extra, _head, orig, suffix = _suffix_info(raw, 2)
            reasons.extend(extra)
            diffs.append(
                _norm_diff(
                    path="unit",
                    reason="QTY_OMITTED_DEFAULT_1",
                    qty_omitted=True,
                    original_length=orig,
                    ignored_suffix=None,
                )
            )
            _emit_suffix("unit", extra, orig, suffix)
            return [op_name, item, 1], tuple(reasons), diffs
        q, q_reasons = _qty_from_raw_token(raw[2])
        if q is None:
            return None, q_reasons, diffs
        extra, _head, orig, suffix = _suffix_info(raw, 3)
        reasons.extend(q_reasons)
        reasons.extend(extra)
        _emit_suffix("unit", extra, orig, suffix)
        return [op_name, item, q], tuple(reasons) or ("EXACT",), diffs

    extra, _head, orig, suffix = _suffix_info(raw, 1)
    reasons.extend(extra)
    _emit_suffix("unit", extra, orig, suffix)
    return [op_name], tuple(reasons) or ("EXACT",), diffs


def canonical_market_entry(
    raw: object,
) -> tuple[list | None, tuple[str, ...], list[dict[str, Any]]]:
    """單筆市場訂單 → C list（HIRE／LAND 單元素；不含 EOS）。"""
    diffs: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return None, ("NON_LIST",) if raw is not None else ("EXPLICIT_NONE", "NON_LIST"), diffs
    if not raw:
        return None, ("EMPTY_ACTION",), diffs
    op_name = raw[0]
    if not isinstance(op_name, str) or op_name not in MARKET_OP_INDEX:
        return None, ("UNKNOWN_OP",), diffs
    if op_name == MARKET_OP_NAMES[MO_NONE]:
        return None, ("EOS_IN_JSON",), diffs
    op = MARKET_OP_INDEX[op_name]
    reasons: list[str] = []

    def _emit_suffix(extra: tuple[str, ...], orig: int, suffix: list | None) -> None:
        if "IGNORED_SUFFIX" in extra:
            diffs.append(
                _norm_diff(
                    path="market",
                    reason="IGNORED_SUFFIX",
                    qty_omitted=False,
                    original_length=orig,
                    ignored_suffix=suffix,
                )
            )

    if op in (MO_HIRE, MO_BUY_LAND):
        extra, _head, orig, suffix = _suffix_info(raw, 1)
        reasons.extend(extra)
        _emit_suffix(extra, orig, suffix)
        return [op_name], tuple(reasons) or ("EXACT",), diffs

    if len(raw) < 3:
        return None, ("MISSING_QTY",) if len(raw) >= 2 else ("MISSING_ITEM",), diffs
    item = raw[1]
    if not isinstance(item, str) or item not in ITEM_INDEX:
        return None, ("UNKNOWN_ITEM",), diffs
    item_i = ITEM_INDEX[item]
    if item_i not in MARKET_ITEM_SUPPORT[op]:
        return None, ("ILLEGAL_ITEM_FOR_OP",), diffs
    q, q_reasons = _qty_from_raw_token(raw[2])
    if q is None:
        return None, q_reasons, diffs
    extra, _head, orig, suffix = _suffix_info(raw, 3)
    reasons.extend(q_reasons)
    reasons.extend(extra)
    _emit_suffix(extra, orig, suffix)
    return [op_name, item, q], tuple(reasons) or ("EXACT",), diffs


def _repath_diffs(diffs: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    out = []
    for d in diffs:
        row = dict(d)
        row["path"] = path
        out.append(row)
    return out


def canonicalize_action(
    raw_action: object,
    *,
    n_alive: int,
    max_market_orders_per_turn: int = MAX_ORDERS,
) -> CanonicalResult:
    """引擎預設與缺欄補齊 → C JSON；不合併訂單、不改可表示數量。

    缺欄（key 不在）與明示 ``None`` 分開：後者是 ``EXPLICIT_NONE``／``NON_LIST``，
    不得當缺欄 ``[]``。非法 market 不進可監督 prefix；
    原 slot 仍完整入 components。
    """
    n_alive = require_n_alive(n_alive)
    require_default_max_orders(max_market_orders_per_turn)
    diffs: list[dict[str, Any]] = []
    components: list[ComponentRecord] = []
    market_compacted = False
    unit_safety_placeholder = False
    engine_noop_indices: list[int] = []
    engine_noop_raw: list[Any] = []

    if not isinstance(raw_action, dict):
        raise OrderedV3Error(
            E_ILLEGAL_ACTION, f"raw_action 必須是 dict，得到 {type(raw_action).__name__}"
        )

    n_hands = n_alive - 1
    farmer_missing = "farmer" not in raw_action
    hands_missing = "hands" not in raw_action
    market_missing = "market" not in raw_action
    raw_farmer = None if farmer_missing else raw_action["farmer"]
    raw_hands = None if hands_missing else raw_action["hands"]
    raw_market = None if market_missing else raw_action["market"]
    for key in raw_action:
        if key not in _KNOWN_ACTION_KEYS:
            diffs.append(
                _norm_diff(
                    path=str(key),
                    reason="IGNORED_KEY",
                    qty_omitted=False,
                    original_length=None,
                    ignored_suffix=copy.deepcopy(raw_action[key]),
                )
            )

    if farmer_missing:
        diffs.append(
            _norm_diff(
                path="farmer",
                reason="MISSING_FARMER_DEFAULT_PASS",
                qty_omitted=False,
                original_length=None,
                ignored_suffix=None,
            )
        )

    if hands_missing:
        diffs.append(
            _norm_diff(
                path="hands",
                reason="MISSING_HANDS_DEFAULT_EMPTY",
                qty_omitted=False,
                original_length=None,
                ignored_suffix=None,
            )
        )
        raw_hands = []
    elif raw_hands is None or not isinstance(raw_hands, list):
        reason = "EXPLICIT_NONE" if raw_hands is None else "NON_LIST"
        components.append(
            _record("hands", "unit_overflow", -1, raw_hands, None, (reason, "NON_LIST"))
        )
        diffs.append(
            _norm_diff(
                path="hands",
                reason=reason,
                qty_omitted=False,
                original_length=None,
                ignored_suffix=None,
                raw=copy.deepcopy(raw_hands),
            )
        )
        raw_hands = []

    if market_missing:
        diffs.append(
            _norm_diff(
                path="market",
                reason="MISSING_MARKET_DEFAULT_EMPTY",
                qty_omitted=False,
                original_length=None,
                ignored_suffix=None,
            )
        )
        raw_market = []
    elif raw_market is None or not isinstance(raw_market, list):
        reason = "EXPLICIT_NONE" if raw_market is None else "NON_LIST"
        components.append(
            _record("market", "market_overflow", -1, raw_market, None, (reason, "NON_LIST"))
        )
        diffs.append(
            _norm_diff(
                path="market",
                reason=reason,
                qty_omitted=False,
                original_length=None,
                ignored_suffix=None,
                raw=copy.deepcopy(raw_market),
            )
        )
        raw_market = []

    farmer_src = ["PASS"] if farmer_missing else raw_farmer
    farmer_c, farmer_reasons, farmer_diffs = canonical_unit_entry(farmer_src)
    farmer_diffs = _repath_diffs(farmer_diffs, "farmer")
    diffs.extend(farmer_diffs)
    if farmer_missing:
        farmer_reasons = ("MISSING_FARMER_DEFAULT_PASS",) + (
            () if farmer_reasons == ("EXACT",) else farmer_reasons
        )
        farmer_c = ["PASS"]
    components.append(
        _record(
            "unit:0", "unit", 0, None if farmer_missing else raw_farmer, farmer_c, farmer_reasons
        )
    )
    if farmer_c is None:
        farmer_c = ["PASS"]
        unit_safety_placeholder = True
        components[-1] = _record(
            "unit:0",
            "unit",
            0,
            None if farmer_missing else raw_farmer,
            farmer_c,
            farmer_reasons + ("SAFETY_PLACEHOLDER",),
        )

    hand_cs: list[list] = []
    for i, hand in enumerate(raw_hands):
        cid = f"unit:{i + 1}"
        if i >= n_hands:
            components.append(
                _record(cid, "unit_overflow", i + 1, hand, None, ("UNITS_EXCEED_N_ALIVE",))
            )
            diffs.append(
                _norm_diff(
                    path=cid,
                    reason="UNITS_EXCEED_N_ALIVE",
                    qty_omitted=False,
                    original_length=len(hand) if isinstance(hand, list) else None,
                    ignored_suffix=None,
                )
            )
            continue
        c, reasons, u_diffs = canonical_unit_entry(hand)
        diffs.extend(_repath_diffs(u_diffs, cid))
        if c is None:
            unit_safety_placeholder = True
            reasons = reasons + ("SAFETY_PLACEHOLDER",)
            c = ["PASS"]
        components.append(_record(cid, "unit", i + 1, hand, c, reasons))
        hand_cs.append(c)

    while len(hand_cs) < n_hands:
        idx = len(hand_cs) + 1
        diffs.append(
            _norm_diff(
                path=f"unit:{idx}",
                reason="PAD_PASS_TO_N_ALIVE",
                qty_omitted=False,
                original_length=None,
                ignored_suffix=None,
            )
        )
        components.append(
            _record(f"unit:{idx}", "unit", idx, None, ["PASS"], ("PAD_PASS_TO_N_ALIVE",))
        )
        hand_cs.append(["PASS"])

    market_cs: list[list] = []
    for k, order in enumerate(raw_market):
        cid = f"market:{k}"
        if k >= MAX_ORDERS:
            components.append(
                _record(cid, "market_overflow", k, order, None, ("ORDERS_EXCEED_MAX",))
            )
            diffs.append(
                _norm_diff(
                    path=cid,
                    reason="ORDERS_EXCEED_MAX",
                    qty_omitted=False,
                    original_length=len(order) if isinstance(order, list) else None,
                    ignored_suffix=None,
                )
            )
            market_compacted = True
            continue
        c, reasons, m_diffs = canonical_market_entry(order)
        diffs.extend(_repath_diffs(m_diffs, cid))
        bucket = market_component_bucket(order, reasons)
        if bucket == BUCKET_ENGINE_NOOP and c is not None:
            # 不可能：MARKET_ITEM_SUPPORT ⊆ ENGINE_ALLOWED_ITEMS（測試斷言）。
            # 真的發生就**不**當 no-op，寧可維持原桶與原拒收。
            bucket = primary_bucket(reasons)
        components.append(_record(cid, "market", k, order, c, reasons, bucket=bucket))
        if bucket == BUCKET_ENGINE_NOOP:
            # §2.3：已證明無效的訂單直接移除（壓縮），**不**算空洞；原 index／raw 另冊保留。
            engine_noop_indices.append(k)
            engine_noop_raw.append(copy.deepcopy(order))
            continue
        if c is None:
            # 原 slot 已入 components；不得把前移後的安全序列當可監督 prefix。
            market_compacted = True
            continue
        market_cs.append(c)

    if len(market_cs) > MAX_ORDERS:
        raise OrderedV3Error(
            E_MARKET_TRUNCATION,
            f"canonical market 長度 {len(market_cs)} > {MAX_ORDERS}；禁止靜默截斷",
        )

    canonical = {"farmer": farmer_c, "hands": hand_cs, "market": market_cs}
    return CanonicalResult(
        canonical=canonical,
        diffs=tuple(diffs),
        n_alive=n_alive,
        components=tuple(components),
        market_compacted=market_compacted,
        unit_safety_placeholder=unit_safety_placeholder,
        engine_noop_indices=tuple(engine_noop_indices),
        engine_noop_raw=tuple(engine_noop_raw),
    )


def _zero_decision(batch: int) -> DecisionV3:
    b = batch
    return DecisionV3(
        unit_verb=np.zeros((b, MAX_UNITS), dtype=I32),
        unit_crop=np.zeros((b, MAX_UNITS), dtype=I32),
        unit_item=np.zeros((b, MAX_UNITS), dtype=I32),
        unit_qty_bytes=np.zeros((b, MAX_UNITS, N_QTY_BYTES), dtype=I32),
        market_op=np.zeros((b, MAX_ORDERS), dtype=I32),
        market_item=np.zeros((b, MAX_ORDERS), dtype=I32),
        market_qty_bytes=np.zeros((b, MAX_ORDERS, N_QTY_BYTES), dtype=I32),
        market_length=np.zeros((b,), dtype=I32),
    )


def empty_decision(batch: int = 1) -> DecisionV3:
    """全 PASS／EOS padding 的 DecisionV3。"""
    if not is_strict_int(batch) or batch < 1:
        raise OrderedV3Error(E_INVALID_SHAPE, f"batch 必須是嚴格正 int，得到 {batch!r}")
    decision = _zero_decision(batch)
    validate_decision_v3(decision, batch=batch)
    return decision


def _fill_unit_slot(decision: DecisionV3, b: int, slot: int, c_unit: list) -> None:
    op_name = c_unit[0]
    op = UNIT_OP_INDEX[op_name]
    decision.unit_verb[b, slot] = op
    decision.unit_crop[b, slot] = 0
    decision.unit_item[b, slot] = 0
    decision.unit_qty_bytes[b, slot] = 0
    if op == OP_PLANT:
        decision.unit_crop[b, slot] = CROP_INDEX[c_unit[1]]
        return
    if op in UNIT_NEEDS_ITEM_QTY:
        decision.unit_item[b, slot] = ITEM_INDEX[c_unit[1]]
        decision.unit_qty_bytes[b, slot] = qty_to_u31_bytes(c_unit[2])


def _fill_market_slot(decision: DecisionV3, b: int, slot: int, c_order: list) -> None:
    op_name = c_order[0]
    op = MARKET_OP_INDEX[op_name]
    decision.market_op[b, slot] = op
    decision.market_item[b, slot] = 0
    decision.market_qty_bytes[b, slot] = 0
    if op in (MO_HIRE, MO_BUY_LAND, MO_NONE):
        return
    decision.market_item[b, slot] = ITEM_INDEX[c_order[1]]
    decision.market_qty_bytes[b, slot] = qty_to_u31_bytes(c_order[2])


def _fill_eos(decision: DecisionV3, b: int, slot: int) -> None:
    decision.market_op[b, slot] = EOS_OP
    decision.market_item[b, slot] = 0
    decision.market_qty_bytes[b, slot, :] = 0


def _n_alive_rows(n_alive: Any, batch: int) -> tuple[int, ...]:
    if is_strict_int(n_alive) or isinstance(n_alive, (tuple, list)):
        return require_n_alive_rows(n_alive, batch)
    if hasattr(n_alive, "shape"):
        a = _require_int32_ndarray(n_alive, (batch,), "n_alive")
        return require_n_alive_rows(tuple(int(a[i]) for i in range(batch)), batch)
    raise OrderedV3Error(
        E_INVALID_N_ALIVE,
        f"n_alive 必須是嚴格 int、int 列或 int32[B]，得到 {type(n_alive).__name__}",
    )


def _validate_row_semantics(decision: DecisionV3, b: int, live: int) -> None:
    live = require_n_alive(live)
    ops = _py_ints_1d(decision.unit_verb[b], MAX_UNITS, f"unit_verb[{b}]")
    crops = _py_ints_1d(decision.unit_crop[b], MAX_UNITS, f"unit_crop[{b}]")
    items = _py_ints_1d(decision.unit_item[b], MAX_UNITS, f"unit_item[{b}]")
    qty_u = _py_qty_rows(decision.unit_qty_bytes[b], MAX_UNITS, f"unit_qty_bytes[{b}]")
    for u in range(MAX_UNITS):
        reasons = unit_slot_token_reasons(
            op=ops[u],
            crop=crops[u],
            item=items[u],
            qty_bytes=qty_u[u],
            alive=u < live,
        )
        if reasons:
            raise OrderedV3Error(
                E_TOKEN_SEMANTICS,
                f"batch {b} unit {u} n_alive={live}：{reasons}",
            )
    m_ops = _py_ints_1d(decision.market_op[b], MAX_ORDERS, f"market_op[{b}]")
    m_items = _py_ints_1d(decision.market_item[b], MAX_ORDERS, f"market_item[{b}]")
    m_qty = _py_qty_rows(decision.market_qty_bytes[b], MAX_ORDERS, f"market_qty_bytes[{b}]")
    stored_L = _py_ints_1d(decision.market_length[b : b + 1], 1, f"market_length[{b}]")[0]
    m_reasons = market_row_token_reasons(m_ops, m_items, m_qty, stored_L)
    if m_reasons:
        raise OrderedV3Error(
            E_TOKEN_SEMANTICS,
            f"batch {b} market n_alive={live}：{m_reasons}",
        )


def validate_decision_semantics(decision: DecisionV3, n_alive: Any) -> tuple[int, ...]:
    """JSON 與 SimAction 共用：L／padding／op／item／qty／sentinel／死 slot。

    ``n_alive`` 每列明示；禁止省略或猜全部 29。
    """
    batch = validate_decision_v3(decision)
    rows = _n_alive_rows(n_alive, batch)
    for b, live in enumerate(rows):
        _validate_row_semantics(decision, b, live)
    return rows


def encode_action(
    raw_action: object,
    *,
    n_alive: int,
    max_market_orders_per_turn: int = MAX_ORDERS,
    batch_index: int = 0,
    decision: DecisionV3 | None = None,
) -> EncodeResult:
    """raw JSON → DecisionV3（預設 B=1）＋六桶 sidecar。不呼叫舊 encoder。"""
    n_alive = require_n_alive(n_alive)
    require_default_max_orders(max_market_orders_per_turn)
    if not is_strict_int(batch_index) or batch_index < 0:
        raise OrderedV3Error(E_INVALID_SHAPE, f"batch_index {batch_index!r}")
    raw_copy = copy.deepcopy(raw_action) if isinstance(raw_action, dict) else raw_action
    canon = canonicalize_action(
        raw_action,
        n_alive=n_alive,
        max_market_orders_per_turn=max_market_orders_per_turn,
    )
    if decision is None:
        if batch_index != 0:
            raise OrderedV3Error(E_INVALID_SHAPE, "未提供 decision 時 batch_index 必須為 0")
        decision = empty_decision(1)
        b = 0
    else:
        b_size = validate_decision_v3(decision)
        if batch_index >= b_size:
            raise OrderedV3Error(E_INVALID_SHAPE, f"batch_index {batch_index} >= batch {b_size}")
        b = batch_index

    units = [canon.canonical["farmer"], *canon.canonical["hands"]]
    if len(units) != n_alive:
        raise OrderedV3Error(
            E_ILLEGAL_ACTION,
            f"C units {len(units)} != n_alive {n_alive}",
        )
    for slot, c_unit in enumerate(units):
        _fill_unit_slot(decision, b, slot, c_unit)
    for slot in range(n_alive, MAX_UNITS):
        decision.unit_verb[b, slot] = OP_PASS
        decision.unit_crop[b, slot] = 0
        decision.unit_item[b, slot] = 0
        decision.unit_qty_bytes[b, slot] = 0

    market = list(canon.canonical["market"])
    if len(market) > MAX_ORDERS:
        raise OrderedV3Error(
            E_MARKET_TRUNCATION,
            f"encode_action 拒收 market 長度 {len(market)} > {MAX_ORDERS}；禁止 [:MAX] 截斷",
        )
    L = len(market)
    for slot, c_order in enumerate(market):
        _fill_market_slot(decision, b, slot, c_order)
    if L < MAX_ORDERS:
        _fill_eos(decision, b, L)
        for slot in range(L + 1, MAX_ORDERS):
            _fill_eos(decision, b, slot)
    decision.market_length[b] = L
    _validate_row_semantics(decision, b, n_alive)

    inferred = inferred_market_length(decision.market_op[b])
    keep_ok = _market_keep_eligible(canon)
    roundtrip_ok = _roundtrip_ok(canon, decision, b, n_alive, inferred)
    representation_success = (
        roundtrip_ok and not canon.market_compacted and not canon.unit_safety_placeholder
    )
    return EncodeResult(
        decision=decision,
        canonical=canon.canonical,
        raw_action=raw_copy if isinstance(raw_copy, dict) else {"_non_dict": raw_copy},
        normalization_diffs=canon.diffs,
        components=canon.components,
        roundtrip_ok=roundtrip_ok,
        n_alive=n_alive,
        inferred_market_length=inferred,
        market_compacted=canon.market_compacted,
        unit_safety_placeholder=canon.unit_safety_placeholder,
        market_keep_eligible=keep_ok and not canon.market_compacted,
        representation_success=representation_success,
        engine_noop_indices=canon.engine_noop_indices,
        engine_noop_raw=copy.deepcopy(canon.engine_noop_raw),
    )


def inferred_market_length(market_op_row: Any) -> int:
    """不信外部 L：第一次 EOS 的位置，或 10。先驗型別再比較。"""
    ops = _py_ints_1d(market_op_row, MAX_ORDERS, "market_op")
    return inferred_market_length_ops(ops)


def _rec_engine_noop(rec: ComponentRecord) -> bool:
    """只有 market component 會是 `engine_noop`（見 :func:`market_component_bucket`）。"""
    return rec.primary_bucket == BUCKET_ENGINE_NOOP


def _rec_supported(rec: ComponentRecord) -> bool:
    return rec.primary_bucket in (BUCKET_EXACT, BUCKET_C_DIFF)


def _market_keep_eligible(canon: CanonicalResult) -> bool:
    """§2.3：``not market_compacted``，且所有**非** engine_noop 的 market 類
    component 都是 supported。
    """
    if canon.market_compacted:
        return False
    for rec in canon.components:
        if rec.kind not in ("market", "market_overflow"):
            continue
        if _rec_engine_noop(rec):
            continue
        if not _rec_supported(rec):
            return False
    return True


def _roundtrip_ok(
    canon: CanonicalResult,
    decision: DecisionV3,
    b: int,
    n_alive: int,
    inferred: int,
) -> bool:
    if canon.market_compacted or canon.unit_safety_placeholder:
        return False
    # §2.3：roundtrip 只對**非 engine_noop** component 比對（no-op 沒有 canonical slot）。
    if any(not _rec_supported(c) for c in canon.components if not _rec_engine_noop(c)):
        return False
    if int(decision.market_length[b]) != inferred:
        return False
    try:
        decoded = engine_json_from_decision(decision, n_alive=n_alive, batch_index=b)
    except OrderedV3Error:
        return False
    return decoded == canon.canonical


def engine_json_from_decision(
    decision: DecisionV3,
    *,
    n_alive: int,
    batch_index: int = 0,
) -> dict[str, Any]:
    """只輸出 farmer＋實際在場 hands＋真訂單；不含死 slot／EOS／padding。"""
    n_alive = require_n_alive(n_alive)
    b_size = validate_decision_v3(decision)
    if not is_strict_int(batch_index) or not 0 <= batch_index < b_size:
        raise OrderedV3Error(E_INVALID_SHAPE, f"batch_index {batch_index!r}")
    b = batch_index
    _validate_row_semantics(decision, b, n_alive)
    units: list[list] = []
    for slot in range(n_alive):
        units.append(_unit_json_from_slot(decision, b, slot))
    market: list[list] = []
    L = inferred_market_length(decision.market_op[b])
    for slot in range(L):
        market.append(_market_json_from_slot(decision, b, slot))
    return {"farmer": units[0], "hands": units[1:], "market": market}


def engine_json_batch(
    decision: DecisionV3,
    *,
    n_alive: Any,
) -> list[dict[str, Any]]:
    """B 列 JSON；``n_alive`` 每列明示，可不同。"""
    rows = validate_decision_semantics(decision, n_alive)
    out: list[dict[str, Any]] = []
    for b, live in enumerate(rows):
        units = [_unit_json_from_slot(decision, b, slot) for slot in range(live)]
        L = inferred_market_length(decision.market_op[b])
        market = [_market_json_from_slot(decision, b, slot) for slot in range(L)]
        out.append({"farmer": units[0], "hands": units[1:], "market": market})
    return out


def _unit_json_from_slot(decision: DecisionV3, b: int, slot: int) -> list:
    op = _py_ints_1d(decision.unit_verb[b, slot : slot + 1], 1, f"unit_verb[{b},{slot}]")[0]
    if not 0 <= op < len(UNIT_OP_NAMES):
        raise OrderedV3Error(E_ILLEGAL_ACTION, f"unit_verb[{slot}]={op} 越界")
    name = UNIT_OP_NAMES[op]
    if op == OP_PLANT:
        crop_i = _py_ints_1d(decision.unit_crop[b, slot : slot + 1], 1, f"unit_crop[{b},{slot}]")[0]
        if not 0 <= crop_i < len(CROPS):
            raise OrderedV3Error(E_ILLEGAL_ACTION, f"unit crop {crop_i} 越界")
        return [name, CROPS[crop_i]]
    if op in UNIT_NEEDS_ITEM_QTY:
        item_i = _py_ints_1d(decision.unit_item[b, slot : slot + 1], 1, f"unit_item[{b},{slot}]")[0]
        if not 0 <= item_i < len(ITEMS):
            raise OrderedV3Error(E_ILLEGAL_ACTION, f"unit item {item_i} 越界")
        qty = u31_bytes_to_qty(decision.unit_qty_bytes[b, slot])
        return [name, ITEMS[item_i], qty]
    return [name]


def _market_json_from_slot(decision: DecisionV3, b: int, slot: int) -> list:
    op = _py_ints_1d(decision.market_op[b, slot : slot + 1], 1, f"market_op[{b},{slot}]")[0]
    if not 0 <= op < len(MARKET_OP_NAMES) or op == EOS_OP:
        raise OrderedV3Error(E_ILLEGAL_ACTION, f"market_op[{slot}]={op} 不是真訂單")
    name = MARKET_OP_NAMES[op]
    if op in (MO_HIRE, MO_BUY_LAND):
        return [name]
    item_i = _py_ints_1d(decision.market_item[b, slot : slot + 1], 1, f"market_item[{b},{slot}]")[0]
    support = MARKET_ITEM_SUPPORT[op]
    if item_i not in support:
        raise OrderedV3Error(
            E_ILLEGAL_ACTION, f"{name} item index {item_i} 不在語法域 {sorted(support)}"
        )
    qty = u31_bytes_to_qty(decision.market_qty_bytes[b, slot])
    return [name, ITEMS[item_i], qty]


def decode_action(
    decision: DecisionV3,
    *,
    n_alive: int,
    batch_index: int = 0,
) -> dict[str, Any]:
    """tokens → 引擎 JSON（C）。不輸出 padding／EOS／死 slot。"""
    return engine_json_from_decision(decision, n_alive=n_alive, batch_index=batch_index)


def simaction_numpy_from_decision(
    decision: DecisionV3,
    *,
    seat: int,
    n_alive: Any,
) -> dict[str, np.ndarray]:
    """DecisionV3 → 與 ``SimAction`` 同名的 NumPy 欄位，shape ``[B,2,U]``／``[B,2,10]``。

    不 import ``rl.actions``／JAX。未指定座為 PASS／NONE。
    ``n_alive`` 每列明示（int32[B] 或 B=1 的嚴格 int），不得省略或猜 29。
    先跑與 JSON 相同的語意驗證，再填欄位。
    """
    seat = require_seat(seat)
    b_size = validate_decision_v3(decision)
    live_rows = validate_decision_semantics(decision, n_alive)
    unit_op = np.zeros((b_size, 2, MAX_UNITS), dtype=I32)
    unit_arg = np.full((b_size, 2, MAX_UNITS), -1, dtype=I32)
    unit_qty = np.zeros((b_size, 2, MAX_UNITS), dtype=I32)
    order_op = np.zeros((b_size, 2, MAX_ORDERS), dtype=I32)
    order_item = np.full((b_size, 2, MAX_ORDERS), -1, dtype=I32)
    order_qty = np.zeros((b_size, 2, MAX_ORDERS), dtype=I32)
    verbs = _require_int32_ndarray(decision.unit_verb, (b_size, MAX_UNITS), "unit_verb")
    crops = _require_int32_ndarray(decision.unit_crop, (b_size, MAX_UNITS), "unit_crop")
    u_items = _require_int32_ndarray(decision.unit_item, (b_size, MAX_UNITS), "unit_item")
    u_qty_b = _require_int32_ndarray(
        decision.unit_qty_bytes, (b_size, MAX_UNITS, N_QTY_BYTES), "unit_qty_bytes"
    )
    m_ops = _require_int32_ndarray(decision.market_op, (b_size, MAX_ORDERS), "market_op")
    m_items = _require_int32_ndarray(decision.market_item, (b_size, MAX_ORDERS), "market_item")
    m_qty_b = _require_int32_ndarray(
        decision.market_qty_bytes, (b_size, MAX_ORDERS, N_QTY_BYTES), "market_qty_bytes"
    )
    for b, live in enumerate(live_rows):
        for u in range(live):
            op = int(verbs[b, u])
            unit_op[b, seat, u] = op
            if op == OP_PLANT:
                unit_arg[b, seat, u] = int(crops[b, u])
            elif op in UNIT_NEEDS_ITEM_QTY:
                unit_arg[b, seat, u] = int(u_items[b, u])
                unit_qty[b, seat, u] = np.int32(u31_bytes_to_qty(u_qty_b[b, u]))
        L = inferred_market_length(m_ops[b])
        for s in range(L):
            op = int(m_ops[b, s])
            order_op[b, seat, s] = op
            if op not in (MO_NONE, MO_HIRE, MO_BUY_LAND):
                order_item[b, seat, s] = int(m_items[b, s])
                order_qty[b, seat, s] = np.int32(u31_bytes_to_qty(m_qty_b[b, s]))
        for s in range(L, MAX_ORDERS):
            order_op[b, seat, s] = EOS_OP
    out = {
        "unit_op": unit_op,
        "unit_arg": unit_arg,
        "unit_qty": unit_qty,
        "order_op": order_op,
        "order_item": order_item,
        "order_qty": order_qty,
    }
    if tuple(out) != SIMACTION_FIELD_NAMES:
        raise OrderedV3Error(E_INVALID_SHAPE, "SimAction 欄位名與契約不一致")
    return out


def stack_decisions(rows: list[DecisionV3]) -> DecisionV3:
    """沿 batch 軸拼接已核過的單列 DecisionV3。"""
    if not rows:
        raise OrderedV3Error(E_INVALID_SHAPE, "stack_decisions 需要至少一列")
    parts: dict[str, list] = {name: [] for name, _tail in DECISION_V3_FIELDS}
    for row in rows:
        validate_decision_v3(row)
        if int(_as_batch(row)) != 1:
            raise OrderedV3Error(E_INVALID_SHAPE, "stack_decisions 每列 batch 必須為 1")
        for name, _tail in DECISION_V3_FIELDS:
            parts[name].append(getattr(row, name))
    stacked = DecisionV3(
        **{name: np.concatenate(parts[name], axis=0) for name, _tail in DECISION_V3_FIELDS}
    )
    validate_decision_v3(stacked, batch=len(rows))
    return stacked


def _as_batch(decision: DecisionV3) -> int:
    shape = getattr(decision.market_length, "shape", None)
    if shape is None:
        raise OrderedV3Error(E_INVALID_SHAPE, "market_length 缺少 shape")
    return int(shape[0])


def component_bucket_counts(
    components: tuple[ComponentRecord, ...] | list[ComponentRecord],
) -> dict[str, int]:
    """五桶計數；分母是 component 數，不用 row 主桶抹平。"""
    counts = {bucket: 0 for bucket in PRIMARY_BUCKET_ORDER}
    for rec in components:
        counts[rec.primary_bucket] = counts.get(rec.primary_bucket, 0) + 1
    return counts


__all__ = (
    "CanonicalResult",
    "ComponentRecord",
    "EncodeResult",
    "canonicalize_action",
    "canonical_market_entry",
    "canonical_unit_entry",
    "component_bucket_counts",
    "decode_action",
    "empty_decision",
    "encode_action",
    "engine_json_batch",
    "engine_json_from_decision",
    "inferred_market_length",
    "qty_to_u31_bytes",
    "require_bool_ndarray",
    "require_marked_bool",
    "require_n_alive_host",
    "simaction_numpy_from_decision",
    "stack_decisions",
    "u31_bytes_to_qty",
    "validate_decision_semantics",
)
