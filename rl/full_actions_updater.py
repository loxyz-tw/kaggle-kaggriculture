#!/usr/bin/env python
"""完整動作 PPO 的最小 updater（規格 內部規格文件）。

**這一段只做 updater 本身**：逐 component clipped surrogate ＋ value ＋ 條件 entropy，
沿用既有 bridge（`rl/full_actions_ppo.py`）與 P1 的純 reward／GAE。
不是正式 PPO 配方、不含 KL 錨 loss、不新增 critic、不導入舊 runner。

三件容易做錯、這裡明確釘住的事
------------------------------

1. **ratio 的分母要代表產生 decision 的行為策略**（審查者 §49.2）。
   「重算用同一組 params 所以 ratio 必為 1」是**構造性**論證，不能替代對帳：
   `behaviour_denominator()` 用 rollout 當時的 params、**同一支 teacher 函式**、
   同 batch shape 與同精度重算一次 `teacher_old_logp_h`，並**先**對 rollout 存下來的
   `logp_h` 逐 token 硬核（≤1e-5、active 逐位同、`teacher_drop` 全 0），不過就 STOP。
   算完 `stop_gradient`，**整輪只算一次**，不得每個 minibatch 拿已更新的 params 重算。

2. **ratio 逐 component，不用 joint**。joint 把 263 項的 Δ 加起來取 exp，100 格各動
   0.15 就 e^15（Loop 4 smoke C 實測），clip 完全咬不住。這裡沿 `P1.factored_surrogate`：
   每格一個 ratio、各自 clip、一列把 active 項**相加**。

3. **advantage 在全 valid 列一次 normalize**（`P1.normalize_advantages(adv, valid)`），
   不是每個 minibatch 各自 normalize；padding 列的 adv／ret 已由 `P1.gae` 歸 0。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl import caps as caps_mod
from rl import full_actions_decode as D
from rl import full_actions_ppo as PB
from rl import loop4_ppo as P1
from rl import rules

F32 = jnp.float32
#: **診斷線**：teacher 重算與 rollout 行為機率的逐 token 差。CPU 上仍是硬檻（嚴格對帳），
#: GPU 上只記錄超線槽數，數值出口改用下面的 ratio 風險預算。GPU 要 `highest`。
DENOM_ATOL = 1e-5
#: **零更新 ratio 偏差預算**（09-21 新契約）：未更新策略時，單 component 比率誤差上限 0.1%
#: （clip 0.2 的 1/200）。這是看過資料後新訂的**工程風險預算**，不是浮點等價容差，
#: 不證根因、也不宣稱對學習無影響。
RATIO_BUDGET = 1e-3
#: PPO 的分母來源：rollout 當時**實際抽樣**保存的逐 token 行為機率（`rows["logp_h"]`）。
#: teacher 重算只作診斷，不得替換、校正或以「構造 ratio=1」冒充等價。
DENOMINATOR_SOURCE = "rollout_logp_v1"


class HealthBudget(NamedTuple):
    """短程 leg 的 host 停損預算（規格 內部規格文件）。

    **看過階段 A 之後新訂的保守試驗預算**，不是已證最優超參數、也不是精確信賴區間；
    只做 host 判斷，不改 loss／decoder／reward，也不降低任何全量檢查。
    成本量測模式**不帶** budget（`None`），舊 run 的判定不因此改寫。
    """

    name: str = "short_leg_v1"
    ratio_lo: float = 0.1  # 候選接受前：active component ratio 粗偏離停止線
    ratio_hi: float = 10.0  # 階段 A 觀測 [0.216, 3.403]；不宣稱 clip 保證此界線
    clip_frac_mean_max: float = 0.20  # 輪末：逐批 clip_frac_h 的**算術**均值（階段 A 0.02094）
    kl_to_il_mean_max: float = 2.0  # 輪前：每座 32 列 fresh KL-to-IL MC proxy 的均值（nats）
    kl_probe_rows: int = 32
    #: T3 穩定化（`t3_stabilization_v1`）才開：輪末對同輪**全部**有效 minibatch 做一次
    #: 無梯度 teacher 檢查（post-update ratio 同界、anchor KL row-sum 均值上限）。
    post_update_check: bool = False
    anchor_kl_rowsum_max: float | None = None
    #: 尾部比例 gate：輪末全域 active token 落在 [tail_lo, tail_hi] 外的比例須
    #: **<= tail_frac_max**（等號接受、超出即 STOP；審查者 ε 定案 09-25 §2）。N=0 或非有限不放行。
    #: 只在輪末全域 enforce；候選、逐座、逐 head 固定 report-only（`TAIL_PHASE_MODES`）。
    #: `tail_frac_max=None` ＝ 全部 phase report-only（只報不擋）。
    tail_lo: float | None = None
    tail_hi: float | None = None
    tail_frac_max: float | None = None


HEALTH_BUDGET_V1 = HealthBudget()
#: T3 PPO 穩定化計畫 §6.2 的預登記預算（與舊 v1 分名）：ratio 收緊到 [0.2, 5]、輪末全量
#: post-update 檢查、anchor KL（對新 IL 的 exact 條件 KL，row-sum nats）均值 ≤ 2.0。
HEALTH_BUDGET_T3 = HealthBudget(
    name="t3_stabilization_v1",
    ratio_lo=0.2,
    ratio_hi=5.0,
    clip_frac_mean_max=0.20,
    kl_to_il_mean_max=2.0,
    kl_probe_rows=32,
    post_update_check=True,
    anchor_kl_rowsum_max=2.0,
)
#: T3 校準煙測（審查者 09-25 裁定）：ratio 硬界改為**極值災難界** [0.05, 20]（候選與輪末同用）；
#: [0.2, 5] 改成尾部比例 gate，本次 report-only（`tail_frac_max=None`，ε 待預登記）；
#: clip／anchor KL／fresh proxy 預算與 v1 相同。v1 保留不改。
HEALTH_BUDGET_T3_V2 = HEALTH_BUDGET_T3._replace(
    name="t3_stabilization_v2",
    ratio_lo=0.05,
    ratio_hi=20.0,
    tail_lo=0.2,
    tail_hi=5.0,
    tail_frac_max=None,
)
#: 正式系列預算（審查者 ε 定案 09-25、審查者 確認）：
#: 極值界 [0.1, 10]（候選與輪末）；內層 [0.2, 5] 輪末全域尾部比例 ε＝5e-5 硬擋（`<=` 過）；
#: 候選／逐座／逐 head 只報；clip／anchor KL／fresh proxy 同 v2。
#: 依據：452acac 校準煙測兩輪 [0.2,5] 超界比例 8.5e-6／1.36e-5、[0.1,10] 零超界。
HEALTH_BUDGET_T3_V3 = HEALTH_BUDGET_T3_V2._replace(
    name="t3_stabilization_v3",
    ratio_lo=0.1,
    ratio_hi=10.0,
    tail_lo=0.2,
    tail_hi=5.0,
    tail_frac_max=5e-5,
)
#: T3 effect-mask（內部規格文件 v2）：數值全同 v3；差別在
#: surrogate 與兩個 enforce 項（極值界、輪末全域尾部）的納入集合改 score_active
#: （見 `effect_contract`）
HEALTH_BUDGET_T3_V4 = HEALTH_BUDGET_T3_V3._replace(name="t3_stabilization_v4")
#: 開 effect-mask 的 budget（由名稱決定；不在 HealthBudget 加欄，免得改動 v1–v3 的身分）
EFFECT_BUDGETS = frozenset({HEALTH_BUDGET_T3_V4.name})
BUDGETS = {
    "v1": HEALTH_BUDGET_V1,
    "t3_stabilization_v1": HEALTH_BUDGET_T3,
    "t3_stabilization_v2": HEALTH_BUDGET_T3_V2,
    "t3_stabilization_v3": HEALTH_BUDGET_T3_V3,
    "t3_stabilization_v4": HEALTH_BUDGET_T3_V4,
}


def uses_effect_mask(budget: HealthBudget | None) -> bool:
    return budget is not None and budget.name in EFFECT_BUDGETS


def health_budget_meta(budget: HealthBudget | None) -> dict[str, Any] | None:
    """寫進 result／identity 的 budget 身分；沒開就是 `None`（成本量測模式）。"""
    if budget is None:
        return None
    provenance = {
        HEALTH_BUDGET_T3.name: "T3 PPO 穩定化計畫 §6.2 預登記風險預算（非已證最優、非信賴區間）",
        HEALTH_BUDGET_T3_V3.name: (
            "審查者 ε 定案 09-25（審查者 確認）：依 452acac 校準煙測，極值界 [0.1,10]、"
            "[0.2,5] 輪末全域尾部 ε=5e-5 硬擋（<= 過）、候選／逐座／逐 head 只報"
        ),
        HEALTH_BUDGET_T3_V4.name: (
            "T3 effect-mask 規格 v2（審查者 規格審 PASS）：數值同 v3；surrogate 與極值界／輪末全域"
            "尾部的納入集合改 score_active（active & ~noeffect）"
        ),
        HEALTH_BUDGET_T3_V2.name: (
            "審查者 09-25：B16／B64 煙測被 [0.2,5] 硬界擋後改為災難界 [0.05,20]＋尾部比例 gate"
            "（本次 report-only，ε 待校準後預登記）"
        ),
    }.get(budget.name, "看過階段 A（06996f3）後新訂的保守試驗預算，非已證最優、非信賴區間")
    meta = {
        **budget._asdict(),
        "scope": "host 停損；不改 loss／decoder／reward，不降低全量檢查",
        "provenance": provenance,
    }
    if budget.tail_lo is not None:
        meta["tail_contract"] = tail_contract(budget)
    if uses_effect_mask(budget):
        meta["effect_contract"] = effect_contract()
    return meta


# ------------------------------------------------------------- ratio 分布統計
#: 計數界線：[0.2,5]（v1 硬界／v2 尾部 gate）、[0.1,10]（歷史）、[0.05,20]（v2 災難界）
RATIO_BOUNDS = {"b0.2_5": (0.2, 5.0), "b0.1_10": (0.1, 10.0), "b0.05_20": (0.05, 20.0)}
#: log ratio 直方圖：ln0.05–ln20 等寬 40 格，另加 underflow（格 0）與 overflow（格 41）
HIST_EDGES = np.linspace(np.log(0.05), np.log(20.0), 41)
N_HIST = HIST_EDGES.size + 1  # 42
_UNIT_PARTS = ("verb", "crop", "item", "qty_b0", "qty_b1", "qty_b2", "qty_b3")
_MARKET_PARTS = ("op", "item", "qty_b0", "qty_b1", "qty_b2", "qty_b3")
#: 依 `D.pack_one` 槽序：29 單位 × 7（verb／crop／item／qty b0–b3）
#: ＋ 10 訂單 × 6（op／item／qty b0–b3）⇒ 13 組
HEAD_GROUPS = tuple(f"unit_{p}" for p in _UNIT_PARTS) + tuple(f"market_{p}" for p in _MARKET_PARTS)


def _slot_group() -> np.ndarray:
    g = []
    for _u in range(PB.CAPS.max_units):
        g += list(range(len(_UNIT_PARTS)))
    for _m in range(PB.CAPS.max_orders):
        g += [len(_UNIT_PARTS) + i for i in range(len(_MARKET_PARTS))]
    if len(g) != D.N_SLOTS:
        raise AssertionError(f"槽數 {len(g)} ≠ {D.N_SLOTS}")
    return np.asarray(g, np.int32)


SLOT_GROUP = _slot_group()
_GROUP_ONEHOT = np.eye(len(HEAD_GROUPS), dtype=np.float32)[SLOT_GROUP]  # [263, 13]
#: 固定 13 組映射的版本身分（組名＋逐槽組號的 SHA256）；映射一變，resume 身分就不同
HEAD_GROUP_MAP = {
    "version": "head13_v1",
    "groups": list(HEAD_GROUPS),
    "slot_group_sha256": hashlib.sha256(SLOT_GROUP.tobytes()).hexdigest(),
}
#: 尾部比例的 phase 模式（審查者 ε 定案 09-25 §2）：只有輪末全域 enforce；
#: 候選小 minibatch、逐座、逐 head 固定 report-only，系列中不得改成 enforce。
TAIL_PHASE_MODES = {
    "global_post": "enforce",
    "candidate": "report-only",
    "seat": "report-only",
    "head": "report-only",
}


#: `tail_contract` 的必填欄位；resume 時缺任一即拒（不以預設補齊）
TAIL_CONTRACT_KEYS = (
    "eps_global",
    "phase_modes",
    "accept_rule",
    "ratio_definition",
    "count_denominator",
    "phase_definitions",
    "head_group_map",
)


def tail_contract(budget: HealthBudget) -> dict[str, Any]:
    """尾部 gate 的完整契約（進 config／result／resume 身分；審查者 16a83f1 審 B3）。

    effect-mask（v4）的 n_out／N 集合是 score_active（與 `effect_contract` 一致；審查者 e887eb8
    審 B3）；v1–v3 的序列化逐字不變。
    """
    enforced = budget.tail_frac_max is not None
    s_ = "score_active" if uses_effect_mask(budget) else "policy_active"
    return {
        "eps_global": budget.tail_frac_max,
        "phase_modes": (
            dict(TAIL_PHASE_MODES) if enforced else {k: "report-only" for k in TAIL_PHASE_MODES}
        ),
        "accept_rule": "n_out/N <= eps_global（等號接受、超出 STOP）；N=0 或非有限 STOP",
        "ratio_definition": (
            "r=exp(logp_theta−logp_rollout)；分母 logp_rollout＝原 rollout 記錄的 logp，不重算；"
            f"n_out=count({s_} & (r < tail_lo or r > tail_hi))"
        ),
        "count_denominator": (
            f"N=sum({s_})，只收已排定 valid 列（scheduled minibatches）；"
            "不重置、不接 IL loss mask；低／高尾合計一份 ε"
        ),
        "phase_definitions": {
            "candidate": "接受更新前、loss forward 的當前 params（單一 minibatch）",
            "global_post": "輪末最終 params 對原 rollout 全量已排定 minibatches，無梯度",
            "seat": "global_post 統計按座位逐列加總",
            "head": "global_post 統計按固定 13 組映射加總",
        },
        "head_group_map": HEAD_GROUP_MAP,
    }


#: effect 規則判定依賴的來源檔（SHA 進契約）
EFFECT_SOURCE_FILES = ("rl/full_actions_decode.py", "rl/rules.py", "rl/ordered_v3_contract.py")
EFFECT_CONTRACT_KEYS = (
    "rules_version",
    "rules",
    "cash_rule",
    "price_constants",
    "source_sha256",
    "surrogate",
    "inclusion_sets",
)


def effect_contract() -> dict[str, Any]:
    """T3 effect-mask 契約（進 config／result／resume 身分）。"""
    root = Path(__file__).resolve().parents[1]
    return {
        "rules_version": D.EFFECT_RULES_VERSION,
        "rules": {
            "E1_empty_pickup_qty": "enabled",
            "E2_a5_place_animal": "count-only（qty 本就 inactive）",
            "E3_sell_upper_zero_qty": "enabled",
            "E4_buy_cash_upper_below_price_qty": "enabled",
            "E5_hire_buy_land": "count-only（qty 本就 inactive）",
            "masked_slots": "只遮 qty 四 byte；verb／op／item 不遮",
            "decided_before_qty": True,
        },
        "cash_rule": (
            "cash_up 起點＝單位相後、市場相開始時的該座 money；可能有收入的 SELL（shed 上界>0）"
            "之後 unknown；BUY／HIRE／BUY_LAND 不減；負值 unknown；unknown 不遮"
        ),
        "price_constants": {
            "ANIMAL_COST": list(rules.ANIMAL_COST),
            "PRICE_FLOOR": rules.PRICE_FLOOR,
            "ANIMAL_ITEM_BASE": rules.ANIMAL_ITEM_BASE,
        },
        "source_sha256": {
            f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in EFFECT_SOURCE_FILES
        },
        "surrogate": (
            "pg = −mean_rows( Σ_{h∈score_active} min(r_h·A, clip(r_h)·A) )；score_active＝"
            "active & ~noeffect；row 內求和、row 間原聚合，不除以 score 數"
        ),
        "inclusion_sets": {
            "pg_surrogate": "score_active",
            "candidate_ratio_extreme_[0.1,10]": "score_active（enforce）",
            "post_ratio_extreme_[0.1,10]": "score_active（enforce）",
            "post_global_tail_[0.2,5]": "score_active（enforce）",
            "candidate_tail／seat／head": "score_active（report-only）",
            "clip_fraction": "policy_active（同 v3）",
            "fresh_kl_proxy": "policy_active（同 v3）",
            "anchor_kl": "policy_active（同 v3）",
            "entropy_bonus": "policy_active（同 v3）",
            "raw_stats": "policy_active（report-only，永遠落盤）",
            "noeffect_stats": "active & noeffect（report-only，無門檻）",
            "score_empty": "enforce 集合為空即 STOP（v3 零分母政策）",
        },
    }


def check_effect_contract(meta: dict[str, Any] | None, *, where: str) -> None:
    """effect budget 的身分必須帶完整 `effect_contract` 且與本版一致；否則 `ValueError`。"""
    if meta is None or meta.get("name") not in EFFECT_BUDGETS:
        return
    ec = meta.get("effect_contract")
    if not isinstance(ec, dict):
        raise ValueError(f"{where}：health_budget 缺 effect_contract，拒絕")
    missing = [k for k in EFFECT_CONTRACT_KEYS if k not in ec]
    if missing:
        raise ValueError(f"{where}：effect_contract 缺欄位 {missing}，拒絕")
    if ec != effect_contract():
        raise ValueError(f"{where}：effect_contract 與本版不同，拒絕")


def check_tail_contract(meta: dict[str, Any] | None, *, where: str) -> None:
    """budget 有尾部界線時，`tail_contract` 必填且模式／映射須與本版一致；否則 `ValueError`。

    v3 等有 ε 的 budget：模式須恰為 `TAIL_PHASE_MODES`；無 ε（v2）：四個 phase 全 report-only。
    """
    if meta is None or meta.get("tail_lo") is None:
        return
    tc = meta.get("tail_contract")
    if not isinstance(tc, dict):
        raise ValueError(f"{where}：health_budget 缺 tail_contract（必填模式），拒絕")
    missing = [k for k in TAIL_CONTRACT_KEYS if k not in tc]
    if missing:
        raise ValueError(f"{where}：tail_contract 缺必填欄位 {missing}，拒絕")
    if tc["eps_global"] != meta.get("tail_frac_max"):
        raise ValueError(f"{where}：tail_contract ε 與 budget 不一致，拒絕")
    want = (
        dict(TAIL_PHASE_MODES)
        if meta.get("tail_frac_max") is not None
        else {k: "report-only" for k in TAIL_PHASE_MODES}
    )
    if tc["phase_modes"] != want:
        raise ValueError(f"{where}：tail phase 模式 {tc['phase_modes']} ≠ 本版 {want}，拒絕")
    if tc["head_group_map"] != HEAD_GROUP_MAP:
        raise ValueError(f"{where}：13 組映射版本與本版不同，拒絕")
    if meta.get("name") in EFFECT_BUDGETS:
        # v4：n_out／N 的集合描述須與本版（score_active）逐字一致（審查者 e887eb8 審 B3）
        want_tc = tail_contract(BUDGETS[meta["name"]])
        for k in ("ratio_definition", "count_denominator"):
            if tc[k] != want_tc[k]:
                raise ValueError(f"{where}：tail_contract {k} 與本版（score_active）不同，拒絕")


def slot_describe(slot: int) -> dict[str, Any]:
    """槽號 → head／單位或訂單序。"""
    n_u = PB.CAPS.max_units * len(_UNIT_PARTS)
    if slot < n_u:
        return {"slot": slot, "phase": "unit", "index": slot // 7, "head": _UNIT_PARTS[slot % 7]}
    j = slot - n_u
    return {"slot": slot, "phase": "market", "index": j // 6, "head": _MARKET_PARTS[j % 6]}


def _packed_decision(rows) -> jax.Array | None:
    """rollout decision → 與 `pack_one` 同序的 263 槽 token（qty 為各 byte）。"""
    dec = rows.get("decision") if isinstance(rows, dict) else None
    if not dec or any(k not in dec for k in PB.LABEL_FIELDS):
        return None
    unit = {
        "verb": dec["unit_verb"],
        "crop": dec["unit_crop"],
        "item": dec["unit_item"],
        "qty": dec["unit_qty_bytes"],
    }
    market = {"op": dec["market_op"], "item": dec["market_item"], "qty": dec["market_qty_bytes"]}
    return D.pack_one(unit, market).astype(jnp.int32)


def ratio_stats(
    log_ratio, active, rows, *, new_lp=None, old_lp=None, ref_lp=None
) -> dict[str, jax.Array]:
    """一批的 active ratio 分布（jit 內，全部 stop_gradient；只回計數與極值位置，不存 logits）。

    審查者 M1：另回**逐列**計數（active、三組界線 below／above、直方圖）供 host 端依座位精確加總；
    逐列 × 13 組的 active 旗標（算各 head 的 distinct episode 支持數）；最差／最好 token 的
    decision token、old／new／reference logp 與該列 advantage（給定才有，否則 NaN／−1）。
    """
    d = jax.lax.stop_gradient(log_ratio)
    a = active
    af = a.astype(F32)
    b = d.shape[0]
    g1h = jnp.asarray(_GROUP_ONEHOT)
    out: dict[str, Any] = {"n_active": jnp.sum(af), "n_active_group": jnp.sum(af, 0) @ g1h}
    for name, (lo, hi) in RATIO_BOUNDS.items():
        below = a & (d < np.log(lo))
        above = a & (d > np.log(hi))
        out[f"{name}/below"] = jnp.sum(below.astype(F32))
        out[f"{name}/above"] = jnp.sum(above.astype(F32))
        out[f"{name}/below_group"] = jnp.sum(below.astype(F32), 0) @ g1h
        out[f"{name}/above_group"] = jnp.sum(above.astype(F32), 0) @ g1h
        out[f"{name}/row_any"] = jnp.any(below | above, axis=-1)  # [B]
        out[f"{name}/row_below"] = jnp.sum(below.astype(F32), -1)  # [B]
        out[f"{name}/row_above"] = jnp.sum(above.astype(F32), -1)
    out["row_n_active"] = jnp.sum(af, -1)
    out["row_group_active"] = (af @ g1h) > 0  # [B, 13]
    dmin = jnp.where(a, d, jnp.inf)
    dmax = jnp.where(a, d, -jnp.inf)
    gm = g1h.T > 0  # [13, 263]
    out["group_log_min"] = jnp.min(jnp.where(gm, jnp.min(dmin, 0)[None, :], jnp.inf), -1)
    out["group_log_max"] = jnp.max(jnp.where(gm, jnp.max(dmax, 0)[None, :], -jnp.inf), -1)
    idx = jnp.searchsorted(jnp.asarray(HIST_EDGES, F32), d, side="right")  # 0..41
    row_hist = jnp.zeros((b, N_HIST), F32).at[jnp.arange(b)[:, None], idx].add(af)
    out["row_hist"] = row_hist
    out["hist"] = jnp.sum(row_hist, 0)
    i_min, i_max = jnp.argmin(dmin.reshape(-1)), jnp.argmax(dmax.reshape(-1))
    packed = _packed_decision(rows)
    nan = jnp.asarray(jnp.nan, F32)
    adv = jnp.asarray(rows["adv"], F32) if "adv" in rows else None
    for tag, i, v in (("min", i_min, dmin), ("max", i_max, dmax)):
        r_, c_ = i // d.shape[1], i % d.shape[1]
        out[f"{tag}_log"] = v.reshape(-1)[i]
        out[f"{tag}_row"] = r_.astype(jnp.int32)
        out[f"{tag}_slot"] = c_.astype(jnp.int32)
        out[f"{tag}_token"] = packed[r_, c_] if packed is not None else jnp.asarray(-1, jnp.int32)
        for nm, lp in (("new", new_lp), ("old", old_lp), ("ref", ref_lp)):
            out[f"{tag}_logp_{nm}"] = (
                jax.lax.stop_gradient(lp)[r_, c_].astype(F32) if lp is not None else nan
            )
        out[f"{tag}_adv"] = adv[r_] if adv is not None else nan
    for k in ("seat", "env_id", "turn"):
        out[f"row_{k}"] = (
            jnp.asarray(rows[k]).astype(jnp.int32) if k in rows else jnp.full((b,), -1, jnp.int32)
        )
    return out


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """metrics 去掉巢狀的 `ratio_stats`／`ratio_stats_raw`（給逐項 float 化的呼叫端）。"""
    return {k: v for k, v in metrics.items() if not k.startswith("ratio_stats")}


def aggregate_ratio_stats(
    per_batch: list[dict[str, Any]],
    budget: HealthBudget | None = None,
    *,
    teacher_env_id: int | None = None,
    phase: str = "global_post",
) -> dict[str, Any]:
    """host 端彙總多批 `ratio_stats`（每筆需附 `batch`＝批序、`stats`）。回可 JSON 的報告。

    - 三組界線的 below／above／active 與佔比；13 個 head 組的 min／max ratio、超界數、active 數。
    - log ratio 直方圖（ln0.05–ln20、40 格＋上下溢）。
    - 全體最小／最大 ratio 的 token 位置：批序、槽號→head、座位、turn、env。
    - 受影響 row 數與 episode（座位, env）數（任一 active token 超界）。
    - budget 有尾部 gate 欄位時報 `tail`；`phase` 決定模式（`TAIL_PHASE_MODES`）：只有
      `global_post` 且 `tail_frac_max` 有值才 enforce（`n_out/N <= ε`，N=0／非有限不過），
      `candidate` 一律 report-only（`pass=None`）。
    - 審查者 M1：`per_seat`（active、三組界線 below／above、直方圖，逐列精確加總）；各 head 組的
      distinct episode 支持數（`teacher_env_id` 給定時 fresh／teacher 分列）；episode 分母清單與
      各界線的受影響 episode 清單；極值 token 的 decision／old／new／reference logp／advantage。
    """
    if not per_batch:
        raise ValueError("aggregate_ratio_stats：沒有任何批")
    if phase not in ("global_post", "candidate"):
        raise ValueError(f"aggregate_ratio_stats：未知 phase {phase!r}")
    st = [{k: np.asarray(v) for k, v in x["stats"].items()} for x in per_batch]
    n_act = float(sum(s["n_active"] for s in st))
    rep_: dict[str, Any] = {"n_batches": len(st), "n_active": n_act, "bounds": {}}
    n_act_g = np.sum([s["n_active_group"] for s in st], 0)
    groups = {name: {"n_active": float(n_act_g[i])} for i, name in enumerate(HEAD_GROUPS)}
    for name, (lo, hi) in RATIO_BOUNDS.items():
        below = float(sum(s[f"{name}/below"] for s in st))
        above = float(sum(s[f"{name}/above"] for s in st))
        rows_hit, eps = 0, set()
        for s in st:
            hit = s[f"{name}/row_any"].astype(bool)
            rows_hit += int(hit.sum())
            eps |= {
                (int(a), int(e))
                for a, e in zip(s["row_seat"][hit], s["row_env_id"][hit], strict=True)
            }
        rep_["bounds"][name] = {
            "lo": lo,
            "hi": hi,
            "n_below": below,
            "n_above": above,
            "frac_out": (below + above) / max(n_act, 1.0),
            "n_rows_affected": rows_hit,
            "n_episodes_affected": len(eps),
            "episodes_affected": sorted([a, e] for a, e in eps),
        }
        bg = np.sum([s[f"{name}/below_group"] for s in st], 0)
        ag = np.sum([s[f"{name}/above_group"] for s in st], 0)
        for i, g in enumerate(HEAD_GROUPS):
            groups[g][f"{name}/n_below"] = float(bg[i])
            groups[g][f"{name}/n_above"] = float(ag[i])
    gmin = np.min([s["group_log_min"] for s in st], 0)
    gmax = np.max([s["group_log_max"] for s in st], 0)
    for i, g in enumerate(HEAD_GROUPS):
        groups[g]["ratio_min"] = float(np.exp(gmin[i])) if np.isfinite(gmin[i]) else None
        groups[g]["ratio_max"] = float(np.exp(gmax[i])) if np.isfinite(gmax[i]) else None
    # episode 分母清單與各 head 的 distinct episode 支持數（fresh／teacher 分列）
    all_eps: set = set()
    group_eps: list[set] = [set() for _ in HEAD_GROUPS]
    for s in st:
        ra = s["row_n_active"] > 0
        for r in np.nonzero(ra)[0]:
            all_eps.add((int(s["row_seat"][r]), int(s["row_env_id"][r])))
        rga = s["row_group_active"].astype(bool)
        for r, gi in zip(*np.nonzero(rga), strict=True):
            group_eps[gi].add((int(s["row_seat"][r]), int(s["row_env_id"][r])))

    def _split(eps: set) -> dict[str, int]:
        if teacher_env_id is None:
            return {"n_episodes": len(eps)}
        t = sum(1 for _a, e in eps if e == teacher_env_id)
        return {"n_episodes": len(eps), "n_fresh": len(eps) - t, "n_teacher": t}

    for gi, g in enumerate(HEAD_GROUPS):
        groups[g]["episode_support"] = _split(group_eps[gi])
    rep_["groups"] = groups
    rep_["episodes"] = {
        **_split(all_eps),
        "teacher_env_id": teacher_env_id,
        "list": sorted([a, e] for a, e in all_eps),
        "key": "(seat, env_id)；同一輪內每座每 env 一局",
    }
    # 逐座（逐列精確加總，不假設一批單座）
    per_seat: dict[str, Any] = {}
    for seat in sorted({int(x) for s in st for x in s["row_seat"]}):
        acc = {"n_active": 0.0, "hist": np.zeros(N_HIST), "bounds": {}}
        for name in RATIO_BOUNDS:
            acc["bounds"][name] = {"n_below": 0.0, "n_above": 0.0}
        for s in st:
            m = s["row_seat"] == seat
            acc["n_active"] += float(s["row_n_active"][m].sum())
            acc["hist"] += s["row_hist"][m].sum(0)
            for name in RATIO_BOUNDS:
                acc["bounds"][name]["n_below"] += float(s[f"{name}/row_below"][m].sum())
                acc["bounds"][name]["n_above"] += float(s[f"{name}/row_above"][m].sum())
        for name in RATIO_BOUNDS:
            b_ = acc["bounds"][name]
            b_["frac_out"] = (b_["n_below"] + b_["n_above"]) / max(acc["n_active"], 1.0)
        acc["hist"] = [float(x) for x in acc["hist"]]
        per_seat[f"seat{seat}"] = acc
    rep_["per_seat"] = per_seat
    hist = np.sum([s["hist"] for s in st], 0)
    rep_["hist"] = {
        "edges_log": [float(x) for x in HIST_EDGES],
        "counts": [float(x) for x in hist],
        "layout": "c[0]<e[0]；c[i]∈[e[i-1],e[i])；c[41]≥e[40]（searchsorted right）",
    }
    for tag, pick in (("min", np.argmin), ("max", np.argmax)):
        vals = np.array([float(s[f"{tag}_log"]) for s in st])
        j = int(pick(vals))
        s, x = st[j], per_batch[j]
        r = int(s[f"{tag}_row"])
        rep_[f"extreme_{tag}"] = {
            "ratio": float(np.exp(vals[j])) if np.isfinite(vals[j]) else None,
            "batch": x.get("batch"),
            "row_in_batch": r,
            **slot_describe(int(s[f"{tag}_slot"])),
            "seat": int(s["row_seat"][r]),
            "env_id": int(s["row_env_id"][r]),
            "turn": int(s["row_turn"][r]),
            "decision_token": int(s.get(f"{tag}_token", -1)),
            **{
                f"logp_{nm}": _fnone(s.get(f"{tag}_logp_{nm}", np.nan))
                for nm in ("old", "new", "ref")
            },
            "advantage": _fnone(s.get(f"{tag}_adv", np.nan)),
            "advantage_sign": (
                None
                if not np.isfinite(float(s.get(f"{tag}_adv", np.nan)))
                else int(np.sign(float(s[f"{tag}_adv"])))
            ),
        }
    if budget is not None and budget.tail_lo is not None:
        key = next(
            (
                k
                for k, (lo, hi) in RATIO_BOUNDS.items()
                if (lo, hi) == (budget.tail_lo, budget.tail_hi)
            ),
            None,
        )
        frac = rep_["bounds"][key]["frac_out"] if key else None
        enforce = budget.tail_frac_max is not None and TAIL_PHASE_MODES[phase] == "enforce"
        rep_["tail"] = {
            "bounds": [budget.tail_lo, budget.tail_hi],
            "phase": phase,
            "n_active": n_act,
            "frac_out": frac,
            "frac_max": budget.tail_frac_max,
            "mode": "enforce" if enforce else "report-only",
            "pass": (
                bool(
                    frac is not None
                    and n_act > 0
                    and np.isfinite(n_act)
                    and np.isfinite(frac)
                    and frac <= budget.tail_frac_max
                )
                if enforce
                else None
            ),
            "rule": (
                "輪末全域 n_out/N <= frac_max 才過（等號接受）；N=0／非有限 STOP；"
                "候選／逐座／逐 head 只報"
            ),
        }
    return rep_


def _fnone(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


class HealthGateStop(ValueError):
    """**已凍結健康門**觸發（計畫 winloss v3.1 §5 真值表第 1 項；審查者 實作審 B1）。

    只在預登記的健康停止線被**有限值**越過時丟：候選 ratio 越界、輪末 clip fraction 超限、
    fresh KL-to-IL 均值超限。契約／資料／程式異常（非有限、缺值、列數、座位集合、score_empty、
    noeffect ⊄ active、shape、padding 等）一律仍是一般 `ValueError`＝異常。`gate` 記觸發的門名。
    仍是 `ValueError` 子類：既有呼叫端與訊息不變。"""

    def __init__(self, msg: str, gate: str):
        super().__init__(msg)
        self.gate = gate


def check_candidate_ratio(
    metrics: dict[str, Any], budget: HealthBudget, *, where: str
) -> dict[str, Any]:
    """**候選 params 被接受之前**：active component ratio 要落在 `[ratio_lo, ratio_hi]`。

    非有限一律拒。超限就停當輪——不自動降 lr、不重跑。

    **口徑（不要讀成別的）**：這裡的 ratio 來自這一梯度步**更新前**那次 loss 前向
    （`new_lp` 用的是 `params`，不是候選 `cand_p`）。所以這道門擋的是「**用這個偏離度算出來的
    候選不准被接受**」，**不保證**更新之後再前向一次的 ratio 也在界內——本檔不為此另加 forward。
    """
    lo, hi = float(metrics["ratio_active_min"]), float(metrics["ratio_active_max"])
    rep_ = {
        "where": where,
        "ratio_active_min": lo,
        "ratio_active_max": hi,
        "bounds": [budget.ratio_lo, budget.ratio_hi],
    }
    if "n_score_total" in metrics:  # effect-mask：enforce 集合為空不得當 PASS（v3 零分母政策）
        n_sc = float(metrics["n_score_total"])
        rep_["n_score_total"] = n_sc
        rep_["n_noeffect_total"] = float(metrics["n_noeffect_total"])
        rep_["ratio_raw_minmax"] = [
            float(metrics["ratio_raw_min"]),
            float(metrics["ratio_raw_max"]),
        ]
        if not n_sc > 0:
            raise ValueError(f"候選 score_empty（STOP）：{rep_}")
        if float(metrics["noeffect_subset_active"]) != 1.0:
            raise ValueError(f"候選 noeffect ⊄ active（STOP）：{rep_}")
        # 免分不免 finite：noeffect 的 ratio 非有限照拒；有限而越界的 noeffect 只報（審查者 B1）
        if float(metrics["ratio_policy_active_finite"]) != 1.0:
            raise ValueError(f"候選 policy_active ratio 非有限（含 noeffect，STOP）：{rep_}")
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"候選 ratio 非有限（STOP）：{rep_}")
    if lo > hi:
        raise ValueError(f"候選 ratio min > max（STOP）：{rep_}")
    if "ratio_stats" in metrics:
        agg = aggregate_ratio_stats(
            [{"batch": where, "stats": metrics["ratio_stats"]}], budget, phase="candidate"
        )
        rep_["stats"] = {
            "n_active": agg["n_active"],
            "bounds": agg["bounds"],
            "tail": agg.get("tail"),
        }
    if lo < budget.ratio_lo or hi > budget.ratio_hi:
        raise HealthGateStop(
            f"候選 ratio 超出停止線（STOP，不接受此候選）：{rep_}", "candidate_ratio_out_of_bounds"
        )
    # 候選小 minibatch 的尾部比例只報不擋（審查者 ε 定案 §2：不把整輪 ε 套到 32 列）
    return rep_


def check_round_clip_fraction(
    hist: list[dict[str, Any]], budget: HealthBudget, *, where: str
) -> dict[str, Any]:
    """**寫完成 checkpoint 之前**：逐批 `clip_frac_h` 的算術均值 <= 上限。

    這是每批比例的算術平均，**不是**全 token 加權比例；逐批分布另外留存。
    """
    vals = [float(h["clip_frac_h"]) for h in hist if h.get("clip_frac_h") is not None]
    rep_ = {
        "where": where,
        "n_batches": len(vals),
        "n_batches_missing_clip_frac": len(hist) - len(vals),
        "mean_of_batch_clip_frac": float(np.mean(vals)) if vals else None,
        "max_of_batch_clip_frac": float(np.max(vals)) if vals else None,
        "limit": budget.clip_frac_mean_max,
        "method": "每批 clip_frac_h 的算術均值（非 token 加權）",
    }
    if not vals or len(vals) != len(hist):
        raise ValueError(f"clip_frac_h 缺值，無法判定輪末健康（STOP）：{rep_}")
    if not np.isfinite(rep_["mean_of_batch_clip_frac"]):
        raise ValueError(f"clip_frac_h 均值非有限（STOP）：{rep_}")
    if rep_["mean_of_batch_clip_frac"] > budget.clip_frac_mean_max:
        raise HealthGateStop(
            f"輪末 clip fraction 超限（STOP，本輪不算完整輪）：{rep_}",
            "round_clip_frac_over_budget",
        )
    return rep_


def check_kl_to_il(
    probes: dict[str, dict[str, Any]], budget: HealthBudget, *, where: str
) -> dict[str, Any]:
    """**本輪更新之前**：兩座各自 fresh KL-to-IL MC proxy 的均值 <= 上限、且有限。

    這是**有抽樣誤差的漂移警報**，不是精確 KL；負值照報、不取絕對值觸發。
    更新後的 stale proxy 只作診斷，不進這道門。

    座位與列數是**硬拒**不是記錄：必須恰好 `{"seat0", "seat1"}`、每座 `n_rows` 等於
    `budget.kl_probe_rows`——少一座、多一座、或任一座列數不符都不算量到。
    """
    rep_: dict[str, Any] = {"where": where, "limit": budget.kl_to_il_mean_max, "per_seat": {}}
    if set(probes) != {"seat0", "seat1"}:
        raise ValueError(
            f"fresh KL proxy 的座位集合必須恰好 seat0／seat1（STOP）：得到 {sorted(probes)}"
        )
    bad, contract_bad = [], []
    for seat_key, r in sorted(probes.items()):
        mean = float(r["mean"])
        rep_["per_seat"][seat_key] = {
            "mean": mean,
            "n_rows": int(r["n_rows"]),
            "expected_rows": budget.kl_probe_rows,
            "rows_as_expected": int(r["n_rows"]) == budget.kl_probe_rows,
            "kind": r.get("kind"),
        }
        if int(r["n_rows"]) != budget.kl_probe_rows:
            bad.append(f"{seat_key}: n_rows {r['n_rows']} ≠ {budget.kl_probe_rows}")
            contract_bad.append(seat_key)
        if not np.isfinite(mean):
            bad.append(f"{seat_key}: 非有限 {mean}")
            contract_bad.append(seat_key)
        elif mean > budget.kl_to_il_mean_max:
            bad.append(f"{seat_key}: mean {mean:.4f} > {budget.kl_to_il_mean_max}")
    if bad:
        msg = f"KL-to-IL 漂移警報觸發（STOP）：{bad}；{rep_}"
        if contract_bad:  # 列數／非有限＝量測契約異常，不是健康門
            raise ValueError(msg)
        raise HealthGateStop(msg, "fresh_kl_to_il_over_budget")
    return rep_


def denominator_policy(backend: str) -> dict[str, Any]:
    """寫進 identity／報表的分母政策（語意硬拒與數值出口分開列）。"""
    return {
        "source": DENOMINATOR_SOURCE,
        "hard_reject": [
            "active 逐位相同",
            "teacher_drop 全 0",
            "兩邊 logp 有限",
            "inactive 槽 logp 為 0",
            "shape 一致",
        ],
        "numeric_exit": {
            "cpu": f"逐 token |Δlogp| <= {DENOM_ATOL}（嚴格對帳）",
            "gpu": f"active 槽 float64 |expm1(Δ)| <= {RATIO_BUDGET}（零更新比率偏差預算）",
            "diagnostic_line": DENOM_ATOL,
        },
        "backend": backend,
        "note": "語意檢查通過只代表本次檢查項目通過，不能宣稱已排除所有邏輯 bug；"
        "數值預算不是浮點等價容差，也不指認任何 kernel 根因",
    }


def ratio_budget(teacher_lp, rollout_lp, active) -> dict[str, Any]:
    """`|expm1(teacher_old − rollout_logp)|` 在 **active 槽**上的最大值（host float64）。

    **兩端各自先 cast float64 再相減**：先在 float32 相減會把差本身量化掉。
    未更新策略時，這個量就是單 component 的比率誤差；用 max，不用 mean／p99 掩蓋最差槽。
    """
    d = np.asarray(teacher_lp, np.float64) - np.asarray(rollout_lp, np.float64)
    act = np.asarray(active)
    vals = np.abs(np.expm1(d))[act]
    return {
        "max_abs_expm1_active": float(vals.max()) if vals.size else 0.0,
        "n_active_slots": int(act.sum()),
        "budget": RATIO_BUDGET,
    }


def ratio_budget_from_metrics(metrics: dict[str, Any]) -> float:
    """由 `make_loss` 已回的 `ratio_active_min/max` 取比率偏差的 bound（不另存 logits）。

    **呼叫端要先確認兩者有限且 `min <= max`**（`check_zero_update_budget` 會做）：
    NaN 進 `max()` 會被吞掉。
    """
    lo, hi = float(metrics["ratio_active_min"]), float(metrics["ratio_active_max"])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    return max(abs(hi - 1.0), abs(lo - 1.0))


class Cfg(NamedTuple):
    """煙測用係數。**不是正式配方**——正式 leg 的係數要另按成本與健康數據定。"""

    clip: float = P1.CLIP  # 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    lr: float = 1e-4
    grad_clip: float = 1.0
    reward: str = "networth"  # 或 "networth_own"／"dense"／"winloss"
    gamma: float = P1.GAMMA  # 1.0
    lam: float = P1.LAMBDA  # 0.95（winloss 強制 1.0，見 `check_cfg`）
    reward_scale: float = P1.REWARD_SCALE  # 1e-4（winloss 強制 1.0）
    minibatch: int = 2
    epochs: int = 1
    #: T3 KL 錨：`β · mean_rows Σ_active KL(π_IL,h ‖ π_θ,h)`（exact categorical、同 prefix、
    #: row-sum nats，不除 active 數）。0 ＝ 不加錨（舊行為；C1 的 β=0 路線也走這裡）。
    anchor_beta: float = 0.0


ANCHOR_META = {
    "direction": "KL(pi_IL || pi_theta)（forward，IL 側 stop_gradient）",
    "estimator": "exact categorical，完整合法類別；同一 rollout decision／prefix、同一真投影",
    "aggregation": "mean_valid_rows Σ_active_slots（row-sum nats，不除 active 數、不除 263）",
    "active": "rollout 的 policy active（PPO 同一份）；IL loss mask（M2／空 PICKUP）不進",
    "reference": "固定原 IL（整段訓練不換）",
}


#: reward enum（`rewards_of` 分流；trainer／critic 暖身 CLI 的 choices 取自這裡的子集）
REWARDS = ("networth", "networth_own", "dense", "winloss")
#: winloss（計畫 內部規格文件 v3.1 §1／§2）唯一允許的係數
WINLOSS_GAMMA = 1.0
WINLOSS_LAMBDA = 1.0
WINLOSS_SCALE = 1.0


def check_cfg(cfg: Cfg) -> None:
    """`lam`／`clip` 實際由 `rl/loop4_ppo.py` 的**模組常數**決定
    （`P1.gae` 與 `P1.factored_surrogate` 不吃參數），所以這裡**不接受**與常數不同的值——
    寫了卻被忽略比直接拒還危險。

    `gamma` 例外（計畫 內部規格文件）：γ==1 走原版 `P1.gae`，
    γ<1 走本檔的 `gae_gamma`（`targets()` 分流），所以接受 `0 < γ ≤ 1`。

    winloss（計畫 v3.1 §1／§2）：**只**接受 γ＝1、λ＝1、reward_scale＝1，`targets()` 走
    `gae_gamma(γ=1, λ=1)`；`P1.LAMBDA` 不改，其他 reward 的檢查與分流逐行不變。"""
    if cfg.reward not in REWARDS:
        raise ValueError(f"reward 只支援 {REWARDS}，得到 {cfg.reward!r}")
    winloss = cfg.reward == "winloss"
    fixed = (
        ("lam", cfg.lam, WINLOSS_LAMBDA if winloss else P1.LAMBDA),
        ("clip", cfg.clip, P1.CLIP),
    )
    bad = [(n, got, want) for n, got, want in fixed if float(got) != float(want)]
    if bad:
        raise ValueError(f"這些係數由 P1 模組常數決定，給了不同值會被忽略，明拒：{bad}")
    if not 0.0 < float(cfg.gamma) <= 1.0:
        raise ValueError(f"gamma 必須在 (0, 1]，得到 {cfg.gamma}")
    if winloss and (float(cfg.gamma) != WINLOSS_GAMMA or float(cfg.reward_scale) != WINLOSS_SCALE):
        raise ValueError(
            f"winloss 只准 γ={WINLOSS_GAMMA}、λ={WINLOSS_LAMBDA}、reward_scale={WINLOSS_SCALE}："
            f"得到 γ={cfg.gamma}、reward_scale={cfg.reward_scale}"
        )
    if not (np.isfinite(float(cfg.anchor_beta)) and float(cfg.anchor_beta) >= 0.0):
        raise ValueError(f"anchor_beta 必須是有限且 ≥ 0，得到 {cfg.anchor_beta}")


def gae_gamma(values, last_value, reward, valid, done, *, gamma: float, lam: float):
    """`P1.gae` 的同一套遮罩／done 語義，只把 γ 變成參數；回 `(adv, ret)` 同 `P1.gae`。

    給 γ<1 用；γ==1 的非 winloss 一律走原版 `P1.gae`（舊行為逐位不變）。winloss 以
    `(γ=1, λ=1)` 走這裡（`P1.gae` 的 λ 是模組常數）。`reward` 必須是 `[T, B]`。
    """
    vmask = valid.astype(values.dtype)
    cont = 1.0 - done.astype(values.dtype)

    def back(carry, i):
        adv_next, v_next = carry
        m, c = vmask[i], cont[i]
        r = reward[i] * m
        delta = r + gamma * v_next * c * m - values[i] * m
        adv = (delta + gamma * lam * adv_next * c * m) * m
        return (adv, values[i] * m), adv

    _, advs = jax.lax.scan(
        back, (jnp.zeros_like(last_value), last_value), jnp.arange(values.shape[0]), reverse=True
    )
    return advs, (advs + values) * vmask


def make_optimizer(cfg: Cfg) -> optax.GradientTransformation:
    """與 IL 同形狀的 optimizer（先 clip 再 adam），resume 用 `opt.init` 對葉數。"""
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.lr))


def opt_config(cfg: Cfg) -> dict[str, float]:
    """寫進 checkpoint 的 optimizer 配置（`load_train_state` 會逐欄比對）。"""
    return {"lr": float(cfg.lr), "clip": float(cfg.grad_clip)}


# ------------------------------------------------------------------ targets
def rewards_of(traj, final, *, seat: int, cfg: Cfg) -> jax.Array:
    if cfg.reward == "networth":
        return P1.networth_rewards(traj, final, seat, scale=cfg.reward_scale, valid=traj.valid)
    if cfg.reward == "networth_own":
        return own_networth_rewards(traj, final, seat, scale=cfg.reward_scale, valid=traj.valid)
    if cfg.reward == "dense":
        return P1.dense_rewards(traj, final, seat, traj.valid)
    if cfg.reward == "winloss":
        return winloss_rewards(final, seat, valid=traj.valid)
    raise ValueError(f"reward 只支援 {REWARDS}，得到 {cfg.reward!r}")


def winloss_outcome(final, seat) -> jax.Array:
    """``z = sign(own_final_money − opp_final_money)`` ``[B]``：勝 +1、敗 −1、和 0。

    與 networth_own 同一終局來源（`final.money`）、同一 `seat_vec`／`pick_seat` 取座；
    非有限現金直接拒（host 檢查），不讓 NaN 以 sign(NaN) 混進 advantage。"""
    money = final.money
    if not bool(jnp.all(jnp.isfinite(money.astype(jnp.float32)))):
        raise ValueError("winloss：final.money 含非有限值，拒收")
    sv = P1.seat_vec(seat, money.shape[0])
    own = P1.pick_seat(money, sv).astype(jnp.float32)
    opp = P1.pick_seat(money, 1 - sv).astype(jnp.float32)
    return jnp.sign(own - opp)


def winloss_rewards(final, seat, *, valid) -> jax.Array:
    """計畫 v3.1 §1：z 只放在每條軌跡**最後有效 transition**（`P1.last_valid_step`），
    其他步與 padding 全 0；scale 固定 1（`check_cfg` 強制）。"""
    return P1.scatter_to_last_valid(winloss_outcome(final, seat), valid).astype(F32)


def own_networth_rewards(traj, final, seat, *, scale: float, valid) -> jax.Array:
    """`P1.networth_rewards` 刪掉對手現金項：``r_t = [Φ(s_{t+1}) − Φ(s_t)] × scale``。

    Φ、真局末非現金歸零、valid 遮罩全照 P1
    （計畫 內部規格文件）；
    ＝ `networth_rewards` ＋ 對手現金變動 × scale。
    """
    phi_pre = jax.vmap(lambda st: P1.asset_value(st, seat))(traj.state)  # [T, B]
    sv = P1.seat_vec(seat, final.money.shape[0])
    phi_fin = P1.pick_seat(final.money, sv).astype(jnp.float32)
    phi_post = jnp.concatenate([phi_pre[1:], phi_fin[None]], axis=0)
    is_last = jnp.arange(phi_pre.shape[0])[:, None] == P1.last_valid_step(valid)[None, :]
    phi_post = jnp.where(is_last, phi_fin[None, :], phi_post)
    r = (phi_post - phi_pre) * scale
    return r * valid.astype(r.dtype)


def targets(traj, final, *, seat: int, cfg: Cfg) -> dict[str, Any]:
    """`[T, B]` 上算 reward → GAE。**這裡不 normalize**——advantage 要把兩座的 valid 列
    **合在一起**做一次（見 `normalize_joint`），各座各自 normalize 等於給兩座不同尺度。

    `last_value=0` 只有在每個 env 的最後有效步真的 `done` 時才成立，所以這裡**明檢**，
    不靠「煙測大概會跑到局末」。截短的 rollout 不得用這條路徑。
    """
    valid = traj.valid
    last = P1.last_valid_step(valid)
    done_at_last = jnp.take_along_axis(traj.done, last[None, :], axis=0)[0]
    if not bool(jnp.all(done_at_last)):
        raise ValueError(
            "有 env 的最後有效步不是 done：last_value=0 不成立，"
            "截短 rollout 不得走 P1 的真局末勢函數路徑"
        )
    check_cfg(cfg)
    r = rewards_of(traj, final, seat=seat, cfg=cfg)
    adv, ret = advantages(cfg, traj.value, r, valid, traj.done)
    return {"reward": r, "adv_raw": adv, "ret": ret, "valid": valid, "last_valid_t": last}


def advantages(cfg: Cfg, values, reward, valid, done):
    """`targets()` 的 GAE 分流（last_value＝0）；critic 暖身重現 adv 也走這一支。

    - winloss：`gae_gamma(γ=1, λ=1)`（計畫 v3.1 §2）；
    - 其他 reward、γ==1：原版 `P1.gae`（λ＝`P1.LAMBDA`，逐位不變）；γ<1：`gae_gamma`。"""
    last0 = jnp.zeros(values.shape[1:], F32)
    if cfg.reward == "winloss":
        return gae_gamma(
            values, last0, reward, valid, done, gamma=WINLOSS_GAMMA, lam=WINLOSS_LAMBDA
        )
    if float(cfg.gamma) == 1.0:
        return P1.gae(values, last0, reward, valid, done)
    return gae_gamma(values, last0, reward, valid, done, gamma=float(cfg.gamma), lam=float(cfg.lam))


def normalize_joint(tgts: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """**兩座合起來**的 valid raw adv 一次 normalize，再分回各座。

    每座各自 normalize 會讓同一個 batch 裡兩座的 advantage 尺度不同（座位間的優劣被
    各自的 std 抹掉）；這裡先把所有 valid 列串起來算 mean／std，再套回去。
    """
    parts = [np.asarray(t["adv_raw"])[np.asarray(t["valid"])].reshape(-1) for t in tgts.values()]
    allv = np.concatenate(parts) if parts else np.zeros((0,), np.float64)
    mean = float(allv.mean()) if allv.size else 0.0
    std = float(allv.std()) if allv.size else 0.0
    for t in tgts.values():
        v = np.asarray(t["valid"])
        a = (np.asarray(t["adv_raw"], np.float32) - mean) / (std + 1e-8)
        t["adv"] = jnp.asarray(np.where(v, a, 0.0).astype(np.float32))
    return {
        "n_valid_rows_all_seats": int(allv.size),
        "adv_mean_before": mean,
        "adv_std_before": std,
        "scope": "兩座合併一次 normalize（不是逐座、也不是逐 minibatch）",
    }


def batch_from(traj, tgt: dict[str, Any]) -> dict[str, Any]:
    """`PB.flatten_bridge`（valid 過濾）＋ 逐列 adv／ret。欄位不自己另造。"""
    flat = PB.flatten_bridge(traj)
    keep = np.asarray(traj.valid).reshape(-1)
    for name in ("adv", "ret", "adv_raw", "reward"):
        flat[name] = np.asarray(tgt[name]).reshape(-1)[keep]
    return flat


def take_rows(batch: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    """抽列：`state`／`decision` 是 pytree／dict，其餘是 `[N, ...]`。"""
    out: dict[str, Any] = {
        "state": jax.tree.map(lambda x: jnp.asarray(np.asarray(x)[idx]), batch["state"]),
        "decision": {k: jnp.asarray(np.asarray(v)[idx]) for k, v in batch["decision"].items()},
    }
    keep = ("logp_h", "active", "value", "adv", "ret", "seat", "env_id", "turn", "game_step")
    for k in keep:
        out[k] = jnp.asarray(np.asarray(batch[k])[idx])
    if "teacher_old_logp_h" in batch:
        out["teacher_old_logp_h"] = jnp.asarray(np.asarray(batch["teacher_old_logp_h"])[idx])
    if "noeffect" in batch:  # T3 effect-mask：來自 rollout，不重算
        out["noeffect"] = jnp.asarray(np.asarray(batch["noeffect"])[idx])
    return out


# ------------------------------------------------- §49.2 行為分母（只算一次）
def _denominator_rows(
    params_old,
    cfg_model,
    rows: dict[str, Any],
    *,
    seat: int,
    decoders,
    caps,
    atol: float,
    backend: str,
) -> tuple[jax.Array, dict[str, Any]]:
    """對**一個 minibatch**（實際 shuffle 後的形狀）用舊 params 重算 teacher logp。

    **這份重算是診斷，不是分母**：分母是 rollout 保存的 `rows["logp_h"]`（見
    `DENOMINATOR_SOURCE`）。這裡把兩件事分開：

    * **語意硬拒（任何平台）**：`active` 逐位相同、`teacher_drop` 全 0、兩邊 logp 有限、
      inactive 槽 logp 為 0、shape 一致——任一不符 `ValueError`（STOP）。
    * **數值出口**：CPU 逐 token `|Δ| <= atol` 嚴格對帳；GPU 改核 active 槽的
      float64 `|expm1(Δ)| <= RATIO_BUDGET`，並**照實記錄**超過 `atol` 的槽數。
    """
    out = PB.recompute(
        params_old,
        cfg_model,
        rows["state"],
        rows["decision"],
        seat=seat,
        decoders=decoders,
        caps=caps,
    )
    rec_lp, rec_act = out["logp"], out["active"]
    old_lp, old_act = rows["logp_h"], rows["active"]
    # **shape 先檢**：形狀不合時 `rec_lp - old_lp` 會靜默廣播，後面的數字全部沒有意義
    shapes = [tuple(x.shape) for x in (rec_lp, old_lp, rec_act, old_act)]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"分母重算的語意檢查失敗（STOP）：['shape_match']；"
            f"rec_lp/old_lp/rec_act/old_act 形狀 {shapes}"
        )
    drop = out["teacher_drop"]
    n_drop = sum(int(jnp.sum(v)) for grp in ("unit", "market") for v in drop[grp].values())
    diff = jnp.abs(rec_lp - old_lp)
    budget = ratio_budget(rec_lp, old_lp, old_act)
    rep = {
        "n_rows": int(old_lp.shape[0]),
        "shape_match": True,  # 不合早就 raise 了，這裡只是把結論記下來
        "shapes": [list(x) for x in shapes],
        "n_slots_over_diagnostic_line": int(jnp.sum(diff > atol)),
        "ratio_budget": budget,
        "backend": backend,
        "denominator_source": DENOMINATOR_SOURCE,
        "active_bitwise_equal": bool(jnp.all(rec_act == old_act)),
        "teacher_drop_total": n_drop,
        "max_abs_token_logp_diff": float(jnp.max(diff)),
        "mean_abs_token_logp_diff": float(jnp.mean(diff)),
        "max_abs_joint_logp_diff": float(
            jnp.max(jnp.abs(PB.joint_logp(rec_lp, rec_act) - PB.joint_logp(old_lp, old_act)))
        ),
        "rollout_logp_finite": bool(jnp.all(jnp.isfinite(old_lp))),
        "teacher_old_logp_finite": bool(jnp.all(jnp.isfinite(rec_lp))),
        "inactive_logp_all_zero": bool(
            jnp.all(jnp.where(rec_act, 0.0, jnp.abs(rec_lp)) == 0.0)
            and jnp.all(jnp.where(old_act, 0.0, jnp.abs(old_lp)) == 0.0)
        ),
        "atol": atol,
    }
    semantic = [
        name
        for name, ok in (
            ("active_bitwise_equal", rep["active_bitwise_equal"]),
            ("teacher_drop_zero", n_drop == 0),
            ("rollout_logp_finite", rep["rollout_logp_finite"]),
            ("teacher_old_logp_finite", rep["teacher_old_logp_finite"]),
            ("inactive_logp_all_zero", rep["inactive_logp_all_zero"]),
        )
        if not ok
    ]
    if semantic:
        raise ValueError(f"分母重算的語意檢查失敗（STOP）：{semantic}；{rep}")
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"未知 backend={backend!r}：沒有對應的數值出口，拒絕（STOP）")
    if backend == "cpu":
        if rep["max_abs_token_logp_diff"] > atol:
            raise ValueError(f"CPU 嚴格對帳失敗（STOP）：{rep}")
    elif budget["max_abs_expm1_active"] > RATIO_BUDGET:
        raise ValueError(f"零更新比率偏差超預算（STOP）：{rep}")
    return jax.lax.stop_gradient(rec_lp), rep


def precompute_denominators(
    params_old,
    cfg_model: dict[str, Any],
    batches: dict[int, dict[str, Any]],
    plan: list[dict[str, Any]],
    *,
    decoders,
    caps: caps_mod.Caps = PB.CAPS,
    atol: float = DENOM_ATOL,
    backend: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """**首次 update 之前**，照 `schedule()` 排好的**每一個實際 minibatch** 用凍結的舊 params
    重算一次 teacher logp，作為**診斷**與對帳。

    **分母不是這份重算**：PPO 的分母是產生樣本的行為機率，也就是 rollout 保存的
    `rows["logp_h"]`（見 `DENOMINATOR_SOURCE`）。這裡照 minibatch 形狀算，是為了讓診斷與
    update 當下吃的 batch **同形狀同精度**；整批一次算再切開在 GPU 上不保證逐位相同。

    回 `(rows_list, report)`；每個 `rows` 保留 `logp_h`（**分母**）與
    `teacher_old_logp_h`（**診斷用**的重算值），以及逐 token 差與比率偏差的統計。
    """
    backend = backend or jax.devices()[0].platform
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"未知 backend={backend!r}：沒有對應的數值出口，拒絕（STOP）")
    rows_list, reports = [], []
    for item in plan:
        seat = int(item["seat"])
        rows = take_rows(batches[seat]["batch"], item["idx"])
        rows["teacher_old_logp_h"], rep = _denominator_rows(
            params_old,
            cfg_model,
            rows,
            seat=seat,
            decoders=decoders,
            caps=caps,
            atol=atol,
            backend=backend,
        )
        rep = {**rep, "update": item["update"], "seat": seat, "minibatch": item["minibatch"]}
        rows_list.append(rows)
        reports.append(rep)
    summary = {
        "n_minibatches": len(reports),
        "rows_per_minibatch": [r["n_rows"] for r in reports],
        "max_abs_token_logp_diff": max(
            (r["max_abs_token_logp_diff"] for r in reports), default=0.0
        ),
        "max_abs_joint_logp_diff": max(
            (r["max_abs_joint_logp_diff"] for r in reports), default=0.0
        ),
        "teacher_drop_total": sum(r["teacher_drop_total"] for r in reports),
        "atol": atol,
        "backend": backend,
        "denominator_source": DENOMINATOR_SOURCE,
        "n_slots_over_diagnostic_line": sum(r["n_slots_over_diagnostic_line"] for r in reports),
        "max_ratio_budget_abs_expm1": max(
            (r["ratio_budget"]["max_abs_expm1_active"] for r in reports), default=0.0
        ),
        "ratio_budget": RATIO_BUDGET,
        "per_minibatch": reports,
        "note": "**分母是 rollout 保存的 logp_h**；這裡的 teacher 重算只作診斷。"
        "語意檢查任何平台都硬拒；CPU 逐 token 嚴格對帳，GPU 核零更新比率偏差預算。"
        "全量逐批，不抽樣避開失敗。",
    }
    return rows_list, summary


# ------------------------------------------------------------------ loss
def anchor_kl(out_theta: dict, out_ref: dict, active: jax.Array) -> dict[str, jax.Array]:
    """同一 prefix 下逐槽 exact forward KL：`Σ_c p_IL(c)·(log p_IL(c) − log p_θ(c))`。

    - 只在兩邊合法 mask 都為 True 的類別求和（非法／補齊類別不進，避免 0×(−inf−−inf)）。
    - IL 側整份 `stop_gradient`；只在 `active`（rollout 的 policy active）槽計入。
    - 回每列 row-sum ``[B]``（nats，不除 active 數）與逐槽 ``[B, 263]``，外加 mask 一致性。
    """
    lp_r = jax.lax.stop_gradient(out_ref["dist"]["lp"])
    m_r = out_ref["dist"]["mask"]
    lp_t, m_t = out_theta["dist"]["lp"], out_theta["dist"]["mask"]
    both = m_r & m_t
    p_r = jnp.where(both, jnp.exp(lp_r), 0.0)
    kl_slot = jnp.sum(jnp.where(both, p_r * (lp_r - lp_t), 0.0), axis=-1)  # [B, 263]
    kl_slot = jnp.where(active, kl_slot, 0.0)
    act_b = active[..., None]
    return {
        "row_sum": jnp.sum(kl_slot, axis=-1),
        "slot": kl_slot,
        "mask_equal": jnp.all(jnp.where(act_b, m_r == m_t, True)),
    }


def make_loss(
    cfg_model: dict[str, Any],
    cfg: Cfg,
    *,
    seat: int,
    caps: caps_mod.Caps = PB.CAPS,
    effect_mask: bool = False,
):
    """回一支 **static seat** 的 `(params, rows[, ref]) → (loss, metrics)`（外層自己 jit）。

    分母 ＝ `stop_gradient(rows["logp_h"])`（rollout 的行為機率，見 `DENOMINATOR_SOURCE`）；
    梯度只從 `new_lp` 流出去。`teacher_old_logp_h` 若在 rows 裡只被當診斷欄位量一個差。

    `cfg.anchor_beta > 0` 時必須給 `ref`（固定原 IL 參數）：兩模型沿同一 rollout decision／
    prefix 各自前向（**不共享模型 cache**），加 `β · mean_rows Σ_active KL(π_IL ‖ π_θ)`。
    """
    use_anchor = float(cfg.anchor_beta) > 0.0
    teacher = D.decode("teacher", dist=use_anchor)

    def loss_fn(params, rows, ref=None):
        if use_anchor and ref is None:
            raise ValueError("anchor_beta > 0 必須給 ref（原 IL 參數）")
        labels = {k: rows["decision"][k] for k in PB.LABEL_FIELDS}
        out = teacher(params, cfg_model, rows["state"], seat, caps, labels=labels)
        new_lp, act = out["logp"], rows["active"]
        drop = out["teacher_drop"]
        n_drop = sum(jnp.sum(v) for grp in ("unit", "market") for v in drop[grp].values())
        # **PPO 的分母是產生樣本的行為機率**：rollout 當下抽樣保存的 `logp_h`。
        # teacher 重算（`teacher_old_logp_h`）只作診斷，不進 loss。
        old_lp = jax.lax.stop_gradient(rows["logp_h"])
        adv = jax.lax.stop_gradient(rows["adv"])
        ret = jax.lax.stop_gradient(rows["ret"])
        pg, pg_stats = P1.factored_surrogate(new_lp, old_lp, act, adv)
        eff_metrics = {}
        score = act
        if effect_mask:
            # T3 effect-mask：surrogate 只對 score_active 求和（row 內求和、row 間原聚合）；
            # clip 比例等 pg_stats 仍用 policy_active（同 v3 公式）
            noeff = rows["noeffect"]
            score = act & ~noeff
            pg, _ = P1.factored_surrogate(new_lp, old_lp, score, adv)
            ne = act & noeff
            r_ = jnp.exp(new_lp - old_lp)
            eff_metrics = {
                "n_score_total": jnp.sum(score.astype(F32)),
                "n_noeffect_total": jnp.sum(ne.astype(F32)),
                "noeffect_subset_active": jnp.all(~noeff | act).astype(F32),
                # 所有 policy_active（含 noeffect）的 ratio 有限；inactive 不計（審查者 B1）
                "ratio_policy_active_finite": jnp.all(~act | jnp.isfinite(r_)).astype(F32),
                "ratio_raw_min": jnp.min(jnp.where(act, r_, jnp.inf)),
                "ratio_raw_max": jnp.max(jnp.where(act, r_, -jnp.inf)),
                "ratio_noeffect_min": jnp.min(jnp.where(ne, r_, jnp.inf)),
                "ratio_noeffect_max": jnp.max(jnp.where(ne, r_, -jnp.inf)),
                "ratio_stats_raw": ratio_stats(
                    new_lp - old_lp, act, rows, new_lp=new_lp, old_lp=old_lp
                ),
            }
        value = out["value"]
        value_loss = 0.5 * jnp.mean((value - ret) ** 2)
        ent_rows = jnp.sum(jnp.where(act, out["entropy"], 0.0), axis=-1)
        entropy = jnp.mean(ent_rows)
        loss = pg + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
        anchor_metrics = {}
        if use_anchor:
            ref_p = jax.lax.stop_gradient(ref)
            out_ref = teacher(ref_p, cfg_model, rows["state"], seat, caps, labels=labels)
            kl = anchor_kl(out, out_ref, act)
            kl_mean = jnp.mean(kl["row_sum"])
            loss = loss + cfg.anchor_beta * kl_mean
            n_act = jnp.sum(act.astype(F32))
            anchor_metrics = {
                "anchor_kl_rowsum_mean": kl_mean,
                "anchor_kl_rowsum_max": jnp.max(kl["row_sum"]),
                "anchor_kl_per_active_mean": jnp.sum(kl["row_sum"]) / jnp.maximum(n_act, 1.0),
                "anchor_term": cfg.anchor_beta * kl_mean,
                "anchor_kl_finite": jnp.all(jnp.isfinite(kl["slot"])).astype(F32),
                "anchor_mask_equal": kl["mask_equal"].astype(F32),
                "anchor_ref_active_equal": jnp.all(out_ref["active"] == act).astype(F32),
            }
        ratio = jnp.exp(new_lp - old_lp)
        var_ret = jnp.var(ret)
        metrics = {
            "loss": loss,
            "pg": pg,
            "value_loss": value_loss,
            "entropy": entropy,
            "n_active_per_row_mean": jnp.mean(jnp.sum(act, axis=-1).astype(F32)),
            "n_active_total": jnp.sum(act.astype(F32)),
            # enforce 用的極值：v3 為 policy_active；effect-mask 開時為 score_active
            "ratio_active_max": jnp.max(jnp.where(score, ratio, -jnp.inf)),
            "ratio_active_min": jnp.min(jnp.where(score, ratio, jnp.inf)),
            "value_mean": jnp.mean(value),
            "ret_mean": jnp.mean(ret),
            "ret_var": var_ret,
            # ret 方差 0 時 EV 沒有定義——回 nan，讓呼叫端照實記，不冒充 1.0
            "explained_variance": jnp.where(
                var_ret > 0, 1.0 - jnp.var(ret - value) / (var_ret + 1e-12), jnp.nan
            ),
            "offmask_logp_abs_max": jnp.max(jnp.where(act, 0.0, jnp.abs(new_lp))),
            # 診斷：teacher 重算與行為機率的差（**不參與 loss**），沒帶就記 nan
            "teacher_old_vs_rollout_max_abs": (
                jnp.max(jnp.where(act, jnp.abs(rows["teacher_old_logp_h"] - rows["logp_h"]), 0.0))
                if "teacher_old_logp_h" in rows
                else jnp.asarray(jnp.nan, F32)
            ),
            # 候選更新在 host 端被硬拒時要看得到原因（jit 內不能 raise）
            "active_bitwise_equal": jnp.all(out["active"] == act).astype(F32),
            "teacher_drop_total": n_drop.astype(F32),
            "new_logp_finite": jnp.all(jnp.isfinite(new_lp)).astype(F32),
            "entropy_finite": jnp.all(jnp.isfinite(out["entropy"])).astype(F32),
            "loss_finite": jnp.isfinite(loss).astype(F32),
            **pg_stats,
            **anchor_metrics,
            **eff_metrics,
            # 候選 ratio 的分布統計（計數／直方圖／極值位置；不存 logits）
            "ratio_stats": ratio_stats(new_lp - old_lp, score, rows, new_lp=new_lp, old_lp=old_lp),
        }
        return loss, metrics

    return loss_fn


def check_zero_update_budget(
    metrics: dict[str, Any], *, where: str, budget: float = RATIO_BUDGET
) -> dict[str, Any]:
    """**第一次接受任何更新之前**，用實際融合前向（`make_update` 那支）的輸出核零更新偏差。

    未更新策略時 ratio 應該是 1；這裡用 `make_loss` 已回的 `ratio_active_min/max` 取
    偏差 bound（不另存整張 logits），超過預算就 `ValueError`（STOP）。
    同時把 active／drop／finite 的語意檢查一起做——融合反傳的前向與獨立 teacher 編譯
    **不保證逐位相同**，所以這一關不能靠分母預算那次代勞。
    """
    m = {k: float(v) for k, v in scalar_metrics(metrics).items()}
    # effect-mask（v4）：`ratio_active_*` 是 score 極值；零更新數值同值檢查不因免分縮小，
    # 改讀 policy_active 的 raw 極值（審查者 B1）。v1–v3 沒有 raw 欄，走原欄位。
    raw = "ratio_raw_min" in m
    lo, hi = (
        (m["ratio_raw_min"], m["ratio_raw_max"])
        if raw
        else (m["ratio_active_min"], m["ratio_active_max"])
    )
    if raw and m["ratio_policy_active_finite"] != 1.0:
        raise ValueError(f"零更新校驗失敗（STOP）：policy_active ratio 非有限；{where}")
    # NaN 會被 max() 吞掉 ⇒ 先硬核有限與 min<=max，再算 bound
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"零更新校驗失敗（STOP）：ratio 非有限 min={lo}、max={hi}；{where}")
    if lo > hi:
        raise ValueError(f"零更新校驗失敗（STOP）：ratio_min {lo} > ratio_max {hi}；{where}")
    dev = ratio_budget_from_metrics({"ratio_active_min": lo, "ratio_active_max": hi})
    report = {
        "where": where,
        "ratio_active_min": m["ratio_active_min"],
        "ratio_active_max": m["ratio_active_max"],
        **({"ratio_raw_minmax": [lo, hi], "deviation_set": "policy_active（raw）"} if raw else {}),
        "max_abs_ratio_deviation": dev,
        "budget": budget,
        "active_bitwise_equal": m["active_bitwise_equal"] == 1.0,
        "teacher_drop_total": m["teacher_drop_total"],
        "new_logp_finite": m["new_logp_finite"] == 1.0,
        "entropy_finite": m["entropy_finite"] == 1.0,
        "loss_finite": m["loss_finite"] == 1.0,
        "teacher_old_vs_rollout_max_abs": m.get("teacher_old_vs_rollout_max_abs"),
        "note": "候選 params／opt **一律丟棄**，這一步不接受任何更新；成本照實計入",
    }
    bad = [
        k
        for k in (
            "active_bitwise_equal",
            "new_logp_finite",
            "entropy_finite",
            "loss_finite",
        )
        if not report[k]
    ]
    if report["teacher_drop_total"] != 0.0:
        bad.append("teacher_drop_total")
    if dev > budget:
        bad.append(f"max_abs_ratio_deviation {dev:.3e} > {budget:.0e}")
    if bad:
        raise ValueError(f"零更新校驗失敗（STOP）：{bad}；{report}")
    return report


def accept_or_reject(metrics: dict[str, Any], grads, params) -> dict[str, Any]:
    """**host 端硬檢**：不符就 `ValueError`，不讓壞的候選 params 被接受。

    jit 裡不能 raise，所以 `make_loss` 把 active／drop／finite 以 0/1 回出來，這裡判。
    """
    m = {k: float(v) for k, v in scalar_metrics(metrics).items()}
    grad_finite = all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))
    param_finite = all(bool(jnp.all(jnp.isfinite(p))) for p in jax.tree.leaves(params))
    report = {
        "active_bitwise_equal": m["active_bitwise_equal"] == 1.0,
        "teacher_drop_total": m["teacher_drop_total"],
        "new_logp_finite": m["new_logp_finite"] == 1.0,
        "entropy_finite": m["entropy_finite"] == 1.0,
        "loss_finite": m["loss_finite"] == 1.0,
        "grads_finite": grad_finite,
        "params_finite": param_finite,
    }
    # KL 錨開著時：KL 有限、兩模型合法 mask 在 active 槽一致、IL 側 active 與 rollout 相同
    for k in ("anchor_kl_finite", "anchor_mask_equal", "anchor_ref_active_equal"):
        if k in m:
            report[k] = m[k] == 1.0
    bad = [k for k, v in report.items() if k != "teacher_drop_total" and not v]
    if bad or report["teacher_drop_total"] != 0.0:
        raise ValueError(f"候選更新被拒（STOP）：{bad or 'teacher_drop != 0'}；{report}")
    return report


def make_update(
    cfg_model: dict[str, Any],
    cfg: Cfg,
    opt,
    *,
    seat: int,
    caps=PB.CAPS,
    ref_params=None,
    effect_mask: bool = False,
):
    """一支 static seat 的 jit update step：`step(params, opt_state, rows)`。

    `cfg.anchor_beta > 0` 時必須給 `ref_params`（固定原 IL）；它以 **jit 引數**傳入
    （不烘成常數、不跨輪快取），梯度只對 `params` 取。metrics 另附 `grad_norm`（clip 前全域範數）。
    """
    use_anchor = float(cfg.anchor_beta) > 0.0
    if use_anchor and ref_params is None:
        raise ValueError("anchor_beta > 0 必須給 ref_params（原 IL）")
    if not use_anchor and ref_params is not None:
        raise ValueError("anchor_beta == 0 不收 ref_params（避免以為錨有作用）")
    loss_fn = make_loss(cfg_model, cfg, seat=seat, caps=caps, effect_mask=effect_mask)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def _step(params, opt_state, rows, ref):
        (loss, metrics), grads = grad_fn(params, rows, ref)
        metrics = {**metrics, "grad_norm": optax.tree.norm(grads)}
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics, grads

    def step(params, opt_state, rows):
        return _step(params, opt_state, rows, ref_params)

    step._cache_size = _step._cache_size  # `cost.cache_size` 讀這個
    step.anchor = use_anchor
    return step


def make_post_check(
    cfg_model: dict[str, Any], *, seat: int, caps=PB.CAPS, effect_mask: bool = False
):
    """輪末全量 post-update 檢查的**單批**無梯度前向（T3 §6.2 第 2、3 點）。

    `check(params, ref, rows)` → 用**原 rollout 分母** `rows["logp_h"]` 算 post-update ratio 的
    active min／max、clip 計數、anchor KL row-sum（對 `ref`＝原 IL），以及語意檢查。
    不回傳 logits；host 端逐批彙總後由 `check_post_update` 判定。
    """
    teacher = D.decode("teacher", dist=True)

    @jax.jit
    def check(params, ref, rows):
        labels = {k: rows["decision"][k] for k in PB.LABEL_FIELDS}
        out = teacher(params, cfg_model, rows["state"], seat, caps, labels=labels)
        out_ref = teacher(ref, cfg_model, rows["state"], seat, caps, labels=labels)
        act = rows["active"]
        m = act.astype(F32)
        r = jnp.exp(out["logp"] - rows["logp_h"])
        kl = anchor_kl(out, out_ref, act)
        # enforce 集合：v3＝policy_active；effect-mask＝score_active（clip／anchor 仍用 active）
        score = act & ~rows["noeffect"] if effect_mask else act
        eff = {}
        if effect_mask:
            ne = act & rows["noeffect"]
            kl_ne = jnp.where(ne, kl["slot"], 0.0)
            eff = {
                "n_score": jnp.sum(score.astype(F32)),
                "n_noeffect": jnp.sum(ne.astype(F32)),
                "noeffect_subset_active": jnp.all(~rows["noeffect"] | act).astype(F32),
                "ratio_policy_active_finite": jnp.all(~act | jnp.isfinite(r)).astype(F32),
                "ratio_raw_min": jnp.min(jnp.where(act, r, jnp.inf)),
                "ratio_raw_max": jnp.max(jnp.where(act, r, -jnp.inf)),
                "ratio_noeffect_min": jnp.min(jnp.where(ne, r, jnp.inf)),
                "ratio_noeffect_max": jnp.max(jnp.where(ne, r, -jnp.inf)),
                "noeffect_kl_sum": jnp.sum(kl_ne),
                "noeffect_kl_rowsum_max": jnp.max(jnp.sum(kl_ne, axis=-1)),
                "n_rows_with_noeffect": jnp.sum(jnp.any(ne, axis=-1).astype(F32)),
                "ratio_stats_raw": ratio_stats(
                    out["logp"] - rows["logp_h"],
                    act,
                    rows,
                    new_lp=out["logp"],
                    old_lp=rows["logp_h"],
                    ref_lp=out_ref["logp"],
                ),
            }
        drop = sum(
            jnp.sum(v)
            for o in (out, out_ref)
            for g in ("unit", "market")
            for v in o["teacher_drop"][g].values()
        )
        return {
            "n_rows": jnp.asarray(act.shape[0], F32),
            "n_active": jnp.sum(m),
            "ratio_active_min": jnp.min(jnp.where(score, r, jnp.inf)),
            "ratio_active_max": jnp.max(jnp.where(score, r, -jnp.inf)),
            "n_clipped": jnp.sum(m * (jnp.abs(r - 1.0) > P1.CLIP)),
            "anchor_kl_rowsum_sum": jnp.sum(kl["row_sum"]),
            "anchor_kl_rowsum_max": jnp.max(kl["row_sum"]),
            "logp_finite": jnp.all(jnp.isfinite(out["logp"])).astype(F32),
            "anchor_kl_finite": jnp.all(jnp.isfinite(kl["slot"])).astype(F32),
            "anchor_mask_equal": kl["mask_equal"].astype(F32),
            "active_equal": (
                jnp.all(out["active"] == act) & jnp.all(out_ref["active"] == act)
            ).astype(F32),
            "teacher_drop_total": drop.astype(F32),
            "ratio_stats": ratio_stats(
                out["logp"] - rows["logp_h"],
                score,
                rows,
                new_lp=out["logp"],
                old_lp=rows["logp_h"],
                ref_lp=out_ref["logp"],
            ),
            **eff,
        }

    return check


def check_post_update(
    per_batch: list[dict[str, Any]],
    budget: HealthBudget,
    *,
    where: str,
    teacher_env_id: int | None = None,
) -> dict[str, Any]:
    """彙總輪末全量 post-update 檢查；任一不過 ⇒ `ValueError`（STOP，本輪不寫完整 checkpoint）。

    - ratio：所有批 active 的 min ≥ `ratio_lo`、max ≤ `ratio_hi`（原 rollout 分母）。
    - clip：逐批 clip fraction 的**算術**均值 ≤ `clip_frac_mean_max`；token 加權值另報、不混分母。
    - anchor KL：全部有效列的 row-sum 均值 ≤ `anchor_kl_rowsum_max`；逐座另報。
    - 尾部（budget 有 `tail_frac_max`）：全域 `n_out/N <= ε`（等號接受）；N=0／非有限不過。
    - 語意：logp／KL 有限、active 與 rollout 逐位相同、兩模型 mask 一致、teacher drop 0。
    """
    if not budget.post_update_check:
        raise ValueError("此 budget 未開 post_update_check")
    if not per_batch:
        raise ValueError(f"post-update 檢查沒有任何批（STOP）：{where}")
    f = [
        {
            k: (v if k in ("seat", "batch") else float(v))
            for k, v in b.items()
            if not k.startswith("ratio_stats")
        }
        for b in per_batch
    ]
    n_rows = sum(b["n_rows"] for b in f)
    n_act = sum(b["n_active"] for b in f)
    fracs = [b["n_clipped"] / max(b["n_active"], 1.0) for b in f]
    per_seat = {}
    for seat in sorted({b["seat"] for b in f}):
        fs = [b for b in f if b["seat"] == seat]
        per_seat[f"seat{seat}"] = {
            "n_batches": len(fs),
            "n_rows": sum(b["n_rows"] for b in fs),
            "anchor_kl_rowsum_mean": sum(b["anchor_kl_rowsum_sum"] for b in fs)
            / max(sum(b["n_rows"] for b in fs), 1.0),
            "ratio_active_min": min(b["ratio_active_min"] for b in fs),
            "ratio_active_max": max(b["ratio_active_max"] for b in fs),
        }
    rep_ = {
        "where": where,
        "budget": budget.name,
        "n_batches": len(f),
        "n_rows": n_rows,
        "n_active": n_act,
        "ratio_active_min": min(b["ratio_active_min"] for b in f),
        "ratio_active_max": max(b["ratio_active_max"] for b in f),
        "ratio_bounds": [budget.ratio_lo, budget.ratio_hi],
        "clip_frac_mean_of_batches": float(np.mean(fracs)),
        "clip_frac_token_weighted": sum(b["n_clipped"] for b in f) / max(n_act, 1.0),
        "clip_frac_limit": budget.clip_frac_mean_max,
        "anchor_kl_rowsum_mean": sum(b["anchor_kl_rowsum_sum"] for b in f) / max(n_rows, 1.0),
        "anchor_kl_rowsum_max": max(b["anchor_kl_rowsum_max"] for b in f),
        "anchor_kl_per_active_mean": sum(b["anchor_kl_rowsum_sum"] for b in f) / max(n_act, 1.0),
        "anchor_kl_limit": budget.anchor_kl_rowsum_max,
        "per_seat": per_seat,
        "method": "同輪全部有效 minibatch、更新後參數、原 rollout 分母；無梯度",
    }
    bad = []
    for k in ("logp_finite", "anchor_kl_finite", "anchor_mask_equal", "active_equal"):
        if not all(b[k] == 1.0 for b in f):
            bad.append(k)
    if any(b["teacher_drop_total"] != 0.0 for b in f):
        bad.append("teacher_drop_total")
    # 逐批核有限：Python 的 min／max 會吞掉 NaN，不能只看彙總值
    per_vals = [
        b[k]
        for b in f
        for k in ("ratio_active_min", "ratio_active_max", "anchor_kl_rowsum_sum", "n_clipped")
    ]
    if not all(np.isfinite(v) for v in per_vals):
        bad.append("non_finite")
    else:
        if rep_["ratio_active_min"] < budget.ratio_lo or rep_["ratio_active_max"] > budget.ratio_hi:
            bad.append("post_update_ratio_out_of_bounds")
        if rep_["anchor_kl_rowsum_mean"] > budget.anchor_kl_rowsum_max:
            bad.append("anchor_kl_over_budget")
    if rep_["clip_frac_mean_of_batches"] > budget.clip_frac_mean_max:
        bad.append("post_update_clip_frac_over_budget")
    if uses_effect_mask(budget):
        n_sc = sum(b["n_score"] for b in f)
        n_ne = sum(b["n_noeffect"] for b in f)
        ident = all(b["n_active"] == b["n_score"] + b["n_noeffect"] for b in f)
        mins = [b["ratio_noeffect_min"] for b in f if b["n_noeffect"] > 0]
        maxs = [b["ratio_noeffect_max"] for b in f if b["n_noeffect"] > 0]
        rep_["effect"] = {
            "rules_version": D.EFFECT_RULES_VERSION,
            "n_raw_active": n_act,
            "n_score": n_sc,
            "n_noeffect": n_ne,
            "raw_equals_score_plus_noeffect": ident and n_act == n_sc + n_ne,
            "noeffect_subset_active": all(b["noeffect_subset_active"] == 1.0 for b in f),
            "ratio_policy_active_finite": all(b["ratio_policy_active_finite"] == 1.0 for b in f),
            "enforce_ratio_minmax_score": [rep_["ratio_active_min"], rep_["ratio_active_max"]],
            "raw_ratio_minmax": [
                min(b["ratio_raw_min"] for b in f),
                max(b["ratio_raw_max"] for b in f),
            ],
            "noeffect_ratio_minmax": [min(mins) if mins else None, max(maxs) if maxs else None],
            "noeffect_kl_sum": sum(b["noeffect_kl_sum"] for b in f),
            "noeffect_kl_rowsum_max": max(b["noeffect_kl_rowsum_max"] for b in f),
            "n_rows_with_noeffect": sum(b["n_rows_with_noeffect"] for b in f),
            "noeffect_kl_rowsum_mean_over_rows_with_noeffect": (
                sum(b["noeffect_kl_sum"] for b in f)
                / max(sum(b["n_rows_with_noeffect"] for b in f), 1.0)
            ),
            "noeffect_kl_token_mean": sum(b["noeffect_kl_sum"] for b in f) / max(n_ne, 1.0),
            "mode": "report-only（noeffect 讀數無門檻）；enforce 項用 score_active",
        }
        if not rep_["effect"]["raw_equals_score_plus_noeffect"]:
            bad.append("effect_count_identity")
        if not rep_["effect"]["noeffect_subset_active"]:
            bad.append("noeffect_not_subset_active")
        # 免分不免 finite（審查者 B1）：有限而越界的 noeffect 只報
        if not rep_["effect"]["ratio_policy_active_finite"]:
            bad.append("ratio_policy_active_nonfinite")
        if not n_sc > 0:
            bad.append("score_empty")
        if all("ratio_stats_raw" in b for b in per_batch):
            rep_["stats_raw"] = aggregate_ratio_stats(
                [{"batch": b.get("batch"), "stats": b["ratio_stats_raw"]} for b in per_batch],
                None,
                teacher_env_id=teacher_env_id,
            )
    if all("ratio_stats" in b for b in per_batch):
        rep_["stats"] = aggregate_ratio_stats(
            [{"batch": b.get("batch"), "stats": b["ratio_stats"]} for b in per_batch],
            budget,
            teacher_env_id=teacher_env_id,
            phase="global_post",
        )
        tail = rep_["stats"].get("tail")
        if tail and tail["pass"] is False:
            bad.append("post_update_tail_frac_over_budget")
    rep_["failed"] = bad
    if bad:
        brief = {k: v for k, v in rep_.items() if k not in ("stats", "stats_raw")}
        raise PostUpdateStop(
            f"輪末 post-update 檢查未過（STOP，本輪不算完整輪）：{bad}；{brief}", rep_
        )
    return rep_


class PostUpdateStop(ValueError):
    """輪末檢查 STOP；`report` 帶完整彙總（含分布統計），供 trainer 落盤。"""

    def __init__(self, msg: str, report: dict[str, Any]):
        super().__init__(msg)
        self.report = report


#: 輪末 post-update 的**健康門**失敗理由（有限值越過預登記停止線）；其餘 failed 理由
#: （logp／KL 非有限、mask／active 不一致、teacher_drop、effect 計數恆等式、score_empty 等）
#: ＝契約異常
POST_UPDATE_HEALTH_REASONS = frozenset(
    {
        "post_update_ratio_out_of_bounds",
        "anchor_kl_over_budget",
        "post_update_clip_frac_over_budget",
        "post_update_tail_frac_over_budget",
    }
)


def health_stop_info(exc: BaseException) -> dict[str, Any] | None:
    """計畫 winloss v3.1 §5 第 1 項＋審查者 實作審 B1：**明確型別**分類，未知一律異常（None）。

    - `HealthGateStop` ⇒ 健康（gate 名）；
    - `PostUpdateStop` ⇒ failed 非空且**全部**屬 `POST_UPDATE_HEALTH_REASONS` 才算健康；
      混有任一結構理由即異常；
    - 其他（含訊息帶「STOP」的一般 `ValueError`：rollout／shape／列數／padding 契約）⇒ 異常。"""
    if isinstance(exc, HealthGateStop):
        return {"kind": "health_gate", "gates": [exc.gate]}
    if isinstance(exc, PostUpdateStop):
        failed = list((exc.report or {}).get("failed") or [])
        if failed and set(failed) <= POST_UPDATE_HEALTH_REASONS:
            return {"kind": "post_update_health", "gates": failed}
    return None


def is_health_stop(exc: BaseException) -> bool:
    return health_stop_info(exc) is not None


def component_grad_norms(
    cfg_model: dict[str, Any], cfg: Cfg, rows, *, seat: int, params, caps=PB.CAPS
):
    """pg／value／entropy 三項各自的 grad norm（**同一個小批算一次**，
    不在每個正式 update 額外做三次反傳）。"""
    teacher = D.decode("teacher")

    def parts(p):
        labels = {k: rows["decision"][k] for k in PB.LABEL_FIELDS}
        out = teacher(p, cfg_model, rows["state"], seat, caps, labels=labels)
        act = rows["active"]
        pg, _ = P1.factored_surrogate(
            out["logp"],
            jax.lax.stop_gradient(rows["teacher_old_logp_h"]),
            act,
            jax.lax.stop_gradient(rows["adv"]),
        )
        v = 0.5 * jnp.mean((out["value"] - jax.lax.stop_gradient(rows["ret"])) ** 2)
        h = jnp.mean(jnp.sum(jnp.where(act, out["entropy"], 0.0), axis=-1))
        return pg, v, h

    out = {}
    for i, name in enumerate(("pg", "value", "entropy")):
        g = jax.grad(lambda p, i=i: parts(p)[i])(params)
        out[f"grad_norm_{name}"] = float(optax.global_norm(g))
    return out


# ------------------------------------------------------------- resume 身分
#: resume 時**逐項比較**的身分欄（不是只存起來好看）。
IDENTITY_KEYS = (
    "denominator_source",
    "denominator_policy",
    "health_budget",
    "cfg",
    "caps",
    "coefs",
    "reference_sha256",
    "il_start_sha256",
    "init_state",
    "source_files",
    "matmul_precision",
    "backend",
    # 係數遷移歷史（ppo150k iter3 修復 R1）：未遷移的 run 為 None；
    # 舊 checkpoint 無此欄 ⇒ .get 為 None，照舊全等
    "coef_migrations",
)


def resume_identity(
    *,
    cfg_model: dict[str, Any],
    caps_meta: dict[str, Any],
    cfg: Cfg,
    reference_sha256: str,
    il_start_sha256: str,
    init_state: dict[str, Any],
    source_files: dict[str, str],
    matmul_precision: str,
    backend: str,
    denominator_source: str,
    denominator_policy_: dict[str, Any],
    health_budget: dict[str, Any] | None,
    coef_migrations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """checkpoint 要記、resume 要比的那一份。內容為準，`git_head` 另記但不單獨當判準。

    `denominator_source`／`denominator_policy` 進身分 ⇒ **舊 teacher 分母的 checkpoint
    不可能靜默續接**（欄位不同就 `check_resume_identity` 直接拒）。
    """
    return {
        "denominator_source": denominator_source,
        "denominator_policy": denominator_policy_,
        # 成本量測模式是 None；開了 budget 的 leg 與它 resume 身分不同 ⇒ 不可互接
        "health_budget": health_budget,
        "cfg": cfg_model,
        "caps": caps_meta,
        "coefs": cfg._asdict(),
        "reference_sha256": reference_sha256,
        "il_start_sha256": il_start_sha256,
        "init_state": init_state,
        "source_files": source_files,
        "matmul_precision": matmul_precision,
        "backend": backend,
        "coef_migrations": coef_migrations,
    }


def check_resume_identity(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """逐欄比對；任一不同 ⇒ `ValueError`（換 reference／caps／精度／依賴都要擋）。"""
    if not saved:
        raise ValueError("checkpoint 沒有 identity：拒絕 resume（不靜默 fresh start）")
    check_tail_contract(current.get("health_budget"), where="resume 目前身分")
    check_tail_contract(saved.get("health_budget"), where="resume checkpoint 身分")
    check_effect_contract(current.get("health_budget"), where="resume 目前身分")
    check_effect_contract(saved.get("health_budget"), where="resume checkpoint 身分")
    diff = {
        k: {"saved": saved.get(k), "current": current.get(k)}
        for k in IDENTITY_KEYS
        if saved.get(k) != current.get(k)
    }
    if diff:
        raise ValueError(f"resume 身分不符，拒載：{sorted(diff)}；{diff}")
    return {"compared_keys": list(IDENTITY_KEYS), "all_equal": True}


def schedule(
    batches: dict[int, dict[str, Any]],
    rng: np.random.Generator,
    n_updates: int,
    *,
    start_update: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把「哪一輪、哪一座、吃哪幾列」**先排完**，因為分母要照實際 minibatch 形狀預算。

    RNG 的消耗順序就是這裡的順序。**resume 必須用還原的 RNG 重新排**下一輪，
    不能沿用記憶體裡還沒存進 checkpoint 的未來 plan——那樣 RNG 比對會恆真、
    等於沒測到重播。`start_update` 讓續跑接在正確的輪次編號上。
    """
    plan, dropped = [], {}
    for it in range(start_update, start_update + n_updates):
        for seat in sorted(batches):
            b = batches[seat]
            idx_list, n_drop = minibatches(int(b["n_rows"]), int(b["minibatch"]), rng)
            dropped[f"update{it}_seat{seat}"] = n_drop
            for mb, idx in enumerate(idx_list):
                plan.append({"update": it, "seat": seat, "minibatch": mb, "idx": idx})
    return plan, {
        "n_steps": len(plan),
        "n_rows_dropped_tail": dropped,
        "order": [(p_["update"], p_["seat"], p_["minibatch"]) for p_ in plan],
    }


def plans_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """兩份排程逐項相同（輪次／座位／批序／列索引）。resume 重建後要對得上。"""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if (x["update"], x["seat"], x["minibatch"]) != (y["update"], y["seat"], y["minibatch"]):
            return False
        if not np.array_equal(np.asarray(x["idx"]), np.asarray(y["idx"])):
            return False
    return True


def minibatches(n: int, size: int, rng: np.random.Generator) -> tuple[list[np.ndarray], int]:
    """固定 RNG 的 shuffle 切批；**尾批照實丟並回報行數**，不補列、不繞回。"""
    perm = rng.permutation(n)
    n_full = n // size
    if n_full == 0:
        raise ValueError(f"這一座只有 {n} 列，湊不出一個 {size} 列的批：明拒，不縮批")
    return [perm[i * size : (i + 1) * size] for i in range(n_full)], int(n - n_full * size)


__all__ = [
    "DENOM_ATOL",
    "HEALTH_BUDGET_V1",
    "HealthBudget",
    "check_candidate_ratio",
    "check_kl_to_il",
    "check_round_clip_fraction",
    "health_budget_meta",
    "HEALTH_BUDGET_T3_V4",
    "EFFECT_BUDGETS",
    "uses_effect_mask",
    "effect_contract",
    "check_effect_contract",
    "check_tail_contract",
    "tail_contract",
    "DENOMINATOR_SOURCE",
    "RATIO_BUDGET",
    "Cfg",
    "check_zero_update_budget",
    "denominator_policy",
    "ratio_budget",
    "ratio_budget_from_metrics",
    "batch_from",
    "accept_or_reject",
    "IDENTITY_KEYS",
    "check_cfg",
    "REWARDS",
    "advantages",
    "is_health_stop",
    "health_stop_info",
    "HealthGateStop",
    "POST_UPDATE_HEALTH_REASONS",
    "winloss_outcome",
    "winloss_rewards",
    "check_resume_identity",
    "component_grad_norms",
    "make_loss",
    "make_optimizer",
    "make_update",
    "minibatches",
    "normalize_joint",
    "opt_config",
    "plans_equal",
    "precompute_denominators",
    "resume_identity",
    "schedule",
    "rewards_of",
    "take_rows",
    "targets",
]
