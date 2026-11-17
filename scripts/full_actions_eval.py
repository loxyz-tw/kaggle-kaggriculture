#!/usr/bin/env python
"""full-actions runtime 驗收：**官方引擎裡**跑 NumPy agent，只讀 hook 記真效果。

規格：內部規格文件（審查者 §31 PASS 的必要規格）。
唯讀盤點在 內部量測紀錄——**本檔依規格取代該稿的「回合淨差推定成交」**。

為什麼不能用回合淨差
--------------------
同回合 `PLACE→SELL` 會讓 shed 淨差為零；`SELL→BUY` 同品項也會；日末清 `watered_today`、
自動倒包、工人重置都會遮住當步效果。所以「shed 減少且 money 增加 ⇒ 賣掉了」是錯的量測，
不做。改成在**官方原函式前後**只讀記錄。

只讀 hook 的邊界
----------------
* 包 `_apply_unit_action` 與 `_commit_unit`：**原函式各呼叫一次**、回傳原結果、
  不改參數／state／RNG；`finally` 還原（例外也還原）。
* 外層 `interpreter` wrapper 只建立「當步 turn／player／farm 參照 ＋ 原 request 對位」，
  再呼叫原 interpreter **一次**，不改它的執行流程。wrapper 必須是**明確 `(state, env)`
  兩參數 closure**——`kaggle_environments` 會看 `__code__.co_argcount`，
  綁定 method 或 `*args` 會被判成別的 arity。
* 記錄實際載入的官方模組路徑與 SHA256，並核三個函式的介面；**缺介面明拒**，
  不退回回合淨差冒稱 actual。
* audit **不向 policy 傳任何資訊**：agent 只吃自己的 observation。

三欄並列
--------
`raw`（我方送出的原 request）／`allowed`（引擎 atomic PLANT 預驗後真正傳進
`_apply_unit_action` 的動作）／`effect`（前後差分分類出來的成功事件）。
三者分開報，不因為 intent 出現就記成功。

本檔**第一段只做**：hook ＋ `env.run` 真 agent ＋ 兩局 ＋ 100 次 obs 解析計時。
32 局 launcher、教師 128 列探針、PPO 都不在這一段。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import os
import resource
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

from rl import caps as caps_mod  # noqa: E402
from rl import rules  # noqa: E402
from scripts.loop4_eval import _daily_probe, fold_daily  # noqa: E402
from scripts.loop4_parity import check_action_json  # noqa: E402

CAPS = caps_mod.EXACT_REFERENCE
#: 官方函式的介面契約（參數名逐字比對；不符就拒，不猜）
EXPECTED_SIGNATURES = {
    "_apply_unit_action": (
        "farm",
        "private",
        "idx",
        "action",
        "board_size",
        "day",
        "turns_per_day",
        "shed_capacity",
    ),
    "_commit_unit": ("op", "item", "price", "farm", "private", "market", "shed_capacity"),
    # 市場成交落盤（2026-09-26）：HIRE／BUY_LAND 不經 `_commit_unit`，另包這兩支（無回傳值）
    "_do_hire": ("farm", "private", "board_size", "mult"),
    "_do_buy_land": ("farm", "board_size"),
    "interpreter": ("state", "env"),
}
MOVE_OPS = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
OPPOSITE = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}


def official_module():
    """實際被 `kaggle_environments` 載入的那一份 kaggriculture 模組（不是 docs 的快照）。"""
    import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: PLC0415

    return K


def module_identity(mod) -> dict[str, Any]:
    path = Path(mod.__file__).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_interfaces(mod) -> dict[str, list[str]]:
    """三個函式的參數名逐字核；缺一或對不上就 raise（**不退回淨差**）。"""
    got: dict[str, list[str]] = {}
    for name, want in EXPECTED_SIGNATURES.items():
        fn = getattr(mod, name, None)
        if fn is None:
            raise SystemExit(f"官方模組沒有 {name}：介面不符，拒絕用回合淨差冒稱 actual")
        params = tuple(inspect.signature(fn).parameters)
        if params != want:
            raise SystemExit(f"{name} 介面不符：{params} ≠ {want}")
        got[name] = list(params)
    return got


# ------------------------------------------------------------------ 事件記錄
def _tile_at(farm, pos):
    if pos is None:
        return None
    x, y = map(int, pos)
    if not 0 <= y < len(farm["tiles"]):
        return None
    row = farm["tiles"][y]
    return row[x] if 0 <= x < len(row) else None


def _snapshot_tile(tile) -> dict[str, Any] | None:
    if not isinstance(tile, dict):
        return {"kind": None if tile is None else str(tile)}
    return {
        "kind": tile.get("kind"),
        "crop": tile.get("crop"),
        "animal": tile.get("animal"),
        "watered_today": bool(tile.get("watered_today")),
        "yield_units": int(tile.get("yield_units", 0) or 0),
        "planted_day": tile.get("planted_day"),
    }


def _inventory(private, idx):
    inventories = private.get("inventories", [])
    return dict(inventories[idx]) if 0 <= idx < len(inventories) else {}


class Recorder:
    """把官方原函式前後的**實際差分**記下來。policy 看不到這裡的任何東西。"""

    def __init__(self) -> None:
        self.unit_events: list[dict[str, Any]] = []
        self.sales: list[dict[str, Any]] = []
        #: 雙方**每一次**官方單件成交嘗試（`_commit_unit` 全部 op，含 ok=False）＋ HIRE／BUY_LAND
        self.trades: list[dict[str, Any]] = []
        self.ctx: dict[str, Any] = {"turn": -1, "day": -1, "player_of_farm": {}, "raw": {}}

    # -- 當步上下文（由 interpreter wrapper 設定；只是參照與對位，不改執行流程）
    def begin_turn(self, turn: int, day: int, farms: list, raw: dict[int, dict]) -> None:
        self.ctx = {
            "turn": int(turn),
            "day": int(day),
            "player_of_farm": {id(f): i for i, f in enumerate(farms)},
            "raw": raw,
        }

    def _player(self, farm) -> int:
        return int(self.ctx["player_of_farm"][id(farm)])

    def _raw_of(self, player: int, idx: int):
        a = self.ctx["raw"].get(player) or {}
        if idx == 0:
            return a.get("farmer")
        hands = a.get("hands") or []
        return hands[idx - 1] if idx - 1 < len(hands) else None

    def unit_before(self, farm, private, idx, action):
        from kaggle_environments.envs.kaggriculture import kaggriculture as K  # noqa: PLC0415

        pos = K._farmer_position(farm, idx)
        inv = _inventory(private, idx) if pos is not None else {}
        return {
            "player": self._player(farm),
            "day": self.ctx["day"],
            "turn": self.ctx["turn"],
            "unit": int(idx),
            "raw": copy.deepcopy(self._raw_of(self._player(farm), idx)),
            "allowed": copy.deepcopy(action),
            "pos_before": None if pos is None else list(map(int, pos)),
            "tile_before": _snapshot_tile(_tile_at(farm, pos)),
            "inv_before": inv,
            "shed_before": dict(private.get("shed", {})),
        }

    def unit_after(self, rec: dict[str, Any], farm, private, idx) -> None:
        from kaggle_environments.envs.kaggriculture import kaggriculture as K  # noqa: PLC0415

        pos = K._farmer_position(farm, idx)
        rec["pos_after"] = None if pos is None else list(map(int, pos))
        rec["tile_after"] = _snapshot_tile(_tile_at(farm, pos))
        rec["inv_after"] = _inventory(private, idx) if pos is not None else {}
        rec["shed_after"] = dict(private.get("shed", {}))
        rec["effect"] = classify(rec)
        self.unit_events.append(rec)

    def trade(self, farm, op: str, item, price, ok: bool, *, cost=None, reason=None) -> None:
        """一次官方成交嘗試（單件）。`price` 是官方報價；`cost` 是實際現金變化（成交才非 0）。"""
        self.trades.append(
            {
                "turn": self.ctx["turn"],
                "day": self.ctx["day"],
                "player": self._player(farm),
                "op": op,
                "item": item,
                "price": None if price is None else float(price),
                "ok": bool(ok),
                "cash_delta": float(cost) if cost is not None else 0.0,
                "reason": reason,
            }
        )

    def sale(self, farm, item: str, price: int) -> None:
        """`_commit_unit` 的 SELL **回 True 才算成交一件**（按 player／item 累計）。"""
        self.sales.append(
            {
                "turn": self.ctx["turn"],
                "day": self.ctx["day"],
                "player": self._player(farm),
                "item": item,
                "price": int(price),
            }
        )


def classify(rec: dict[str, Any]) -> dict[str, Any]:
    """依 `allowed` 的操作**與實際差分**分類。intent 出現不等於成功。"""
    a = rec["allowed"]
    op = a[0] if isinstance(a, list) and a else None
    pb, pa = rec["pos_before"], rec["pos_after"]
    tb, ta = rec["tile_before"] or {}, rec["tile_after"] or {}
    ib, ia = rec["inv_before"], rec["inv_after"]
    sb, sa = rec["shed_before"], rec["shed_after"]
    inv_delta = {
        k: int(ia.get(k, 0)) - int(ib.get(k, 0))
        for k in set(ib) | set(ia)
        if int(ia.get(k, 0)) != int(ib.get(k, 0))
    }
    shed_delta = {
        k: int(sa.get(k, 0)) - int(sb.get(k, 0))
        for k in set(sb) | set(sa)
        if int(sa.get(k, 0)) != int(sb.get(k, 0))
    }
    out: dict[str, Any] = {"op": op, "inv_delta": inv_delta, "shed_delta": shed_delta}
    if op in MOVE_OPS:
        out["kind"] = "MOVE" if pb != pa else None
        out["dir"] = op
    elif op == "PLANT":
        out["kind"] = "PLANT" if tb.get("kind") != "PLANT" and ta.get("kind") == "PLANT" else None
        out["crop"] = ta.get("crop")
    elif op == "WATER":
        # 在日末清旗**之前**觀測（hook 在 unit 相位裡）
        out["kind"] = "WATER" if (not tb.get("watered_today")) and ta.get("watered_today") else None
    elif op == "HARVEST":
        gained = {k: v for k, v in inv_delta.items() if v > 0}
        out["kind"] = (
            "HARVEST"
            if gained and int(ta.get("yield_units", 0)) < int(tb.get("yield_units", 0))
            else None
        )
        out["items"] = gained
    elif op in ("PLACE", "DROP"):
        into = {k: v for k, v in shed_delta.items() if v > 0}
        outof = {k: -v for k, v in inv_delta.items() if v < 0}
        # **DROP 的背包消失與真入倉分開**；容量溢出不算入倉
        out["kind"] = (
            "PLACE_ANIMAL"
            if op == "PLACE" and not tb.get("animal") and ta.get("animal")
            else "STORE"
            if into
            else "INV_LOST"
            if op == "DROP" and outof
            else None
        )
        out["stored"] = into
        out["left_inventory"] = outof
        out["overflow"] = (
            {k: v - into.get(k, 0) for k, v in outof.items() if v > into.get(k, 0)}
            if op == "DROP"
            else {}
        )
    elif op == "PICKUP":
        got = {k: v for k, v in inv_delta.items() if v > 0}
        out["kind"] = "PICKUP" if got else None
        out["items"] = got
    else:
        out["kind"] = None
    return out


@contextlib.contextmanager
def read_only_hooks(env, rec: Recorder):
    """安裝五個只讀 hook；**離開時（含例外）一定還原**。

    `_commit_unit`（SELL／BUY_*，逐件）與 `_do_hire`／`_do_buy_land`：原函式各呼叫一次、參數
    原樣轉交；只在前後讀 money／shed／hands／quadrants 判定成交與實際現金變化。
    """
    K = official_module()
    orig_apply, orig_commit = K._apply_unit_action, K._commit_unit
    orig_hire, orig_land = K._do_hire, K._do_buy_land
    orig_interp = env.interpreter

    def apply_hook(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
        before = rec.unit_before(farm, private, idx, action)
        out = orig_apply(  # 原函式**只呼叫一次**，參數原樣轉交
            farm, private, idx, action, board_size, day, turns_per_day, shed_capacity
        )
        rec.unit_after(before, farm, private, idx)
        return out

    def commit_hook(op, item, price, farm, private, market, shed_capacity=100):
        m0 = farm["money"]
        shed_item0 = private["shed"].get(item, 0)
        shed_total0 = sum(private["shed"].values())
        out = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if op == "SELL" and out is True:
            rec.sale(farm, item, price)
        reason = None
        if out is not True:
            if op == "SELL":
                reason = "shed_empty" if shed_item0 <= 0 else "other"
            elif m0 < price:
                reason = "money"
            elif op in ("BUY_PRODUCT", "BUY_ANIMAL") and shed_total0 >= shed_capacity:
                reason = "shed_full"
            else:
                reason = "other"
        rec.trade(farm, op, item, price, out is True, cost=farm["money"] - m0, reason=reason)
        return out

    def hire_hook(farm, private, board_size, mult=K.FARM_HAND_COST_MULT):
        m0, h0 = farm["money"], len(farm["hands"])
        quote = K._hire_cost(farm["hires_today"], mult)
        out = orig_hire(farm, private, board_size, mult)
        ok = len(farm["hands"]) == h0 + 1
        rec.trade(
            farm,
            "HIRE",
            None,
            quote,
            ok,
            cost=farm["money"] - m0,
            reason=None if ok else ("money" if m0 < quote else "other"),
        )
        return out

    def land_hook(farm, board_size):
        m0, q0 = farm["money"], len(farm["unlocked_quadrants"])
        n_extra = q0 - 1
        quote = K.LAND_PRICES[n_extra] if n_extra < len(K.LAND_ORDER) else None
        out = orig_land(farm, board_size)
        ok = len(farm["unlocked_quadrants"]) == q0 + 1
        reason = None
        if not ok:
            reason = "all_unlocked" if quote is None else ("money" if m0 < quote else "other")
        rec.trade(
            farm,
            "BUY_LAND",
            farm["unlocked_quadrants"][-1] if ok else None,
            quote,
            ok,
            cost=farm["money"] - m0,
            reason=reason,
        )
        return out

    def interpreter(state, env):  # noqa: A002  # 明確兩參數 closure（co_argcount == 2）
        obs0 = state[0].observation
        tpd = max(1, int(K.get(env.configuration, "turnsPerDay", 24)))
        step = int(K.get(obs0, "step", 0))
        raw = {i: (s.action if isinstance(s.action, dict) else {}) for i, s in enumerate(state)}
        farms = list(getattr(obs0, "farms", []) or [])
        rec.begin_turn(step, step // tpd, farms, raw)
        return orig_interp(state, env)  # 原 interpreter **只呼叫一次**

    assert interpreter.__code__.co_argcount == 2, "wrapper 必須是兩參數函式"
    K._apply_unit_action = apply_hook
    K._commit_unit = commit_hook
    K._do_hire = hire_hook
    K._do_buy_land = land_hook
    env.interpreter = interpreter
    try:
        yield rec
    finally:
        K._apply_unit_action = orig_apply
        K._commit_unit = orig_commit
        K._do_hire = orig_hire
        K._do_buy_land = orig_land
        env.interpreter = orig_interp


# ------------------------------------------------------------------ 成功鏈
def worker_key(e: dict[str, Any]) -> tuple:
    """工人身分**逐日分代**：日末回家／重生不跨代，也不計成 MOVE。"""
    return (e["player"], e["day"], e["unit"])


def chains(events: list[dict[str, Any]], player: int) -> dict[str, Any]:
    """鏈 A：同單位成功 MOVE ≥1 → 停靠格成功 PLANT → 該格後續成功 WATER。

    鏈 B：同品項 HARVEST → **手動**入倉 → SELL 的成功事件序列。
    明示這是**同品項的操作序列**，不冒稱能辨識混合庫存裡那批貨的來源。
    """
    moves, pending, completed, broken = {}, {}, [], []
    for e in events:
        if e["player"] != player:
            continue
        eff = e["effect"]
        key = worker_key(e)
        pos = tuple(e["pos_after"]) if e["pos_after"] is not None else None
        tile = e["tile_after"] or {}
        if eff["kind"] == "MOVE":
            moves[key] = moves.get(key, 0) + 1
        # Removal/replacement invalidates the crop even when its op is not a success kind.
        st = pending.get(pos)
        identity = (tile.get("kind"), tile.get("crop"), tile.get("planted_day"))
        if st and (identity != st["identity"] or eff["kind"] == "PLANT"):
            broken.append({**pending.pop(pos), "reason": "crop_replaced_or_removed"})
        if eff["kind"] == "PLANT" and moves.get(key, 0):
            pending[pos] = {
                "player": player,
                "day": e["day"],
                "unit": e["unit"],
                "moves": moves.pop(key),
                "plant_turn": e["turn"],
                "tile": list(pos),
                "crop": eff["crop"],
                "identity": identity,
            }
        elif eff["kind"] == "WATER" and pos in pending:
            completed.append({**pending.pop(pos), "water_turn": e["turn"], "water_unit": e["unit"]})
    return {
        "chain_a": completed,
        "chain_a_count": len(completed),
        "chain_a_incomplete": list(pending.values()),
        "chain_a_broken": broken,
    }


def chain_b(events: list[dict[str, Any]], sales: list[dict[str, Any]], player: int) -> dict:
    # Unit phase precedes market phase; stable sort preserves official slot order.
    timeline = [(e["turn"], 0, e) for e in events if e["player"] == player]
    timeline += [(s["turn"], 1, s) for s in sales if s["player"] == player]
    got, out = {}, []
    for _turn, phase, e in sorted(timeline, key=lambda x: (x[0], x[1])):
        if phase == 1:
            st = got.get(e["item"])
            if st and "store_turn" in st:
                out.append(
                    {
                        **got.pop(e["item"]),
                        "player": player,
                        "item": e["item"],
                        "sell_turn": e["turn"],
                        "sold_units": 1,
                        "price": e["price"],
                    }
                )
            continue
        eff = e["effect"]
        if eff["kind"] == "HARVEST":
            for item, qty in eff.get("items", {}).items():
                if item not in got:
                    got[item] = {"harvest_turn": e["turn"], "unit": e["unit"], "harvest_qty": qty}
        elif eff["kind"] == "STORE":
            for item, qty in eff.get("stored", {}).items():
                if item in got and "store_turn" not in got[item]:
                    got[item].update(store_turn=e["turn"], store_unit=e["unit"], store_qty=qty)
    return {
        "chain_b": out,
        "chain_b_count": len(out),
        "chain_b_broken_after_harvest": {k: v for k, v in got.items() if "store_turn" not in v},
        "chain_b_broken_after_store": {k: v for k, v in got.items() if "store_turn" in v},
        "note": "同品項的操作序列；不冒稱混合庫存中特定貨物的來源可辨；sold_units 為完成序列的首件",
    }


def linked_chains(events, player, chain_a, chain_b_rows):
    """Join A/B only through an observed crop generation and a causal harvest.

    Crop provenance is exact for the observed tile events. The later store/sale
    remains a same-item sequence, not proof that those particular goods were sold.
    """
    successful_plants = {(a["plant_turn"], a["unit"]): a for a in chain_a}
    plots, harvests = {}, {}
    for e in events:
        if e["player"] != player or e["pos_after"] is None:
            continue
        pos = tuple(e["pos_after"])
        before, after = e["tile_before"] or {}, e["tile_after"] or {}
        kind = e["effect"]["kind"]
        if kind == "MOVE":
            continue  # tile_before is the departure tile, not the destination crop.
        st = plots.get(pos)
        ident_before = (before.get("kind"), before.get("crop"), before.get("planted_day"))
        if st and st["identity"] != ident_before:
            plots.pop(pos)
            st = None
        if kind == "PLANT":
            plots[pos] = {
                "key": (e["turn"], e["unit"]),
                "watered": False,
                "identity": (after.get("kind"), after.get("crop"), after.get("planted_day")),
            }
        elif st:
            if kind == "WATER":
                st["watered"] = True
            elif kind == "HARVEST" and st["watered"] and st["key"] in successful_plants:
                harvests[(e["turn"], e["unit"])] = successful_plants[st["key"]]
            ident_after = (after.get("kind"), after.get("crop"), after.get("planted_day"))
            if ident_after != st["identity"]:
                plots.pop(pos, None)
    joined = []
    for b in chain_b_rows:
        a = harvests.get((b["harvest_turn"], b["unit"]))
        if a and a["crop"] == b["item"]:
            joined.append({"crop_chain": a, "item_chain": b})
    return {
        "linked_chains": joined,
        "linked_chain_count": len(joined),
        "linked_chain_scope": "observed MOVE/PLANT/WATER/HARVEST on same crop generation; "
        "later STORE/SELL is same-item sequence, not cargo provenance",
    }


def turn_around(events: list[dict[str, Any]], player: int) -> dict[str, Any]:
    """掉頭率：同一 (day, unit) 的**相鄰成功 MOVE** 反向次數 ÷ 可比較的相鄰 MOVE 對數。"""
    by_worker: dict[tuple, list[str]] = {}
    for e in events:
        if e["player"] == player and e["effect"]["kind"] == "MOVE":
            by_worker.setdefault(worker_key(e), []).append(e["effect"]["dir"])
    pairs = rev = 0
    for dirs in by_worker.values():
        for a, b in zip(dirs, dirs[1:], strict=False):
            pairs += 1
            rev += OPPOSITE.get(a) == b
    return {
        "move_events": sum(len(v) for v in by_worker.values()),
        "adjacent_pairs": pairs,
        "reversals": rev,
        "turn_around_rate": (rev / pairs) if pairs else None,
    }


def event_counts(events: list[dict[str, Any]], player: int) -> dict[str, dict[str, int]]:
    """raw／allowed／effect 三欄分別計數（intent 與成功分開）。"""
    raw: dict[str, int] = {}
    allowed: dict[str, int] = {}
    effect: dict[str, int] = {}
    for e in events:
        if e["player"] != player:
            continue
        for col, d in (("raw", raw), ("allowed", allowed)):
            a = e[col]
            op = a[0] if isinstance(a, list) and a else "NONE"
            d[op] = d.get(op, 0) + 1
        k = e["effect"]["kind"]
        if k:
            effect[k] = effect.get(k, 0) + 1
    return {"raw": raw, "allowed": allowed, "effect": effect}


def sell_summary(sales: list[dict], player: int, requests: list[int]) -> dict[str, Any]:
    """actual 是**每回合每品項的成交總量**；不聲稱有逐筆訂單歸因。"""
    mine = [s for s in sales if s["player"] == player]
    per: dict[str, dict[str, int]] = {}
    for s in mine:
        d = per.setdefault(s["item"], {"units": 0, "revenue": 0})
        d["units"] += 1
        d["revenue"] += s["price"]
    by_turn = {}
    for sale in mine:
        d = by_turn.setdefault((sale["turn"], sale["item"]), {"units": 0, "revenue": 0})
        d["units"] += 1
        d["revenue"] += sale["price"]
    q = sorted(requests)

    def pct(p: float):
        return q[min(len(q) - 1, int(p * len(q)))] if q else None

    return {
        "actual_units_total": len(mine),
        "actual_by_item": per,
        "actual_by_turn_item": [{"turn": t, "item": item, **d} for (t, item), d in by_turn.items()],
        "request_orders": len(requests),
        "request_qty_p50": pct(0.5),
        "request_qty_p95": pct(0.95),
        "request_qty_max": max(q) if q else None,
        "attribution": "per-turn per-item totals；無逐筆 actual/request 比值",
    }


MARKET_OPS = ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND")


def market_summary(trades: list[dict], player: int, *, starting_money: float, final_money: float):
    """一方的官方成交彙總（逐 op、逐品項、逐日、逐回合品項）＋現金對帳。

    成交＝官方函式回報成功（`_commit_unit` 回 True；HIRE／BUY_LAND 以 hands／quadrants 增加判定）；
    金額＝實際現金變化（`cash_delta`，SELL 為正、買為負）。被拒＝官方拒收（不動現金）。
    `cash_identity`：起始現金＋Σcash_delta 必須等於終局現金（官方只有這三支函式改 money）。
    """
    mine = [t for t in trades if t["player"] == player]
    by_op: dict[str, Any] = {}
    for op in MARKET_OPS:
        ts = [t for t in mine if t["op"] == op]
        ok = [t for t in ts if t["ok"]]
        by_item: dict[str, Any] = {}
        for t in ts:
            d = by_item.setdefault(
                str(t["item"]),
                {"n_ok": 0, "amount": 0.0, "n_rejected": 0, "rejected_by_reason": {}},
            )
            if t["ok"]:
                d["n_ok"] += 1
                d["amount"] += abs(t["cash_delta"])
            else:
                d["n_rejected"] += 1
                rr = d["rejected_by_reason"]
                rr[t["reason"]] = rr.get(t["reason"], 0) + 1
        for d in by_item.values():
            d["unit_price_mean"] = d["amount"] / d["n_ok"] if d["n_ok"] else None
        rej: dict[str, int] = {}
        for t in ts:
            if not t["ok"]:
                rej[t["reason"]] = rej.get(t["reason"], 0) + 1
        by_op[op] = {
            "n_ok": len(ok),
            "amount": sum(abs(t["cash_delta"]) for t in ok),
            "n_rejected": len(ts) - len(ok),
            "rejected_by_reason": rej,
            "by_item": by_item,
        }
    by_day: dict[int, dict[str, float]] = {}
    by_turn_item: dict[tuple, dict[str, float]] = {}
    for t in mine:
        d = by_day.setdefault(t["day"], {})
        k = by_turn_item.setdefault(
            (t["turn"], t["op"], str(t["item"])), {"n_ok": 0, "amount": 0.0, "n_rejected": 0}
        )
        if t["ok"]:
            d[t["op"]] = d.get(t["op"], 0.0) + abs(t["cash_delta"])
            k["n_ok"] += 1
            k["amount"] += abs(t["cash_delta"])
        else:
            k["n_rejected"] += 1
    income = sum(t["cash_delta"] for t in mine if t["cash_delta"] > 0)
    spend = -sum(t["cash_delta"] for t in mine if t["cash_delta"] < 0)
    implied = starting_money + income - spend
    return {
        "by_op": by_op,
        "by_day_amount": [{"day": d, **v} for d, v in sorted(by_day.items())],
        "by_turn_item": [
            {"turn": tu, "op": op, "item": it, **v}
            for (tu, op, it), v in sorted(by_turn_item.items())
        ],
        "cash_identity": {
            "starting_money": starting_money,
            "income": income,
            "spend": spend,
            "implied_final": implied,
            "final_money": final_money,
            "diff": final_money - implied,
        },
        "scope": (
            "官方 _commit_unit 逐件＋_do_hire／_do_buy_land；成交金額＝實際現金變化；被拒不動現金"
        ),
    }


def final_farm_summary(last) -> list[dict[str, Any]]:
    """終局雙方 farm 摘要（取最後一步 observation；private 取各方自己的 observation）。"""
    obs0 = last[0].observation
    out = []
    for i, s in enumerate(last):
        farm = obs0.farms[i]
        priv = getattr(s.observation, "private", None) or {}
        kinds: dict[str, int] = {}
        crops: dict[str, int] = {}
        animals: dict[str, int] = {}
        n_locked = n_empty = 0
        for row in farm["tiles"]:
            for t in row:
                if t == "LOCKED":
                    n_locked += 1
                elif t is None:
                    n_empty += 1
                elif isinstance(t, dict):
                    kinds[str(t.get("kind"))] = kinds.get(str(t.get("kind")), 0) + 1
                    if "crop" in t:
                        crops[t["crop"]] = crops.get(t["crop"], 0) + 1
                    if "animal" in t:
                        animals[t["animal"]] = animals.get(t["animal"], 0) + 1
        shed = {k: int(v) for k, v in (priv.get("shed") or {}).items() if v}
        seeds = {k: int(v) for k, v in (priv.get("seeds") or {}).items() if v}
        out.append(
            {
                "player": i,
                "money": float(farm["money"]),
                "n_units": 1 + len(farm["hands"]),
                "hires_today": int(farm.get("hires_today", 0)),
                "unlocked_quadrants": list(farm["unlocked_quadrants"]),
                "tiles": {"locked": n_locked, "empty": n_empty, "by_kind": kinds},
                "crops": crops,
                "animals": animals,
                "shed": shed,
                "shed_total": sum(shed.values()),
                "seeds": seeds,
            }
        )
    return out


def daily_planted(probes, day=29):
    return next((d["planted_h23"] for d in fold_daily(probes) if d["day"] == day), None)


def check_full_action(js):
    bad = check_action_json(js)
    for i, a in enumerate([js.get("farmer"), *(js.get("hands") or [])]):
        if not isinstance(a, list) or not a or a[0] not in rules.UNIT_OP_NAMES:
            bad.append(f"unit {i}: invalid operation {a}")
            continue
        op = a[0]
        expected = 3 if op in ("PICKUP", "PLACE") else 2 if op == "PLANT" else 1
        if len(a) != expected:
            bad.append(f"unit {i}: invalid arity {a}")
        elif op == "PLANT" and a[1] not in rules.CROPS:
            bad.append(f"unit {i}: invalid crop {a}")
        elif op in ("PICKUP", "PLACE") and (
            a[1] not in rules.ITEMS or type(a[2]) is not int or not 0 < a[2] <= 2147483647
        ):
            bad.append(f"unit {i}: invalid item or quantity {a}")
    return bad


# ------------------------------------------------- 封存對手（真提交 tar．§38）
#
# 為什麼要這一整段：`kaggle_environments/agent.py` 用 `exec(code, {})` 跑對手的
# `main.py`，命名空間裡**沒有 `__file__`**；v27 包內的 `main.py`（其實是 `v27.py`
# 改名）因此退回相對字串 `"v27.py"`，`os.path.abspath()` 接的是**行程 cwd**，
# 再 `sys.path.insert(0, cwd)`。本線 repo 根目錄剛好也有一支 `agent_core.py`
# （SHA `e073d548…`，不是包內的 `53e80eb6…`），所以從 repo 根跑會**靜默載到錯的
# core**、基準根本不是 v27。對策：每局獨立子行程、cwd＝解包目錄、且**載入後實查
# `sys.modules["agent_core"].__file__` 與其 SHA**——約定靠機制，不靠記得。

OPP_MEMBERS = ("main.py", "agent_core.py")


def kaggle_env_version() -> str:
    """只取 `__version__` 字串。`kaggle_environments.version` 是它從
    `importlib.metadata` import 進來的**函式**，塞進 result 會讓 `json.dumps` 直接炸。"""
    import kaggle_environments  # noqa: PLC0415

    v = getattr(kaggle_environments, "__version__", None)
    return v if isinstance(v, str) else "unknown"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract_opponent(tar_path: Path, tar_sha256: str, dest: Path) -> dict[str, Any]:
    """核 tar SHA → **只**解兩個 regular member 到 `dest`（每 run 新建、不覆蓋）。

    不用 `extractall`：那會跟著包裡的任何路徑／連結／目錄成員走。這裡逐一取名，
    非 regular member 直接拒，其餘成員不解。
    """
    tar_path = Path(tar_path).resolve()
    actual = sha256_file(tar_path)
    if actual != tar_sha256:
        raise ValueError(f"對手 tar SHA 不符：期望 {tar_sha256}，實算 {actual}（{tar_path}）")
    if dest.exists():
        raise FileExistsError(f"解包目錄已存在，拒絕覆蓋：{dest}")
    dest.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        for name in OPP_MEMBERS:
            try:
                m = tf.getmember(name)
            except KeyError as e:
                raise ValueError(f"對手 tar 缺成員 {name}：{tar_path}") from e
            if not m.isreg():
                raise ValueError(f"對手 tar 成員 {name} 不是 regular file（{m.type!r}）")
            src = tf.extractfile(m)
            if src is None:
                raise ValueError(f"對手 tar 成員 {name} 無法讀取")
            (dest / name).write_bytes(src.read())
    return {
        "kind": "archived-tar",
        "tar_path": str(tar_path),
        "tar_sha256": actual,
        "members": {n: sha256_file(dest / n) for n in OPP_MEMBERS},
        "extract_dir": str(dest.resolve()),
        "main_py": str((dest / "main.py").resolve()),
    }


def verify_loaded_core(extract_dir: Path, expect_sha256: str) -> dict[str, Any]:
    """實查**真的被載進來的** `agent_core`：必須在解包目錄下，且 SHA 等於包內成員。

    只信 `sys.modules` 的證據，不信 cwd 或 `sys.path` 的安排。
    """
    mod = sys.modules.get("agent_core")
    if mod is None:
        raise RuntimeError("對手的 agent_core 還沒被載入（無法核身分）")
    f = getattr(mod, "__file__", None)
    if not f:
        raise RuntimeError(f"已載入的 agent_core 沒有 __file__：{mod!r}")
    p = Path(f).resolve()
    d = Path(extract_dir).resolve()
    if not p.is_relative_to(d):
        raise RuntimeError(f"已載入的 agent_core 不在解包目錄：{p}（應在 {d} 下）")
    sha = sha256_file(p)
    if sha != expect_sha256:
        raise RuntimeError(f"已載入的 agent_core SHA 不符：期望 {expect_sha256}，實算 {sha}（{p}）")
    return {"loaded_agent_core_file": str(p), "loaded_agent_core_sha256": sha}


def prepare_opponent_load(extract_dir: Path, *, purge_module_cache: bool) -> None:
    """載對手前的明確準備（worker 與 probe 共用）。

    為什麼不能只靠 cwd：`kaggle_environments/agent.py` 在 `exec` **之前**就
    `sys.path.append(exec_dir)`，所以解包目錄已經在 `sys.path`（尾端）；包內 `main.py`
    的 `insert(0, cwd)` 有 `_p not in sys.path` 條件，於是根本不執行，`sys.path[0]`
    仍是本線 root ⇒ `import agent_core` 命中 repo 根那支。這裡把解包目錄明確排到首位。

    但**先檢 cwd**：cwd 錯（例如從 repo 根跑）本身就是配置錯誤，不能被這裡的硬置路徑
    蓋掉；`purge_module_cache` 只在獨立子行程用，父行程的 `sys.modules` 不動。
    """
    d = Path(extract_dir).resolve()
    cwd = Path.cwd().resolve()
    if cwd != d:
        raise RuntimeError(f"cwd 不是解包目錄：{cwd}（應為 {d}）")
    if purge_module_cache:
        sys.modules.pop("agent_core", None)
    if str(d) in sys.path:
        sys.path.remove(str(d))
    sys.path.insert(0, str(d))


def load_raw_callable(main_py: Path):
    """走官方 `get_last_callable` 的同一條路（`exec(code, {})`、把 dirname 加進
    `sys.path`），這樣缺 `__file__` 的真語意保持不變。"""
    from kaggle_environments.agent import get_last_callable  # noqa: PLC0415

    path = str(Path(main_py).resolve())
    return get_last_callable(Path(path).read_text(), path=path)


def opponent_argcount(fn) -> int | None:
    """官方 `Agent.act`／`build_agent` 的呼叫協定：有 `__code__.co_argcount` 就把
    ``[observation, configuration]`` 截到該長度（0／1／2 參數皆可），沒有就兩個都傳（回 None）。"""
    code = getattr(fn, "__code__", None)
    return code.co_argcount if code is not None and hasattr(code, "co_argcount") else None


def build_opponent(raw: str, env) -> tuple[Any, dict[str, Any]]:
    """對手 callable ＋ 逐 call 計時。用官方 `build_agent` 的 path 分支，不自己改寫載入。

    2026-09-28 修正：builtin（例如 `starter_agent(obs)`）以前固定用兩參數呼叫 ⇒ 首步 TypeError、
    座位 ERROR、終局被 interpreter 覆蓋成 DONE。現在照官方協定依 `co_argcount` 截參數；**不捕捉
    TypeError 換 arity 重試**（函式內真 TypeError 必須浮出）。`n`＝呼叫次數（沿用舊欄位），
    `ok`／`exceptions` 另記成功與例外次數。檔案對手的 `callable_agent` 是兩參數，行為與舊版相同。
    """
    from kaggle_environments.agent import build_agent  # noqa: PLC0415

    inner, _parallel = build_agent(raw, env.agents, env.name)
    nargs = opponent_argcount(inner)
    stats = {"n": 0, "ok": 0, "exceptions": 0, "total_s": 0.0, "max_s": 0.0, "argcount": nargs}

    def opponent(observation, configuration):
        args = (
            (observation, configuration) if nargs is None else (observation, configuration)[:nargs]
        )
        t0 = time.perf_counter()
        try:
            r = inner(*args)
        except BaseException:
            stats["exceptions"] += 1
            raise
        finally:
            dt = time.perf_counter() - t0
            stats["n"] += 1
            stats["total_s"] += dt
            stats["max_s"] = max(stats["max_s"], dt)
        stats["ok"] += 1
        return r

    return opponent, stats


def status_history_check(history: list[list[str]]) -> dict[str, Any]:
    """逐步雙方 status：非末步任一座不是 ACTIVE（ERROR／INVALID／TIMEOUT／INACTIVE…）、
    或末步不是全 DONE ⇒ 不合格。interpreter 終局會把早期 ERROR 覆蓋成 DONE，所以不能只看末步。"""
    bad = [[i, h] for i, h in enumerate(history[:-1]) if any(x != "ACTIVE" for x in h)]
    last_ok = bool(history) and all(x == "DONE" for x in history[-1])
    return {
        "ok": not bad and last_ok,
        "n_bad_steps": len(bad),
        "first_bad": bad[:5],
        "last": history[-1] if history else None,
    }


def stderr_steps(logs: list, k: int) -> list[int]:
    """`env.logs` 裡第 k 座 stderr 非空的步號。

    官方 `Agent.act` 在 debug=False 時收 stdout／stderr。
    """
    return [
        i
        for i, lg in enumerate(logs)
        if isinstance(lg, list) and len(lg) > k and isinstance(lg[k], dict) and lg[k].get("stderr")
    ]


def game_validity(
    *,
    status_ok: bool,
    expected_calls: int,
    ours_calls: int,
    opp_calls_ok: int,
    opp_exceptions: int,
    ours_stderr: int = 0,
    opp_stderr: int = 0,
) -> dict[str, Any]:
    """任一不合格 ⇒ INVALID。stderr 不設字串豁免：任一座任一步 stderr 非空即 INVALID。"""
    reasons = []
    if not status_ok:
        reasons.append("status_history")
    if ours_calls != expected_calls:
        reasons.append(f"ours_calls {ours_calls}≠{expected_calls}")
    if opp_calls_ok != expected_calls:
        reasons.append(f"opponent_calls {opp_calls_ok}≠{expected_calls}")
    if opp_exceptions:
        reasons.append(f"opponent_exceptions {opp_exceptions}")
    if ours_stderr:
        reasons.append(f"ours_stderr {ours_stderr}")
    if opp_stderr:
        reasons.append(f"opponent_stderr {opp_stderr}")
    return {"validity": "VALID" if not reasons else "INVALID", "invalid_reasons": reasons}


# ------------------------------------------------------------------ 跑一局
def play(
    ckpt: Path,
    seed: int,
    seat: int,
    opponent: str,
    collect_obs: int = 0,
    *,
    opp_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    agent = FN.make_agent(ckpt, seat=seat, caps=CAPS)
    bad: list[dict] = []
    probes: list[dict] = []
    timing: list[float] = []
    obs_kept: list[dict] = []
    sell_requests: list[int] = []
    box = {"t": 0}

    def ours(obs, _config=None):
        if len(obs_kept) < collect_obs:
            obs_kept.append(json.loads(json.dumps(obs)))  # 深拷貝，audit 不動原件
        t0 = time.perf_counter()
        js = agent(obs)  # **計時涵蓋官方 obs dict 解析 → JSON**
        timing.append(time.perf_counter() - t0)
        probes.append(_daily_probe(obs, seat))
        for o in js.get("market") or []:
            if o and o[0] == "SELL" and len(o) == 3:
                sell_requests.append(int(o[2]))
        v = check_full_action(js)
        if v:
            bad.append({"turn": box["t"], "bad": v})
        box["t"] += 1
        return js

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    rec = Recorder()
    raw_opp, opp_stats = build_opponent(opp_identity["main_py"] if opp_identity else opponent, env)
    core_check: dict[str, Any] = {}
    core_err: list[BaseException] = []
    if opp_identity:
        extract_dir = Path(opp_identity["extract_dir"])
        core_sha = opp_identity["members"]["agent_core.py"]
        first = {"done": False}

        def opp_callable(observation, configuration):
            r = raw_opp(observation, configuration)
            if not first["done"]:
                first["done"] = True
                # 引擎會把 agent 丟出的例外吞成 ERROR status，所以先收起來、跑完再爆
                try:
                    core_check.update(verify_loaded_core(extract_dir, core_sha))
                    core_check["checked_at_first_call"] = True
                except BaseException as e:  # noqa: BLE001
                    core_err.append(e)
            return r
    else:
        opp_callable = raw_opp

    agents = [ours, opp_callable] if seat == 0 else [opp_callable, ours]
    with read_only_hooks(env, rec):
        env.run(agents)
    if core_err:
        raise core_err[0]
    if opp_identity:
        # 局後再核一次：確認整局中途沒有被換掉
        after = verify_loaded_core(Path(opp_identity["extract_dir"]), core_sha)
        if after != {k: core_check[k] for k in after}:
            raise RuntimeError(f"對手 agent_core 在局中被替換：{core_check} → {after}")
        core_check["checked_after_game"] = True

    last = env.steps[-1]
    money = [int(s.observation.farms[i]["money"]) for i, s in enumerate(last)]
    # ---- 2026-09-28：逐步 status／stderr／呼叫數進 gate（不只看終局 DONE）
    hist = status_history_check([[s.status for s in st] for st in env.steps])
    logs = getattr(env, "logs", None) or []
    opp_err, ours_err = stderr_steps(logs, 1 - seat), stderr_steps(logs, seat)
    validity = game_validity(
        status_ok=hist["ok"],
        expected_calls=len(env.steps) - 1,
        ours_calls=box["t"],
        opp_calls_ok=opp_stats["ok"],
        opp_exceptions=opp_stats["exceptions"],
        ours_stderr=len(ours_err),
        opp_stderr=len(opp_err),
    )
    # 逐日折疊只取**存量**；最後再補一筆 final observation，避免少掉 h23
    probes.append(_daily_probe(last[seat].observation, seat))
    a_chains = chains(rec.unit_events, seat)
    b_chains = chain_b(rec.unit_events, rec.sales, seat)
    start_money = float(official_module().get(env.configuration, "startingMoney", 3000))
    final_money = [float(s.observation.farms[i]["money"]) for i, s in enumerate(last)]
    return {
        "seed": seed,
        "seat": seat,
        "opponent": opp_identity["main_py"] if opp_identity else opponent,
        "opponent_identity": {**(opp_identity or {}), **core_check},
        "opponent_call_seconds": {
            "n": opp_stats["n"],
            "mean": round(opp_stats["total_s"] / opp_stats["n"], 5) if opp_stats["n"] else None,
            "max": round(opp_stats["max_s"], 5),
        },
        "kaggle_environments_version": kaggle_env_version(),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3),
        "entries": len(env.steps),
        "statuses": [s.status for s in last],
        "money": money,
        "ours": money[seat],
        "opp": money[1 - seat],
        "win": money[seat] > money[1 - seat],
        "result": ("W" if money[seat] > money[1 - seat] else "L")
        if money[seat] != money[1 - seat]
        else "D",
        "syntax_violations": bad,
        "step_seconds": {
            "n": len(timing),
            "p50": round(statistics.median(timing), 4) if timing else None,
            "p99": round(sorted(timing)[int(0.99 * (len(timing) - 1))], 4) if timing else None,
            "max": round(max(timing), 4) if timing else None,
        },
        "events": event_counts(rec.unit_events, seat),
        "unit_trace": [e for e in rec.unit_events if e["player"] == seat],
        **a_chains,
        **b_chains,
        **linked_chains(rec.unit_events, seat, a_chains["chain_a"], b_chains["chain_b"]),
        "turn_around": turn_around(rec.unit_events, seat),
        "sell": sell_summary(rec.sales, seat, sell_requests),
        "daily_planted_d29": daily_planted(probes),
        "auto_drop_recorded": False,  # 自動倒包不經 `_apply_unit_action`，這裡不記，也不冒稱有記
        "obs_kept": obs_kept,
        # ---- 2026-09-26 新增（舊欄位不動）：雙方官方成交與終局 farm
        "market": {
            f"p{p}": market_summary(
                rec.trades, p, starting_money=start_money, final_money=final_money[p]
            )
            for p in range(len(last))
        },
        "final_farms": final_farm_summary(last),
        # ---- 2026-09-28 新增（對手呼叫協定修正；舊欄位不改名）
        "opponent_calls": opp_stats["ok"],
        "opponent_call_attempts": opp_stats["n"],
        "opponent_exceptions": opp_stats["exceptions"],
        "opponent_argcount": opp_stats["argcount"],
        "ours_calls": box["t"],
        "status_history_ok": hist["ok"],
        "status_history": hist,
        "opponent_stderr_steps": {
            "n": len(opp_err),
            "first": opp_err[:5],
            "first_text": (logs[opp_err[0]][1 - seat]["stderr"][:2000] if opp_err else None),
        },
        "ours_stderr_steps": {
            "n": len(ours_err),
            "first": ours_err[:5],
            "first_text": (logs[ours_err[0]][seat]["stderr"][:2000] if ours_err else None),
        },
        **validity,
    }


def parse_latency(ckpt: Path, seat: int, obs_list: list[dict]) -> dict[str, Any]:
    """真 obs dict → JSON 的 100 次計時：補既有 latency **沒含解析**的缺口。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    agent = FN.make_agent(ckpt, seat=seat, caps=CAPS)
    ts = []
    for obs in obs_list:
        t0 = time.perf_counter()
        agent(obs)
        ts.append(time.perf_counter() - t0)
    q = sorted(ts)

    def pct(p):
        return round(q[min(len(q) - 1, int(p * len(q)))] * 1000, 2) if q else None

    return {
        "n": len(ts),
        "includes": "official obs dict 解析 → decode → 引擎 JSON",
        "first_ms": round(ts[0] * 1000, 2) if ts else None,
        "p50_ms": pct(0.5),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": round(max(q) * 1000, 2) if q else None,
    }


def run_one_game_subprocess(
    *,
    ckpt: Path,
    seed: int,
    seat: int,
    collect_obs: int,
    opponent: str,
    identity_path: Path | None,
    cwd: Path,
    out_path: Path,
) -> dict[str, Any]:
    """**每局一個獨立子行程**：cwd 換成解包目錄，`sys.modules` 從零開始。

    子行程的 `PYTHONPATH` 用**絕對**的本線 root（cwd 已經不是 repo 根，寫 `.` 會指錯）。
    子行程非零退出一律往上拋，不吞：局末沒有 JSON 也算失敗。
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--ckpt",
        str(Path(ckpt).resolve()),
        "--seed",
        str(seed),
        "--seat",
        str(seat),
        "--collect-obs",
        str(collect_obs),
        "--out",
        str(out_path.resolve()),
    ]
    if identity_path is not None:
        cmd += ["--opp-identity", str(identity_path.resolve())]
    else:
        cmd += ["--opponent", opponent]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"每局子行程失敗：seed={seed} seat={seat} rc={proc.returncode} "
            f"out_exists={out_path.exists()}；stderr 末段：{proc.stderr[-2000:]}"
        )
    return json.loads(out_path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponent", default="starter", help="引擎內建名（starter）或 .py 路徑")
    ap.add_argument("--opponent-tar", help="真提交封存包（.tar.gz），與 --opponent-sha 同用")
    ap.add_argument("--opponent-sha", help="封存包的預期 tar SHA256（不符即拒）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5000, 5001])
    ap.add_argument("--seat0", type=int, default=0)
    ap.add_argument("--parse-latency-n", type=int, default=100)
    ap.add_argument("--out", required=True)
    # 單局 worker（父層自己叫，不給人手動當入口用）
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--seat", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--collect-obs", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--opp-identity", help=argparse.SUPPRESS)
    # 診斷：只載對手並核身分，不跑對局
    ap.add_argument("--probe-opponent", help=argparse.SUPPRESS)
    ap.add_argument("--opp-core-sha", help=argparse.SUPPRESS)
    ap.add_argument("--extract-dir", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.probe_opponent:
        prepare_opponent_load(Path(args.extract_dir), purge_module_cache=True)
        load_raw_callable(Path(args.probe_opponent))
        ident = verify_loaded_core(Path(args.extract_dir), args.opp_core_sha)
        print(json.dumps(ident, ensure_ascii=False))
        return 0

    if args.worker:
        ident = json.loads(Path(args.opp_identity).read_text()) if args.opp_identity else None
        if ident:
            prepare_opponent_load(Path(ident["extract_dir"]), purge_module_cache=True)
        g = play(
            Path(args.ckpt),
            args.seed,
            args.seat,
            args.opponent,
            collect_obs=args.collect_obs,
            opp_identity=ident,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(g, ensure_ascii=False))
        return 0

    out_path = Path(args.out)
    run_dir = out_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    opp_identity: dict[str, Any] | None = None
    identity_path: Path | None = None
    child_cwd = ROOT
    if args.opponent_tar:
        if not args.opponent_sha:
            ap.error("--opponent-tar 必須同時給 --opponent-sha")
        stamp = time.strftime("%Y%m%dT%H%M%S")
        dest = run_dir / f"opponent_{args.opponent_sha[:12]}_{stamp}_{os.getpid()}"
        opp_identity = extract_opponent(Path(args.opponent_tar), args.opponent_sha, dest)
        identity_path = run_dir / f"{dest.name}_identity.json"
        identity_path.write_text(json.dumps(opp_identity, ensure_ascii=False, indent=1))
        child_cwd = Path(opp_identity["extract_dir"])

    K = official_module()
    head = {
        "official_module": module_identity(K),
        "interfaces": check_interfaces(K),
        "kaggle_environments_version": kaggle_env_version(),
        "act_timeout": None,
        "caps": CAPS.name,
        "ckpt": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        "weight_role": "smoke" if "smoke" in str(args.ckpt) else "candidate",
        "opponent": opp_identity or {"kind": "builtin-or-path", "raw": args.opponent},
        "source": {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "eval_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    games = []
    for i, seed in enumerate(args.seeds):
        seat = (args.seat0 + i) % 2  # 兩座各半
        g = run_one_game_subprocess(
            ckpt=Path(args.ckpt),
            seed=seed,
            seat=seat,
            collect_obs=args.parse_latency_n if i == 0 else 0,
            opponent=args.opponent,
            identity_path=identity_path,
            cwd=child_cwd,
            out_path=run_dir / f"game{i:03d}_seed{seed}_seat{seat}.json",
        )
        games.append(g)
    obs_list = games[0].pop("obs_kept", [])
    for g in games[1:]:
        g.pop("obs_kept", None)
    env = make("kaggriculture", debug=False)
    head["act_timeout"] = getattr(env.configuration, "actTimeout", None)

    core_ok = opp_identity is None or all(
        g["opponent_identity"].get("checked_at_first_call")
        and g["opponent_identity"].get("checked_after_game")
        and g["opponent_identity"].get("loaded_agent_core_sha256")
        == opp_identity["members"]["agent_core.py"]
        for g in games
    )
    out = {
        **head,
        "games": games,
        "wld": {
            "W": sum(g["result"] == "W" for g in games),
            "L": sum(g["result"] == "L" for g in games),
            "D": sum(g["result"] == "D" for g in games),
        },
        "all_720_entries": all(g["entries"] == 720 for g in games),
        "all_done": all(set(g["statuses"]) == {"DONE"} for g in games),
        "all_valid": all(g.get("validity") == "VALID" for g in games),
        "syntax_clean": all(not g["syntax_violations"] for g in games),
        "opponent_identity_verified": core_ok,
        "peak_rss_gib": max(g["peak_rss_gib"] for g in games) if games else None,
        "parse_latency": parse_latency(Path(args.ckpt), args.seat0, obs_list),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "games"}, ensure_ascii=False, indent=1))
    return (
        0
        if (
            out["all_720_entries"]
            and out["all_done"]
            and out["all_valid"]
            and out["syntax_clean"]
            and out["opponent_identity_verified"]
            and out["parse_latency"]["n"] == args.parse_latency_n
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
