"""完整動作表示第一段的必要正確性測試。

規格 內部規格文件；審查者 §14／§14.1 的落點。
真 fixture 定位與 SHA：內部量測紀錄
（A 同格雙 PLANT、B 同格多 PLACE ＋ SELL 後 BUY、C overlay 還原的 WATER；列號、shard、overlay、
canonical rows 路徑都讀這個檔。2026-09-28 起指向 `s2a_shards_20260927` 的 dev 片，原 09-19 shard12
版本存為 `full_actions_fixture_refs_20260919_superseded.json`）。
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl import caps as caps_mod  # noqa: E402
from rl import full_actions_decode as D  # noqa: E402
from rl import full_actions_head as H  # noqa: E402
from rl import full_actions_obs as FO  # noqa: E402
from rl import ordered_v3_contract as C  # noqa: E402
from rl import rules  # noqa: E402
from rl import state as S  # noqa: E402

CAPS = caps_mod.BOUNDED_TRAINING
FIXREF = Path(os.environ.get("KAGGRI_FIXTURE_REFS") or ROOT / "fixtures/full_actions_fixture_refs.json")


@pytest.fixture(scope="module")
def cfg_params():
    cfg = H.config()
    return cfg, H.init_params(cfg, seed=0)


def _fixture_refs():
    if not FIXREF.exists():
        pytest.skip(f"缺 fixture 定位檔 {FIXREF}")
    return json.loads(FIXREF.read_text())


def _load_row(shard: Path, row: int):
    """讀**單一列**的 `s/*` 與 `d/*`。

    用 `scripts.full_actions_il_train.RowSlicer`：ZIP_STORED 成員直接 seek 到那一列，
    **不把整個成員載進 RAM**、也不重讀前綴（`s/tiles.npy` 是 822.8 MiB）。
    """
    from scripts.full_actions_il_train import RowSlicer  # noqa: PLC0415

    z = RowSlicer(shard)
    st = S.SimState(
        **{f: jnp.asarray(z.read(f"s/{f}.npy", row, row + 1)) for f in S.SimState._fields}
    )
    d = {
        k: jnp.asarray(z.read(f"d/{k}.npy", row, row + 1))
        for k in (
            "unit_verb",
            "unit_crop",
            "unit_item",
            "unit_qty_bytes",
            "market_op",
            "market_item",
            "market_qty_bytes",
            "market_length",
        )
    }
    n_alive = z.read("n_alive.npy", row, row + 1)
    return st, d, n_alive


def test_row_slicer_matches_zipfile_read():
    """seek 讀出來的那一列，要與既有 `_read_slice` 逐位元相同（證明 offset 算對了）。"""
    from scripts.full_actions_il_train import RowSlicer  # noqa: PLC0415
    from scripts.loop4_il_labels import _read_slice  # noqa: PLC0415

    ref = _fixture_refs()
    shard = ROOT / ref["shard"]
    row = ref["A_two_units_same_tile_PLANT"]["row"]
    rs = RowSlicer(shard)
    zf = zipfile.ZipFile(shard)
    for name in ("s/tiles.npy", "s/inv_seq.npy", "d/unit_qty_bytes.npy", "row_ids.npy"):
        assert np.array_equal(rs.read(name, row, row + 1), _read_slice(zf, name, row, row + 1)), (
            name
        )
    assert rs.info("s/tiles.npy")["stored"], "s/tiles 應該是 ZIP_STORED（才走得到 seek 路徑）"
    rs.close()


# ---------------------------------------------------------------- 觀測
def test_obs_dims_and_new_fields():
    st = S.reset(2, CAPS, seed=0)
    o = FO.observation(st, 0, CAPS)
    assert o["tiles"].shape[-1] == 35
    assert o["units"].shape[-1] == 42, "6 基本 ＋ 12 log 量 ＋ 12 線性量 ＋ 12 順位（T3 F5）"
    assert o["products"].shape[-1] == 10
    assert o["global"].shape[-1] == 32, "24 ＋ 5 種子 ＋ 3 shed 動物"
    assert FO.dims()["in_progress"] == "always_zero"


def test_seed_feature_is_not_clipped_at_200():
    """種子尺度是 int32 正域；200 與 400 不得被壓成同一個值。"""
    st = S.reset(2, CAPS, seed=0)
    seeds = np.asarray(st.seeds).copy()
    seeds[0, 0, 0], seeds[1, 0, 0] = 200, 400
    st = st._replace(seeds=jnp.asarray(seeds))
    f = np.asarray(FO.seed_features(st, 0))
    assert f[0, 0] < f[1, 0], "400 的特徵必須大於 200"
    assert f[1, 0] < 1.0


def test_inv_rank_uses_inv_ctr_and_masks_opponent_and_dead():
    st = S.reset(1, CAPS, seed=0)
    inv = np.asarray(st.inv).copy()
    seq = np.asarray(st.inv_seq).copy()
    ctr = np.asarray(st.inv_ctr).copy()
    inv[0, 0, 0, 3] = 2
    seq[0, 0, 0, 3] = 2
    ctr[0, 0, 0] = 3
    inv[0, 1, 0, 3] = 5
    seq[0, 1, 0, 3] = 1
    st = st._replace(inv=jnp.asarray(inv), inv_seq=jnp.asarray(seq), inv_ctr=jnp.asarray(ctr))
    o = FO.observation(st, 0, CAPS)
    own = np.asarray(o["units"])[0, 0, 0, 30:]
    opp = np.asarray(o["units"])[0, 1, :, 6:]
    # T3 F4：規範化順位 rank/(N_present+1)；單一存在品項 → 1/2（舊式 seq/ctr 會是 2/3）
    assert own[3] == pytest.approx(1.0 / 2.0)
    assert (opp == 0).all(), "對手背包順位是 private"
    dead = np.asarray(o["units"])[0, 0, 5, 6:]
    assert (dead == 0).all(), "死單位整組歸零"


# ---------------------------------------------------------------- head
def test_unit_slot_emb_distinguishes_identical_units(cfg_params):
    """沒有 slot embedding 的話，同格同背包的兩個雇工會拿到相同 logits。"""
    cfg, p = cfg_params
    assert p["unit_slot_emb"].shape == (cfg["max_units"], cfg["d_model"])
    st = S.reset(1, CAPS, seed=0)
    st = st._replace(hires_today=jnp.asarray(np.full((1, 2), 4, np.int32)))
    o = FO.observation(st, 0, CAPS)
    h = H.encode_tokens(p, o, cfg)
    lg = np.asarray(H.unit_logits(p, h["unit"])["verb"])[0]
    assert not np.allclose(lg[1], lg[2]), "unit_slot_emb 必須讓不同槽可區分"


def test_no_legacy_heads_in_params(cfg_params):
    _cfg, p = cfg_params
    for k in ("task_w", "tqty_w", "sell_w", "farm_w"):
        assert k not in p, f"舊 head {k} 不得留在新參數樹"


# ---------------------------------------------------------------- 解碼
def test_sample_teacher_share_mask_and_prefix(cfg_params):
    cfg, p = cfg_params
    st = S.reset(2, CAPS, seed=0)
    o = D.decode("sample")(p, cfg, st, 0, CAPS, key=jax.random.PRNGKey(0))
    lab = {k: o["decision"][k] for k in o["decision"] if k != "market_length"}
    t = D.decode("teacher")(p, cfg, st, 0, CAPS, labels=lab)
    assert np.allclose(np.asarray(t["logp"]), np.asarray(o["logp"]), atol=1e-6)
    assert bool(jnp.all(t["active"] == o["active"]))
    assert int(np.asarray(t["teacher_drop"]["unit"]["verb"]).sum()) == 0


def test_teacher_offmask_token_leaves_denominator_but_keeps_raw(cfg_params):
    cfg, p = cfg_params
    st = S.reset(2, CAPS, seed=0)
    o = D.decode("sample")(p, cfg, st, 0, CAPS, key=jax.random.PRNGKey(0))
    lab = {k: o["decision"][k] for k in o["decision"] if k != "market_length"}
    bad = dict(lab)
    v = np.asarray(lab["unit_verb"]).copy()
    v[:, 0] = C.UNIT_OP_INDEX["HARVEST"]  # day 0 空地上不可能收成
    bad["unit_verb"] = jnp.asarray(v)
    t = D.decode("teacher")(p, cfg, st, 0, CAPS, labels=bad)
    drop = np.asarray(t["teacher_drop"]["unit"]["verb"])
    assert drop[:, 0].all(), "落在遮罩外的 teacher token 要退出分母"
    assert not np.asarray(t["active"])[:, 0].any(), "該 token 的 active 必須 False"
    assert (np.asarray(t["decision"]["unit_verb"])[:, 0] == C.UNIT_OP_INDEX["HARVEST"]).all()
    by_op = np.asarray(t["teacher_drop"]["unit_verb_by_op"])
    assert by_op[:, C.UNIT_OP_INDEX["HARVEST"]].sum() == 2, "逐 op 計數"


def test_qty_domain_only_u31_no_stock_clamp(cfg_params):
    """qty 只受 U31 域限制：b0 ≤ 127、不得全零；**不因持有量被遮或截**。"""
    cfg, p = cfg_params
    st = S.reset(4, CAPS, seed=1)
    o = D.decode("sample")(p, cfg, st, 0, CAPS, key=jax.random.PRNGKey(3))
    qb = np.asarray(o["decision"]["unit_qty_bytes"])
    act = np.asarray(o["active"]).reshape(4, -1)
    n_unit = cfg["max_units"] * D.UNIT_SLOTS
    used = act[:, :n_unit].reshape(4, cfg["max_units"], D.UNIT_SLOTS)[..., 3]
    if used.any():
        assert (qb[used][:, 0] <= C.QTY_HIGH_BYTE_MAX).all()
        assert (qb[used].sum(-1) > 0).all(), "四 byte 全零是 sentinel，真數量不得為 0"
    mq = np.asarray(o["decision"]["market_qty_bytes"])
    assert (mq[..., 0] <= C.QTY_HIGH_BYTE_MAX).all()


def test_market_mask_is_syntax_only(cfg_params):
    """先 SELL 再 BUY、重複同品項下單都必須能表示；不得用資金／庫存遮 intent。"""
    cfg, p = cfg_params
    st = S.reset(1, CAPS, seed=0)
    poor = np.asarray(st.money).copy()
    poor[0, :] = 0  # 沒錢
    st = st._replace(money=jnp.asarray(poor), shed=jnp.zeros_like(st.shed))  # 也沒貨
    lab_op = np.zeros((1, C.MAX_ORDERS), np.int32)
    lab_op[0, :4] = [C.MO_SELL, C.MO_BUY_SEED, C.MO_SELL, C.MO_BUY_SEED]
    lab = {
        "unit_verb": jnp.zeros((1, cfg["max_units"]), jnp.int32),
        "unit_crop": jnp.zeros((1, cfg["max_units"]), jnp.int32),
        "unit_item": jnp.zeros((1, cfg["max_units"]), jnp.int32),
        "unit_qty_bytes": jnp.zeros((1, cfg["max_units"], 4), jnp.int32),
        "market_op": jnp.asarray(lab_op),
        "market_item": jnp.zeros((1, C.MAX_ORDERS), jnp.int32),
        "market_qty_bytes": jnp.zeros((1, C.MAX_ORDERS, 4), jnp.int32).at[:, :4, 3].set(1),
    }
    t = D.decode("teacher")(p, cfg, st, 0, CAPS, labels=lab)
    assert int(np.asarray(t["teacher_drop"]["market"]["op"]).sum()) == 0, (
        "沒錢沒貨也不得遮掉 SELL／BUY 的 intent"
    )
    assert int(np.asarray(t["decision"]["market_length"])[0]) == 4


def test_value_comes_from_pre_action_state(cfg_params):
    """V(s) 必須吃動作**前**的 state；投影後的 encoder 只給市場頭。"""
    cfg, p = cfg_params
    st = S.reset(2, CAPS, seed=0)
    o = D.decode("sample")(p, cfg, st, 0, CAPS, key=jax.random.PRNGKey(0))
    h1 = H.encode_tokens(p, FO.observation(st, 0, CAPS), cfg)
    assert np.allclose(np.asarray(o["value"]), np.asarray(H.value(p, h1["glob"])), atol=1e-6)
    h2 = H.encode_tokens(p, FO.observation(o["projected"], 0, CAPS), cfg)
    assert not np.allclose(np.asarray(o["value"]), np.asarray(H.value(p, h2["glob"])), atol=1e-6)


def test_logp_slots_are_263_and_inactive_are_zero(cfg_params):
    cfg, p = cfg_params
    st = S.reset(2, CAPS, seed=0)
    o = D.decode("sample")(p, cfg, st, 0, CAPS, key=jax.random.PRNGKey(0))
    assert o["logp"].shape[-1] == 263 == D.N_SLOTS
    lp, act = np.asarray(o["logp"]), np.asarray(o["active"])
    assert (lp[~act] == 0).all(), "inactive 位置不得有 logp"


# ------------------------------------------------- 真 shard fixture（三欄：intent／合法／效果）
def test_fixture_A_two_units_same_tile_plant(cfg_params):
    """row 729：unit2 與 unit4 同格 (1,3) 都 PLANT MELON。

    三欄分別報：**intent**（raw token 保留）、**合法**（真 prefix 後第二個單位是否仍可種）、
    **效果**（引擎 `apply_unit_actions` 後那一格的 kind／seeds）。
    """
    cfg, p = cfg_params
    ref = _fixture_refs()
    a = ref["A_two_units_same_tile_PLANT"]
    st, d, _n = _load_row(ROOT / ref["shard"], a["row"])
    seat = a["teacher_seat"]
    u2, u4 = a["units"][0]["unit_index"], a["units"][1]["unit_index"]

    # intent：raw token
    assert int(d["unit_verb"][0, u2]) == C.UNIT_OP_INDEX["PLANT"]
    assert int(d["unit_verb"][0, u4]) == C.UNIT_OP_INDEX["PLANT"]

    # 合法：第二個單位的遮罩要從**真 prefix state**（第一個單位種完）算
    fields = {
        "unit_verb": jnp.asarray(np.asarray(d["unit_verb"])),
        "unit_crop": jnp.asarray(np.asarray(d["unit_crop"])),
        "unit_item": jnp.asarray(np.asarray(d["unit_item"])),
        "unit_qty_bytes": jnp.asarray(np.asarray(d["unit_qty_bytes"])),
    }
    from rl import obs as rl_obs  # noqa: PLC0415

    st_before = D.prefix_state(st, fields, seat, CAPS, u4)
    leg = rl_obs.legality(st_before, CAPS)
    plant_ok_second = bool(np.asarray(leg["unit_ops"][0, seat, u4, rules.OP_PLANT]))

    # 效果：整組一次進引擎（引擎的 atomic PLANT 預檢在這裡生效）
    after = D.prefix_state(st, fields, seat, CAPS, cfg["max_units"])
    flat = a["tile_xy"]["y"] * rules.BOARD_SIZE + a["tile_xy"]["x"]
    kind_after = int(np.asarray(after.tiles)[0, seat, flat, rules.C_KIND])
    seeds_before = int(np.asarray(st.seeds)[0, seat, a["units"][0]["crop_id"]])
    seeds_after = int(np.asarray(after.seeds)[0, seat, a["units"][0]["crop_id"]])

    # teacher decode：第二個單位的 PLANT 若已不在遮罩內，該 token 必須 off-mask／inactive，
    # 但 raw token 要保留（不得換成 PASS）
    lab = {k: jnp.asarray(np.asarray(v)) for k, v in d.items() if k != "market_length"}
    t = D.decode("teacher")(p, cfg, st, seat, CAPS, labels=lab)
    drop_verb = np.asarray(t["teacher_drop"]["unit"]["verb"])[0]
    act_slots = np.asarray(t["active"]).reshape(1, -1)[0]
    second_slot = u4 * D.UNIT_SLOTS  # 該單位的 verb 位置
    assert drop_verb[u4] == (not plant_ok_second), "off-mask 計數要對到遮罩結果"
    assert bool(act_slots[second_slot]) == plant_ok_second, "inactive 要與遮罩一致"
    assert int(np.asarray(t["decision"]["unit_verb"])[0, u4]) == C.UNIT_OP_INDEX["PLANT"], (
        "raw token 必須保留"
    )
    assert int(np.asarray(t["decision"]["unit_verb"])[0, u2]) == C.UNIT_OP_INDEX["PLANT"]

    # **decode 輸出的** projected 要與「拿 decode 的 decision 直接跑引擎 kernel」逐欄一致。
    # （比 `prefix_state` 對 `build_action` 是恆真式——同一個呼叫——抓不到 decoder 改了 qty／item。）
    from rl import kernels as K  # noqa: PLC0415

    t_fields = {
        k: t["decision"][k] for k in ("unit_verb", "unit_crop", "unit_item", "unit_qty_bytes")
    }
    t_direct = K.apply_unit_actions(st, D.build_action(st, t_fields, seat, CAPS), CAPS)
    for f in S.SimState._fields:
        assert np.array_equal(
            np.asarray(getattr(t["projected"], f)), np.asarray(getattr(t_direct, f))
        ), f
    g = D.decode("greedy")(p, cfg, st, seat, CAPS)
    g_fields = {
        k: g["decision"][k] for k in ("unit_verb", "unit_crop", "unit_item", "unit_qty_bytes")
    }
    g_direct = K.apply_unit_actions(st, D.build_action(st, g_fields, seat, CAPS), CAPS)
    for f in S.SimState._fields:
        assert np.array_equal(
            np.asarray(getattr(g["projected"], f)), np.asarray(getattr(g_direct, f))
        ), f

    assert kind_after in (rules.TK_PLANT, rules.TK_EMPTY)
    # 三欄一起報，讓 reviewer 讀得到；不對「效果」硬斷言引擎語意
    print(
        json.dumps(
            {
                "intent": {"u2": "PLANT", "u4": "PLANT", "tile": a["tile_xy"]},
                "legal_second_unit_plant_after_prefix": plant_ok_second,
                "effect": {
                    "tile_kind_after": kind_after,
                    "seeds_before": seeds_before,
                    "seeds_after": seeds_after,
                },
            },
            ensure_ascii=False,
        )
    )
    assert seeds_after <= seeds_before


def test_fixture_B_place_then_sell_and_sell_before_buy(cfg_params):
    """row 150：PLACE WOOL 12／FERTILIZER 1 後同回合 SELL，且 SELL 之後還有 BUY。"""
    cfg, p = cfg_params
    ref = _fixture_refs()
    bfx = ref["B_multi_PLACE_same_tile_plus_SELL"]
    st, d, _n = _load_row(ROOT / ref["shard"], bfx["row"])
    seat = bfx["teacher_seat"]
    ops = np.asarray(d["market_op"])[0]
    names = [C.MARKET_OP_NAMES[int(o)] for o in ops[: bfx["market_orders"][-1]["slot"] + 1]]
    assert "SELL" in names and names.index("SELL") < max(
        i for i, n in enumerate(names) if n.startswith("BUY")
    ), "SELL 之後還有 BUY：這個順序必須能表示"

    lab = {k: jnp.asarray(np.asarray(v)) for k, v in d.items() if k != "market_length"}
    t = D.decode("teacher")(p, cfg, st, seat, CAPS, labels=lab)
    mdrop = int(np.asarray(t["teacher_drop"]["market"]["op"]).sum())
    assert mdrop == 0, "市場只有語法域遮罩，真教師訂單序列不得被判違規"

    # --- 真 qty：U31 往返，且與 fixture 記的數量相同 ---
    u0 = bfx["place_units"][0]  # 第一個 PLACE 單位（item／qty 讀 refs）
    qty_u = int(np.asarray(D.u31_from_bytes(lab["unit_qty_bytes"]))[0, u0["unit_index"]])
    assert qty_u == u0["qty"]
    from rl.ordered_v3_codec import qty_to_u31_bytes, u31_bytes_to_qty  # noqa: PLC0415

    raw_bytes = tuple(int(x) for x in np.asarray(lab["unit_qty_bytes"])[0, u0["unit_index"]])
    assert u31_bytes_to_qty(raw_bytes) == qty_u, "codec 與 decode 的 U31 定義必須一致"
    assert tuple(qty_to_u31_bytes(qty_u)) == raw_bytes, "原請求的 byte 要能往返"
    mq = np.asarray(D.u31_from_bytes(lab["market_qty_bytes"]))[0]
    for o in bfx["market_orders"]:
        assert int(mq[o["slot"]]) == o["qty"], f"slot {o['slot']} 的 qty 要往返相同"

    # --- 效果：PLACE 12 投影後真的進 shed；投影與直接 kernel 逐欄一致 ---
    fields = {k: lab[k] for k in ("unit_verb", "unit_crop", "unit_item", "unit_qty_bytes")}
    after = D.prefix_state(st, fields, seat, CAPS, cfg["max_units"])
    from rl import kernels as K  # noqa: PLC0415

    direct = K.apply_unit_actions(st, D.build_action(st, fields, seat, CAPS), CAPS)
    for f in S.SimState._fields:
        assert np.array_equal(np.asarray(getattr(after, f)), np.asarray(getattr(direct, f))), f
    wool = int(u0["item_id"])  # 第一個 PLACE 單位的 item（原 fixture 為 WOOL）
    shed_before = int(np.asarray(st.shed)[0, seat, wool])
    shed_after = int(np.asarray(after.shed)[0, seat, wool])
    inv_before = int(np.asarray(st.inv)[0, seat, u0["unit_index"], wool])
    inv_after = int(np.asarray(after.inv)[0, seat, u0["unit_index"], wool])
    moved = shed_after - shed_before
    assert moved == inv_before - inv_after, "入倉數量要等於單位背包減少量"
    shed_room = rules.SHED_CAPACITY - int(np.asarray(st.shed)[0, seat].sum())
    assert moved == min(u0["qty"], shed_room, inv_before), (
        f"引擎是截量不是拒絕：moved 應為 min({u0['qty']}, room={shed_room}, inv={inv_before})"
    )

    # --- 市場成交另列：投影只做單位相，市場相還沒動（與 env.step 的市場前狀態一致） ---
    assert np.array_equal(np.asarray(after.market_inv), np.asarray(st.market_inv)), (
        "單位相投影不得改市場庫存——成交要另外由 env.step 的市場相處理"
    )

    # --- 市場成交：真的跑 `rl.env.step`（單位＋訂單一次進引擎，含順序） ---
    from rl import env as E  # noqa: PLC0415

    full = {**fields, **{k: lab[k] for k in ("market_op", "market_item", "market_qty_bytes")}}
    raw_act = D.build_action(st, full, seat, CAPS, market=True)
    nxt_raw = E.step(st, raw_act, CAPS)
    dec_fields = {k: t["decision"][k] for k in full}
    nxt_dec = E.step(st, D.build_action(st, dec_fields, seat, CAPS, market=True), CAPS)
    settle = {
        "money_before": int(np.asarray(st.money)[0, seat]),
        "money_after_raw": int(np.asarray(nxt_raw.money)[0, seat]),
        "shed_wool_after_step": int(np.asarray(nxt_raw.shed)[0, seat, wool]),
        "market_inv_changed": bool(
            not np.array_equal(np.asarray(nxt_raw.market_inv), np.asarray(st.market_inv))
        ),
    }
    for f in S.SimState._fields:
        assert np.array_equal(np.asarray(getattr(nxt_raw, f)), np.asarray(getattr(nxt_dec, f))), (
            f"教師原請求與 decode 決策跑 env.step 後 {f} 不同"
        )
    print(
        json.dumps(
            {
                "intent": {
                    "market_ops": names,
                    "place_wool_qty": qty_u,
                    "market_qty": [int(mq[o["slot"]]) for o in bfx["market_orders"]],
                },
                "legal": {
                    "unit_drop": int(np.asarray(t["teacher_drop"]["unit"]["verb"]).sum()),
                    "market_drop": mdrop,
                    "market_length": int(np.asarray(t["decision"]["market_length"])[0]),
                },
                "effect_market_settled_via_env_step": settle,
                "effect": {
                    "shed_wool_before": shed_before,
                    "shed_wool_after": shed_after,
                    "unit_inv_wool_before": inv_before,
                    "unit_inv_wool_after": inv_after,
                    "market_inv_unchanged_by_unit_phase": True,
                },
            },
            ensure_ascii=False,
        )
    )


# ---------------------------------------------------------------- NumPy parity
def test_numpy_obs_matches_jax(cfg_params):
    """部署路徑的觀測要跟 JAX 的一致（f32 同路徑 ≤ 1e-4）。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415
    from rl import loop4_numpy as LNP  # noqa: PLC0415

    st = S.reset(1, CAPS, seed=0)
    nps = LNP.NpState(
        **{f: np.asarray(getattr(st, f))[0] for f in LNP.NpState.__slots__ if hasattr(st, f)}
    )
    o_np = FN.observation(nps, CAPS)
    o_jx = FO.observation(st, 0, CAPS)
    for k in ("tiles", "units", "products", "global"):
        assert o_np[k].shape == tuple(o_jx[k].shape), k
        assert np.max(np.abs(o_np[k] - np.asarray(o_jx[k]))) < 1e-4, k


def test_parent_conditioning_changes_child_logits(cfg_params):
    """item／qty 的 logits 必須**真的**吃已選 parent，不是只靠 mask。"""
    cfg, p = cfg_params
    st = S.reset(1, CAPS, seed=0)
    h = H.encode_tokens(p, FO.observation(st, 0, CAPS), cfg)
    hu = h["unit"][:, 0]
    lg_pick = H.unit_item_logits(p, hu, jnp.asarray([C.OP_PICKUP]))
    lg_place = H.unit_item_logits(p, hu, jnp.asarray([C.OP_PLACE]))
    assert not np.allclose(np.asarray(lg_pick), np.asarray(lg_place)), "unit item 要條件 verb"

    q_a = H.qty_init_unit(p, hu, jnp.asarray([C.OP_PICKUP]), jnp.asarray([0]))
    q_b = H.qty_init_unit(p, hu, jnp.asarray([C.OP_PICKUP]), jnp.asarray([7]))
    assert not np.allclose(np.asarray(q_a), np.asarray(q_b)), "qty hidden 要條件 item"

    hg = h["glob"]
    ptr = H.pointer_keys(p, h["prod"], hg, H.pointer_aux(st, 0))  # T3 A1 pointer 頭
    m0 = H.market_item_logits(p, H.market_init(p, hg), jnp.asarray([C.MO_SELL]), ptr)
    m1 = H.market_item_logits(p, H.market_init(p, hg), jnp.asarray([C.MO_BUY_SEED]), ptr)
    assert not np.allclose(np.asarray(m0), np.asarray(m1)), "market item 要條件 op"
    qa = H.qty_init_market(p, H.market_init(p, hg), jnp.asarray([C.MO_SELL]), jnp.asarray([0]), ptr)
    qb = H.qty_init_market(p, H.market_init(p, hg), jnp.asarray([C.MO_SELL]), jnp.asarray([3]), ptr)
    assert not np.allclose(np.asarray(qa), np.asarray(qb)), "market qty 要條件選定 candidate"

    # 逐 byte：第二個 byte 的 logits 必須隨第一個 byte 改變
    h0 = H.qty_init_unit(p, hu, jnp.asarray([C.OP_PICKUP]), jnp.asarray([0]))
    l_a = H.byte_logits(p, H.byte_advance(p, h0, jnp.asarray([1])))
    l_b = H.byte_logits(p, H.byte_advance(p, h0, jnp.asarray([2])))
    assert not np.allclose(np.asarray(l_a), np.asarray(l_b)), "qty 是逐 byte 條件鏈"


def test_gradient_flows_only_through_active_tokens(cfg_params):
    """inactive／off-mask 位置不得有梯度；改 active 的 teacher token 會改 loss。"""
    cfg, p = cfg_params
    st = S.reset(2, CAPS, seed=0)
    o = D.decode("sample")(p, cfg, st, 0, CAPS, key=jax.random.PRNGKey(0))
    lab = {k: o["decision"][k] for k in o["decision"] if k != "market_length"}

    def loss_of(params, labels):
        r = D.decode("teacher")(params, cfg, st, 0, CAPS, labels=labels)
        n = jnp.maximum(jnp.sum(r["active"]), 1.0)
        return -jnp.sum(jnp.where(r["active"], r["logp"], 0.0)) / n

    g = jax.grad(loss_of)(p, lab)
    assert float(jnp.sum(jnp.abs(g["verb_w"]))) > 0, "active token 必須有梯度"
    base = float(loss_of(p, lab))
    changed = dict(lab)
    v = np.asarray(lab["unit_verb"]).copy()
    v[:, 0] = C.OP_PASS  # PASS 在 day 0 一定合法 ⇒ 仍是 active，loss 應改變
    changed["unit_verb"] = jnp.asarray(v)
    assert abs(float(loss_of(p, changed)) - base) > 1e-6

    # inactive 的 label 怎麼改都不得影響 loss 或任何梯度（day 0 只有 1 個活單位，
    # 槽 5 是死的；padding／未來動作不得污染當步的輸出）
    dead = dict(lab)
    dv = np.asarray(lab["unit_verb"]).copy()
    dv[:, 5] = C.UNIT_OP_INDEX["HARVEST"]
    dq = np.asarray(lab["unit_qty_bytes"]).copy()
    dq[:, 5] = [0, 0, 1, 7]
    dead["unit_verb"] = jnp.asarray(dv)
    dead["unit_qty_bytes"] = jnp.asarray(dq)
    assert float(loss_of(p, dead)) == base, "改 inactive label 不得改 loss"
    g2 = jax.grad(loss_of)(p, dead)
    for k in g:
        assert np.array_equal(np.asarray(g[k]), np.asarray(g2[k])), f"改 inactive label 改了 {k}"


def _np_state(st):
    from rl import loop4_numpy as LNP  # noqa: PLC0415

    return LNP.NpState(
        **{f: np.asarray(getattr(st, f))[0] for f in LNP.NpState.__slots__ if hasattr(st, f)}
    )


def test_numpy_jax_per_token_logp_parity(cfg_params):
    """部署路徑與 JAX 的**逐 263 token logp／active** 要對得上（greedy 同路徑，f32 ≤ 1e-4）。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    st = S.reset(1, CAPS, seed=0)
    pnp = {k: np.asarray(v) for k, v in p.items()}
    got = FN.act(pnp, cfg, _np_state(st), CAPS, greedy=True)
    jx = D.decode("greedy")(p, cfg, st, 0, CAPS)
    for k in ("unit_verb", "unit_crop", "unit_item", "unit_qty_bytes", "market_op", "market_item"):
        assert np.array_equal(np.asarray(got["decision"][k]), np.asarray(jx["decision"][k])[0]), k
    assert got["logp"].shape == (263,) == (FN.N_LOGP_SLOTS,)
    assert np.array_equal(got["active"], np.asarray(jx["active"])[0]), "active 要逐格相同"
    assert np.max(np.abs(got["logp"] - np.asarray(jx["logp"])[0])) < 1e-4
    assert (got["logp"][~got["active"]] == 0).all(), "inactive 位置不得有 logp"
    assert abs(got["value"] - float(np.asarray(jx["value"])[0])) < 1e-4


def test_numpy_evaluate_scores_jax_decision(cfg_params):
    """**雙向耦合之一**：JAX 抽出的 decision 交給 NumPy `evaluate`，逐 token 分數要一致。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    st = S.reset(1, CAPS, seed=2)
    pnp = {k: np.asarray(v) for k, v in p.items()}
    jx = D.decode("sample")(p, cfg, st, 0, CAPS, key=jax.random.PRNGKey(7))
    dec = {k: np.asarray(v)[0] for k, v in jx["decision"].items() if k != "market_length"}
    ev = FN.evaluate(pnp, cfg, _np_state(st), CAPS, dec)
    assert np.array_equal(ev["active"], np.asarray(jx["active"])[0])
    assert np.max(np.abs(ev["logp"] - np.asarray(jx["logp"])[0])) < 1e-4
    for k in dec:
        assert np.array_equal(np.asarray(ev["decision"][k]), dec[k]), f"raw token 要保留：{k}"


def test_jax_teacher_scores_numpy_decision(cfg_params):
    """**雙向耦合之二**：NumPy 抽出的 decision 交給 JAX teacher/recompute，分數要一致。

    不要求跨 PRNG 產生同一個 decision——要求的是**同一個 decision 的分數相同**。
    """
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    st = S.reset(1, CAPS, seed=3)
    pnp = {k: np.asarray(v) for k, v in p.items()}
    got = FN.act(pnp, cfg, _np_state(st), CAPS, rng=np.random.default_rng(0), greedy=False)
    lab = {
        k: jnp.asarray(np.asarray(v)[None])
        for k, v in got["decision"].items()
        if k != "market_length"
    }
    t = D.decode("teacher")(p, cfg, st, 0, CAPS, labels=lab)
    assert np.array_equal(np.asarray(t["active"])[0], got["active"])
    assert np.max(np.abs(np.asarray(t["logp"])[0] - got["logp"])) < 1e-4
    assert int(np.asarray(t["teacher_drop"]["unit"]["verb"]).sum()) == 0, (
        "自己抽出來的動作一定在遮罩內"
    )


def test_numpy_evaluate_drops_offmask_token_from_denominator(cfg_params):
    """NumPy `evaluate` 與 JAX teacher 同一套資格規則：off-mask 的 token 退出分母、raw 保留。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    st = S.reset(1, CAPS, seed=0)
    pnp = {k: np.asarray(v) for k, v in p.items()}
    got = FN.act(pnp, cfg, _np_state(st), CAPS, greedy=True)
    dec = {k: np.asarray(v).copy() for k, v in got["decision"].items() if k != "market_length"}
    dec["unit_verb"][0] = C.UNIT_OP_INDEX["HARVEST"]  # day 0 空地不可能收成
    ev = FN.evaluate(pnp, cfg, _np_state(st), CAPS, dec)
    assert not ev["active"][0], "off-mask 的 token 必須 inactive"
    assert ev["logp"][0] == 0.0
    assert int(ev["decision"]["unit_verb"][0]) == C.UNIT_OP_INDEX["HARVEST"], "raw token 要保留"
    assert ev["teacher_drop"]["unit_verb_by_op"][C.UNIT_OP_INDEX["HARVEST"]] == 1

    lab = {k: jnp.asarray(v[None]) for k, v in dec.items()}
    t = D.decode("teacher")(p, cfg, st, 0, CAPS, labels=lab)
    assert np.array_equal(np.asarray(t["active"])[0], ev["active"]), "兩路資格規則要一致"


# ---------------------------------------------------------------- schema／checkpoint／agent
def test_schema_and_signature_have_one_definition(cfg_params):
    """schema 字串與參數簽章只有一份定義（jax-free 那支），head 只是 re-export。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    _cfg, p = cfg_params
    assert H.SCHEMA is FN.SCHEMA == "full-actions-v1"
    assert H.params_signature is FN.params_signature
    assert H.N_LOGP_SLOTS == FN.N_LOGP_SLOTS == D.N_SLOTS == 263
    sig = FN.params_signature({k: np.asarray(v) for k, v in p.items()})
    assert sig == H.params_signature(p), "jnp 與 np 的簽章要相同"
    assert sig["layout"] == "flat" and sig["count"] == H.param_count(p)


def test_checkpoint_roundtrip_and_rejects_foreign_weights(cfg_params, tmp_path):
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    pnp = {k: np.asarray(v) for k, v in p.items()}
    path = tmp_path / "ckpt.npz"
    FN.save_params(path, pnp, H.meta(cfg, seed=0, params=p))
    got, meta = FN.load_params(path)
    assert meta["schema"] == FN.SCHEMA
    assert meta["params"]["count"] == H.param_count(p)
    assert set(got) == set(pnp)
    for k in pnp:
        assert np.array_equal(got[k], pnp[k]), k

    # 1) 舊格式（沒有 meta）明拒
    old = tmp_path / "old.npz"
    np.savez(old, **{f"p/{k}": v for k, v in pnp.items()})
    with pytest.raises(ValueError, match="meta"):
        FN.load_params(old)
    # 2) schema 不符明拒
    other = tmp_path / "other.npz"
    with pytest.raises(ValueError, match="schema"):
        FN.save_params(other, pnp, {**H.meta(cfg, seed=0), "schema": "loop4-v4"})
    # 3) shape 不符明拒
    bad = tmp_path / "bad.npz"
    FN.save_params(bad, pnp, H.meta(cfg, seed=0, params=p))
    z = dict(np.load(bad, allow_pickle=False))
    z["p/verb_w"] = z["p/verb_w"][:, :3]
    np.savez(bad, **z)
    with pytest.raises(ValueError, match="shape"):
        FN.load_params(bad)
    # 4) SHA256 不符明拒
    with pytest.raises(ValueError, match="SHA256"):
        FN.load_params(path, expect_sha256="0" * 64)


def test_obs_dims_single_source(cfg_params):
    """jax-free 模組裡照抄的維度常數，必須與 `full_actions_obs`／`loop4_model` 相同。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415
    from rl import loop4_model as M  # noqa: PLC0415

    assert (FN.TILE_DIM, FN.UNIT_DIM, FN.PRODUCT_DIM, FN.GLOBAL_DIM) == (
        FO.TILE_DIM,
        FO.UNIT_DIM,
        FO.PRODUCT_DIM,
        FO.GLOBAL_DIM,
    )
    assert FN.N_SEGMENTS == M.N_SEGMENTS
    assert FN.OBS_SCHEMA == FO.OBS_VERSION


def test_expected_param_shapes_matches_init_params(cfg_params):
    """cfg 導出的預期鍵／shape 表，要與 `init_params` 實際建出來的逐項相同。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    assert FN.expected_param_shapes(cfg) == {k: list(np.shape(v)) for k, v in p.items()}


def test_load_requires_full_metadata_and_cfg_derived_shapes(cfg_params, tmp_path):
    """meta 少一塊必填欄位、或參數與 cfg 導出的不符，都必須拒載（不是跳過檢查）。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    pnp = {k: np.asarray(v) for k, v in p.items()}
    good = H.meta(cfg, seed=0, params=p)
    path = tmp_path / "ok.npz"
    FN.save_params(path, pnp, good)

    for field in ("obs", "trunk", "unit_slot_emb", "action_space", "params"):
        broken = {k: v for k, v in json.loads(json.dumps(good)).items() if k != field}
        bad = tmp_path / f"no_{field}.npz"
        np.savez(
            bad,
            meta=np.asarray(json.dumps(broken)),
            **{f"p/{k}": v for k, v in pnp.items()},
        )
        with pytest.raises(ValueError, match=field if field != "params" else "params"):
            FN.load_params(bad)

    # 自報簽章「自己一致」但架構不是本版：參數少一片 trunk layer ⇒ 仍要被 cfg 導出的表擋下
    short = {k: v for k, v in pnp.items() if not k.startswith(f"l{cfg['n_layers'] - 1}_")}
    m2 = H.meta(cfg, seed=0)
    m2["params"] = FN.params_signature(short)
    bad = tmp_path / "short.npz"
    np.savez(bad, meta=np.asarray(json.dumps(m2)), **{f"p/{k}": v for k, v in short.items()})
    with pytest.raises(ValueError, match="參數鍵"):
        FN.load_params(bad)

    # caps 槽數不相容要拒
    with pytest.raises(ValueError, match="max_orders"):
        FN.load_params(path, caps=CAPS._replace(max_orders=CAPS.max_orders - 1))
    with pytest.raises(ValueError, match="max_units"):
        FN.load_params(path, caps=CAPS._replace(max_units=CAPS.max_units - 1))


def test_make_agent_reads_n_heads_from_meta_not_weight_shape(tmp_path):
    """`n_heads` 不在任何權重 shape 裡；正式載入必須讀 metadata，不得用 d%6 猜。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    micro = H.config(d_model=12, n_heads=4, d_ff=24)
    p = H.init_params(micro, seed=0)
    path = tmp_path / "micro.npz"
    FN.save_params(path, {k: np.asarray(v) for k, v in p.items()}, H.meta(micro, seed=0, params=p))
    agent = FN.make_agent(path, seat=0, caps=CAPS)
    assert agent.cfg["n_heads"] == 4, "要讀 meta"
    guessed = FN.config_from_params({k: np.asarray(v) for k, v in p.items()}, CAPS)
    assert guessed["n_heads"] == 6, "shape 反推會猜錯——所以正式路徑不准用它"
    assert agent.cfg["d_model"] == 12 and agent.cfg["d_ff"] == 24


def test_make_agent_produces_engine_json(cfg_params, tmp_path):
    """`make_agent` → `agent(obs)` 要回引擎吃得下的 JSON，而且是同一顆 decode 的輸出。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    path = tmp_path / "ckpt.npz"
    FN.save_params(path, {k: np.asarray(v) for k, v in p.items()}, H.meta(cfg, seed=0, params=p))
    agent = FN.make_agent(path, seat=0, caps=CAPS)
    assert agent.cfg == {k: cfg[k] for k in agent.cfg}, "cfg 來自 meta，逐項相同"
    assert agent.cfg["d_model"] == cfg["d_model"]
    assert agent.cfg["max_units"] == cfg["max_units"]
    assert agent.cfg["n_layers"] == cfg["n_layers"]

    st = S.reset(1, CAPS, seed=0)
    out = FN.act(agent.params, agent.cfg, _np_state(st), CAPS, greedy=True)
    d = out["decision"]
    orders = [
        (int(d["market_op"][k]), int(d["market_item"][k]), int(FN._u31(d["market_qty_bytes"])[k]))
        for k in range(int(d["market_length"]))
    ]
    from rl import loop4_numpy as LNP  # noqa: PLC0415

    js = LNP.action_to_engine_json(
        d["unit_verb"],
        FN._unit_arg(d["unit_verb"], d["unit_crop"], d["unit_item"]),
        FN._u31(d["unit_qty_bytes"]),
        orders,
        int(out["n_alive"]),
    )
    assert isinstance(js["farmer"], list) and js["farmer"][0] in rules.UNIT_OP_NAMES
    assert len(js["market"]) == int(d["market_length"])


# ------------------------------------------------- B2/B3：部署出口與真列 parity
def _np_state_of(st, i: int, seat: int):
    """`SimState[i]` → `NpState`，**自己放索引 0**（部署的 canonical 擺法）。"""
    from rl import loop4_numpy as LNP  # noqa: PLC0415

    two_seat = {
        "money",
        "tiles",
        "unit_x",
        "unit_y",
        "hires_today",
        "n_quads",
        "shed",
        "seeds",
        "inv",
        "inv_seq",
        "inv_ctr",
    }
    out = {}
    for f in LNP.NpState.__slots__:
        a = np.asarray(getattr(st, f))[i]
        out[f] = a[[seat, 1 - seat]] if f in two_seat else a
    return LNP.NpState(**out)


def _decision_np(lab, i: int = 0):
    return {k: np.asarray(v)[i] for k, v in lab.items()}


@pytest.mark.parametrize(
    "fixture_key", ["A_two_units_same_tile_PLANT", "B_multi_PLACE_same_tile_plus_SELL"]
)
def test_engine_json_matches_codec_canonical(cfg_params, fixture_key):
    """**B2**：部署出口的 JSON 要與 `ordered_v3_codec` 由 `d/*` 還原的正典 JSON 逐字相同。

    這證明的是「**訓練吃的 token** ＝ 部署送出的動作」。**不是 raw→JSON**：
    `ordered_v3_data.py:727` 在寫 `d/*` 前跑過 `sanitize_unit_slots`（依原 state 的遮罩把
    不合格 slot 改成 PASS），真正的原請求在 sidecar 的 `raw_action`／`canonical_action`／
    `components[].pre_sanitize_tokens`。raw→JSON 的對帳等最小恢復契約。
    """
    from rl import full_actions_numpy as FN  # noqa: PLC0415
    from rl.ordered_v3_codec import engine_json_from_decision  # noqa: PLC0415
    from rl.ordered_v3_contract import DecisionV3  # noqa: PLC0415

    cfg, p = cfg_params
    ref = _fixture_refs()
    fx = ref[fixture_key]
    st, d, n_alive = _load_row(ROOT / ref["shard"], fx["row"])
    seat, live = fx["teacher_seat"], int(np.asarray(n_alive)[0])
    lab = {k: jnp.asarray(np.asarray(v)) for k, v in d.items() if k != "market_length"}
    t = D.decode("teacher")(p, cfg, st, seat, CAPS, labels=lab)

    canon = engine_json_from_decision(
        DecisionV3(**{k: np.asarray(v, np.int32) for k, v in d.items()}), n_alive=live
    )
    dec = {k: np.asarray(v)[0] for k, v in t["decision"].items() if k != "market_length"}
    dec["market_length"] = int(np.asarray(t["decision"]["market_length"])[0])
    assert dec["market_length"] == int(np.asarray(d["market_length"])[0]), (
        "token 推出的訂單數要等於 shard 記的 market_length"
    )
    assert FN.decision_to_engine_json(dec, live) == canon, "部署 JSON 與正典 JSON 必須逐字相同"


#: 真列 NumPy↔JAX parity 的列：A、B 在 0927 dev 片都是 teacher seat 1，C 是 seat 0，
#: 三者合起來兩座都覆蓋。
#: C 在這裡吃的是 `d/*` 的 **raw sanitized labels**（未套 overlay；其 WATER 槽在 `d/*` 是 PASS），
#: 與 A／B 同一讀法——本測試比的是兩路 decode 對同一組 labels 的逐格結果，不是 overlay 還原。
PARITY_KEYS = [
    "A_two_units_same_tile_PLANT",
    "B_multi_PLACE_same_tile_plus_SELL",
    "C_sanitized_WATER_restored_by_overlay",
]


def test_real_row_parity_keys_cover_both_seats():
    """真列 parity 的 refs 鍵，teacher seat 集合必須是 {0, 1}（兩座都要有真列覆蓋）。"""
    ref = _fixture_refs()
    assert {int(ref[k]["teacher_seat"]) for k in PARITY_KEYS} == {0, 1}


@pytest.mark.parametrize("fixture_key", PARITY_KEYS)
def test_real_row_numpy_jax_parity(cfg_params, fixture_key):
    """**B3**：真列（多活單位、真 qty、真訂單）的雙向逐 263 格 parity ＋ projected 逐欄相等。

    row 729 的 teacher 座位是 1，NumPy 只吃 canonical 狀態，所以先把座位換成自己＝0。
    """
    from rl import full_actions_numpy as FN  # noqa: PLC0415
    from rl import loop4_numpy as LNP  # noqa: PLC0415

    cfg, p = cfg_params
    pnp = {k: np.asarray(v) for k, v in p.items()}
    ref = _fixture_refs()
    fx = ref[fixture_key]
    st, d, _n = _load_row(ROOT / ref["shard"], fx["row"])
    seat = fx["teacher_seat"]
    lab = {k: jnp.asarray(np.asarray(v)) for k, v in d.items() if k != "market_length"}

    jx = D.decode("teacher")(p, cfg, st, seat, CAPS, labels=lab)
    ev = FN.evaluate(pnp, cfg, _np_state_of(st, 0, seat), CAPS, _decision_np(lab))

    assert np.array_equal(ev["active"], np.asarray(jx["active"])[0]), "active 要逐格相同"
    assert np.max(np.abs(ev["logp"] - np.asarray(jx["logp"])[0])) < 1e-4
    assert abs(ev["value"] - float(np.asarray(jx["value"])[0])) < 1e-4

    two_seat = {
        "money",
        "tiles",
        "unit_x",
        "unit_y",
        "hires_today",
        "n_quads",
        "shed",
        "seeds",
        "inv",
        "inv_seq",
        "inv_ctr",
    }
    for f in LNP.NpState.__slots__:
        got = np.asarray(getattr(ev["projected"], f))
        want = np.asarray(getattr(jx["projected"], f))[0]
        if f in two_seat:
            want = want[[seat, 1 - seat]]
        assert np.array_equal(got, want), f"projected 的 {f} 兩路不同"


def test_numpy_decode_rejects_non_canonical_seat(cfg_params):
    """**B4**：NumPy 觀測永遠讀索引 0，seat≠0 會變成「觀測看 0、遮罩看 1」——必須明拒。"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    cfg, p = cfg_params
    st = S.reset(1, CAPS, seed=0)
    pnp = {k: np.asarray(v) for k, v in p.items()}
    got = FN.act(pnp, cfg, _np_state(st), CAPS, greedy=True)
    dec = {k: np.asarray(v) for k, v in got["decision"].items() if k != "market_length"}
    with pytest.raises(ValueError, match="canonical"):
        FN.evaluate(pnp, cfg, _np_state(st), CAPS, dec, seat=1)


# ------------------------------------------------- overlay（§18／§20）
#: 路徑與 WATER 列讀 fixture 定位檔（缺檔時指向不存在的路徑 ⇒ 各測試照舊 skip）。
_REF = json.loads(FIXREF.read_text()) if FIXREF.exists() else {}
_MISSING = ROOT / "fixtures/__fixture_refs_missing__"
OVERLAY_NPZ = ROOT / _REF["overlay_v1"]["npz"] if _REF else _MISSING
OVERLAY_MAN = ROOT / _REF["overlay_v1"]["manifest"] if _REF else _MISSING
CANON_ROWS = ROOT / _REF["canonical_rows"] if _REF else _MISSING
#: reader 的正式路徑用 v2（審查者 §23 的 keep 規則）；拒載那幾條仍用 v1 單片，
#: 這樣不必為了選 entry 改測試。
V2_NPZ = ROOT / _REF["overlay_v2"]["npz"] if _REF else _MISSING
V2_MAN = ROOT / _REF["overlay_v2"]["manifest"] if _REF else _MISSING
#: 原 WATER 被 sanitize 成 PASS、真 prefix 之後才合法的列（refs 的 C 鍵；原 fixture 為 row 333）
ROW_WATER = _REF["C_sanitized_WATER_restored_by_overlay"]["row"] if _REF else -1


SHARDS_MANIFEST = ROOT / _REF["shards_manifest"] if _REF else _MISSING


def _overlay_args():
    """`Overlay` 需要的四個身分：shard SHA、原 manifest 記的 sidecar SHA、原 manifest SHA。"""
    from scripts.full_actions_il_train import manifest_sha_map, sha256_file  # noqa: PLC0415

    if not (OVERLAY_NPZ.exists() and OVERLAY_MAN.exists() and SHARDS_MANIFEST.exists()):
        pytest.skip(f"缺 overlay 或 manifest：{OVERLAY_NPZ}")
    ref = _fixture_refs()
    shard = (ROOT / ref["shard"]).resolve()
    sm = manifest_sha_map(SHARDS_MANIFEST)
    return {
        "shard_sha256": sha256_file(shard),
        "audit_sidecar_sha256": sm[str(shard)]["sidecar_sha"],
        "source_manifest_sha256": sha256_file(SHARDS_MANIFEST),
    }


def _overlay(man_path: Path | None = None):
    from scripts.full_actions_il_train import Overlay  # noqa: PLC0415

    a = _overlay_args()
    return Overlay(
        OVERLAY_NPZ,
        man_path or OVERLAY_MAN,
        a["shard_sha256"],
        audit_sidecar_sha256=a["audit_sidecar_sha256"],
        source_manifest_sha256=a["source_manifest_sha256"],
    )


def _restored_labels(ov, npz_row: int, d: dict):
    """把 overlay 的原 token 疊回這一列的 `d/*`（回 NumPy dict，batch 1）。"""
    lab = {k: np.asarray(v).copy() for k, v in d.items() if k != "market_length"}
    n = 0
    for j in ov.by_row.get(int(npz_row), ()):
        sl = int(ov.slot[j])
        for k, arr in ov.tok.items():
            lab[k][0, sl] = arr[j]
        n += 1
    return lab, n


def test_overlay_rejects_wrong_sha_and_wrong_shard():
    """overlay 必須對 manifest 的 expected SHA；綁錯片或改過檔案都要拒，**不 fallback**。"""
    from scripts.full_actions_il_train import Overlay  # noqa: PLC0415

    a = _overlay_args()
    _overlay()  # 正常路徑不得丟
    with pytest.raises(ValueError, match="別片 shard"):
        Overlay(
            OVERLAY_NPZ,
            OVERLAY_MAN,
            "0" * 64,
            audit_sidecar_sha256=a["audit_sidecar_sha256"],
            source_manifest_sha256=a["source_manifest_sha256"],
        )


def test_overlay_rejects_broken_sha_chain(tmp_path):
    """§23.1：改掉 entry 的 sidecar SHA 一個字元、或換一份 source manifest，都要拒載。"""
    from scripts.full_actions_il_train import Overlay  # noqa: PLC0415

    a = _overlay_args()
    man = json.loads(OVERLAY_MAN.read_text())
    e = man["overlays"][0]
    sc = e["sidecar_sha256"]
    e["sidecar_sha256"] = ("0" if sc[0] != "0" else "1") + sc[1:]
    bad = tmp_path / "bad_sidecar.json"
    bad.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="sidecar SHA"):
        Overlay(
            OVERLAY_NPZ,
            bad,
            a["shard_sha256"],
            audit_sidecar_sha256=a["audit_sidecar_sha256"],
            source_manifest_sha256=a["source_manifest_sha256"],
        )
    with pytest.raises(ValueError, match="別份 shards_manifest"):
        Overlay(
            OVERLAY_NPZ,
            OVERLAY_MAN,
            a["shard_sha256"],
            audit_sidecar_sha256=a["audit_sidecar_sha256"],
            source_manifest_sha256="f" * 64,
        )


def test_shard_reader_requires_overlay():
    """正式路徑沒給 overlay 就拒；只有明示診斷旗標才准吃 sanitized 的 `d/*`。"""
    from scripts.full_actions_il_train import ShardReader  # noqa: PLC0415

    ref = _fixture_refs()
    shard = ROOT / ref["shard"]
    if not shard.exists():
        pytest.skip("缺 shard")
    with pytest.raises(ValueError, match="overlay"):
        ShardReader(shard, split="dev", manifest={})


def test_overlay_restores_row333_water_and_makes_it_active(cfg_params):
    """ROW_WATER（refs C 鍵；原 fixture 為 row 333 unit 8）：`d/*` 是 PASS，overlay 還原成 WATER。

    **走正式路徑**：`ShardReader(..., overlay=v2).rows([ROW_WATER])` 出來的 state／labels／seat／
    unit_keep 直接餵 teacher decode。原 state 遮罩 false、真 prefix 之後 true，
    所以還原後那個 token 必須 active（進得了 loss 分母）。
    """
    from rl import obs as rl_obs  # noqa: PLC0415
    from scripts.full_actions_il_train import (  # noqa: PLC0415
        Overlay,
        ShardReader,
        manifest_sha_map,
        sha256_file,
    )

    cfg, p = cfg_params
    if not (V2_NPZ.exists() and V2_MAN.exists() and SHARDS_MANIFEST.exists()):
        pytest.skip(f"缺 v2 overlay：{V2_NPZ}")
    ref = _fixture_refs()
    shard = (ROOT / ref["shard"]).resolve()
    sm = manifest_sha_map(SHARDS_MANIFEST)
    ov = Overlay(
        V2_NPZ,
        V2_MAN,
        sha256_file(shard),
        audit_sidecar_sha256=sm[str(shard)]["sidecar_sha"],
        source_manifest_sha256=sha256_file(SHARDS_MANIFEST),
    )
    reader = ShardReader(shard, split="dev", manifest=sm, overlay=ov)
    batch = reader.rows(np.asarray([ROW_WATER]))

    assert batch["n_overlaid_slots"] >= 1, "這一列應該有被 sanitize 的槽，reader 要還原它"
    seat = batch["seat"]
    lab = batch["labels"]
    lab_np = {k: np.asarray(v) for k, v in lab.items()}
    slots = [int(ov.slot[j]) for j in ov.by_row[ROW_WATER]]
    water = C.UNIT_OP_INDEX["WATER"]

    _st_raw, d_raw, _n = _load_row(shard, ROW_WATER)  # 未套 overlay 的 d/*
    assert any(int(np.asarray(d_raw["unit_verb"])[0, sl]) == C.OP_PASS for sl in slots), (
        "d/* 原本是 PASS"
    )
    assert any(int(lab_np["unit_verb"][0, sl]) == water for sl in slots), "reader 還原成 WATER"

    st = batch["state"]
    t = D.decode("teacher")(p, cfg, st, seat, CAPS, labels=lab)
    act = np.asarray(t["active"])[0]
    leg0 = np.asarray(rl_obs.legality(st, CAPS)["unit_ops"])[0, seat]
    fields = {k: lab[k] for k in ("unit_verb", "unit_crop", "unit_item", "unit_qty_bytes")}
    checked = 0
    for sl in slots:
        if int(lab_np["unit_verb"][0, sl]) != water:
            continue
        pre = D.prefix_state(st, fields, seat, CAPS, sl)
        legp = np.asarray(rl_obs.legality(pre, CAPS)["unit_ops"])[0, seat]
        assert not bool(leg0[sl, water]), "原 state 遮罩應該是 false（這就是被 sanitize 的原因）"
        assert bool(legp[sl, water]), "真 prefix 之後應該合法"
        assert bool(act[sl * D.UNIT_SLOTS]), "還原後的 WATER 必須 active，才進得了 loss 分母"
        assert bool(batch["unit_keep"][0, sl]), "§17.1 的 keep 要把這個 slot 收回來"
        checked += 1
    assert checked >= 1, "沒有驗到任何還原後的 WATER 槽"


@pytest.mark.parametrize(
    "fixture_key",
    [
        "C_sanitized_WATER_restored_by_overlay",
        "A_two_units_same_tile_PLANT",
        "B_multi_PLACE_same_tile_plus_SELL",
    ],
)
def test_overlay_tokens_match_sidecar_canonical_json(fixture_key):
    """raw→tokens→JSON：overlay 還原後的 token 轉成引擎 JSON，要等於 sidecar 的
    `canonical_action`。（比對 `d/*` 只證 token→出口，那是 sanitize 過的。）"""
    from rl import full_actions_numpy as FN  # noqa: PLC0415

    if not CANON_ROWS.exists():
        pytest.skip(f"缺 {CANON_ROWS}（用 full_actions_raw_overlay.py canonical-rows 產）")
    ov = _overlay()
    ref = _fixture_refs()
    npz_row = ref[fixture_key]["row"]
    canon = json.loads(CANON_ROWS.read_text())
    row = canon["rows"][str(npz_row)]
    _st, d, n_alive = _load_row(ROOT / ref["shard"], npz_row)
    lab_np, _n = _restored_labels(ov, npz_row, d)
    dec = {k: v[0] for k, v in lab_np.items()}
    dec["market_length"] = int(np.asarray(d["market_length"])[0])
    got = FN.decision_to_engine_json(dec, int(row["n_alive"]))
    assert got == row["canonical_action"], f"row {npz_row} 的 raw→JSON 不符 sidecar canonical"
    assert int(np.asarray(n_alive)[0]) == int(row["n_alive"])


# ------------------------------------------------- resume 資料身分／髒工作樹（§26、§26.1）
def _real_source():
    """用**真 reader**（v2 overlay）組 source identity，不是手捏 dict。"""
    from scripts.full_actions_il_train import (  # noqa: PLC0415
        Overlay,
        ShardReader,
        manifest_sha_map,
        sha256_file,
        source_identity,
    )

    if not (V2_NPZ.exists() and V2_MAN.exists() and SHARDS_MANIFEST.exists()):
        pytest.skip(f"缺 v2 overlay：{V2_NPZ}")
    ref = _fixture_refs()
    shard = (ROOT / ref["shard"]).resolve()
    sm = manifest_sha_map(SHARDS_MANIFEST)
    ov = Overlay(
        V2_NPZ,
        V2_MAN,
        sha256_file(shard),
        audit_sidecar_sha256=sm[str(shard)]["sidecar_sha"],
        source_manifest_sha256=sha256_file(SHARDS_MANIFEST),
    )
    return source_identity([ShardReader(shard, split="dev", manifest=sm, overlay=ov)])


def test_resume_identity_accepts_same_and_rejects_changed_overlay():
    """resume 的資料身分：同身分要過；overlay／manifest／shard 任一 SHA 變了就拒。

    `git_head` 不同**不算**不同——repo 改了但資料沒變，resume 是合法的。
    """
    import copy  # noqa: PLC0415

    from scripts.full_actions_il_train import check_resume_identity  # noqa: PLC0415

    source = _real_source()
    check_resume_identity({"source": source}, source)  # 同身分不得丟

    moved_head = copy.deepcopy(source)
    moved_head["git_head"] = "0" * 40
    moved_head["dirty_paths"] = ["whatever.py"]
    check_resume_identity({"source": moved_head}, source)  # 只有 repo 變：仍可 resume

    for field in ("overlay_sha256", "overlay_manifest_sha256", "source_manifest_sha256"):
        bad = copy.deepcopy(source)
        bad["inputs"][0]["overlay"][field] = "0" * 64
        with pytest.raises(ValueError, match="資料身分"):
            check_resume_identity({"source": bad}, source)

    bad_npz = copy.deepcopy(source)
    bad_npz["inputs"][0]["npz_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="資料身分"):
        check_resume_identity({"source": bad_npz}, source)

    with pytest.raises(ValueError, match="meta.source"):
        check_resume_identity({}, source)


def test_resume_identity_rejects_reordered_shards():
    """**順序也是身分**：`rng` 用 `trains` 的索引抽片，換序＝換了續跑的抽樣序列。"""
    import copy  # noqa: PLC0415

    from scripts.full_actions_il_train import check_resume_identity  # noqa: PLC0415

    one = _real_source()
    two = copy.deepcopy(one)
    second = copy.deepcopy(one["inputs"][0])
    second["npz_sha256"] = "a" * 64
    second["split"] = "train"
    two["inputs"] = [one["inputs"][0], second]

    check_resume_identity({"source": two}, two)  # 同順序要過
    swapped = copy.deepcopy(two)
    swapped["inputs"] = list(reversed(swapped["inputs"]))
    with pytest.raises(ValueError, match="資料身分"):
        check_resume_identity({"source": swapped}, two)


def test_check_dirty_only_blocks_dependency_files():
    """髒工作樹：與依賴集合有交集才拒；作者 既有的無關腳本照實 dirty=true 但不擋。"""
    from scripts.full_actions_il_train import (  # noqa: PLC0415
        DEPENDENCY_FILES,
        check_dirty,
    )

    unrelated = [
        "scripts/run_loop4_arch_sweep_chain.sh",
        "scripts/loop4_plant_sampling_probe.py",
        "scripts/wait_then_chain2_task27.sh",
        "內部規格文件",
    ]
    check_dirty(unrelated)  # 不得丟
    for dep in ("rl/kernels.py", "rl/obs.py", "scripts/full_actions_il_train.py"):
        with pytest.raises(SystemExit, match=dep.split("/")[-1]):
            check_dirty([*unrelated, dep])
    for f in DEPENDENCY_FILES:
        assert (ROOT / f).exists(), f"依賴集合列到不存在的檔：{f}（改名了？）"


def test_git_porcelain_parses_untracked_and_renames(monkeypatch):
    """`dirty_paths` 要含 untracked，rename 取新舊兩邊——不然依賴檔改名會躲過硬檢。"""
    import subprocess  # noqa: PLC0415

    from scripts import full_actions_il_train as T  # noqa: PLC0415

    sample = " M rl/kernels.py\n?? scripts/new_thing.py\nR  rl/obs.py -> rl/obs2.py\n"

    class _R:
        stdout = sample

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    got = T._git_porcelain()
    assert got == sorted(["rl/kernels.py", "scripts/new_thing.py", "rl/obs.py", "rl/obs2.py"])
