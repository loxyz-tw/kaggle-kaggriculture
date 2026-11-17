"""`scripts/full_actions_eval.py` 的只讀 hook 驗證。

規格：內部規格文件。

要證的事只有兩件：

1. **包裝不改行為**——同一個 seed、同一組動作，原官方 env 與包過 hook 的 env
   逐步的 `observation`／`reward`／`status` 完全相同。
2. **事件數量對得上**——三個會讓「回合淨差」失效的情境（同回合 PLACE→SELL、
   同品項 SELL→BUY、日末 WATER）都要記出正確的事件數，而不是被淨差抵銷掉。

用明確的原生 state fixture（直接寫官方 state 的欄位），不重建市場模擬器。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

kaggle_environments = pytest.importorskip("kaggle_environments")
from kaggle_environments import make  # noqa: E402

from scripts import full_actions_eval as E  # noqa: E402

K = E.official_module()
TPD = 24


def _fresh(seed: int = 5000):
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(2)
    return env


def _dump(env) -> str:
    """逐步 state：observation／reward／status 全部進來，不只比最後一格。"""
    return json.dumps(env.steps, sort_keys=True, default=str)


def _shed_adjacent_pos(board_size: int = 10) -> list:
    x, y = K._shed_access_tiles(board_size)[0]
    return [int(x), int(y)]


def _mut_place_then_sell(env) -> None:
    """同回合 PLACE→SELL：入倉與賣出互相抵銷，shed 淨差可能為 0。"""
    for s in env.state:
        farm = s.observation.farms[s.observation.player]
        farm["farmer"] = _shed_adjacent_pos()
        s.observation.private["shed"]["WHEAT"] = 0
        K._farmer_inventory(s.observation.private, 0)["WHEAT"] = 3


def _mut_sell_then_buy(env) -> None:
    """同品項 SELL→BUY：庫存淨差可能為 0。"""
    for s in env.state:
        s.observation.private["shed"]["WHEAT"] = 5
        s.observation.farms[s.observation.player]["money"] = 10_000


def _mut_day_end_water(env) -> None:
    """日末 WATER：`watered_today` 會在日末被清掉，事後從 observation 看不出來。"""
    for s in env.state:
        farm = s.observation.farms[s.observation.player]
        pos = farm["farmer"]
        x, y = map(int, pos)
        farm["tiles"][y][x] = K._new_plant("WHEAT", 0, TPD)
        farm["tiles"][y][x]["watered_today"] = False
        s.observation.step = TPD - 1  # h23：這一步結束就跑 `_end_of_day`


SCENARIOS = {
    "place_then_sell": (
        _mut_place_then_sell,
        {"farmer": ["PLACE", "WHEAT", 3], "hands": [], "market": [["SELL", "WHEAT", 2]]},
    ),
    "sell_then_buy": (
        _mut_sell_then_buy,
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "WHEAT", 2], ["BUY_PRODUCT", "WHEAT", 2]],
        },
    ),
    "day_end_water": (
        _mut_day_end_water,
        {"farmer": ["WATER"], "hands": [], "market": []},
    ),
}


def _run(scenario: str, hooked: bool):
    mutate, action = SCENARIOS[scenario]
    env = _fresh()
    mutate(env)
    rec = E.Recorder()
    if hooked:
        with E.read_only_hooks(env, rec):
            env.step([action, action])
    else:
        env.step([action, action])
    return env, rec


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_hooks_do_not_change_engine_behaviour(scenario):
    """原 env 與包過 hook 的 env，逐步 state／reward／status 必須完全相同。"""
    plain, _ = _run(scenario, hooked=False)
    hooked, _rec = _run(scenario, hooked=True)
    assert _dump(plain) == _dump(hooked), f"{scenario}：包裝改變了引擎行為"


def test_hooks_are_restored_even_on_exception():
    """`finally` 一定還原三個函式——例外也不能留下髒的 module。"""
    env = _fresh()
    before = (K._apply_unit_action, K._commit_unit, env.interpreter)
    with pytest.raises(RuntimeError), E.read_only_hooks(env, E.Recorder()):
        raise RuntimeError("boom")
    assert (K._apply_unit_action, K._commit_unit, env.interpreter) == before


def test_wrapper_interpreter_is_two_arg_closure():
    """`kaggle_environments` 看 `__code__.co_argcount`；綁定 method／`*args` 會被判錯 arity。"""
    env = _fresh()
    with E.read_only_hooks(env, E.Recorder()):
        fn = env.interpreter
        assert fn.__code__.co_argcount == 2
        assert fn.__code__.co_varnames[:2] == ("state", "env")
        assert fn.__code__.co_flags & 0x04 == 0, "不得有 *args"
    assert E.EXPECTED_SIGNATURES["interpreter"] == ("state", "env")


def test_interfaces_checked_and_module_identity_recorded():
    ident = E.module_identity(K)
    assert Path(ident["path"]).exists() and len(ident["sha256"]) == 64
    assert set(E.check_interfaces(K)) == set(E.EXPECTED_SIGNATURES)


def test_place_then_sell_counts_both_events_not_net_diff():
    """同回合入倉 3 ＋ 賣出 2：shed 淨差只有 +1，但兩個事件都要各自記到。"""
    _env, rec = _run("place_then_sell", hooked=True)
    stores = [e for e in rec.unit_events if e["player"] == 0 and e["effect"]["kind"] == "STORE"]
    assert len(stores) == 1, "PLACE 的入倉要記成一個 STORE 事件"
    assert stores[0]["effect"]["stored"].get("WHEAT") == 3
    sold = [s for s in rec.sales if s["player"] == 0 and s["item"] == "WHEAT"]
    assert len(sold) == 2, "`_commit_unit` 回 True 才算一件，SELL 2 就是兩件"
    net = int(stores[0]["shed_after"].get("WHEAT", 0)) - int(
        stores[0]["shed_before"].get("WHEAT", 0)
    )
    assert net == 3, "單位相位的 shed 差是 +3；市場相位另外減 2——淨差 +1 不能當任何一方的 actual"


def test_sell_then_buy_same_item_counts_sales():
    """同品項先賣後買：庫存淨差可能為 0，但成交件數必須是 2。"""
    _env, rec = _run("sell_then_buy", hooked=True)
    sold = [s for s in rec.sales if s["player"] == 0 and s["item"] == "WHEAT"]
    assert len(sold) == 2
    assert all(s["price"] >= 1 for s in sold), "$1 賣出也算成交一件"


def test_day_end_water_is_observed_before_the_flag_is_cleared():
    """日末 WATER：hook 在單位相位觀測，所以記得到；事後看 observation 已被清旗。"""
    env, rec = _run("day_end_water", hooked=True)
    waters = [e for e in rec.unit_events if e["player"] == 0 and e["effect"]["kind"] == "WATER"]
    assert len(waters) == 1, "日末澆水要記成一次成功 WATER"
    farm = env.state[0].observation.farms[0]
    pos = farm["farmer"]
    tile = farm["tiles"][int(pos[1])][int(pos[0])]
    assert isinstance(tile, dict) and not tile.get("watered_today"), (
        "日末已清旗——這就是為什麼不能靠事後 observation 差分"
    )


def test_raw_allowed_effect_are_three_separate_columns():
    """三欄並列：intent 出現不等於成功。"""
    _env, rec = _run("day_end_water", hooked=True)
    e = next(x for x in rec.unit_events if x["player"] == 0 and x["unit"] == 0)
    assert e["raw"] == ["WATER"] and e["allowed"] == ["WATER"]
    assert set(e["effect"]) >= {"op", "kind", "inv_delta", "shed_delta"}
    counts = E.event_counts(rec.unit_events, 0)
    assert set(counts) == {"raw", "allowed", "effect"}
    assert counts["raw"].get("WATER") == 1 and counts["effect"].get("WATER") == 1


def test_snapshot_native_animal_and_missing_inventory_are_read_only():
    env = _fresh()
    farm, private = env.state[0].observation.farms[0], env.state[0].observation.private
    x, y = farm["farmer"]
    farm["tiles"][y][x] = K._new_animal("GOOSE", 0)
    private["inventories"] = []
    before = _dump(env)
    rec = E.Recorder()
    rec.begin_turn(0, 0, [farm], {})
    e = rec.unit_before(farm, private, 0, [])
    rec.unit_after(e, farm, private, 0)
    assert _dump(env) == before
    assert e["tile_before"]["animal"] == "GOOSE"
    assert e["pos_before"] == farm["farmer"]
    # Invalid empty request returns before the official inventory grow helper.
    with E.read_only_hooks(env, rec):
        rec.begin_turn(0, 0, [farm], {})
        K._apply_unit_action(farm, private, 0, [], 10, 0, TPD)
    assert _dump(env) == before


def _event(turn, kind, unit=0, pos=(1, 1), crop="WHEAT", **effect):
    return {
        "player": 0,
        "turn": turn,
        "day": turn // 24,
        "unit": unit,
        "pos_after": list(pos),
        "tile_after": {"kind": "PLANT", "crop": crop, "planted_day": 0},
        "effect": {"kind": kind, "crop": crop, **effect},
    }


def _sale(turn):
    return {"player": 0, "turn": turn, "item": "WHEAT", "price": 1}


def test_chain_b_rejects_future_and_accepts_unit_before_market():
    ev = [_event(10, "HARVEST", items={"WHEAT": 3}), _event(11, "STORE", stored={"WHEAT": 2})]
    assert E.chain_b(ev, [_sale(0)], 0)["chain_b_count"] == 0
    result = E.chain_b(ev, [_sale(0), _sale(11), _sale(11)], 0)
    assert result["chain_b_count"] == 1
    link = result["chain_b"][0]
    assert (link["harvest_turn"], link["store_turn"], link["sell_turn"]) == (10, 11, 11)
    assert link["harvest_qty"] == 3 and link["store_qty"] == 2
    # No stale store state can be inherited by a new harvest sequence.
    ev += [_event(12, "HARVEST", items={"WHEAT": 1})]
    result = E.chain_b(ev, [_sale(11), _sale(13)], 0)
    assert result["chain_b_count"] == 1
    assert "WHEAT" in result["chain_b_broken_after_harvest"]


def test_chain_a_water_can_cross_day_and_unit_but_not_crop_replacement():
    ev = [_event(1, "MOVE"), _event(2, "PLANT"), _event(25, "WATER", unit=1)]
    result = E.chains(ev, 0)
    assert result["chain_a_count"] == 1
    assert result["chain_a"][0]["water_unit"] == 1
    assert result["chain_a"][0]["unit"] == 0
    removed = _event(3, None)
    removed["tile_after"] = {"kind": None}
    assert E.chains(ev[:2] + [removed, ev[-1]], 0)["chain_a_count"] == 0
    # A fresh same-crop PLANT at the same tile cannot reuse the earlier walk.
    assert E.chains(ev[:2] + [_event(3, "PLANT", unit=1), ev[-1]], 0)["chain_a_count"] == 0


def test_sell_summary_preserves_turn_bins():
    out = E.sell_summary([_sale(2), _sale(3)], 0, [1, 1])
    assert out["actual_units_total"] == 2
    assert [(r["turn"], r["units"]) for r in out["actual_by_turn_item"]] == [(2, 1), (3, 1)]


def test_interpreter_audit_failure_is_not_silently_accepted(monkeypatch):
    env, rec = _fresh(), E.Recorder()

    def fail(*args):
        raise ValueError("audit broken")

    monkeypatch.setattr(rec, "begin_turn", fail)
    with E.read_only_hooks(env, rec), pytest.raises(ValueError, match="audit broken"):
        env.interpreter(env.state, env)


def test_animal_placement_is_not_drop_overflow():
    env = _fresh()
    farm, private = env.state[0].observation.farms[0], env.state[0].observation.private
    x, y = farm["farmer"]
    farm["tiles"][y][x] = {"kind": "COOP"}
    private["inventories"][0]["GOOSE"] = 1
    rec = E.Recorder()
    with E.read_only_hooks(env, rec):
        rec.begin_turn(0, 0, [farm], {})
        K._apply_unit_action(farm, private, 0, ["PLACE", "GOOSE", 1], 10, 0, TPD)
    assert rec.unit_events[0]["effect"]["kind"] == "PLACE_ANIMAL"
    assert rec.unit_events[0]["effect"]["overflow"] == {}


def test_daily_planted_uses_real_fold_field():
    env = _fresh()
    probe = E._daily_probe(env.state[0].observation, 0)
    probes = [dict(probe) for _ in range(720)]
    probes[-1]["planted"] = 7
    assert E.daily_planted(probes) == 7


def test_full_action_checks_units():
    base = {"farmer": ["PASS"], "hands": [], "market": []}
    assert E.check_full_action(base) == []
    assert E.check_full_action({**base, "farmer": ["PLACE", "WHEAT", 0]})
    assert E.check_full_action({**base, "hands": [["BOGUS"]]})


def test_main_records_weight_sha_and_rejects_failed_game(tmp_path, monkeypatch):
    import hashlib

    ckpt = tmp_path / "smoke.npz"
    ckpt.write_bytes(b"identity fixture")
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval", "--ckpt", str(ckpt), "--seeds", "1", "--parse-latency-n", "0", "--out", str(out)],
    )
    monkeypatch.setattr(
        E,
        "run_one_game_subprocess",
        lambda **kw: {
            "entries": 2,
            "statuses": ["ERROR", "DONE"],
            "syntax_violations": [],
            "obs_kept": [],
            "result": "L",
            "peak_rss_gib": 0.1,
        },
    )
    monkeypatch.setattr(E, "parse_latency", lambda *a: {"n": 0})
    assert E.main() == 1
    result = json.loads(out.read_text())
    assert result["ckpt_sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert result["weight_role"] == "smoke"


def test_linked_chain_requires_observed_crop_and_water_before_harvest():
    ev = [
        _event(1, "MOVE"),
        _event(2, "PLANT"),
        _event(3, "WATER", unit=1),
        _event(4, "HARVEST", items={"WHEAT": 2}),
        _event(5, "STORE", stored={"WHEAT": 2}),
    ]
    for e in ev:
        e["tile_before"] = dict(e["tile_after"])
    ev[1]["tile_before"] = {"kind": None}
    move_other = _event(2, "MOVE", unit=1)
    move_other["tile_before"] = {"kind": None}
    ev.insert(2, move_other)
    a = E.chains(ev, 0)["chain_a"]
    b = E.chain_b(ev, [_sale(5)], 0)["chain_b"]
    assert E.linked_chains(ev, 0, a, b)["linked_chain_count"] == 1
    # Same kind on another plot is not the planted/watered crop.
    ev[4]["pos_after"] = [2, 2]
    assert E.linked_chains(ev, 0, a, b)["linked_chain_count"] == 0
    # Water later in a turn must not retroactively qualify an earlier harvest.
    ev[4]["pos_after"] = [1, 1]
    ev[3]["turn"] = ev[4]["turn"]
    reordered = [ev[0], ev[1], ev[2], ev[4], ev[3], ev[5]]
    assert E.linked_chains(reordered, 0, a, b)["linked_chain_count"] == 0


# ------------------------------------------------ §38 封存對手（真提交 tar）
#
# 為什麼要這四支：`exec(code, {})` 沒有 `__file__` ⇒ 包內 `main.py` 退回相對字串、
# `abspath()` 接到 **cwd**、`sys.path.insert(0, cwd)`。本線 repo 根目錄也有一支
# `agent_core.py`，從根跑會靜默載錯 core，基準就不是真提交版本。

_FAKE_MAIN = """\
import os
import sys

_THIS = globals().get("__file__", "v27.py")
for _p in (os.path.dirname(os.path.abspath(_THIS)), "/kaggle_simulations/agent"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import agent_core


def agent(obs):
    return agent_core.plan(obs)
"""


def _make_tar(tmp_path: Path, core_body: str = "def plan(obs):\n    return {}\n"):
    import tarfile

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(_FAKE_MAIN)
    (src / "agent_core.py").write_text(core_body)
    tar = tmp_path / "pkg.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(src / "main.py", arcname="main.py")
        tf.add(src / "agent_core.py", arcname="agent_core.py")
    return tar, E.sha256_file(tar)


def test_extract_opponent_rejects_wrong_tar_sha_and_existing_dir(tmp_path):
    tar, sha = _make_tar(tmp_path)
    with pytest.raises(ValueError, match="tar SHA 不符"):
        E.extract_opponent(tar, "0" * 64, tmp_path / "d1")
    ident = E.extract_opponent(tar, sha, tmp_path / "d2")
    assert set(ident["members"]) == {"main.py", "agent_core.py"}
    assert ident["tar_sha256"] == sha
    assert Path(ident["main_py"]).is_file()
    # 每 run 新建唯一目錄：既存目錄一律拒，不覆蓋既有證據
    with pytest.raises(FileExistsError):
        E.extract_opponent(tar, sha, tmp_path / "d2")


def test_verify_loaded_core_rejects_outside_dir_and_sha_mismatch(tmp_path, monkeypatch):
    import types

    good = tmp_path / "pkg"
    good.mkdir()
    core = good / "agent_core.py"
    core.write_text("X = 1\n")
    sha = E.sha256_file(core)

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "agent_core.py").write_text("X = 1\n")

    mod = types.ModuleType("agent_core")
    mod.__file__ = str(outside / "agent_core.py")
    monkeypatch.setitem(sys.modules, "agent_core", mod)
    with pytest.raises(RuntimeError, match="不在解包目錄"):
        E.verify_loaded_core(good, sha)

    mod.__file__ = str(core)
    with pytest.raises(RuntimeError, match="SHA 不符"):
        E.verify_loaded_core(good, "1" * 64)
    assert E.verify_loaded_core(good, sha)["loaded_agent_core_sha256"] == sha

    monkeypatch.delitem(sys.modules, "agent_core")
    with pytest.raises(RuntimeError, match="還沒被載入"):
        E.verify_loaded_core(good, sha)


def _probe(main_py: Path, core_sha: str, extract_dir: Path, cwd: Path):
    import os
    import subprocess

    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "full_actions_eval.py"),
            "--ckpt",
            "unused",
            "--out",
            "unused",
            "--probe-opponent",
            str(main_py),
            "--opp-core-sha",
            core_sha,
            "--extract-dir",
            str(extract_dir),
        ],
        cwd=str(cwd),
        env={**os.environ, "PYTHONPATH": str(ROOT), "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
    )


def test_fresh_process_loads_packaged_core_even_with_namesake_on_path(tmp_path, monkeypatch):
    """三件事：父行程 `sys.modules` 已被同名 decoy 佔住時，每局的 **fresh subprocess**
    仍載到包內那份；cwd＝解包目錄的載入正確；cwd＝別處（模擬 repo 根、同名 core 在
    cwd）則**明拒**，不靜默用錯的 core。"""
    import types

    tar, sha = _make_tar(tmp_path, core_body="def plan(obs):\n    return {'packaged': True}\n")
    dest = tmp_path / "pkg"
    ident = E.extract_opponent(tar, sha, dest)
    core_sha = ident["members"]["agent_core.py"]

    decoy = tmp_path / "decoy"
    decoy.mkdir()
    decoy_core = decoy / "agent_core.py"
    decoy_core.write_text("def plan(obs):\n    return {'decoy': True}\n")

    # 父行程先被同名模組污染（實測會發生的情況：本線 repo 根也有一支 agent_core.py）。
    # 子行程是全新 interpreter，這個污染不該傳下去；monkeypatch 結束自動還原。
    poison = types.ModuleType("agent_core")
    poison.__file__ = str(decoy_core)
    monkeypatch.setitem(sys.modules, "agent_core", poison)
    assert E.sha256_file(decoy_core) != core_sha

    ok = _probe(Path(ident["main_py"]), core_sha, dest, cwd=dest)
    assert ok.returncode == 0, ok.stderr[-2000:]
    loaded = json.loads(ok.stdout.strip().splitlines()[-1])
    assert Path(loaded["loaded_agent_core_file"]).parent == dest.resolve()
    assert loaded["loaded_agent_core_sha256"] == core_sha

    # 父行程的 decoy 還在 sys.modules，子行程照樣載到包內那份
    assert sys.modules["agent_core"] is poison

    # cwd 錯（模擬從 repo 根跑）先被明檢擋下，不被「把解包目錄硬置首位」蓋過去
    bad = _probe(Path(ident["main_py"]), core_sha, dest, cwd=decoy)
    assert bad.returncode != 0
    assert "cwd 不是解包目錄" in bad.stderr


def test_worker_failure_is_not_swallowed_by_parent(tmp_path, monkeypatch):
    """子行程非零 ⇒ 父層拋錯並帶上 stderr，不把該局當成沒發生。"""
    ckpt = tmp_path / "broken.npz"
    ckpt.write_bytes(b"not a real npz")
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--ckpt",
            str(ckpt),
            "--seeds",
            "1",
            "--parse-latency-n",
            "0",
            "--out",
            str(out),
        ],
    )
    with pytest.raises(RuntimeError, match="每局子行程失敗"):
        E.main()
    assert not out.exists()


def test_kaggle_env_version_is_a_plain_string():
    """`kaggle_environments.version` 是 importlib 的函式，不是版本；result 要能 json.dumps。"""
    v = E.kaggle_env_version()
    assert isinstance(v, str) and v == kaggle_environments.__version__
    assert (
        json.loads(json.dumps({"kaggle_environments_version": v}))["kaggle_environments_version"]
        == v
    )


# ------------------------------------------------ 雙方市場成交落盤（2026-09-26，舊欄位不動）
_MARKET_ORDERS = [
    ["HIRE"],
    ["BUY_LAND"],
    ["BUY_ANIMAL", "GOOSE", 2],
    ["SELL", "WHEAT", 2],
    ["BUY_SEED", "WHEAT", 1],
    ["BUY_PRODUCT", "FERTILIZER", 1],
]


def _mut_market_rich_poor(env) -> None:
    """座 0 沒錢、shed 空（全部被拒）；座 1 有錢有貨（全部成交）。"""
    for s in env.state:
        # 引擎用 state[0] 的 farms；各 state 的 farms 不一定是同一物件 ⇒ 每份都設
        for p, farm in enumerate(s.observation.farms):
            farm["money"] = 0.0 if p == 0 else 10_000.0
        p = s.observation.player
        s.observation.private["shed"]["WHEAT"] = 0 if p == 0 else 5


def _run_market(hooked: bool):
    env = _fresh()
    _mut_market_rich_poor(env)
    act = {"farmer": ["PASS"], "hands": [], "market": [list(o) for o in _MARKET_ORDERS]}
    rec = E.Recorder()
    if hooked:
        with E.read_only_hooks(env, rec):
            env.step([act, act])
    else:
        env.step([act, act])
    return env, rec


def test_market_hooks_do_not_change_engine_behaviour():
    plain, _ = _run_market(hooked=False)
    hooked, _rec = _run_market(hooked=True)
    assert _dump(plain) == _dump(hooked), "市場 hook 改變了引擎行為"


def test_market_trades_recorded_for_both_players_with_reasons_and_cash_identity():
    env, rec = _run_market(hooked=True)
    obs0 = env.steps[-1][0].observation
    t0 = [t for t in rec.trades if t["player"] == 0]
    t1 = [t for t in rec.trades if t["player"] == 1]
    # 座 0：每個 op 各一次嘗試、全部被拒、不動現金
    assert {t["op"] for t in t0} == set(E.MARKET_OPS)
    assert all(not t["ok"] and t["cash_delta"] == 0.0 for t in t0)
    reasons = {t["op"]: t["reason"] for t in t0}
    assert reasons == {
        "HIRE": "money",
        "BUY_LAND": "money",
        "BUY_ANIMAL": "money",
        "SELL": "shed_empty",
        "BUY_SEED": "money",
        "BUY_PRODUCT": "money",
    }
    # 座 1：全部成交；逐件（GOOSE 2、SELL 2）
    n_ok = {op: sum(t["ok"] for t in t1 if t["op"] == op) for op in E.MARKET_OPS}
    assert n_ok == {
        "HIRE": 1,
        "BUY_LAND": 1,
        "BUY_ANIMAL": 2,
        "SELL": 2,
        "BUY_SEED": 1,
        "BUY_PRODUCT": 1,
    }
    assert [t["cash_delta"] for t in t1 if t["op"] == "BUY_ANIMAL"] == [-300.0, -300.0]
    assert [t["cash_delta"] for t in t1 if t["op"] == "BUY_LAND"] == [-1000.0]
    # 舊 `sales` 不變：只有 SELL 成交
    assert [(s["player"], s["item"]) for s in rec.sales] == [(1, "WHEAT"), (1, "WHEAT")]
    for p, start in ((0, 0.0), (1, 10_000.0)):
        m = E.market_summary(
            rec.trades, p, starting_money=start, final_money=float(obs0.farms[p]["money"])
        )
        assert m["cash_identity"]["diff"] == 0.0, m["cash_identity"]
    m1 = E.market_summary(rec.trades, 1, starting_money=10_000.0, final_money=0.0)
    assert m1["by_op"]["SELL"]["amount"] == sum(s["price"] for s in rec.sales)
    assert m1["by_op"]["BUY_ANIMAL"]["by_item"]["GOOSE"]["unit_price_mean"] == 300.0
    m0 = E.market_summary(rec.trades, 0, starting_money=0.0, final_money=0.0)
    assert m0["by_op"]["SELL"]["rejected_by_reason"] == {"shed_empty": 1}


def test_market_hooks_are_restored_even_on_exception():
    env = _fresh()
    before = (K._do_hire, K._do_buy_land, K._commit_unit)
    with pytest.raises(RuntimeError), E.read_only_hooks(env, E.Recorder()):
        raise RuntimeError("boom")
    assert (K._do_hire, K._do_buy_land, K._commit_unit) == before


def test_final_farm_summary_both_players():
    env, _rec = _run_market(hooked=True)
    ff = E.final_farm_summary(env.steps[-1])
    assert [f["player"] for f in ff] == [0, 1]
    assert ff[0]["n_units"] == 1 and ff[1]["n_units"] == 2
    assert ff[1]["unlocked_quadrants"] == ["NW", "NE"] and ff[0]["unlocked_quadrants"] == ["NW"]
    assert ff[1]["shed"].get("GOOSE") == 2 and ff[1]["shed"].get("WHEAT") == 3
    assert ff[1]["seeds"] == {"WHEAT": 1}
    assert ff[0]["tiles"]["locked"] > ff[1]["tiles"]["locked"]
    assert ff[1]["money"] == float(env.steps[-1][0].observation.farms[1]["money"])
