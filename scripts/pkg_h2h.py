"""部署包對部署包直接對戰（官方引擎；載入方式同 verify kids/game.py：get_last_callable(main.py 原文, path=main.py)）。

用法：pkg_h2h.py <X_dir> <O_dir> <seed> <X_seat> <out.json>
X＝受測包（A 或 B），O＝上個實際提交真包。兩包內嵌套件名不同（_fa_<identity>），可同程序並存；
各自初始化時核自身 model.npz SHA（fail-fast），並記錄實際載入的 model_path。
"""

import contextlib
import io
import json
import os
import sys
import time

x_dir, o_dir, seed, xseat, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
if os.path.exists(out):
    raise SystemExit(f"已存在：{out}")
if os.path.exists("/kaggle_simulations/agent"):
    raise SystemExit("/kaggle_simulations/agent 存在，會蓋過包內 model 找檔")
from kaggle_environments import make  # noqa: E402
from kaggle_environments.agent import get_last_callable  # noqa: E402


def load(d):
    p = os.path.join(d, "main.py")
    t0 = time.perf_counter()
    fn = get_last_callable(open(p, encoding="utf-8").read(), path=p)
    return fn, time.perf_counter() - t0


xr, x_exec = load(x_dir)
orr, o_exec = load(o_dir)
cnt = {"x_calls": 0, "x_exceptions": 0, "o_calls": 0, "o_exceptions": 0}
tim = {"x": [], "o": []}


def wrap(fn, key, d):
    # get_last_callable 以空 globals exec，包內 __file__ 不存在 ⇒ 權重找檔落到 cwd。
    # 兩包同程序時，每次呼叫前 chdir 到該方自己的包目錄（harness 層；包本體不改，初始化仍核自身 model SHA）。
    def agent(obs, config=None):
        cnt[f"{key}_calls"] += 1
        prev = os.getcwd()
        os.chdir(d)
        t = time.perf_counter()
        try:
            a = fn(obs, config)
        except BaseException:
            cnt[f"{key}_exceptions"] += 1
            raise
        finally:
            tim[key].append(time.perf_counter() - t)
            os.chdir(prev)
        return a
    return agent


env = make("kaggriculture", configuration={"seed": seed, "episodeSteps": 720}, debug=False)
agents = [wrap(xr, "x", x_dir), wrap(orr, "o", o_dir)] if xseat == 0 else [wrap(orr, "o", o_dir), wrap(xr, "x", x_dir)]
buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    env.run(agents)
wall = time.time() - t0
steps_status = [[s.status for s in st] for st in env.steps]
bad_mid = [(i, ss) for i, ss in enumerate(steps_status[:-1]) if any(v != "ACTIVE" for v in ss)]
last = env.steps[-1]
money = [int(s.observation.farms[i]["money"]) for i, s in enumerate(last)]


def lat(q):
    q = sorted(q[1:])
    if not q:
        return None
    return {"n": len(q), "p50": q[len(q) // 2], "p99": q[min(len(q) - 1, int(0.99 * len(q)))], "max": q[-1]}


xm, om = money[xseat], money[1 - xseat]
xmp, omp = xr.__globals__["_FA_STATE"]["model_path"], orr.__globals__["_FA_STATE"]["model_path"]
model_ok = bool(xmp and omp and os.path.dirname(os.path.abspath(xmp)) == os.path.abspath(x_dir)
                and os.path.dirname(os.path.abspath(omp)) == os.path.abspath(o_dir))
res = {"seed": seed, "x_seat": xseat, "x_dir": x_dir, "o_dir": o_dir, "entries": len(env.steps),
       "statuses": [s.status for s in last], "n_non_active_before_last": len(bad_mid), "non_active_before_last": bad_mid[:5],
       "rewards": [s.reward for s in last], "money": money, "x": xm, "o": om, "margin": xm - om,
       "result": "W" if xm > om else ("L" if xm < om else "D"), **cnt,
       "x_model_path": xr.__globals__["_FA_STATE"]["model_path"], "o_model_path": orr.__globals__["_FA_STATE"]["model_path"],
       "x_init_errors": xr.__globals__["_FA_STATE"]["init_errors"], "o_init_errors": orr.__globals__["_FA_STATE"]["init_errors"],
       "x_pkg": xr.__globals__.get("_FA_PKG"), "o_pkg": orr.__globals__.get("_FA_PKG"),
       "x_first_step_s": tim["x"][0] if tim["x"] else None, "o_first_step_s": tim["o"][0] if tim["o"] else None,
       "x_warm": lat(tim["x"]), "o_warm": lat(tim["o"]), "x_loader_exec_s": x_exec, "o_loader_exec_s": o_exec,
       "wall_s": round(wall, 2), "stdout_bytes": len(buf.getvalue()),
       "valid": len(env.steps) == 720 and [s.status for s in last] == ["DONE", "DONE"] and not bad_mid
       and cnt["x_calls"] == 719 and cnt["o_calls"] == 719 and not cnt["x_exceptions"] and not cnt["o_exceptions"] and model_ok,
       "model_paths_own_dir": model_ok}
with open(out, "x") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(f"seed{seed} xseat{xseat} {res['result']} margin={res['margin']:+d} wall={wall:.1f}s valid={res['valid']}", flush=True)
