#!/usr/bin/env python
# ruff: noqa: E501  子程序與 main.py 樣板是字串常數，長行在字串內
"""full-actions NumPy agent 提交包（方案 C）：`build` 產可重現 tar，`verify` 隔離驗證。

**本檔不提交、不碰 Kaggle API**。

方案 C
------
* tar 只有兩個 member：根目錄 `main.py` ＋ `model.npz`。
* `main.py` 內嵌部署閉包的六個模組原始碼（`rl/__init__`、`rules`、`caps`、`ordered_v3_contract`、
  `loop4_numpy`、`full_actions_numpy`），來源讀本 repo 工作樹的 `rl/<m>.py`。唯一改動是 `from rl import X[ as Y]` → `from <pkg>.rl import X[ as Y]`，
  `<pkg>`＝`_fa_<identity12>`，由 loader 以 `types.ModuleType` 掛進 `sys.modules`
  ⇒ 不佔用 `rl` 這個名字，避開同程序載入兩個包時的模組快取陷阱。
* `model.npz`＝checkpoint 的 actor（p/*＋完整 meta，無 o/*），以內嵌的 `FN.load_params`／`FN.save_params`
  抽出並逐葉核相等；載入走 `FN.make_agent(path, seat, caps=EXACT_REFERENCE, rng=None, expect_sha256=…)`
  （`config_from_meta`，不繞守衛）。找檔順序：`/kaggle_simulations/agent` → `dirname(__file__)` → cwd。
* 入口 `agent(obs)`：單執行緒 BLAS 環境變數在 import numpy 前**明確設為 1**（宿主若已先載 numpy 則不生效，
  入口記錄 `numpy_preloaded_at_exec`）；`obs["player"]` 定 seat、按 seat 快取 agent；**fail-fast**：初始化（找檔／SHA／schema／模組身分）失敗寫 stderr 後向外拋出，決策期例外也直接拋出，
  沒有任何 PASS 降級、不寫 stdout。
* `verify`：mandatory checks（重建相等、actor 逐葉、tar 成員與屬性、內嵌原始碼、import 閉包、1,438 obs
  parity、BLAS 宿主設 4、四個負例 fail-fast、子程序結果符合預期）全過才 rc 0。
  有給 `--opponent` 時另跑實際 tar 對該對手雙座 720 步（無中途非 ACTIVE；有給參照局目錄時與參照逐元相同）
  與延遲門；沒給就略過這兩項並記在 `not_applicable`。
* `--source-label`：寫進 main.py 文件字串與 identity 的來源標籤（預設＝本 repo `git rev-parse HEAD`，
  沒有 git 時為 `worktree`）。線上提交包用的標籤是 `8416e0503ead51bd81810e8fc2ce15784582b773`；
  六模組相同、權重相同、標籤相同 ⇒ main.py 與 tar 逐 byte 相同。
* tar：member 依名稱排序、mode 0644、mtime 0、uid／gid 0、uname／gname 空；gzip mtime 0
  ⇒ 同一 checkpoint 重建 sha256 相同（verify 會重建核對）。

用法::

    python scripts/build_full_actions_submission.py build \\
        --checkpoint <ckpt.npz> --expected-sha <sha256> [--source-label <label>] [--out-dir <dir>]
    python scripts/build_full_actions_submission.py verify \\
        --checkpoint <ckpt.npz> --expected-sha <sha256> --obs <obs_replay.json> ... --parity-n <N> \\
        --wrong-model <另一個 actor.npz> [--opponent NAME,PATH/main.py,SEED[,REF_DIR]] ...
"""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
#: manifest 與 verify 結果的根目錄（`--out-dir` 可改）
OUT_DIR = ROOT / "dist" / "submit_candidate"
#: 來源標籤（`--source-label`；main() 開頭設定）
SRC_COMMIT = "worktree"
#: 內嵌模組（依賴序；`__init__` 先）
MODULES = ("__init__", "rules", "caps", "ordered_v3_contract", "loop4_numpy", "full_actions_numpy")
TEMPLATE_VERSION = "fa-submission-c-v2"
MAX_TAR_BYTES = 100 * 1024 * 1024
IMPORT_RE = re.compile(r"^(?P<ind>[ \t]*)from rl import (?P<rest>\w+(?: as \w+)?)[ \t]*$")
BANNED_MODULE_PREFIXES = (
    "jax",
    "jaxlib",
    "kaggle_environments",
    "rl.full_actions_head",
    "rl.full_actions_obs",
)


class HardStop(SystemExit):
    def __init__(self, message: str):
        super().__init__(2)
        self.message = message


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ 來源與改寫
def repo_source(name: str) -> bytes:
    """本 repo 工作樹的 `rl/<name>.py` 原始 bytes。"""
    return (ROOT / "rl" / f"{name}.py").read_bytes()


def default_source_label() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "worktree"


def rewrite(src: bytes, pkg: str) -> tuple[bytes, list[int]]:
    """只改 `from rl import X[ as Y]` 整行；回傳改寫後 bytes 與被改的行號（1-based）。"""
    out, changed = [], []
    for i, line in enumerate(src.decode("utf-8").split("\n"), 1):
        m = IMPORT_RE.match(line)
        if m:
            out.append(f"{m['ind']}from {pkg}.rl import {m['rest']}")
            changed.append(i)
        else:
            out.append(line)
    new = "\n".join(out).encode("utf-8")
    tree = ast.parse(new)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "rl" or node.module.startswith("rl."))
        ):
            raise HardStop(
                f"改寫後仍有 from {node.module} import（第 {node.lineno} 行）：不在允許的單行格式內"
            )
        if isinstance(node, ast.Import) and any(
            a.name == "rl" or a.name.startswith("rl.") for a in node.names
        ):
            raise HardStop(f"改寫後仍有 import rl（第 {node.lineno} 行）")
        mod = node.module if isinstance(node, ast.ImportFrom) else None
        names = (
            [a.name for a in node.names] if isinstance(node, ast.Import) else ([mod] if mod else [])
        )
        for n in names:
            if any(
                n == b or n.startswith(b + ".")
                for b in ("full_actions_head", "full_actions_obs", "jax")
            ):
                raise HardStop(f"內嵌模組 import 了禁止模組 {n}")
    return new, changed


def unrewrite_equal(original: bytes, rewritten: bytes, pkg: str) -> dict[str, Any]:
    """逐行比：除允許的 import 行外逐 byte 相同；import 行須是精確的對應改寫。"""
    a, b = original.decode("utf-8").split("\n"), rewritten.decode("utf-8").split("\n")
    if len(a) != len(b):
        return {"ok": False, "reason": f"行數不同 {len(a)}≠{len(b)}"}
    diff = []
    for i, (x, y) in enumerate(zip(a, b, strict=True), 1):
        if x == y:
            continue
        m = IMPORT_RE.match(x)
        if not m or y != f"{m['ind']}from {pkg}.rl import {m['rest']}":
            return {"ok": False, "reason": f"第 {i} 行非允許改寫", "orig": x, "new": y}
        diff.append(i)
    return {"ok": True, "import_lines_changed": diff}


# ------------------------------------------------------------------ main.py 樣板
MAIN_TEMPLATE = '''"""Kaggriculture full-actions NumPy agent 提交入口（scripts/build_full_actions_submission.py 產生，勿手改）。

identity {identity}；model.npz sha256 {model_sha}；checkpoint sha256 {ckpt_sha}；
source commit {src_commit}（六模組與 {alt_commit} 逐 byte 相同）；template {template}。
內嵌模組以 `{pkg}.rl.*` 掛進 sys.modules（不佔用 `rl`）；權重找檔順序
/kaggle_simulations/agent → dirname(__file__) → cwd。**fail-fast**：初始化（找檔／SHA／schema／模組身分）
失敗寫 stderr 後向外拋出；決策期例外也直接拋出；沒有任何 PASS 降級。
BLAS 執行緒環境變數在本檔 import 任何 numpy 前明確設為 1；宿主若已先載入 numpy，
其 BLAS threadpool 已建立，這幾個變數不再生效（`_FA_STATE["numpy_preloaded_at_exec"]` 記錄）。
"""

import os as _fa_os

for _fa_k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _fa_os.environ[_fa_k] = "1"

import base64 as _fa_b64  # noqa: E402
import sys as _fa_sys  # noqa: E402
import traceback as _fa_tb  # noqa: E402
import types as _fa_types  # noqa: E402

_FA_PKG = "{pkg}"
_FA_IDENTITY = "{identity}"
_FA_MODEL_NAME = "model.npz"
_FA_MODEL_SHA = "{model_sha}"
_FA_SOURCES = (
{sources}
)
_FA_STATE = {{"agents": {{}}, "model_path": None, "init_errors": 0, "numpy_preloaded_at_exec": "numpy" in _fa_sys.modules}}


def _fa_load_package():
    root = _fa_sys.modules.get(_FA_PKG)
    if root is not None:
        if getattr(root, "_fa_identity", None) != _FA_IDENTITY:
            raise RuntimeError("sys.modules[%r] 身分不符本包，拒絕重用" % _FA_PKG)
        return _fa_sys.modules[_FA_PKG + ".rl"]
    names = []
    try:
        root = _fa_types.ModuleType(_FA_PKG)
        root.__path__ = []
        root._fa_identity = _FA_IDENTITY
        _fa_sys.modules[_FA_PKG] = root
        names.append(_FA_PKG)
        pkg = _fa_types.ModuleType(_FA_PKG + ".rl")
        pkg.__path__ = []
        pkg.__package__ = _FA_PKG + ".rl"
        _fa_sys.modules[_FA_PKG + ".rl"] = pkg
        names.append(_FA_PKG + ".rl")
        root.rl = pkg
        for name, b64 in _FA_SOURCES:
            src = _fa_b64.b64decode(b64).decode("utf-8")
            if name == "__init__":
                exec(compile(src, "<%s/rl/__init__.py>" % _FA_PKG, "exec"), pkg.__dict__)
                continue
            full = _FA_PKG + ".rl." + name
            mod = _fa_types.ModuleType(full)
            mod.__package__ = _FA_PKG + ".rl"
            _fa_sys.modules[full] = mod
            names.append(full)
            exec(compile(src, "<%s/rl/%s.py>" % (_FA_PKG, name), "exec"), mod.__dict__)
            setattr(pkg, name, mod)
        return pkg
    except BaseException:
        for n in names:
            _fa_sys.modules.pop(n, None)
        raise


def _fa_model_path():
    here = _fa_os.path.dirname(_fa_os.path.abspath(globals().get("__file__", "main.py")))
    tried = []
    for d in ("/kaggle_simulations/agent", here, _fa_os.getcwd()):
        p = _fa_os.path.join(d, _FA_MODEL_NAME)
        tried.append(p)
        if _fa_os.path.isfile(p):
            return p
    raise FileNotFoundError("找不到 %s：%r" % (_FA_MODEL_NAME, tried))


def _fa_agent_for(seat):
    a = _FA_STATE["agents"].get(seat)
    if a is None:
        pkg = _fa_load_package()
        path = _fa_model_path()
        a = pkg.full_actions_numpy.make_agent(
            path, seat=seat, caps=pkg.caps.EXACT_REFERENCE, rng=None, expect_sha256=_FA_MODEL_SHA
        )
        _FA_STATE["model_path"] = path
        _FA_STATE["agents"][seat] = a
    return a


def agent(obs, config=None):
    seat = int(obs["player"])
    try:
        a = _fa_agent_for(seat)
    except Exception:
        _FA_STATE["init_errors"] += 1
        _fa_sys.stderr.write("[fa-agent] 初始化失敗（seat %d），向外拋出：\\n%s" % (seat, _fa_tb.format_exc()))
        _fa_sys.stderr.flush()
        raise
    return a(obs)
'''


def render_main(
    pkg: str, identity: str, model_sha: str, ckpt_sha: str, rewritten: dict[str, bytes]
) -> str:
    lines = []
    for name in MODULES:
        lines.append(f'    ("{name}", "{base64.b64encode(rewritten[name]).decode("ascii")}"),')
    return MAIN_TEMPLATE.format(
        pkg=pkg,
        identity=identity,
        model_sha=model_sha,
        ckpt_sha=ckpt_sha,
        src_commit=SRC_COMMIT,
        alt_commit=SRC_COMMIT,
        template=TEMPLATE_VERSION,
        sources="\n".join(lines),
    )


def exec_main(text: str, path: str) -> dict:
    ns: dict = {}
    exec(compile(text, path, "exec"), ns)
    return ns


# ------------------------------------------------------------------ tar
def write_tar(out: Path, members: dict[str, bytes]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name in sorted(members):
            data = members[name]
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode, ti.mtime = len(data), 0o644, 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            tf.addfile(ti, io.BytesIO(data))
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(buf.getvalue())
    out.write_bytes(raw.getvalue())
    return sha256_file(out)


# ------------------------------------------------------------------ build
def build(checkpoint: Path, expected_sha: str, dist: Path, out_dir: Path | None) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    ck_sha = sha256_file(checkpoint)
    if ck_sha != expected_sha:
        raise HardStop(f"checkpoint SHA 不符：expect {expected_sha}、實得 {ck_sha}")
    orig = {m: repo_source(m) for m in MODULES}
    # 1) 以暫名載入內嵌碼，抽 actor（與正式包同一份改寫碼、同一個 loader）
    boot_pkg = "_fa_bootstrap_" + ck_sha[:8]
    boot_rw = {m: rewrite(orig[m], boot_pkg)[0] for m in MODULES}
    ns = exec_main(render_main(boot_pkg, "bootstrap", "0" * 64, ck_sha, boot_rw), "<fa-bootstrap>")
    FN = ns["_fa_load_package"]().full_actions_numpy
    params, meta = FN.load_params(checkpoint)
    src_npz = np.load(checkpoint, allow_pickle=False)
    n_o = sum(k.startswith("o/") for k in src_npz.files)
    workdir = dist / f"fa_{ck_sha[:8]}"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    actor = workdir / "model.npz"
    FN.save_params(actor, params, {k: v for k, v in meta.items() if k != "params"})
    a_params, a_meta = FN.load_params(actor)
    az = np.load(actor, allow_pickle=False)
    leaf = {
        "n_leaves_source": sum(k.startswith("p/") for k in src_npz.files),
        "n_leaves_actor": sum(k.startswith("p/") for k in az.files),
        "all_equal": all(
            np.array_equal(src_npz["p/" + k], az["p/" + k])
            and src_npz["p/" + k].dtype == az["p/" + k].dtype
            and src_npz["p/" + k].shape == az["p/" + k].shape
            for k in params
        )
        and set(k for k in src_npz.files if k.startswith("p/"))
        == set(k for k in az.files if k.startswith("p/")),
        "actor_has_o": any(k.startswith("o/") for k in az.files),
        "meta_equal": json.loads(str(src_npz["meta"])) == json.loads(str(az["meta"])),
        "dtypes": sorted({str(az[k].dtype) for k in az.files if k.startswith("p/")}),
    }
    if not (
        leaf["all_equal"]
        and leaf["meta_equal"]
        and not leaf["actor_has_o"]
        and leaf["n_leaves_source"] == leaf["n_leaves_actor"]
    ):
        raise HardStop(f"actor 抽出核對失敗：{leaf}")
    model_sha = sha256_file(actor)
    # 2) identity／正式包名
    ident_src = json.dumps(
        {
            "template": TEMPLATE_VERSION,
            "src_commit": SRC_COMMIT,
            "modules": [[m, sha256_bytes(orig[m])] for m in MODULES],
            "model_sha": model_sha,
        }
    )
    identity = sha256_bytes(ident_src.encode())
    pkg = "_fa_" + identity[:12]
    rw, changed, eq = {}, {}, {}
    for m in MODULES:
        rw[m], changed[m] = rewrite(orig[m], pkg)
        eq[m] = unrewrite_equal(orig[m], rw[m], pkg)
        if not eq[m]["ok"]:
            raise HardStop(f"{m} 改寫核對失敗：{eq[m]}")
    main_text = render_main(pkg, identity, model_sha, ck_sha, rw)
    ast.parse(main_text)
    main_path = workdir / "main.py"
    main_path.write_text(main_text, encoding="utf-8")
    tar_path = dist / f"fa_{ck_sha[:8]}.tar.gz"
    tar_sha = write_tar(
        tar_path, {"main.py": main_path.read_bytes(), "model.npz": actor.read_bytes()}
    )
    if tar_path.stat().st_size >= MAX_TAR_BYTES:
        raise HardStop(f"tar {tar_path.stat().st_size} bytes ≥ 100 MiB")
    manifest = {
        "schema": "fa-submission-manifest-v1",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
            "repo_head": default_source_label(),
        },
        "source_commit": SRC_COMMIT,
        "modules": {
            m: {
                "source_sha256": sha256_bytes(orig[m]),
                "embedded_sha256": sha256_bytes(rw[m]),
                "import_lines_changed": eq[m]["import_lines_changed"],
            }
            for m in MODULES
        },
        "template_version": TEMPLATE_VERSION,
        "identity": identity,
        "package_name": pkg,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": ck_sha,
            "bytes": checkpoint.stat().st_size,
            "n_optimizer_entries": n_o,
        },
        "actor": {
            "sha256": model_sha,
            "bytes": actor.stat().st_size,
            "leaf_check": leaf,
            "param_count": int(sum(int(np.prod(v.shape)) for v in a_params.values())),
            "meta_trunk": a_meta["trunk"],
            "config_from_meta": {
                k: FN.config_from_meta(a_meta)[k]
                for k in ("d_model", "n_layers", "n_heads", "d_ff", "n_tokens")
            },
        },
        "main_py": {"sha256": sha256_file(main_path), "bytes": main_path.stat().st_size},
        "tar": {
            "path": str(tar_path),
            "sha256": tar_sha,
            "bytes": tar_path.stat().st_size,
            "members": ["main.py", "model.npz"],
            "pinned": "PAX; mode 0644; mtime 0; uid/gid 0; uname/gname 空; gzip mtime 0 level 9; 成員依名稱排序",
        },
        "entry": {
            "seat": 'obs["player"]',
            "cache": "per seat",
            "caps": "EXACT_REFERENCE",
            "rng": None,
            "model_search": ["/kaggle_simulations/agent", "dirname(__file__)", "cwd"],
            "on_exception": "fail-fast：初始化失敗寫 stderr 後拋出；決策期例外直接拋出；無 PASS 降級",
        },
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


# ------------------------------------------------------------------ verify（子程序腳本）
CG_SNIPPET = r"""
def _cg():
    import os
    from pathlib import Path
    out = {}
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[2]
        base = Path("/sys/fs/cgroup" + rel)
        out["cgroup"] = rel
        for f in ("cpu.max", "cpu.stat", "memory.peak", "memory.events", "memory.max", "memory.swap.max"):
            try:
                out[f] = (base / f).read_text()
            except OSError as e:
                out[f] = "ERR %s" % e
    except Exception as e:
        out["cgroup_err"] = repr(e)
    out["affinity"] = sorted(os.sched_getaffinity(0))
    st = Path("/proc/self/status").read_text()
    out["threads"] = int(next(l.split()[1] for l in st.splitlines() if l.startswith("Threads:")))
    out["vmhwm_kib"] = int(next(l.split()[1] for l in st.splitlines() if l.startswith("VmHWM:")))
    out["env_threads"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES")}
    return out


def _blas_threads():
    import ctypes
    path = None
    for line in open("/proc/self/maps"):
        f = line.split()
        if f and "openblas" in f[-1] and f[-1].endswith(".so"):
            path = f[-1]
            break
    if path is None:
        return {"lib": None, "threads": None}
    lib = ctypes.CDLL(path)
    for sym in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_", "openblas_get_num_threads"):
        if hasattr(lib, sym):
            return {"lib": path, "symbol": sym, "threads": int(getattr(lib, sym)())}
    return {"lib": path, "threads": None}
"""

CLOSURE_CHILD = (
    CG_SNIPPET
    + r"""
import json, sys, types, time, io, contextlib, importlib.util
agent_dir, obs_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
rl_spec_before = importlib.util.find_spec("rl")
decoy = types.ModuleType("rl"); sys.modules["rl"] = decoy
before = set(sys.modules)
buf = io.StringIO()
obs = json.load(open(obs_file))["steps"]
src = open(agent_dir + "/main.py", encoding="utf-8").read()
ns = {}
t0 = time.perf_counter()
with contextlib.redirect_stdout(buf):
    exec(compile(src, agent_dir + "/main.py", "exec"), ns)
    last = [v for v in ns.values() if callable(v)][-1]
    a0 = ns["agent"](obs[0]["obs"])
t1 = time.perf_counter()
mods = sorted(set(sys.modules) - before)
res = {"last_callable_is_agent": last is ns["agent"], "stdout_bytes": len(buf.getvalue()),
       "cold_exec_import_load_first_call_s": t1 - t0, "init_errors": ns["_FA_STATE"]["init_errors"],
       "numpy_preloaded_at_exec": ns["_FA_STATE"]["numpy_preloaded_at_exec"], "model_path": ns["_FA_STATE"]["model_path"],
       "action_keys": sorted(a0), "new_modules": mods,
       "banned_loaded": sorted(m for m in sys.modules if any(m == b or m.startswith(b + ".") for b in ("jax", "jaxlib", "kaggle_environments", "rl.full_actions_head", "rl.full_actions_obs", "full_actions_head", "full_actions_obs"))),
       "decoy_rl_untouched": sys.modules.get("rl") is decoy and not hasattr(decoy, "full_actions_numpy"),
       "package_modules": sorted(m for m in sys.modules if m.startswith(ns["_FA_PKG"])),
       "repo_rl_importable_before": rl_spec_before is not None, "sys_path": sys.path, "cwd": __import__("os").getcwd(),
       "blas": _blas_threads(), "cg": _cg()}
open(out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
"""
)

BLAS_CHILD = (
    CG_SNIPPET
    + r"""
import json, os, sys
agent_dir, obs_file, mode, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
env_before = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
pre = None
if mode == "preload_numpy":
    import numpy as np
    np.ones((256, 256)) @ np.ones((256, 256))
    pre = _blas_threads()
ns = {}
exec(compile(open(agent_dir + "/main.py", encoding="utf-8").read(), agent_dir + "/main.py", "exec"), ns)
env_after_exec = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
ns["agent"](json.load(open(obs_file))["steps"][0]["obs"])
res = {"mode": mode, "env_before": env_before, "env_after_exec": env_after_exec, "blas_before_exec": pre,
       "blas_after_first_call": _blas_threads(), "numpy_preloaded_at_exec": ns["_FA_STATE"]["numpy_preloaded_at_exec"], "cg": _cg()}
open(out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
"""
)

NEG_CHILD = r"""
import json, sys
main_path, obs_file = sys.argv[1], sys.argv[2]
ns = {"__file__": main_path} if len(sys.argv) > 3 and sys.argv[3] == "--with-file" else {}
exec(compile(open(main_path, encoding="utf-8").read(), main_path, "exec"), ns)
r = ns["agent"](json.load(open(obs_file))["steps"][0]["obs"])
print("RETURNED:" + json.dumps(r), flush=True)
"""

PARITY_PKG_CHILD = (
    CG_SNIPPET
    + r"""
import json, sys
agent_dir, out = sys.argv[1], sys.argv[2]
files = sys.argv[3:]
ns = {}
exec(compile(open(agent_dir + "/main.py", encoding="utf-8").read(), agent_dir + "/main.py", "exec"), ns)
res = []
for f in files:
    rec = json.load(open(f))
    for s in rec["steps"]:
        o = s["obs"]
        res.append({"file": f, "i": s["i"], "seat": int(o["player"]), "action": ns["agent"](o)})
open(out, "w").write(json.dumps({"init_errors": ns["_FA_STATE"]["init_errors"], "rows": res, "cg": _cg()}, ensure_ascii=False))
"""
)

PARITY_REPO_CHILD = (
    CG_SNIPPET
    + r"""
import json, sys
repo, ck, sha, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
files = sys.argv[5:]
sys.path.insert(0, repo)
from rl import caps as caps_mod
from rl import full_actions_numpy as FN
agents = {}
res = []
for f in files:
    rec = json.load(open(f))
    for s in rec["steps"]:
        o = s["obs"]
        seat = int(o["player"])
        if seat not in agents:
            agents[seat] = FN.make_agent(ck, seat=seat, caps=caps_mod.EXACT_REFERENCE, rng=None, expect_sha256=sha)
        res.append({"file": f, "i": s["i"], "seat": seat, "action": agents[seat](o)})
open(out, "w").write(json.dumps({"rl_file": FN.__file__, "rows": res, "cg": _cg()}, ensure_ascii=False))
"""
)

GAME_CHILD = (
    CG_SNIPPET
    + r"""
import json, sys, time, io, contextlib
agent_dir, opp_path, seed, seat, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable
main_path = agent_dir + "/main.py"
t0 = time.perf_counter()
ours_raw = get_last_callable(open(main_path, encoding="utf-8").read(), path=main_path)
t_exec = time.perf_counter() - t0
g = ours_raw.__globals__
timing, actions = [], []
cnt = {"ours_calls": 0, "ours_exceptions": 0, "opp_calls": 0, "opp_exceptions": 0}
def ours(obs, config=None):
    cnt["ours_calls"] += 1
    t = time.perf_counter()
    try:
        a = ours_raw(obs, config)
    except BaseException:
        cnt["ours_exceptions"] += 1
        raise
    timing.append(time.perf_counter() - t)
    actions.append(json.loads(json.dumps(a)))
    return a
opp_raw = get_last_callable(open(opp_path, encoding="utf-8").read(), path=opp_path)
opp_nargs = opp_raw.__code__.co_argcount if hasattr(opp_raw, "__code__") else None
def opp(obs, config=None):
    cnt["opp_calls"] += 1
    try:
        return opp_raw(obs, config) if opp_nargs != 1 else opp_raw(obs)
    except BaseException:
        cnt["opp_exceptions"] += 1
        raise
env = make("kaggriculture", configuration={"seed": seed, "episodeSteps": 720}, debug=False)
agents = [ours, opp] if seat == 0 else [opp, ours]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    env.run(agents)
steps_status = [[s.status for s in st] for st in env.steps]
bad_mid = [(i, ss) for i, ss in enumerate(steps_status[:-1]) if any(x != "ACTIVE" for x in ss)]
last = env.steps[-1]
money = [int(s.observation.farms[i]["money"]) for i, s in enumerate(last)]
q = sorted(timing[1:])
pct = lambda p: q[min(len(q) - 1, int(p * len(q)))]
res = {"seed": seed, "seat": seat, "opponent": opp_path, "opp_nargs": opp_nargs, "entries": len(env.steps), "statuses": [s.status for s in last],
       "non_active_before_last": bad_mid[:5], "n_non_active_before_last": len(bad_mid),
       "status_counts": {k: sum(x == k for ss in steps_status for x in ss) for k in ("ACTIVE", "DONE", "ERROR", "INVALID", "TIMEOUT", "INACTIVE")},
       "rewards": [s.reward for s in last], "money": money, "ours": money[seat], "opp": money[1 - seat], **cnt,
       "init_errors": g["_FA_STATE"]["init_errors"], "loader_exec_s": t_exec, "first_step_s": timing[0] if timing else None,
       "warm": {"n": len(q), "p50": pct(0.5), "p99": pct(0.99), "max": q[-1]} if q else None, "raw_step_s": timing,
       "model_path": g["_FA_STATE"]["model_path"], "stdout_bytes": len(buf.getvalue()), "actions": actions, "cg": _cg()}
open(out, "w").write(json.dumps(res, ensure_ascii=False))
"""
)


def scoped(
    argv: list[str],
    *,
    name: str,
    cwd: Path,
    run_dir: Path,
    wall: int,
    env: dict,
    expect: str = "ok",
) -> dict[str, Any]:
    unit = f"fa-verify-{name}-{os.getpid()}"
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        f"--unit={unit}",
        "-p",
        "CPUQuota=160%",
        "-p",
        "MemoryMax=6656M",
        "-p",
        "MemorySwapMax=0",
        "--",
        "taskset",
        "-c",
        "",
        "/usr/bin/time",
        "-v",
        "-o",
        str(run_dir / f"{name}.time"),
        "timeout",
        "--kill-after=10",
        str(wall),
        *argv,
    ]
    return {"name": name, "unit": unit, "cmd": cmd, "cwd": str(cwd), "env": env, "expect": expect}


def cores_busy(cores: list[int], threshold: float = 30.0) -> dict[int, float]:
    def snap():
        d = {}
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("cpu") and line[3:4].isdigit():
                f = line.split()
                vals = list(map(int, f[1:]))
                d[int(f[0][3:])] = (sum(vals), vals[3] + vals[4])
        return d

    a = snap()
    time.sleep(2)
    b = snap()
    use = {c: 100.0 * (1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0])) for c in cores}
    return {c: u for c, u in use.items() if u > threshold}


def run_specs(
    specs: list[dict], slots: list[str], run_dir: Path, wall: int
) -> tuple[list[dict], str | None]:
    """三槽並行；expect=ok 的子程序 rc≠0 ⇒ 停整批（清其餘 scope）；expect=fail 的 rc≠0 屬預期。"""
    running: dict[str, tuple[subprocess.Popen, dict, Any]] = {}
    free, queue, done, stopped = list(slots), list(specs), [], None
    while queue or running:
        while queue and free and stopped is None:
            spec = queue.pop(0)
            core = free.pop(0)
            spec["cmd"][spec["cmd"].index("taskset") + 2] = core
            spec["cores"] = core
            lf = open(run_dir / f"{spec['name']}.log", "w")
            spec["start"], spec["t0"] = time.strftime("%F %T %Z"), time.time()
            p = subprocess.Popen(
                spec["cmd"],
                cwd=spec["cwd"],
                env=spec["env"],
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[spec["name"]] = (p, spec, lf)
        time.sleep(1)
        for name in list(running):
            p, spec, lf = running[name]
            rc = p.poll()
            if rc is None:
                if time.time() - spec["t0"] <= wall + 60:
                    continue
                subprocess.run(["systemctl", "--user", "stop", spec["unit"] + ".scope"])
                rc = p.wait()
                spec["outer_stopped"] = True
            lf.close()
            spec.update(
                rc=rc, end=time.strftime("%F %T %Z"), wall_s=round(time.time() - spec["t0"], 2)
            )
            try:
                tv = (run_dir / f"{name}.time").read_text()
                spec["time_v_max_rss_kib"] = int(
                    next(
                        line.split(":")[1]
                        for line in tv.splitlines()
                        if "Maximum resident set size" in line
                    )
                )
            except Exception as e:  # noqa: BLE001
                spec["time_v_err"] = repr(e)
            spec["as_expected"] = (
                (rc == 0)
                if spec["expect"] == "ok"
                else (rc not in (0, 124, 137) and not spec.get("outer_stopped"))
            )
            done.append({k: spec[k] for k in spec if k not in ("env", "t0")})
            free.append(spec["cores"])
            del running[name]
            if spec["expect"] == "ok" and rc != 0 and stopped is None:
                stopped = name
                for p2, s2, lf2 in running.values():
                    subprocess.run(["systemctl", "--user", "stop", s2["unit"] + ".scope"])
                    p2.wait()
                    lf2.close()
                    done.append(
                        {
                            **{k: s2[k] for k in s2 if k not in ("env", "t0")},
                            "rc": "stopped_after_failure",
                            "as_expected": False,
                        }
                    )
                running.clear()
                queue.clear()
    return done, stopped


def _cmd_of(action: dict, unit: int) -> list:
    if unit == 0:
        return list(action.get("farmer", ["PASS"]))
    hands = action.get("hands") or []
    return list(hands[unit - 1]) if unit - 1 < len(hands) else ["PASS"]


def _cgstat(cg: dict) -> dict:
    m = dict(line.split() for line in cg["cpu.stat"].strip().splitlines())
    return {
        "cpu.max": cg["cpu.max"].strip(),
        "nr_periods": int(m["nr_periods"]),
        "nr_throttled": int(m["nr_throttled"]),
        "memory.peak": int(cg["memory.peak"]),
        "memory.max": cg["memory.max"].strip(),
        "memory.swap.max": cg["memory.swap.max"].strip(),
        "memory.events": cg["memory.events"].strip().replace("\n", "; "),
        "affinity": cg["affinity"],
        "threads": cg["threads"],
        "vmhwm_kib": cg["vmhwm_kib"],
    }


NEG_CASES = {
    "neg_missing_model": ("FileNotFoundError", "找不到 model.npz"),
    "neg_wrong_sha": ("ValueError", "checkpoint SHA256 不符"),
    "neg_schema": ("ValueError", "schema 是"),
    "neg_priority_dir_wrong_sha": ("ValueError", "checkpoint SHA256 不符"),
}


def verify(args) -> int:
    import numpy as np  # noqa: PLC0415

    ck = Path(args.checkpoint)
    ck8 = args.expected_sha[:8]
    out_dir = OUT_DIR / f"fa_{ck8}"
    man = json.loads((out_dir / "manifest.json").read_text())
    tar_path = Path(man["tar"]["path"])
    vdir = out_dir / f"verify_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = vdir / "runs"
    run_dir.mkdir(parents=True)
    res: dict[str, Any] = {
        "manifest_tar_sha256": man["tar"]["sha256"],
        "template_version": man["template_version"],
        "started": time.strftime("%F %T %Z"),
    }
    # (a) 可重現
    with tempfile.TemporaryDirectory(prefix="fa_rebuild_") as td:
        m2 = build(ck, args.expected_sha, Path(td), None)
    res["a_rebuild"] = {
        "tar_sha256": m2["tar"]["sha256"],
        "model_sha256": m2["actor"]["sha256"],
        "tar_equal": m2["tar"]["sha256"] == man["tar"]["sha256"],
        "model_equal": m2["actor"]["sha256"] == man["actor"]["sha256"],
        "leaf_check": m2["actor"]["leaf_check"],
    }
    # (b) 解包到 repo 外乾淨目錄（正例與四個負例各一份）
    base = Path(tempfile.mkdtemp(prefix=f"fa_verify_{ck8}_"))

    def extract(dst: Path) -> None:
        dst.mkdir(parents=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(dst, filter="data")

    agent_dir = base / "agent"
    extract(agent_dir)
    with tarfile.open(tar_path) as tf:
        infos = tf.getmembers()
    res["b_tar"] = {
        "path": str(tar_path),
        "sha256": sha256_file(tar_path),
        "bytes": tar_path.stat().st_size,
        "members": [
            {
                "name": i.name,
                "size": i.size,
                "mode": oct(i.mode),
                "mtime": i.mtime,
                "uid": i.uid,
                "gid": i.gid,
                "uname": i.uname,
                "gname": i.gname,
                "isfile": i.isfile(),
            }
            for i in infos
        ],
        "extract_dir": str(agent_dir),
        "model_sha256_extracted": sha256_file(agent_dir / "model.npz"),
    }
    neg = {}
    d = base / "neg_missing_model"
    extract(d)
    (d / "model.npz").unlink()
    neg["neg_missing_model"] = {
        "dir": d,
        "main": d / "main.py",
        "cwd": d,
        "with_file": False,
        "setup": "實際 tar 解包後刪除 model.npz；cwd＝解包目錄",
    }
    d = base / "neg_wrong_sha"
    extract(d)
    wrong = Path(args.wrong_model)
    shutil.copyfile(wrong, d / "model.npz")
    neg["neg_wrong_sha"] = {
        "dir": d,
        "main": d / "main.py",
        "cwd": d,
        "with_file": False,
        "setup": f"model.npz 換成另一個合法 actor（{wrong}，sha {sha256_file(wrong)}）",
        "wrong_sha": sha256_file(wrong),
    }
    d = base / "neg_schema"
    extract(d)
    z = np.load(d / "model.npz", allow_pickle=False)
    arrs = {k: z[k] for k in z.files}
    meta = json.loads(str(arrs["meta"]))
    meta["schema"] = "fa-negative-test-schema"
    arrs["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    np.savez(d / "model.npz", **arrs)
    bad_sha = sha256_file(d / "model.npz")
    mt = (d / "main.py").read_text(encoding="utf-8")
    old = f'_FA_MODEL_SHA = "{man["actor"]["sha256"]}"'
    if mt.count(old) != 1:
        raise HardStop("neg_schema：main.py 內 _FA_MODEL_SHA 行不唯一")
    (d / "main.py").write_text(mt.replace(old, f'_FA_MODEL_SHA = "{bad_sha}"'), encoding="utf-8")
    neg["neg_schema"] = {
        "dir": d,
        "main": d / "main.py",
        "cwd": d,
        "with_file": False,
        "bad_model_sha": bad_sha,
        "setup": "model.npz 的 meta.schema 改成 fa-negative-test-schema，main.py 的 _FA_MODEL_SHA 改成此檔 SHA（僅測試用），使 SHA 關通過、由 load_params 的 schema 核對拒收",
    }
    d = base / "neg_priority"
    extract(d / "A")
    shutil.copyfile(wrong, d / "A" / "model.npz")
    extract(d / "B")
    neg["neg_priority_dir_wrong_sha"] = {
        "dir": d,
        "main": d / "A" / "main.py",
        "cwd": d / "B",
        "with_file": True,
        "setup": "__file__＝A/main.py（A 放錯 SHA 權重，優先）、cwd＝B（放正確權重，後序）；/kaggle_simulations/agent 本機不存在",
    }
    # (c) 內嵌原始碼 vs git 物件
    tree = ast.parse((agent_dir / "main.py").read_text(encoding="utf-8"))
    srcs = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_FA_SOURCES" for t in n.targets)
    )
    pkg = man["package_name"]
    cmp = {}
    for name, b64 in srcs:
        emb = base64.b64decode(b64)
        cmp[name] = {
            **unrewrite_equal(repo_source(name), emb, pkg),
            "embedded_sha256": sha256_bytes(emb),
        }
    res["c_source_bytes"] = {"modules_in_order": [n for n, _ in srcs], "per_module": cmp}
    # 子程序
    cenv = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    cenv.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    env4 = {**cenv, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    py = sys.executable
    kids = base / "kids"
    kids.mkdir()
    for fname, body in (
        ("closure.py", CLOSURE_CHILD),
        ("blas.py", BLAS_CHILD),
        ("neg.py", NEG_CHILD),
        ("parity_pkg.py", PARITY_PKG_CHILD),
        ("parity_repo.py", PARITY_REPO_CHILD),
        ("game.py", GAME_CHILD),
    ):
        (kids / fname).write_text(body, encoding="utf-8")
    obs = [str(Path(p).resolve()) for p in args.obs]
    w = args.wall
    specs = [
        scoped(
            [
                py,
                "-I",
                str(kids / "closure.py"),
                str(agent_dir),
                obs[0],
                str(run_dir / "closure.json"),
            ],
            name="closure",
            cwd=agent_dir,
            run_dir=run_dir,
            wall=w,
            env=cenv,
        ),
        scoped(
            [
                py,
                "-I",
                str(kids / "parity_pkg.py"),
                str(agent_dir),
                str(run_dir / "parity_pkg.json"),
                *obs,
            ],
            name="parity_pkg",
            cwd=agent_dir,
            run_dir=run_dir,
            wall=w,
            env=cenv,
        ),
        scoped(
            [
                py,
                "-I",
                str(kids / "parity_repo.py"),
                args.repo_runtime,
                str(ck.resolve()),
                args.expected_sha,
                str(run_dir / "parity_repo.json"),
                *obs,
            ],
            name="parity_repo",
            cwd=base,
            run_dir=run_dir,
            wall=w,
            env=cenv,
        ),
        scoped(
            [
                py,
                "-I",
                str(kids / "blas.py"),
                str(agent_dir),
                obs[0],
                "host4",
                str(run_dir / "blas_host4.json"),
            ],
            name="blas_host4",
            cwd=agent_dir,
            run_dir=run_dir,
            wall=w,
            env=env4,
        ),
        scoped(
            [
                py,
                "-I",
                str(kids / "blas.py"),
                str(agent_dir),
                obs[0],
                "preload_numpy",
                str(run_dir / "blas_preload.json"),
            ],
            name="blas_preload",
            cwd=agent_dir,
            run_dir=run_dir,
            wall=w,
            env=env4,
        ),
    ]
    for nm, c in neg.items():
        specs.append(
            scoped(
                [
                    py,
                    "-I",
                    str(kids / "neg.py"),
                    str(c["main"]),
                    obs[0],
                    *(["--with-file"] if c["with_file"] else []),
                ],
                name=nm,
                cwd=c["cwd"],
                run_dir=run_dir,
                wall=w,
                env=cenv,
                expect="fail",
            )
        )
    games = []
    for opp_name, opp, seed, ref_dir in args.opponent:
        for seat in (0, 1):
            nm = f"game_{opp_name}_s{seat}"
            games.append(
                (
                    nm,
                    opp_name,
                    seed,
                    seat,
                    Path(ref_dir) / f"game{seat:03d}_seed{seed}_seat{seat}.json" if ref_dir else None,
                )
            )
            specs.append(
                scoped(
                    [
                        py,
                        "-I",
                        str(kids / "game.py"),
                        str(agent_dir),
                        str(Path(opp).resolve()),
                        str(seed),
                        str(seat),
                        str(run_dir / f"{nm}.json"),
                    ],
                    name=nm,
                    cwd=agent_dir,
                    run_dir=run_dir,
                    wall=w,
                    env=cenv,
                )
            )
    slots = [c.strip() for c in args.core_slots.split(";")]
    wanted = sorted({int(x) for s in slots for x in s.split(",")})
    t_wait = time.time()
    while busy := cores_busy(wanted):
        if time.time() - t_wait > 600:
            raise HardStop(f"核 {wanted} 等 600 s 仍忙：{busy}")
        time.sleep(15)
    res["cores_wait_s"] = round(time.time() - t_wait, 1)
    done, stopped = run_specs(specs, slots, run_dir, w)
    res["children"] = done
    res["stopped_on"] = stopped
    res["finished"] = time.strftime("%F %T %Z")
    res["tmp_base"] = str(base)
    shutil.copytree(kids, vdir / "kids")
    # ---------------- mandatory checks
    by = {c["name"]: c for c in done}

    def jl(p: Path):
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return None

    chk: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    chk["rebuild_tar_and_model_equal"] = (
        res["a_rebuild"]["tar_equal"] and res["a_rebuild"]["model_equal"]
    )
    lc = man["actor"]["leaf_check"]
    chk["actor_leafwise_equal"] = bool(
        lc["all_equal"]
        and lc["meta_equal"]
        and not lc["actor_has_o"]
        and lc["n_leaves_source"] == lc["n_leaves_actor"]
        and lc["dtypes"] == ["float32"]
    )
    chk["tar_members_exact_and_pinned"] = [
        (m["name"], m["mode"], m["mtime"], m["uid"], m["gid"], m["uname"], m["gname"])
        for m in res["b_tar"]["members"]
    ] == [("main.py", "0o644", 0, 0, 0, "", ""), ("model.npz", "0o644", 0, 0, 0, "", "")]
    chk["tar_lt_100MiB"] = res["b_tar"]["bytes"] < MAX_TAR_BYTES
    chk["embedded_source_bytes_equal_repo"] = [n for n, _ in srcs] == list(MODULES) and all(
        v["ok"] for v in cmp.values()
    )
    cl = jl(run_dir / "closure.json")
    chk["closure_ok"] = (
        bool(cl)
        and cl["last_callable_is_agent"]
        and not cl["banned_loaded"]
        and cl["decoy_rl_untouched"]
        and not cl["repo_rl_importable_before"]
        and cl["stdout_bytes"] == 0
        and cl["init_errors"] == 0
        and str(cl["model_path"]).startswith(str(agent_dir))
    )
    detail["closure"] = cl and {k: cl[k] for k in cl if k not in ("new_modules", "sys_path")} | {
        "numpy_loaded": "numpy" in cl["new_modules"],
        "cg": _cgstat(cl["cg"]),
    }
    pk, rp = jl(run_dir / "parity_pkg.json"), jl(run_dir / "parity_repo.json")
    if pk and rp and len(pk["rows"]) == len(rp["rows"]):
        rows = list(zip(pk["rows"], rp["rows"], strict=True))
        eqv = sum(
            a["action"] == b["action"] and a["seat"] == b["seat"] and a["i"] == b["i"]
            for a, b in rows
        )
        eqj = sum(
            json.dumps(a["action"], ensure_ascii=False)
            == json.dumps(b["action"], ensure_ascii=False)
            for a, b in rows
        )
        detail["parity"] = {
            "n": len(rows),
            "equal_values": eqv,
            "equal_json_order": eqj,
            "init_errors": pk["init_errors"],
            "repo_rl_file": rp["rl_file"],
            "by_seat": {s: sum(1 for a, _ in rows if a["seat"] == s) for s in (0, 1)},
            "days": len({a["i"] // 24 for a, _ in rows}),
            "cg_pkg": _cgstat(pk["cg"]),
            "cg_repo": _cgstat(rp["cg"]),
        }
        chk["parity_all_equal"] = (
            len(rows) == args.parity_n
            and eqv == len(rows)
            and eqj == len(rows)
            and pk["init_errors"] == 0
        )
    else:
        chk["parity_all_equal"] = False
    b4, bp = jl(run_dir / "blas_host4.json"), jl(run_dir / "blas_preload.json")
    detail["blas"] = {
        "host4": b4 and {k: b4[k] for k in b4 if k != "cg"},
        "preload_numpy_limitation": bp and {k: bp[k] for k in bp if k != "cg"},
    }
    chk["blas_explicit_single_thread_when_host_sets_4"] = (
        bool(b4)
        and b4["env_before"]["OPENBLAS_NUM_THREADS"] == "4"
        and all(v == "1" for v in b4["env_after_exec"].values())
        and b4["blas_after_first_call"]["threads"] == 1
        and b4["numpy_preloaded_at_exec"] is False
    )
    negd = {}
    for nm, (exc, msg) in NEG_CASES.items():
        log = (
            (run_dir / f"{nm}.log").read_text(errors="replace")
            if (run_dir / f"{nm}.log").exists()
            else ""
        )
        c = by.get(nm, {})
        negd[nm] = {
            "setup": neg[nm]["setup"],
            "rc": c.get("rc"),
            "stderr_has_init_failure": "[fa-agent] 初始化失敗" in log,
            "stderr_has_exception": exc in log,
            "stderr_has_message": msg in log,
            "returned_action": "RETURNED:" in log,
            "tail": log.strip().splitlines()[-1][:300] if log.strip() else "",
        }
        negd[nm]["ok"] = (
            bool(c.get("as_expected"))
            and negd[nm]["stderr_has_init_failure"]
            and negd[nm]["stderr_has_exception"]
            and negd[nm]["stderr_has_message"]
            and not negd[nm]["returned_action"]
        )
    detail["negatives"] = negd
    chk["negatives_all_fail_fast"] = all(v["ok"] for v in negd.values())
    gd, lat = {}, []
    for nm, _opp_name, _seed, _seat, ref_path in games:
        g = jl(run_dir / f"{nm}.json")
        if not g:
            gd[nm] = {"missing": True, "ok": False}
            continue
        ref = json.loads(ref_path.read_text()) if ref_path else {"unit_trace": [], "money": None}
        acts = g["actions"]
        n = mism = 0
        first = None
        for u in ref["unit_trace"]:
            t = u["turn"]
            n += 1
            if t >= len(acts) or _cmd_of(acts[t], u["unit"]) != list(u["raw"]):
                mism += 1
                first = first or {
                    "turn": t,
                    "unit": u["unit"],
                    "ref": u["raw"],
                    "ours": _cmd_of(acts[t], u["unit"]) if t < len(acts) else None,
                }
        lat += g["raw_step_s"][1:]
        gd[nm] = {
            k: g[k]
            for k in (
                "seed",
                "seat",
                "opponent",
                "opp_nargs",
                "entries",
                "statuses",
                "n_non_active_before_last",
                "non_active_before_last",
                "status_counts",
                "money",
                "ours",
                "opp",
                "ours_calls",
                "ours_exceptions",
                "opp_calls",
                "opp_exceptions",
                "init_errors",
                "loader_exec_s",
                "first_step_s",
                "warm",
                "model_path",
                "stdout_bytes",
            )
        }
        gd[nm].update(
            {
                "n_actions": len(acts),
                "ref_path": str(ref_path) if ref_path else None,
                "ref_money": ref["money"],
                "money_equal_ref": (g["money"] == ref["money"]) if ref_path else None,
                "unit_chain": {"compared": n, "mismatches": mism, "first_mismatch": first},
                "cg": _cgstat(g["cg"]),
            }
        )
        gd[nm]["ok"] = (
            g["entries"] == 720
            and g["statuses"] == ["DONE", "DONE"]
            and g["n_non_active_before_last"] == 0
            and g["ours_exceptions"] == 0
            and g["opp_exceptions"] == 0
            and g["init_errors"] == 0
            and g["stdout_bytes"] == 0
            and g["ours_calls"] == 719
            and g["opp_calls"] == 719
            and gd[nm]["money_equal_ref"] is not False
            and mism == 0
        )
    detail["games"] = gd
    if games:
        chk["games_720_both_done_no_midgame_error_and_equal_ref_if_given"] = len(gd) == len(games) and all(
            v["ok"] for v in gd.values()
        )
        detail["games_ref_compared"] = {
            nm: v.get("money_equal_ref") is not None for nm, v in gd.items()
        }
    q = sorted(lat)
    detail["latency_all_games_warm"] = (
        {
            "n": len(q),
            "p50": q[len(q) // 2],
            "p99": q[min(len(q) - 1, int(0.99 * len(q)))],
            "max": q[-1],
        }
        if q
        else None
    )
    if games:
        chk["latency_gate_p99_lt_0_8_max_lt_1_0_local"] = (
            bool(q)
            and detail["latency_all_games_warm"]["p99"] < 0.8
            and detail["latency_all_games_warm"]["max"] < 1.0
        )
    chk["children_all_as_expected"] = (
        stopped is None and all(c.get("as_expected") for c in done) and len(done) == len(specs)
    )
    res["mandatory_checks"] = chk
    _game_keys = ("games_720_both_done_no_midgame_error_and_equal_ref_if_given",
                  "latency_gate_p99_lt_0_8_max_lt_1_0_local")
    # 靜態檢查＝實戰局與延遲門以外的全部檢查；子程序只看非對局的部分
    # （對局子程序失敗若連帶中止了靜態子程序，靜態檢查視為未完成＝false）。
    _game_names = {g[0] for g in games}
    _static_specs = [sp["name"] for sp in specs if sp["name"] not in _game_names]
    _static_done = {c["name"]: c for c in done if c.get("name") not in _game_names}
    _static_children_ok = all(
        nm in _static_done and _static_done[nm].get("as_expected") for nm in _static_specs
    )
    res["static_checks_passed"] = _static_children_ok and all(
        v for k, v in chk.items() if k not in _game_keys and k != "children_all_as_expected"
    )
    # 完整放行必須包含實戰局與延遲門；未給 --opponent 時不放行。
    res["release_ready"] = bool(games) and all(chk.values())
    res["detail"] = detail
    res["not_applicable"] = {}
    if not games:
        res["not_applicable"]["games_and_latency"] = "未給 --opponent：略過實戰局與延遲門"
    (vdir / "verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(
        json.dumps(
            {
                "vdir": str(vdir),
                "release_ready": res["release_ready"],
                "static_checks_passed": res["static_checks_passed"],
                "not_applicable": res["not_applicable"],
                "mandatory_checks": chk,
                "stopped_on": stopped,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    if res["release_ready"]:
        return 0
    # 4＝未給 --opponent 且靜態檢查全過（未做實戰與延遲驗證）；3＝有檢查失敗
    return 4 if (not games and res["static_checks_passed"]) else 3


def _parse_opponent(text: str) -> tuple[str, str, int, str | None]:
    parts = text.split(",")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("格式：NAME,PATH,SEED[,REF_DIR]")
    return parts[0], parts[1], int(parts[2]), (parts[3] if len(parts) == 4 else None)


def main() -> int:
    global SRC_COMMIT, OUT_DIR
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--checkpoint", required=True)
        sp.add_argument("--expected-sha", required=True)
        sp.add_argument("--source-label", default=None, help="來源標籤（預設 git HEAD 或 worktree）")
        sp.add_argument("--out-dir", default=None, help=f"manifest／verify 輸出根目錄（預設 {OUT_DIR}）")
    v = sub.choices["verify"]
    v.add_argument("--obs", nargs="+", required=True, help="真 obs 重放集 json（steps[].obs）")
    v.add_argument("--parity-n", type=int, required=True, help="預期對帳列數（全數相等才過）")
    v.add_argument(
        "--repo-runtime",
        default=str(ROOT),
        help="parity 參照用的 repo 根目錄（預設本 repo）",
    )
    v.add_argument(
        "--wrong-model", required=True, help="負例用的另一個合法 actor npz（SHA 須不同）"
    )
    v.add_argument(
        "--opponent",
        action="append",
        default=[],
        type=_parse_opponent,
        help="NAME,PATH/main.py,SEED[,REF_DIR]：實戰對手（可重複；REF_DIR 放既有同 ckpt 參照局 "
        "game00{seat}_seed{seed}_seat{seat}.json，給了才逐元比對）",
    )
    v.add_argument("--core-slots", default="0,1;2,3;4,5")
    v.add_argument("--wall", type=int, default=1800)
    a = ap.parse_args()
    SRC_COMMIT = a.source_label or default_source_label()
    if a.out_dir:
        OUT_DIR = Path(a.out_dir).resolve()
    try:
        if a.cmd == "build":
            m = build(
                Path(a.checkpoint), a.expected_sha, DIST, OUT_DIR / f"fa_{a.expected_sha[:8]}"
            )
            print(
                json.dumps(
                    {k: m[k] for k in ("identity", "package_name", "tar", "main_py")}
                    | {
                        "actor": {
                            k: m["actor"][k]
                            for k in ("sha256", "bytes", "param_count", "config_from_meta")
                        }
                    },
                    ensure_ascii=False,
                    indent=1,
                )
            )
            return 0
        return verify(a)
    except HardStop as e:
        print(f"HARD STOP: {e.message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
