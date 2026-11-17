"""打包兩個 Kaggriculture 提交候選。

產出 **self-contained 單檔** `main.py`（agent_core 原始碼 + profile 進入點串接），
tar 根目錄只有這一個檔案。

為什麼不用多檔？多檔版本要靠 `sys.path` + `import agent_core`，而
`kaggle_environments.agent.get_last_callable()` 會把 exec_dir 推進 `sys.path`
再彈掉；一旦同一個 process 內載入兩個各自帶著 `agent_core.py` 的候選（例如本地
anchor vs challenger 對打，或從兩個解壓目錄載入），Python 的模組快取會讓第二個
候選拿到第一個的 `agent_core`，靜默跑錯策略。單檔產物直接消除這一整類風險。
倉庫內的 `main.py` / `challenger.py` 仍是多檔開發入口，並且同時支援
`os.path.dirname(__file__)` 與 `/kaggle_simulations/agent/`。

用法：
    uv run python scripts/build_submissions.py
"""

import gzip
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

TARGETS = {
    "day0_anchor": {
        "profile": "ANCHOR",
        "tar": "kaggriculture-day0-anchor.tar.gz",
        "desc": "crop-only anchor：甜瓜／小麥／蘿蔔／草莓，不養動物",
    },
    "day0_challenger": {
        "profile": "CHALLENGER",
        "tar": "kaggriculture-day0-challenger.tar.gz",
        "desc": "replay-inspired 牛羊混合：9 牛 3 羊 + 甜瓜／草莓／小麥 + 肥料迴圈",
    },
}

ENTRY_TEMPLATE = '''

# --------------------------------------------------------------------------- #
# 提交進入點（由 scripts/build_submissions.py 產生；請勿手改此檔）
# profile: {profile}
# --------------------------------------------------------------------------- #

_PROFILE = {profile}


def agent(obs):
    """Kaggle 進入點。

    必須是本模組最後一個被綁定的 callable：
    `get_last_callable()` 取 `[v for v in env.values() if callable(v)][-1]`。
    """
    return plan(obs, _PROFILE)
'''


def build_source(profile):
    core = (ROOT / "agent_core.py").read_text(encoding="utf-8")
    return core.rstrip("\n") + "\n" + ENTRY_TEMPLATE.format(profile=profile)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source(src, name):
    """在打包前先確認單檔產物自己就能載入，且最後一個 callable 是 agent。"""
    env = {}
    exec(compile(src, f"<{name}>", "exec"), env)
    callables = [k for k, v in env.items() if callable(v)]
    last = callables[-1]
    if last != "agent":
        raise SystemExit(f"{name}: 最後一個 callable 是 {last!r}，不是 'agent'")
    return env["agent"]


def build():
    DIST.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "artifacts": {}}

    for name, spec in TARGETS.items():
        src = build_source(spec["profile"])
        verify_source(src, name)

        staged = DIST / f"{name}-main.py"
        staged.write_text(src, encoding="utf-8")

        tar_path = DIST / spec["tar"]
        if tar_path.exists():
            tar_path.unlink()
        # gzip 標頭會寫入 mtime，若不固定則同樣的原始碼每次打包都會得到不同 sha256，
        # 之後就無法把線上提交對回本地候選。這裡把 gzip mtime 與 tar metadata 都釘死。
        with open(tar_path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tf:
                    info = tf.gettarinfo(str(staged), arcname="main.py")
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with open(staged, "rb") as f:
                        tf.addfile(info, f)

        with tarfile.open(tar_path, "r:gz") as tf:
            members = tf.getnames()
        if members != ["main.py"]:
            raise SystemExit(f"{tar_path}: tar 根目錄應只有 main.py，實際為 {members}")

        size = tar_path.stat().st_size
        if size > 100 * 1024 * 1024:
            raise SystemExit(f"{tar_path}: {size} bytes 超過 100 MiB")

        manifest["artifacts"][name] = {
            "profile": spec["profile"],
            "description": spec["desc"],
            "tar": str(tar_path.relative_to(ROOT)),
            "tar_sha256": sha256(tar_path),
            "tar_bytes": size,
            "members": members,
            "main_py_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
            "main_py_bytes": len(src.encode("utf-8")),
            "main_py_lines": src.count("\n") + 1,
        }
        digest = manifest["artifacts"][name]["tar_sha256"]
        print(f"{name}: {tar_path.relative_to(ROOT)}  {size} bytes  sha256={digest}")

    (DIST / "build-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n寫出 {(DIST / 'build-manifest.json').relative_to(ROOT)}")
    return manifest


if __name__ == "__main__":
    build()
