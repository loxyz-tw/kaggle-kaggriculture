"""Kaggriculture 提交進入點（day0_challenger profile）。

Kaggle 會載入此檔並呼叫 `agent(obs)`；提交環境中檔案位於
`/kaggle_simulations/agent/`，因此兩種載入路徑都要放進 sys.path。
注意：`get_last_callable()` 用 `exec(code, {})` 執行本檔，命名空間中沒有
`__file__`，所以不能直接引用它，也必須讓 `agent` 是最後一個被綁定的 callable。
"""

import os
import sys

_THIS = globals().get("__file__", "challenger.py")
for _p in (os.path.dirname(os.path.abspath(_THIS)), "/kaggle_simulations/agent"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import agent_core  # noqa: E402

_PROFILE = agent_core.CHALLENGER


def agent(obs):
    return agent_core.plan(obs, _PROFILE)
