# Kaggle Kaggriculture — 銀牌（第 122／10,148 名，前 1.20%）

[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) 是 Kaggle 的模擬對戰競賽：兩名玩家回合制經營農場，在 720 回合內透過種植、養殖、加工與市場買賣賺最多錢。本 repo 是我參賽的核心程式碼，最終第 122 名，獲得競賽銀牌（2026-10-15 頒發）。

## 方法

- **模仿學習（主力）**：以天梯強者的官方對局示範做 teacher forcing，訓練 Transformer 策略網路（d_model 384、8 層、6 頭、d_ff 1536）。之後以低學習率（3e-5）把最新幾天的對局資料合併進來續訓。線上兩席之一（LRM3）就是這條線的端點，線上最高約 2,500 分。
- **強化學習（PPO league）**：以 JAX 實作 PPO，從上述 IL 端點起跑，報酬只看終局勝負，對手是凍結的歷代 actor 池。線上另一席（E2）是這條線的端點，但在未見過的 seed 上**沒有證實勝過起點**。
- **規則式 planner**：`agent_core.py` 是不依賴模型的規則式 agent，作為早期提交、基準與對手。

### 與線上提交的對應

模型 agent 部署時只用 NumPy。以下六個模組會被內嵌進提交包的 `main.py`：

```
rl/__init__.py  rl/rules.py  rl/caps.py  rl/ordered_v3_contract.py  rl/loop4_numpy.py  rl/full_actions_numpy.py
```

本 repo 的這六個檔與線上提交包內嵌的原始碼**逐 byte 相同**（內嵌時唯一的改動是 import 路徑改寫，由 `scripts/build_full_actions_submission.py` 自動處理），所以檔內註解保持原樣。

以 `--source-label 8416e0503ead51bd81810e8fc2ce15784582b773` 對同一份權重打包，產出的 `main.py` 與 tar 會和線上提交包逐 byte 相同。我已用 E2 的權重實際驗證過，tar 的 sha256 是 `c2b278e2…`。

## 內容

| 路徑 | 用途 |
|---|---|
| `main.py`、`agent_core.py`、`challenger.py` | 規則式 agent 的提交入口與共用核心 |
| `scripts/evaluate.py`、`scripts/build_submissions.py` | 規則式 agent 的本地對局評估與單檔打包 |
| `rl/`（上述六個部署模組） | 模型 agent 的 NumPy 推論 |
| `rl/full_actions_obs.py`、`full_actions_head.py`、`full_actions_decode.py`、`loop4_*.py`、`ordered_v3_codec.py`、`state.py`、`kernels.py`、`obs.py`、`actions.py`、`pricing.py` | 觀測特徵、動作編碼與解碼、模型（JAX）、模擬器狀態 |
| `scripts/full_actions_il_train.py`、`scripts/full_actions_raw_overlay.py` | IL 訓練（schema-3 shard 串流讀取、raw overlay 標籤修正、續訓時的資料與 lr 遷移） |
| `scripts/full_actions_ppo_train.py`、`full_actions_ppo_cost.py`、`full_actions_critic_init.py`、`full_actions_critic_warmup.py`、`full_actions_collapse_diag.py`；`rl/full_actions_ppo.py`、`full_actions_updater.py`、`loop4_ppo.py`、`loop4_tape.py`、`tape_pool.py`、`gae.py`、`reward.py`、`env.py`、`t3_diag_probe.py` | PPO：critic 暖身、rollout、逐 component clip 的 updater、對手池 |
| `scripts/build_full_actions_submission.py` | checkpoint → 提交包（tar 內是 `main.py`＋`model.npz`），`verify` 做隔離驗證 |
| `scripts/pkg_h2h.py`、`scripts/full_actions_eval.py`、`scripts/loop4_eval.py`、`scripts/loop4_parity.py` | 評估：兩個提交包在官方引擎直接對戰、NumPy agent 對任意對手評估 |
| `scripts/run_capped.sh` | 長跑啟動器（核親和、cgroup RSS 帽、wall timeout） |
| `tests/` | 測試（CPU 可跑；需要未公開存檔的項目會自動 skip） |

## 怎麼執行

```bash
# 安裝（Python ≥ 3.12，使用 uv）
uv sync --group dev --group rl      # 只用 CPU 時，把 pyproject.toml 的 jax[cuda12] 換成 jax

# 測試（CPU）
JAX_PLATFORMS=cpu uv run pytest -q tests/

# 1) 規則式 agent
uv run python scripts/evaluate.py --seeds 10 --out out/day0   # 本地對局評估
uv run python scripts/build_submissions.py                    # 打包成單檔 main.py，輸出到 dist/

# 2) IL 訓練（需自備 schema-3 shard、shards_manifest.json、raw overlay 與 overlay manifest）
uv run python scripts/full_actions_il_train.py --shard <train.npz> ... --manifest <shards_manifest.json> \
  --overlay <overlay.npz> --overlay-manifest <overlay_manifest.json> \
  --d-model 384 --n-layers 8 --n-heads 6 --d-ff 1536 --rows 64 --lr 3e-4 --steps N --out-dir <dir>
# 換資料、降 lr 續訓：
#   --resume <ckpt> --resume-migrate-data OLD:NEW --resume-migrate-data-map <map> [--resume-migrate-lr OLD:NEW]

# 3) PPO（需自備 IL checkpoint、critic 暖身產物、對手池 actor）
#    真教師列的定位檔以環境變數 KAGGRI_FIXTURE_REFS 指定（JSON：shard、shard_sha256）
uv run python scripts/full_actions_critic_init.py ...
#    對手池每個 --pool-actor 都要配一個 --expect-pool-actor-sha；winloss 報酬必須同時給 --lam 1 --reward-scale 1。
#    --out 與它的父目錄都必須尚未存在（訓練程式拒絕覆蓋）。
uv run python scripts/full_actions_ppo_train.py \
  --ckpt <il.npz> --expect-ckpt-sha <sha> \
  --critic-init <critic.npz> --expect-critic-sha <sha> \
  --opponent pool \
  --pool-actor <a1.npz> --expect-pool-actor-sha <sha1> \
  --pool-actor <a2.npz> --expect-pool-actor-sha <sha2> \
  --reward winloss --lam 1 --reward-scale 1 \
  --lr 2.5e-6 --minibatch 96 \
  --out runs/new_run/result.json

# 4) 打包與包對包對戰
uv run python scripts/build_full_actions_submission.py build --checkpoint <ckpt.npz> --expected-sha <sha256>
#    verify 子命令：未給 --opponent 時不做實戰與延遲驗證，release_ready 一律為 false；
#    其餘檢查都通過時 static_checks_passed＝true、結束碼 4，有檢查失敗則結束碼 3。
uv run python scripts/pkg_h2h.py <包A目錄> <包B目錄> <seed> <A的座位> out.json
```

各腳本的完整參數請看 `--help`。

## 不包含的東西

- **資料**：官方每日 replay、轉好的 schema-3 shard、raw overlay、fixture。
- **資料轉換工具**：把官方 replay 轉成 schema-3 shard 的程式沒有公開，使用者要自行準備相同格式的 shard（格式可參考 `scripts/full_actions_il_train.py` 的讀取端）。
- **權重**：IL／PPO checkpoint、critic、PPO 對手池的 actor。
- **其他**：本地評估用的對手 agent、league 的編排與選點腳本、實驗紀錄與審查文件。

因此本 repo 只能**重建配方**，無法逐位元重現線上提交的模型。程式註解多為繁體中文，部分註解提到的內部規格文件沒有隨 repo 公開。
