#!/usr/bin/env bash
# 帶帽啟動器（審查者 2026-09-18）：任務 03／04 的所有長跑一律經這支啟動。
#
# 為什麼要有這支：核親和、執行緒上限、RSS 帽、wall timeout 這些在 `panel.sh` 裡本來就有，
# 但手寫的啟動指令一次又一次漏掉——結果是 loop3_run 開 131 執行緒、診斷開 105 執行緒互搶，
# 負載 28／16 核；以及診斷 RSS 衝到 19.8 GB 把 swap 吃光。靠記得不是機制，所以做成必填參數：
# **缺任一必填參數直接拒絕啟動**，不套預設值放行。
#
# 沿用 內部紀錄 已有的做法（timeout --kill-after=10、
# taskset -c、/usr/bin/time -v），不另創格式。
#
# **記憶體帽為什麼不是 `ulimit -v`**：2026-09-18 實測，診斷腳本在 `ulimit -v 10 GiB` 下、
# **峰值 RSS 只有 5.403 GiB** 時就 `std::bad_alloc` 被 abort（rc 134）。`ulimit -v` 限的是
# **虛擬位址空間**，而 JAX／XLA 一開始編譯多種形狀就會保留遠超 RSS 的位址空間，
# 所以它既擋不住真正的 RSS 成長、又會把正常的 JAX 行程打死。`panel.sh` 用 `ulimit -v` 沒事，
# 是因為它的子行程是 kaggle-environments（純 numpy、不編譯 XLA），不是同一類負載。
# 這裡改用 cgroup v2：`systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0`，
# 那是真的 RSS 上限，而且 `MemorySwapMax=0` 正好對上「swap 不算可用空間」。
# `--vsize-gib` 保留為**選填**，給 panel.sh 那類純 numpy 子行程沿用舊做法。
#
#   scripts/run_capped.sh --cpus 8-11 --mem-gib 8 --wall-s 3600 --name diag-attempt3 -- <指令...>
#
# 產物：reports/scratch/run_capped/<name>.json，含 rc、峰值 RSS（/usr/bin/time -v 的
# "Maximum resident set size"）、實際 wall、套上的核集合與執行緒設定。
set -uo pipefail

usage() {
  cat >&2 <<'USAGE'
用法：scripts/run_capped.sh --cpus <核集合> --mem-gib <N> --wall-s <N> --name <標籤> -- <指令...>

必填（缺任一即拒絕啟動，不套預設）：
  --cpus     taskset -c 的核集合，例如 8-11 或 0,2,4
  --mem-gib  真 RSS 上限（GiB，整數），走 cgroup v2 MemoryMax；swap 設為 0 不計入
  --wall-s   timeout 的秒數（整數）；逾時送 TERM，10 秒後 KILL
  --name     標籤，決定 reports/scratch/run_capped/<name>.json

選填：
  --vsize-gib  ulimit -v 的虛擬位址空間上限（GiB）；**JAX 行程不要用**，見檔頭說明

固定套用（不可關）：
  taskset -c <核集合>
  systemd-run --user --scope -p MemoryMax=<mem-gib>G -p MemorySwapMax=0
  OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS / NUMEXPR_NUM_THREADS = 1
  XLA_FLAGS += --xla_cpu_multi_thread_eigen=false
               --xla_force_host_platform_device_count=1
               intra_op_parallelism_threads=<核數>
  timeout --kill-after=10 <wall-s>
USAGE
}

CPUS="" MEM_GIB="" WALL_S="" NAME="" VSIZE_GIB=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cpus) CPUS="${2-}"; shift 2 || { usage; exit 2; } ;;
    --mem-gib) MEM_GIB="${2-}"; shift 2 || { usage; exit 2; } ;;
    --vsize-gib) VSIZE_GIB="${2-}"; shift 2 || { usage; exit 2; } ;;
    --wall-s) WALL_S="${2-}"; shift 2 || { usage; exit 2; } ;;
    --name) NAME="${2-}"; shift 2 || { usage; exit 2; } ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_capped: 未知參數 $1" >&2; usage; exit 2 ;;
  esac
done

missing=()
test -n "$CPUS" || missing+=(--cpus)
test -n "$MEM_GIB" || missing+=(--mem-gib)
test -n "$WALL_S" || missing+=(--wall-s)
test -n "$NAME" || missing+=(--name)
if [ "${#missing[@]}" -gt 0 ]; then
  echo "run_capped: 缺必填參數 ${missing[*]}；拒絕啟動" >&2
  usage
  exit 2
fi
if [ $# -eq 0 ]; then
  echo "run_capped: -- 之後必須有要跑的指令；拒絕啟動" >&2
  exit 2
fi
case "$MEM_GIB" in ''|*[!0-9]*) echo "run_capped: --mem-gib 必須是正整數，實得 $MEM_GIB" >&2; exit 2 ;; esac
case "$WALL_S" in ''|*[!0-9]*) echo "run_capped: --wall-s 必須是正整數，實得 $WALL_S" >&2; exit 2 ;; esac
if [ -n "$VSIZE_GIB" ]; then
  case "$VSIZE_GIB" in ''|*[!0-9]*) echo "run_capped: --vsize-gib 必須是正整數，實得 $VSIZE_GIB" >&2; exit 2 ;; esac
fi
case "$NAME" in *[!A-Za-z0-9._-]*) echo "run_capped: --name 只准 [A-Za-z0-9._-]，實得 $NAME" >&2; exit 2 ;; esac

# 核數：把 "0-3,8" 這種集合展開來數，供 intra_op_parallelism_threads 用。
n_cpus=$(python3 -c '
import sys
spec = sys.argv[1]
n = 0
for part in spec.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        a, b = part.split("-", 1)
        n += int(b) - int(a) + 1
    else:
        n += 1
print(n)
' "$CPUS") || { echo "run_capped: --cpus 解析失敗：$CPUS" >&2; exit 2; }
test "$n_cpus" -ge 1 || { echo "run_capped: --cpus 展開後是 0 核：$CPUS" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTDIR="$ROOT/reports/scratch/run_capped"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$NAME.log"
TIMEFILE="$OUTDIR/$NAME.time"
JSON="$OUTDIR/$NAME.json"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
XLA_EXTRA="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1 intra_op_parallelism_threads=$n_cpus"
export XLA_FLAGS="${XLA_FLAGS:-} $XLA_EXTRA"

started=$(date -u +%s)
(
  # `--vsize-gib` 只在明確要求時才套；JAX 行程不要用（見檔頭）。
  if [ -n "$VSIZE_GIB" ]; then ulimit -v $((VSIZE_GIB * 1024 * 1024)) || exit 2; fi
  exec /usr/bin/systemd-run --user --scope --quiet --collect \
       -p "MemoryMax=${MEM_GIB}G" -p MemorySwapMax=0 \
       -- /usr/bin/timeout --kill-after=10 "$WALL_S" \
          /usr/bin/taskset -c "$CPUS" \
          /usr/bin/time -v -o "$TIMEFILE" "$@"
) > "$LOG" 2>&1
rc=$?
ended=$(date -u +%s)

peak_kib=$(awk -F': *' '/Maximum resident set size/ {print $2}' "$TIMEFILE" 2>/dev/null | tr -d ' ')
test -n "$peak_kib" || peak_kib=0
python3 - "$JSON" "$NAME" "$CPUS" "$n_cpus" "$MEM_GIB" "$WALL_S" "$rc" "$peak_kib" \
        "$((ended - started))" "$LOG" "$TIMEFILE" "$XLA_FLAGS" <<'PY'
import json, sys
(out, name, cpus, n_cpus, mem_gib, wall_s, rc, peak_kib, wall, log, timefile, xla) = sys.argv[1:13]
open(out, "w", encoding="utf-8").write(
    json.dumps(
        {
            "schema": "run-capped-v1",
            "name": name,
            "cpus": cpus,
            "n_cpus": int(n_cpus),
            "mem_gib_cap": int(mem_gib),
            "mem_cap_mechanism": "cgroup-v2 MemoryMax + MemorySwapMax=0",
            "wall_s_cap": int(wall_s),
            "rc": int(rc),
            "timed_out": int(rc) in (124, 137),
            "peak_rss_gib": round(int(peak_kib) / 1024 / 1024, 3),
            "wall_s": int(wall),
            "log": log,
            "time_v": timefile,
            "threads": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "xla_flags": xla.strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    + "\n"
)
PY
echo "run_capped: $NAME rc=$rc peak_rss_gib=$(python3 -c "print(round($peak_kib/1024/1024,3))") wall_s=$((ended - started)) -> $JSON" >&2
exit "$rc"
