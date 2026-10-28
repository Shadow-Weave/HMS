#!/bin/zsh
# Replay-variant grid for the compact-loss diagnosis (zero API): eight replay_ab.py invocations, each to its own
# results/replay_diag/<name> directory. W* replay the COMPACT run's recorded replies; V* replay r3's FULL replies.
# 紧凑损失诊断的回放变体网格（零 API）：八次 replay_ab.py 调用，各写入 results/replay_diag/<name>。
# W* 回放紧凑 run 记录的回复；V* 回放 r3 的完整回复。
#
# Usage / 用法: ./replay_diag_run.sh [name ...]   (no names = all eight / 不给名字 = 全部八个)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${REPLAY_PY:-/private/tmp/claude-501/-Users-lazy-Documents-Claude-Code/7477fb8f-767d-43d5-86b8-b84b581078ff/scratchpad/bench/venv/bin/python}"
OUT="$HERE/results/replay_diag"
R3="$HERE/results/real/elite_full204_r3"
COMPACT="$HERE/results/real/elite_compact_full204"
mkdir -p "$OUT"

run_one() {
  local name="$1"; shift
  echo "== $name: $PY replay_ab.py --no-answer --out results/replay_diag/$name $*" | tee "$OUT/$name.cmd"
  ( cd "$HERE" && "$PY" replay_ab.py --no-answer --out "$OUT/$name" "$@" ) > "$OUT/$name.log" 2>&1
  echo "== $name exit=$?"
}

variant() {
  case "$1" in
    W0) run_one W0 --replies-run "$COMPACT" --replies-contract compact ;;
    W1) run_one W1 --replies-run "$COMPACT" --replies-contract compact --drop-entities ;;
    W2) run_one W2 --replies-run "$COMPACT" --replies-contract compact --force-type fact ;;
    W3) run_one W3 --replies-run "$COMPACT" --replies-contract compact --drop-when ;;
    W4) run_one W4 --replies-run "$COMPACT" --replies-contract compact --keep-only-text ;;
    V0) run_one V0 --replies-run "$R3" --replies-contract full --arms full ;;
    V2) run_one V2 --replies-run "$R3" --replies-contract full --arms compact --drop-entities ;;
    V3) run_one V3 --replies-run "$R3" --replies-contract full --arms compact --force-type fact ;;
    # Extra axis found during the diagnosis: the compact model's DECISIONS (1,685 NONE vs 23 in r3).
    # 诊断中发现的额外轴：紧凑模型的决策（1,685 个 NONE，r3 只有 23 个）。
    W5) run_one W5 --replies-run "$COMPACT" --replies-contract compact --force-add ;;
    W6) run_one W6 --replies-run "$COMPACT" --replies-contract compact --keep-only-text --force-add ;;
    *) echo "unknown variant $1"; return 2 ;;
  esac
}

if [ $# -eq 0 ]; then set -- W0 W1 W2 W3 W4 V0 V2 V3; fi
for v in "$@"; do variant "$v"; done
