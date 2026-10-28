#!/bin/zsh
# One process per arm into results/final (mem0 and Elite never share a process); then merge the summary.
# 每臂一个进程写入 results/final（mem0 与 Elite 绝不同进程）；最后合并汇总。
PY=/private/tmp/claude-501/-Users-lazy-Documents-Claude-Code/7477fb8f-767d-43d5-86b8-b84b581078ff/scratchpad/bench/venv/bin/python
BENCH="/Users/lazy/Documents/Claude Code/hms-elite/benchmarks/egolife"
OUT="$BENCH/results/final"
LOG="$BENCH/results/final.log"
mkdir -p "$OUT"
: > "$LOG"
for ARM in mem0_faiss mem0_milvus elite_matched elite_full elite_infer; do
  echo "=== $(date -u +%FT%TZ) start $ARM" >> "$LOG"
  T0=$(date +%s)
  "$PY" "$BENCH/run_egolife.py" --arms "$ARM" --limit 204 --repeats 3 --out "$OUT" --run-id final >> "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  echo "=== $(date -u +%FT%TZ) end $ARM rc=$RC shell_wall_s=$((T1-T0))" >> "$LOG"
  if [ $RC -ne 0 ]; then echo "ARM FAILED $ARM rc=$RC" >> "$LOG"; exit $RC; fi
done
echo "=== $(date -u +%FT%TZ) summarize" >> "$LOG"
"$PY" "$BENCH/run_egolife.py" --summarize --out "$OUT" --run-id final >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) done rc=$?" >> "$LOG"
