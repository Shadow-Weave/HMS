#!/bin/zsh
PY=/private/tmp/claude-501/-Users-lazy-Documents-Claude-Code/7477fb8f-767d-43d5-86b8-b84b581078ff/scratchpad/bench/venv/bin/python
BENCH="/Users/lazy/Documents/Claude Code/hms-elite/benchmarks/egolife"
OUT="$BENCH/results/full"
LOG="$BENCH/results/full.log"
mkdir -p "$OUT"
: > "$LOG"
for ARM in mem0_faiss elite_matched elite_full elite_infer; do
  echo "=== $(date -u +%FT%TZ) start $ARM" >> "$LOG"
  T0=$(date +%s)
  "$PY" "$BENCH/run_egolife.py" --arms "$ARM" --limit 204 --repeats 3 --out "$OUT" --run-id full >> "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  echo "=== $(date -u +%FT%TZ) end $ARM rc=$RC shell_wall_s=$((T1-T0))" >> "$LOG"
  if [ $RC -ne 0 ]; then echo "ARM FAILED $ARM rc=$RC" >> "$LOG"; exit $RC; fi
done
echo "=== $(date -u +%FT%TZ) summarize" >> "$LOG"
"$PY" "$BENCH/run_egolife.py" --summarize --out "$OUT" --run-id full >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) done rc=$?" >> "$LOG"
