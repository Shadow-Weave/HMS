#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE_EXPLICIT=0
if [ "${HMS_ENV_FILE+x}" = "x" ]; then
  ENV_FILE="$HMS_ENV_FILE"
  ENV_FILE_EXPLICIT=1
else
  ENV_FILE="$ROOT_DIR/.env"
fi

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ENV_FILE"
  set +a
elif [ "$ENV_FILE_EXPLICIT" = "1" ]; then
  echo "HMS_ENV_FILE does not exist: $ENV_FILE" >&2
  exit 2
fi

if [ "${HMS_BENCHMARK:-longmemeval}" != "longmemeval" ]; then
  echo "This launcher supports only HMS_BENCHMARK=longmemeval." >&2
  exit 2
fi

if [ -n "${HMS_BENCHMARK_DATABASE_URL:-}" ]; then
  export HMS_API_DATABASE_URL="$HMS_BENCHMARK_DATABASE_URL"
fi

if [ -z "${HMS_API_DATABASE_URL:-}" ]; then
  echo "HMS_API_DATABASE_URL is required. See lab/evaluation/benchmarks/longmemeval/README.md." >&2
  exit 2
fi

case "$HMS_API_DATABASE_URL" in
  *@postgres:*)
    echo "HMS_API_DATABASE_URL points to the Compose-only host 'postgres'." >&2
    echo "Use HMS_BENCHMARK_DATABASE_URL with a host-reachable address such as 127.0.0.1." >&2
    exit 2
    ;;
esac

DATA_DIR="${HMS_DATA_DIR:-$ROOT_DIR/.aaaDATA}"
LOG_DIR="${HMS_LOG_DIR:-$ROOT_DIR/.aaaLOG}"
RESULT_DIR="${HMS_RESULT_DIR:-$ROOT_DIR/.aaaRESULT}"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$RESULT_DIR"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${HMS_BENCHMARK_LOG:-$LOG_DIR/longmemeval_${TIMESTAMP}.log}"

RESUME_REQUESTED="${HMS_RESUME:-0}"
CLI_RESULTS_FILENAME=""
EXPECT_RESULTS_FILENAME=0
for argument in "$@"; do
  if [ "$EXPECT_RESULTS_FILENAME" = "1" ]; then
    CLI_RESULTS_FILENAME="$argument"
    EXPECT_RESULTS_FILENAME=0
    continue
  fi
  if [ "$argument" = "--resume" ]; then
    RESUME_REQUESTED=1
  fi
  case "$argument" in
    --results-filename)
      EXPECT_RESULTS_FILENAME=1
      ;;
    --results-filename=*)
      CLI_RESULTS_FILENAME="${argument#--results-filename=}"
      ;;
  esac
done

if [ "$RESUME_REQUESTED" = "1" ] && [ -z "${HMS_RESULTS_FILENAME:-}" ] && [ -z "$CLI_RESULTS_FILENAME" ]; then
  echo "Resume requires HMS_RESULTS_FILENAME or --results-filename to name the existing result artifact." >&2
  exit 2
fi

RESULTS_FILENAME="${CLI_RESULTS_FILENAME:-${HMS_RESULTS_FILENAME:-longmemeval_${TIMESTAMP}.json}}"
CONTEXT_FORMAT="${HMS_CONTEXT_FORMAT:-structured_source}"
PARALLEL="${HMS_PARALLEL:-1}"
MAX_CONCURRENT_QUESTIONS="${HMS_MAX_CONCURRENT_QUESTIONS:-$PARALLEL}"
EVAL_SEMAPHORE_SIZE="${HMS_EVAL_SEMAPHORE_SIZE:-$PARALLEL}"
THINKING_BUDGET="${HMS_THINKING_BUDGET:-500}"
MAX_TOKENS="${HMS_MAX_TOKENS:-8192}"

require_positive_integer() {
  local name="$1"
  local value="$2"
  case "$value" in
    ""|*[!0-9]*|0)
      echo "$name must be a positive integer, got: $value" >&2
      exit 2
      ;;
  esac
}

require_positive_integer HMS_PARALLEL "$PARALLEL"
require_positive_integer HMS_MAX_CONCURRENT_QUESTIONS "$MAX_CONCURRENT_QUESTIONS"
require_positive_integer HMS_EVAL_SEMAPHORE_SIZE "$EVAL_SEMAPHORE_SIZE"
require_positive_integer HMS_THINKING_BUDGET "$THINKING_BUDGET"
require_positive_integer HMS_MAX_TOKENS "$MAX_TOKENS"
if [ -n "${HMS_MAX_INSTANCES:-}" ]; then
  require_positive_integer HMS_MAX_INSTANCES "$HMS_MAX_INSTANCES"
fi
if [ -n "${HMS_MAX_QUESTIONS:-}" ]; then
  require_positive_integer HMS_MAX_QUESTIONS "$HMS_MAX_QUESTIONS"
fi

LONGMEMEVAL_ARGS=(
  --results-dir "$RESULT_DIR"
  --results-filename "$RESULTS_FILENAME"
  --context-format "$CONTEXT_FORMAT"
  --parallel "$PARALLEL"
  --max-concurrent-questions "$MAX_CONCURRENT_QUESTIONS"
  --eval-semaphore-size "$EVAL_SEMAPHORE_SIZE"
  --thinking-budget "$THINKING_BUDGET"
  --max-tokens "$MAX_TOKENS"
)

if [ -n "${HMS_MAX_INSTANCES:-}" ]; then
  LONGMEMEVAL_ARGS+=(--max-instances "$HMS_MAX_INSTANCES")
fi

if [ -n "${HMS_MAX_QUESTIONS:-}" ]; then
  LONGMEMEVAL_ARGS+=(--max-questions "$HMS_MAX_QUESTIONS")
fi

if [ -n "${HMS_DATASET_PATH:-}" ]; then
  LONGMEMEVAL_ARGS+=(--dataset-path "$HMS_DATASET_PATH")
fi

if [ "${HMS_RETRIEVAL_ONLY:-0}" = "1" ]; then
  LONGMEMEVAL_ARGS+=(--skip-ingestion)
fi

if [ "${HMS_ENABLE_QUERY_EXPANSION:-0}" = "1" ]; then
  LONGMEMEVAL_ARGS+=(--enable-query-expansion)
  LONGMEMEVAL_ARGS+=(--query-rewriting-strategy "${HMS_QUERY_REWRITING_STRATEGY:-llm_driven}")
fi

if [ -n "${HMS_SESSION_EXPANSION_WEIGHT:-}" ]; then
  LONGMEMEVAL_ARGS+=(--session-expansion-weight "$HMS_SESSION_EXPANSION_WEIGHT")
fi

case "${HMS_PIPELINE:-ledger}" in
  ledger)
    LONGMEMEVAL_ARGS+=(--oracle-planner-v26)
    ;;
  self_evolution)
    LONGMEMEVAL_ARGS+=(--oracle-planner-v220)
    ;;
  standard)
    ;;
  *)
    echo "Unsupported HMS_PIPELINE: ${HMS_PIPELINE:-}" >&2
    echo "Supported values: standard, ledger, self_evolution" >&2
    exit 2
    ;;
esac

if [ "$RESUME_REQUESTED" = "1" ] && [ "${HMS_RESUME:-0}" = "1" ]; then
  LONGMEMEVAL_ARGS+=(--resume)
fi

PYTHON_BIN="${HMS_PYTHON_BIN:-}"
if [ -n "$PYTHON_BIN" ]; then
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "HMS_PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 2
  fi
  export PYTHONPATH="$ROOT_DIR/core/dataplane:$ROOT_DIR/lab/evaluation${PYTHONPATH:+:$PYTHONPATH}"
  CMD=("$PYTHON_BIN" -m benchmarks.longmemeval.longmemeval_benchmark "${LONGMEMEVAL_ARGS[@]}" "$@")
else
  if ! command -v uv >/dev/null 2>&1; then
    echo "The 'uv' command is required. Install uv or set HMS_PYTHON_BIN." >&2
    exit 2
  fi
  CMD=(
    uv run
    --project "$ROOT_DIR/lab/evaluation"
    python -m benchmarks.longmemeval.longmemeval_benchmark
    "${LONGMEMEVAL_ARGS[@]}"
    "$@"
  )
fi

echo "LongMemEval pipeline: Retain -> Recall -> Answer -> Judge"
echo "Configuration source: $ENV_FILE"
echo "Pipeline profile: ${HMS_PIPELINE:-ledger}"
echo "Context format: $CONTEXT_FORMAT"
echo "Parallel items: $PARALLEL"
echo "Results: $RESULT_DIR/$RESULTS_FILENAME"
echo "Log: $LOG_FILE"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Running: ${CMD[*]}"
  cd "$ROOT_DIR"
  "${CMD[@]}"
} 2>&1 | tee "$LOG_FILE"
