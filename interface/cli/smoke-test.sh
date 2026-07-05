#!/bin/bash
# CLI smoke test - verifies basic CLI functionality against a running API server
#
# Prerequisites:
#   - hms CLI must be in PATH or HMS_CLI env var set
#   - API server must be running at HMS_API_URL (default: http://localhost:8888)
#
# Usage:
#   ./interface/cli/smoke-test.sh
#   HMS_CLI=/path/to/hms ./interface/cli/smoke-test.sh

set -e

# Configuration
HMS_CLI="${HMS_CLI:-hms}"
export HMS_API_URL="${HMS_API_URL:-http://localhost:8888}"
TEST_BANK="cli-smoke-test-$(date +%s)"

echo "=== HMS CLI Smoke Test ==="
echo "CLI: $HMS_CLI"
echo "API URL: $HMS_API_URL"
echo "Test bank: $TEST_BANK"
echo ""

# Helper function
run_test() {
    local name="$1"
    shift
    echo -n "Testing: $name... "
    if "$@" > /tmp/cli-test-output.txt 2>&1; then
        echo "OK"
        return 0
    else
        echo "FAILED"
        echo "  Command: $*"
        echo "  Output:"
        cat /tmp/cli-test-output.txt | sed 's/^/    /'
        return 1
    fi
}

run_test_output() {
    local name="$1"
    local expected="$2"
    shift 2
    echo -n "Testing: $name... "
    if "$@" > /tmp/cli-test-output.txt 2>&1; then
        if grep -qi "$expected" /tmp/cli-test-output.txt; then
            echo "OK"
            return 0
        else
            echo "FAILED (expected '$expected' not found)"
            echo "  Command: $*"
            echo "  Output:"
            cat /tmp/cli-test-output.txt | sed 's/^/    /'
            return 1
        fi
    else
        echo "FAILED"
        echo "  Command: $*"
        echo "  Output:"
        cat /tmp/cli-test-output.txt | sed 's/^/    /'
        return 1
    fi
}

cleanup() {
    echo ""
    echo "Cleaning up test bank..."
    "$HMS_CLI" bank delete "$TEST_BANK" -y 2>/dev/null || true
}
trap cleanup EXIT

FAILED=0

# Test 1: Version
run_test "version" "$HMS_CLI" --version || FAILED=1

# Test 2: Help
run_test "help" "$HMS_CLI" --help || FAILED=1

# Test 3: Configure help
run_test "configure help" "$HMS_CLI" configure --help || FAILED=1

# Test 4: List banks (JSON output)
run_test "list banks" "$HMS_CLI" bank list -o json || FAILED=1

# Test 5: Set bank name (creates the bank)
run_test "set bank name" "$HMS_CLI" bank name "$TEST_BANK" "CLI Smoke Test Bank" || FAILED=1

# Test 6: Get bank disposition
run_test_output "get bank disposition" "CLI Smoke Test Bank" "$HMS_CLI" bank disposition "$TEST_BANK" || FAILED=1

# Test 7: Retain memory
run_test "retain memory" "$HMS_CLI" memory retain "$TEST_BANK" "Alice is a software engineer who loves Rust programming" || FAILED=1

# Test 8: Retain more memories
run_test "retain more memories" "$HMS_CLI" memory retain "$TEST_BANK" "Bob is Alice's colleague who prefers Python" || FAILED=1

# Test 9: Recall memories
run_test_output "recall memories" "Alice" "$HMS_CLI" memory recall "$TEST_BANK" "Who is Alice?" || FAILED=1

# Test 10: Reflect on memories
run_test_output "reflect" "Alice" "$HMS_CLI" memory reflect "$TEST_BANK" "What do you know about Alice?" || FAILED=1

# Test 11: Get bank stats
run_test "bank stats" "$HMS_CLI" bank stats "$TEST_BANK" || FAILED=1

# Test 12: List entities
run_test "list entities" "$HMS_CLI" entity list "$TEST_BANK" || FAILED=1

# Test 13: List documents
run_test "list documents" "$HMS_CLI" document list "$TEST_BANK" || FAILED=1

# Test 14: Clear memories
run_test "clear memories" "$HMS_CLI" memory clear "$TEST_BANK" || FAILED=1

# Test 15: List operations
run_test "list operations" "$HMS_CLI" operation list "$TEST_BANK" || FAILED=1

# --- Coverage-critical commands (added to ensure CLI exercises every endpoint) ---

# Test: Set disposition directly (PUT /profile)
run_test "bank set-disposition" "$HMS_CLI" bank set-disposition "$TEST_BANK" \
    --skepticism 3 --literalism 3 --empathy 3 || FAILED=1

# Test: Recover consolidation (no-op when nothing stalled, but exercises the endpoint)
run_test "bank consolidation-recover" "$HMS_CLI" bank consolidation-recover "$TEST_BANK" || FAILED=1

# Test: Bank template schema
run_test "bank template-schema" "$HMS_CLI" bank template-schema -o json || FAILED=1

# Test: Export bank template
run_test "bank export-template" "$HMS_CLI" bank export-template "$TEST_BANK" -o json || FAILED=1

# Test: Audit log list + stats
run_test "audit list" "$HMS_CLI" audit list "$TEST_BANK" -o json || FAILED=1
run_test "audit stats" "$HMS_CLI" audit stats "$TEST_BANK" -o json || FAILED=1

# Test: Webhook lifecycle (list / create / update / deliveries / delete)
run_test "webhook list (empty)" "$HMS_CLI" webhook list "$TEST_BANK" -o json || FAILED=1

WEBHOOK_OUT=$("$HMS_CLI" webhook create "$TEST_BANK" https://example.invalid/hook -o json 2>/tmp/cli-test-output.txt || true)
if echo "$WEBHOOK_OUT" | grep -q '"id"'; then
    echo "Testing: webhook create... OK"
    WEBHOOK_ID=$(echo "$WEBHOOK_OUT" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)
    run_test "webhook update" "$HMS_CLI" webhook update "$TEST_BANK" "$WEBHOOK_ID" --enabled false || FAILED=1
    run_test "webhook deliveries" "$HMS_CLI" webhook deliveries "$TEST_BANK" "$WEBHOOK_ID" -o json || FAILED=1
    run_test "webhook delete" "$HMS_CLI" webhook delete "$TEST_BANK" "$WEBHOOK_ID" -y || FAILED=1
else
    echo "Testing: webhook create... FAILED"
    cat /tmp/cli-test-output.txt | sed 's/^/    /'
    FAILED=1
fi

# Test 16: Delete bank
run_test "delete bank" "$HMS_CLI" bank delete "$TEST_BANK" -y || FAILED=1

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All smoke tests passed! ==="
    exit 0
else
    echo "=== Some smoke tests failed ==="
    exit 1
fi
