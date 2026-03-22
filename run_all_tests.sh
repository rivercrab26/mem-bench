#!/bin/bash
# Full test suite: all adapters × LongMemEval oracle
# Usage: ./run_all_tests.sh

set -e
VENV="/Users/rayyao/source/project/mem-bench/.venv/bin"
RESULTS_BASE="/Users/rayyao/source/project/mem-bench/test_results"
LIMIT=10  # 10 questions for quick validation

mkdir -p "$RESULTS_BASE"

echo "================================================"
echo "  mem-bench Full Test Suite"
echo "  Date: $(date)"
echo "  Limit: $LIMIT questions per adapter"
echo "================================================"

# 1. BM25 (baseline, fast)
echo -e "\n>>> [1/6] BM25 (baseline)"
$VENV/mem-bench run --adapter bm25 --benchmark longmemeval --split oracle --limit $LIMIT \
  --output-dir "$RESULTS_BASE/bm25" --no-judge 2>&1 | tail -20

# 2. Hindsight (OpenClaw)
echo -e "\n>>> [2/6] Hindsight (OpenClaw)"
$VENV/mem-bench run --adapter hindsight --benchmark longmemeval --split oracle --limit $LIMIT \
  --output-dir "$RESULTS_BASE/hindsight" --no-judge 2>&1 | tail -20

# 3. Mem0 (OSS mode with Ollama)
echo -e "\n>>> [3/6] Mem0 (OSS mode)"
$VENV/mem-bench run --adapter mem0 --benchmark longmemeval --split oracle --limit $LIMIT \
  --output-dir "$RESULTS_BASE/mem0" --no-judge 2>&1 | tail -20

# 4. Graphiti (requires Neo4j on :7687)
echo -e "\n>>> [4/6] Graphiti (Neo4j)"
$VENV/mem-bench run --adapter graphiti --benchmark longmemeval --split oracle --limit $LIMIT \
  --output-dir "$RESULTS_BASE/graphiti" --no-judge 2>&1 | tail -20

# 5. Letta (requires Letta server on :8283)
echo -e "\n>>> [5/6] Letta"
$VENV/mem-bench run --adapter letta --benchmark longmemeval --split oracle --limit $LIMIT \
  --output-dir "$RESULTS_BASE/letta" --no-judge 2>&1 | tail -20

# 6. LangMem (requires OpenAI key for embeddings -- skip if not available)
echo -e "\n>>> [6/6] LangMem"
if [ -n "$OPENAI_API_KEY" ]; then
  $VENV/mem-bench run --adapter langmem --benchmark longmemeval --split oracle --limit $LIMIT \
    --output-dir "$RESULTS_BASE/langmem" --no-judge 2>&1 | tail -20
else
  echo "  SKIPPED: No OPENAI_API_KEY set (LangMem needs OpenAI embeddings)"
fi

# Comparison
echo -e "\n================================================"
echo "  COMPARISON"
echo "================================================"
DIRS=""
for d in "$RESULTS_BASE"/*/; do
  if [ -f "$d/summary.json" ]; then
    DIRS="$DIRS $d"
  fi
done
if [ -n "$DIRS" ]; then
  $VENV/mem-bench compare $DIRS 2>&1
fi

echo -e "\nDone! Results in $RESULTS_BASE/"
