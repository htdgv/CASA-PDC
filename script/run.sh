#!/usr/bin/env bash
set -euo pipefail

# ── Token prompt ──────────────────────────────────────────────────────────────
read -rsp "Enter your Hugging Face token: " TOKEN
echo

read -rsp "Enter the data directory: " DATA_DIR
echo

read -rsp "Enter the data size: " DATA_SIZE
echo

# ── Helper ────────────────────────────────────────────────────────────────────
run_step() {
    local step="$1"; shift
    echo
    echo "────────────────────────────────────────────────────────────"
    echo "▶  ${step}"
    echo "────────────────────────────────────────────────────────────"
    "$@"
    echo "✓  Done: ${step}"
}

# ── Pipeline ──────────────────────────────────────────────────────────────────
run_step "Step 1/4: Data Augmentation" \
    python -m bin.augmentation1 \
        --data-dir $DATA_DIR \
        --save-dir output \
        --size $DATA_SIZE

run_step "Step 2/4: Feature Extraction" \
    python -m bin.feat_extract \
        --data-dir  output/json \
        --token     "$TOKEN" \
        --save-dir  output \
        --data-size $DATA_SIZE

run_step "Step 3/4: DMRS Extraction" \
    python -m bin.dmrs_extract \
        --data-dir  output/json \
        --token     "$TOKEN" \
        --data-size $DATA_SIZE

run_step "Step 4/4: Training" \
    python -m bin.train3 \
        --aug-dir    output/json \
        --save-dir   output/out \
        --size       $DATA_SIZE \
        --token      "$TOKEN" \
        --blind-test $DATA_DIR/test.json

echo
echo "────────────────────────────────────────────────────────────"
echo "✓  All steps completed successfully."
echo "────────────────────────────────────────────────────────────"