#!/usr/bin/env bash
set -euo pipefail

# Run calculate_shapley.py on the 9 perturbed dataset files.
#
# Assumptions:
# - Run this from the MaxShapley repo root, or adjust ROOT below.
# - Perturbed datasets live under $PERTURBED_DIR.
# - calculate_shapley.py is in the repo root.
#
# Notes:
# - The script passes the perturbed file path via a temporary swap:
#   it backs up the expected dataset file, replaces it with the chosen perturbed file,
#   runs calculate_shapley.py, then restores the original.
# - This is needed because calculate_shapley.py currently selects datasets by hardcoded names.

ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CALC_SCRIPT="${CALC_SCRIPT:-$ROOT/calculate_shapley.py}"

DATA_DIR="${DATA_DIR:-$ROOT/data}"
PERTURBED_DIR="${PERTURBED_DIR:-$DATA_DIR/}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/perturbed_runs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/perturbed_runs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

ROUNDS="${ROUNDS:-1}"
SAMPLES_U="${SAMPLES_U:-1}"
SAMPLES_A="${SAMPLES_A:-1}"
LLM="${LLM:-openai}"

run_with_swap() {
  local dataset_key="$1"        # musique | hotpot | msmarco
  local canonical_file="$2"     # original file path expected by calculate_shapley.py
  local perturbed_file="$3"     # perturbed dataset to temporarily swap in
  local tag="$4"                # readable suffix for logs/results

  local backup_file="${canonical_file}.backup_before_perturbed_run"
  local result_csv="$RESULTS_DIR/${dataset_key}_${tag}.csv"
  local run_log_dir="$LOG_DIR/${dataset_key}_${tag}"

  mkdir -p "$run_log_dir"

  echo "============================================================"
  echo "Dataset key   : $dataset_key"
  echo "Canonical file: $canonical_file"
  echo "Perturbed file: $perturbed_file"
  echo "Tag           : $tag"
  echo "============================================================"

  if [[ ! -f "$canonical_file" ]]; then
    echo "Missing canonical dataset file: $canonical_file" >&2
    exit 1
  fi
  if [[ ! -f "$perturbed_file" ]]; then
    echo "Missing perturbed dataset file: $perturbed_file" >&2
    exit 1
  fi

  cp "$canonical_file" "$backup_file"
  trap 'if [[ -f "$backup_file" ]]; then mv -f "$backup_file" "$canonical_file"; fi' RETURN

  cp "$perturbed_file" "$canonical_file"

  "$PYTHON_BIN" "$CALC_SCRIPT" \
    --dataset "$dataset_key" \
    --index 0 \
    --llm "$LLM" \
    --log "$run_log_dir/" \
    --csv "$result_csv" \
    --rounds "$ROUNDS" \
    --samples_u "$SAMPLES_U" \
    --samples_a "$SAMPLES_A"

  mv -f "$backup_file" "$canonical_file"
  trap - RETURN

  echo "Finished: $dataset_key / $tag"
  echo
}

# -------------------------
# MuSiQUE runs
# -------------------------
run_with_swap \
  "musique" \
  "$DATA_DIR/musique_annotated_subset.json" \
  "$PERTURBED_DIR/musique_duplicate_relevant_1.json" \
  "duplicate_relevant_1"

run_with_swap \
  "musique" \
  "$DATA_DIR/musique_annotated_subset.json" \
  "$PERTURBED_DIR/musique_zero_doc_1.json" \
  "zero_doc_1"

run_with_swap \
  "musique" \
  "$DATA_DIR/musique_annotated_subset.json" \
  "$PERTURBED_DIR/musique_contradictory_doc_1.json" \
  "contradictory_doc_1"

# -------------------------
# HotPotQA runs
# -------------------------
run_with_swap \
  "hotpot" \
  "$DATA_DIR/hotpotqa_annotated_subset.json" \
  "$PERTURBED_DIR/hotpotqa_duplicate_relevant_1.json" \
  "duplicate_relevant_1"

run_with_swap \
  "hotpot" \
  "$DATA_DIR/hotpotqa_annotated_subset.json" \
  "$PERTURBED_DIR/hotpotqa_zero_doc_1.json" \
  "zero_doc_1"

run_with_swap \
  "hotpot" \
  "$DATA_DIR/hotpotqa_annotated_subset.json" \
  "$PERTURBED_DIR/hotpotqa_contradictory_doc_1.json" \
  "contradictory_doc_1"

# -------------------------
# MS MARCO runs
# -------------------------
run_with_swap \
  "msmarco" \
  "$DATA_DIR/msmarco_annotated_subset.json" \
  "$PERTURBED_DIR/msmarco_duplicate_relevant_1.json" \
  "duplicate_relevant_1"

run_with_swap \
  "msmarco" \
  "$DATA_DIR/msmarco_annotated_subset.json" \
  "$PERTURBED_DIR/msmarco_zero_doc_1.json" \
  "zero_doc_1"

run_with_swap \
  "msmarco" \
  "$DATA_DIR/msmarco_annotated_subset.json" \
  "$PERTURBED_DIR/msmarco_contradictory_doc_1.json" \
  "contradictory_doc_1"

echo "All perturbed runs completed."
echo "Results: $RESULTS_DIR"
echo "Logs   : $LOG_DIR"
