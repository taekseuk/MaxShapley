#!/usr/bin/env bash
set -euo pipefail

# Generate 9 perturbed dataset files:
#   3 datasets (musique, hotpotqa, msmarco)
# x 3 perturbations (duplicate relevant, zero-doc, contradictory)
#
# Assumptions:
# - This shell script is run from the MaxShapley repo root, or you adjust ROOT below.
# - Your perturbation script is available at $SCRIPT.
# - Input dataset files are under $INPUT_DIR.

ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Update this if you renamed the perturbation script.
SCRIPT="${SCRIPT:-$ROOT/perturb_maxshapley_dataset.py}"

INPUT_DIR="${INPUT_DIR:-$ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data}"
mkdir -p "$OUTPUT_DIR"

# Input datasets
MUSIQUE_IN="$INPUT_DIR/musique_annotated_subset.json"
HOTPOTQA_IN="$INPUT_DIR/hotpotqa_annotated_subset.json"
MSMARCO_IN="$INPUT_DIR/msmarco_annotated_subset.json"

run_job() {
  local input_file="$1"
  local output_file="$2"
  shift 2
  echo "Creating $(basename "$output_file")"
  "$PYTHON_BIN" "$SCRIPT"     --input "$input_file"     --output "$output_file"     "$@"
}

# MuSiQUE
run_job "$MUSIQUE_IN" "$OUTPUT_DIR/musique_duplicate_relevant_1.json"   --duplicate-relevant 1

run_job "$MUSIQUE_IN" "$OUTPUT_DIR/musique_zero_doc_1.json"   --add-zero-docs 1

run_job "$MUSIQUE_IN" "$OUTPUT_DIR/musique_contradictory_doc_1.json"   --add-contradictory-docs 1

# HotPotQA
run_job "$HOTPOTQA_IN" "$OUTPUT_DIR/hotpotqa_duplicate_relevant_1.json"   --duplicate-relevant 1

run_job "$HOTPOTQA_IN" "$OUTPUT_DIR/hotpotqa_zero_doc_1.json"   --add-zero-docs 1

run_job "$HOTPOTQA_IN" "$OUTPUT_DIR/hotpotqa_contradictory_doc_1.json"   --add-contradictory-docs 1

# MS MARCO
run_job "$MSMARCO_IN" "$OUTPUT_DIR/msmarco_duplicate_relevant_1.json"   --duplicate-relevant 1

run_job "$MSMARCO_IN" "$OUTPUT_DIR/msmarco_zero_doc_1.json"   --add-zero-docs 1

run_job "$MSMARCO_IN" "$OUTPUT_DIR/msmarco_contradictory_doc_1.json"   --add-contradictory-docs 1

echo
echo "Done. Generated files:"
ls -1 "$OUTPUT_DIR"
