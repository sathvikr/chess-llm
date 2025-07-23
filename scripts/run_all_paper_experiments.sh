#!/bin/bash

# Run all experimental setups from the paper:
# "Amortized Planning with Large-Scale Transformers: A Case Study on Chess"
# arxiv:2402.04494v2

set -e  # Exit on any error

echo "============================================================"
echo "Running All Paper Experiments"
echo "Amortized Planning with Large-Scale Transformers (Chess)"
echo "============================================================"

# Prepare data first
echo "Step 1: Preparing data for experiments..."
python prepare_data_for_paper_experiments.py --input chess_evals.jsonl --create-all

echo ""
echo "Step 2: Running experiments..."

# Main experiment (Table 1 in paper)
echo ""
echo "=== MAIN EXPERIMENT: Large model (270M params) ==="
echo "10M steps, batch size 4096, learning rate 1e-4"
python run_paper_experiments.py --model-size large_270M --policy action_value --experiment-type main

# Ablation studies (various configurations)
echo ""
echo "=== ABLATION STUDIES ==="

echo ""
echo "--- Model size ablation ---"
echo "Small model (9M params):"
python run_paper_experiments.py --model-size small_9M --policy action_value --experiment-type ablation

echo ""
echo "Medium model (136M params):"
python run_paper_experiments.py --model-size medium_136M --policy action_value --experiment-type ablation

echo ""
echo "Large model (270M params):"
python run_paper_experiments.py --model-size large_270M --policy action_value --experiment-type ablation

echo ""
echo "--- Policy type ablation ---"
echo "State-value policy:"
python run_paper_experiments.py --model-size large_270M --policy state_value --experiment-type ablation

echo ""
echo "Behavioral cloning policy:"
python run_paper_experiments.py --model-size large_270M --policy behavioral_cloning --experiment-type ablation

echo ""
echo "============================================================"
echo "All paper experiments completed!"
echo "============================================================"