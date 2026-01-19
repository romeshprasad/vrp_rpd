#!/bin/bash

# Master script to run Pure BRKGA (Exp 3) on ALL datasets (Sequential, single GPU)
# Vanilla BRKGA with NO warm start, NO GP analysis, NO gene injection
# Provides baseline for comparison against GP-enhanced BRKGA
# For parallel GPU execution, use run_all_pure_brkga_gpu.sh instead

DATASETS=(
  gr17 gr21 gr24 gr48 eil51 berlin52 eil101 kroA100 gr202 gr431 gr666 rat783 dsj1000
)

GPU_ID=${1:-0}  # Default to GPU 0 if not specified

echo "========================================"
echo "COMPREHENSIVE PURE BRKGA EXPERIMENT"
echo "========================================"
echo "GPU ID: ${GPU_ID}"
echo "Total datasets: ${#DATASETS[@]}"
echo ""
echo "Configuration:"
echo "  Warm Start: NO"
echo "  GP Analysis: NO"
echo "  Gene Injection: NO"
echo "  Total generations: 10,000"
echo "  Checkpoints: Every 1,000 generations"
echo "  Seeds: 3 independent runs per instance"
echo ""
echo "Purpose: Establish vanilla BRKGA baseline"
echo "         (Experiment 3 of 3)"
echo ""
echo "NOTE: This runs sequentially. For parallel GPU execution,"
echo "      use: ./run_all_pure_brkga_gpu.sh <num_gpus>"
echo "========================================"
echo ""

# Make script executable
chmod +x run_pure_brkga.sh

START_TIME=$(date +%s)
TOTAL_SUCCESS=0
TOTAL_FAIL=0

for i in "${!DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "[$((i+1))/${#DATASETS[@]}] ${DATASETS[$i]}"
    echo "========================================"

    ./run_pure_brkga.sh ${GPU_ID} $i

    if [ $? -eq 0 ]; then
        ((TOTAL_SUCCESS++))
    else
        ((TOTAL_FAIL++))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================"
echo "ALL PURE BRKGA EXPERIMENTS COMPLETE"
echo "========================================"
echo "Total datasets processed: ${#DATASETS[@]}"
echo "Successful: ${TOTAL_SUCCESS}"
echo "Failed: ${TOTAL_FAIL}"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results: ./results/pure_brkga/"
echo ""
echo "Next steps:"
echo "  1. Analyze results: python analyze_pure_brkga.py --csv"
echo "  2. Compare with Exp 2: python compare_experiments.py"
echo "========================================"
