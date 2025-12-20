#!/bin/bash
#SBATCH -J baseline_model
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/baseline/slurm_%j.out
#SBATCH --error=logs/baseline/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

# Load conda environment
source ~/.bashrc
conda activate vcc

set -euo pipefail

echo "=============================================="
echo "Running all baseline models"
echo "=============================================="

# Run PCA baseline
echo ""
echo "[1/2] Running PCA baseline..."
echo "----------------------------------------------"
python -m src.main --config src/configs/pca.yaml
echo "PCA baseline completed!"

# Run Logistic Regression baseline
echo ""
echo "[2/2] Running Logistic Regression baseline..."
echo "----------------------------------------------"
python -m src.main --config src/configs/logreg.yaml
echo "Logistic Regression baseline completed!"

echo ""
echo "=============================================="
echo "All baselines completed!"
echo "=============================================="

# Generate comparison report
echo ""
echo "Generating comparison report..."
python scripts/compare.py --pattern "results/pca_*" --pattern "results/logreg_*" --output results/reports --name baseline_comparison
echo "Comparison report saved to results/reports/"
