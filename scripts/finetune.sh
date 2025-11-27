#!/bin/bash
#SBATCH -J scGPT_finetune
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIATITANRTX:1
#SBATCH --output=logs/finetune/slurm_%j.out
#SBATCH --error=logs/finetune/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

# Load conda environment
source ~/.bashrc
conda activate vcc

# Navigate to project directory
cd /public/home/wangar2023/VCC_Project

# ========== Step 1: Convert Data to GEARS Format ==========
echo ""
echo "Step 1: Converting data to GEARS format..."
echo "=========================================="

# Check if GEARS data already exists
if [ ! -d "data/processed/gears/vcc" ]; then
    python scripts/convert_to_gears.py \
        --train_path data/processed/train.h5ad \
        --output_dir data/processed/gears \
        --dataset_name vcc
else
    echo "GEARS data already exists, skipping conversion..."
fi

# ========== Step 2: Finetune scGPT ==========
echo ""
echo "Step 2: Finetuning scGPT..."
echo "=========================================="

python src/train.py \
    --config src/configs/finetune.yaml \
    --seed 42

# ========== Step 3: Evaluate Finetuned Model on Test Set ==========
echo ""
echo "Step 3: Evaluating finetuned model on held-out test genes..."
echo "=========================================="

# Create results directory
mkdir -p results/scgpt_finetuned

# Run evaluation using the finetuned model
python src/main.py \
    --config src/configs/config.yaml \
    --model_type scgpt_finetuned \
    --threads 8

echo ""
echo "=========================================="
echo "Finetuning pipeline complete!"
echo ""
echo "Outputs:"
echo "  - Model:   model/scGPT_finetuned/"
echo "  - Results: results/scgpt_finetuned/"
echo "=========================================="

