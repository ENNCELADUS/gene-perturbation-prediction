#!/bin/bash
#SBATCH --job-name=scgpt_finetune
#SBATCH --output=logs/finetune/slurm_%j.out
#SBATCH --error=logs/finetune/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ========== Environment Setup ==========
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

# Load conda environment
source ~/.bashrc
conda activate vcc

# Navigate to project directory
cd /home/richard/projects/VCC

# Create log directory
mkdir -p logs/finetune

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

