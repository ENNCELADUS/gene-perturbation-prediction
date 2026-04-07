## Zero-Shot Perturbation Prediction

### Overview
This implementation performs zero-shot perturbation prediction using a pre-trained scGPT model. It uses control cells ("non-targeting") from the training set as a baseline and modifies the input "perturbation flag" to simulate specific gene knockouts/perturbations.

### Directory Structure
- **`src/configs/`**: Contains `config.yaml` for paths and model parameters.
- **`src/`**:
    - **`data/`**: `loader.py` handles loading `h5ad` files, identifying control cells, and preparing batches with perturbation flags.
    - **`model/`**: `wrapper.py` wraps the scGPT model, handling weight loading and the forward pass (`pred_perturb`).
    - **`utils/`**: `metrics.py` (MSE, Pearson) and `logging.py`.
    - **`main.py`**: The orchestrator that loops through target genes, runs predictions, and saves results.

### Usage

**1. Interactive Run:**
```bash
# Sync the project environment
uv sync

# Run the pipeline
uv run python src/main.py --config src/configs/config.yaml
```

**2. Batch Job (SLURM):**
```bash
sbatch scripts/zeroshot.sh
```

### Data Flow
1. **Load**: `train.h5ad` (Control Cells) and `test.h5ad` (Target List + Ground Truth).
2. **Loop**: For each Target Gene in Test:
    - Sample Control Cells.
    - Create Input: `[Control Expression] + [Target Gene Pert Flag]`.
    - **Predict**: Model generates `Predicted Expression`.
    - **Evaluate**: Compare `Predicted Mean` vs `Ground Truth Mean`.
3. **Output**: `results/perturbation_metrics.csv` and `logs/zeroshot/slurm_*.out`.
