import anndata
import numpy as np
import torch
from pathlib import Path
import logging


class PerturbationDataLoader:
    def __init__(self, config, vocab, logger=None):
        self.config = config
        self.vocab = vocab
        self.logger = logger or logging.getLogger(__name__)

        self.train_adata = None
        self.test_adata = None
        self.control_cells = None

        self._load_data()

    def _load_data(self):
        data_dir = Path(self.config["paths"]["data_dir"])
        train_path = data_dir / self.config["paths"]["train_file"]
        test_path = data_dir / self.config["paths"]["test_file"]

        self.logger.info(f"Loading training data from {train_path}")
        self.train_adata = anndata.read_h5ad(train_path)

        self.logger.info(f"Loading test data from {test_path}")
        self.test_adata = anndata.read_h5ad(test_path)

        # Pre-process vocab mapping for test data genes
        # Use vocab["<pad>"] for unknown genes to strictly avoid OOB indices
        pad_token_id = self.vocab["<pad>"]
        if "id_in_vocab" not in self.test_adata.var.columns:
            # Map gene names to vocab IDs (use pad_token_id for unknown genes)
            self.test_adata.var["id_in_vocab"] = [
                self.vocab[gene] if gene in self.vocab else pad_token_id
                for gene in self.test_adata.var.index
            ]

        # Extract Control Cells (non-targeting)
        ctrl_key = self.config["inference"]["control_target_gene"]
        self.control_cells = self.train_adata[
            self.train_adata.obs["target_gene"] == ctrl_key
        ]
        self.logger.info(f"Found {self.control_cells.n_obs} control cells.")

    def get_test_targets(self):
        """Return unique target genes from test set"""
        return self.test_adata.obs["target_gene"].unique()

    def get_target_ground_truth(self, target_gene):
        """Get actual expression data for a specific target gene"""
        return self.test_adata[self.test_adata.obs["target_gene"] == target_gene]

    def get_control_mean(self) -> np.ndarray:
        """Get mean expression of all control cells (for PDS delta calculation)."""
        if isinstance(self.control_cells.X, np.ndarray):
            return self.control_cells.X.mean(axis=0)
        return np.asarray(self.control_cells.X.mean(axis=0)).flatten()

    def get_gene_names(self) -> np.ndarray:
        """Get gene names from test data."""
        return np.asarray(self.test_adata.var_names)

    def get_train_adata(self):
        """Return the training AnnData object for baseline fitting."""
        return self.train_adata

    def prepare_perturbation_batch(
        self, target_gene, n_cells=None, seed=None, return_control_expr=False
    ):
        """
        Prepare a batch of control cells with the perturbation flag set for target_gene.

        Args:
            target_gene: Target gene for perturbation
            n_cells: Number of cells to sample (default: config batch_size)
            seed: Random seed for reproducibility
            return_control_expr: If True, also return control expression array

        Returns:
            BatchData object, or (BatchData, control_expr) if return_control_expr=True
        """
        if n_cells is None:
            n_cells = self.config["inference"]["batch_size"]

        # Sample control cells with optional seed for reproducibility
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.control_cells.n_obs, n_cells, replace=True)
        batch_ctrl = self.control_cells[indices].copy()

        # Expression values (X)
        # Assuming X is sparse or dense. Convert to dense for model input
        if isinstance(batch_ctrl.X, np.ndarray):
            x = batch_ctrl.X
        else:
            x = batch_ctrl.X.toarray()

        # Convert to float32 for model input (AMP will handle precision)
        x = torch.tensor(x, dtype=torch.float32)

        # Create Perturbation Flags
        # 0: No perturbation, 1: Perturbation
        # We need to map target_gene to its index in the feature list of the adata
        # NOTE: This assumes test_adata and train_adata have same var (genes)

        try:
            target_idx = self.test_adata.var_names.get_loc(target_gene)
        except KeyError:
            self.logger.warning(
                f"Target gene {target_gene} not found in gene list. Skipping."
            )
            return None

        pert_flags = torch.zeros_like(x, dtype=torch.long)
        pert_flags[:, target_idx] = 1

        # Prepare BatchDict structure expected by scGPT
        # We need a simple object or dict that can be accessed via .x and .pert

        # In the tutorial, batch_data is a PyG Data object or similar with .x attribute
        # .x contains [expression, pert_flags] concatenated or separate?
        # Tutorial: x[:, 0] is expression, x[:, 1] is pert_flags

        combined_x = torch.stack([x, pert_flags.float()], dim=1)  # (batch, 2, n_genes)

        # The model.pred_perturb expects:
        # x: (batch, 2, n_genes) -> flattened to (batch, 2*n_genes) maybe?
        # Let's check generation_model.py line 313:
        # x: torch.Tensor = batch_data.x
        # ori_gene_values = x[:, 0].view(batch_size, -1)
        # pert_flags = x[:, 1].long().view(batch_size, -1)

        # So batch_data.x should be shape (batch, 2*n_genes) if view is used like that?
        # Wait, x[:, 0] implies dim 1 is size 2?
        # No, if x is (batch, 2*n_genes), then x[:, 0] is just scalar?
        # "x[:, 0].view(batch_size, -1)" implies x has shape (batch, 2, n_genes) likely

        class BatchData:
            def __init__(self, x, pert):
                self.x = x
                self.pert = pert  # just for len() check

            def to(self, device):
                self.x = self.x.to(device)

        batch_data = BatchData(combined_x, pert_flags)

        if return_control_expr:
            return batch_data, x.numpy()
        return batch_data
