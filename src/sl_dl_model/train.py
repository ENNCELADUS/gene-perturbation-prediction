"""Per-fold training loop (Accelerate/DDP, tqdm) and the StateDlProducer.

StateDlProducer trains the model on one fold's pairs and then embeds every
gene in the universe through the frozen STATE backbone to produce a per-gene
embedding table.  Wave 4 will add score_matrix.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from accelerate import Accelerator
from torch import optim
from tqdm.auto import tqdm

from sl_benchmark_baseline.features import build_pair_features
from sl_dl_model.bags import GwpsBags
from sl_dl_model.config import SLDLConfig
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.losses import bag_loss, combine, sl_bce_loss
from sl_dl_model.model import SlDlModel

logger = logging.getLogger(__name__)


def _epoch_weights(epoch: int, config: SLDLConfig) -> dict[str, float]:
    """Return the loss-component weights for a given epoch.

    Warmup epochs suppress SL BCE (``lambda_sl=0``) and run full distill +
    bag supervision.  Post-warmup epochs activate SL BCE and reduce distill.

    Args:
        epoch: Zero-indexed current epoch number.
        config: Training configuration.

    Returns:
        Dict with keys ``"sl"``, ``"distill"``, ``"bag"``.
    """
    if epoch < config.warmup_epochs:
        return {
            "sl": 0.0,
            "distill": config.lambda_distill,
            "bag": config.lambda_bag,
        }
    return {
        "sl": config.lambda_sl,
        "distill": config.lambda_distill_after_warmup,
        "bag": config.lambda_bag,
    }


class StateDlProducer:
    """Train the DL model on a fold's train pairs; emit per-gene embeddings.

    Implements the :class:`~sl_dl_model.evaluate.EmbeddingProducer` protocol.
    After :meth:`produce` completes, the trained model is cached on the
    instance for use by Wave 4's ``score_matrix`` method.

    Args:
        config: Training configuration.
        esm: Precomputed ESM2 embedding table.
        bags: Per-gene gwps response bags + control template.
        train_pairs: List of 5-tuples ``(gene_a, gene_b, label, ea, eb)``
            where ``ea``/``eb`` are per-gene GeneEffect scalars for this pair.
        input_dim: Feature dimension of the control cells / STATE input.
        output_dim: Feature dimension of the STATE output (= input_dim for the
            HVG checkpoint).
    """

    def __init__(
        self,
        config: SLDLConfig,
        *,
        esm: Esm2EmbeddingTable,
        bags: GwpsBags,
        train_pairs: list[tuple[str, str, int, float, float]],
        input_dim: int,
        output_dim: int,
    ) -> None:
        self.config = config
        self.esm = esm
        self.bags = bags
        self.train_pairs = train_pairs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._model: SlDlModel | None = None

    def _build_model(self) -> SlDlModel:
        """Construct a fresh :class:`SlDlModel` from config + ESM dim."""
        checkpoint = (
            None
            if self.config.state_backend == "linear_mock"
            else self.config.state_checkpoint
        )
        return SlDlModel(
            backend=self.config.state_backend,
            checkpoint=checkpoint,
            esm_dim=self.esm.dim,
            adapter_hidden=self.config.adapter_hidden,
            pert_dim=self.config.pert_dim,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            pooling=self.config.pooling,
            pair_hidden=self.config.pair_hidden,
            include_coverage_flag=self.config.include_coverage_flag,
        )

    def produce(
        self,
        symbols: np.ndarray,
        train_symbols: set[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train on this fold then embed all universe genes.

        Genes without an ESM2 vector stay as zero embeddings (mask=0).
        Held-out genes are embedded purely via ``adapter(ESM2)`` + frozen
        STATE; they receive no bag or distill supervision (leakage rule).

        Args:
            symbols: Universe gene symbols in canonical order, shape
                ``(n_gene,)``.
            train_symbols: Set of upper-case gene symbols that appear in the
                training pairs for this fold.

        Returns:
            Tuple ``(embeddings, mask)`` where ``embeddings`` has shape
            ``(n_gene, emb_dim)`` and ``mask`` is an int array of shape
            ``(n_gene,)`` with 1 for ESM2-covered genes and 0 otherwise.
        """
        torch.manual_seed(self.config.seed)
        accelerator = Accelerator()
        model = self._build_model()
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=self.config.lr,
        )
        model, optimizer = accelerator.prepare(model, optimizer)
        self._train(model, optimizer, accelerator, train_symbols)
        self._model = accelerator.unwrap_model(model)

        device = accelerator.device
        control = torch.tensor(self.bags.control_template, device=device)
        pooled_dim = self._model.emb_dim
        embeddings = np.zeros((len(symbols), pooled_dim), dtype=float)
        mask = np.zeros(len(symbols), dtype=int)

        self._model.eval()
        with torch.no_grad():
            for row, symbol in enumerate(
                tqdm(
                    symbols,
                    desc="embed-universe",
                    disable=not accelerator.is_main_process,
                )
            ):
                key = str(symbol).upper()
                vec = self.esm.vectors_by_symbol.get(key)
                if vec is None:
                    continue
                esm_vec = torch.tensor(vec, device=device)
                e_g = self._model.embed_gene(esm_vec, control)
                embeddings[row] = e_g.cpu().numpy()
                mask[row] = 1

        return embeddings, mask

    def _train(
        self,
        model: SlDlModel,
        optimizer: optim.Optimizer,
        accelerator: Accelerator,
        train_symbols: set[str],
    ) -> None:
        """Run the training loop over all epochs.

        Bag supervision is applied only for genes in
        ``train_symbols ∩ bags.bags_by_symbol`` (leakage rule).

        Args:
            model: Model prepared by Accelerate (possibly wrapped).
            optimizer: Optimizer prepared by Accelerate.
            accelerator: The Accelerator instance for backward + device.
            train_symbols: Upper-case gene symbols present in train pairs.
        """
        device = accelerator.device
        control = torch.tensor(self.bags.control_template, device=device)
        covered_train = {
            s.upper() for s in train_symbols if s.upper() in self.bags.bags_by_symbol
        }

        for epoch in range(self.config.max_epochs):
            weights = _epoch_weights(epoch, self.config)
            model.train()
            inner = accelerator.unwrap_model(model)
            pbar = tqdm(
                self.train_pairs,
                desc=f"epoch {epoch}",
                disable=not accelerator.is_main_process,
            )
            for a, b, label, ea, eb in pbar:
                key_a, key_b = a.upper(), b.upper()
                vec_a = self.esm.vectors_by_symbol.get(key_a)
                vec_b = self.esm.vectors_by_symbol.get(key_b)
                if vec_a is None or vec_b is None:
                    continue

                esm_a = torch.tensor(vec_a, device=device)
                esm_b = torch.tensor(vec_b, device=device)
                e_a = inner.embed_gene(esm_a, control)
                e_b = inner.embed_gene(esm_b, control)

                ge = torch.tensor(
                    build_pair_features(np.array([ea]), np.array([eb])),
                    device=device,
                    dtype=torch.float32,
                )
                logit = inner.score_pairs(
                    e_a.unsqueeze(0),
                    e_b.unsqueeze(0),
                    ge,
                )
                parts: dict[str, torch.Tensor] = {
                    "sl": sl_bce_loss(
                        logit,
                        torch.tensor([float(label)], device=device),
                    ),
                }

                # Bag supervision: only for covered train genes (leakage rule).
                if weights["bag"] > 0:
                    bag_part = _bag_part(
                        inner,
                        covered_train,
                        control,
                        device,
                        key_a,
                        vec_a,
                        key_b,
                        vec_b,
                        self.bags,
                    )
                    if bag_part is not None:
                        parts["bag"] = bag_part

                total = combine(parts, weights)
                optimizer.zero_grad()
                accelerator.backward(total)
                optimizer.step()


def _bag_part(
    model: SlDlModel,
    covered_train: set[str],
    control: torch.Tensor,
    device: torch.device | str,
    key_a: str,
    vec_a: np.ndarray,
    key_b: str,
    vec_b: np.ndarray,
    bags: GwpsBags,
) -> torch.Tensor | None:
    """Compute bag supervision loss for covered genes in this pair.

    Accumulates bag_loss contributions for key_a and key_b when each is in
    covered_train.  Returns None if neither gene is covered.

    Args:
        model: Unwrapped SlDlModel.
        covered_train: Upper-case symbols eligible for bag supervision.
        control: Control cell tensor on device.
        device: Torch device.
        key_a: Upper-case symbol for gene A.
        vec_a: ESM2 vector for gene A.
        key_b: Upper-case symbol for gene B.
        vec_b: ESM2 vector for gene B.
        bags: GwpsBags containing real response bags.

    Returns:
        Scalar bag loss tensor, or None if no covered gene in this pair.
    """
    total: torch.Tensor | None = None
    for key, vec in ((key_a, vec_a), (key_b, vec_b)):
        if key not in covered_train:
            continue
        pred = model.encoder(torch.tensor(vec, device=device), control)
        real = torch.tensor(bags.bags_by_symbol[key], device=device)
        term = bag_loss(pred, real)
        total = term if total is None else total + term
    return total
