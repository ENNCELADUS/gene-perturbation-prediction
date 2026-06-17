"""Per-fold training loop (Accelerate/DDP, tqdm) and the StateDlProducer.

StateDlProducer trains the model on one fold's pairs and then embeds every
gene in the universe through the frozen STATE backbone to produce a per-gene
embedding table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch import optim
from tqdm.auto import tqdm

from sl_benchmark_baseline.features import build_pair_features
from sl_dl_model.bags import GwpsBags
from sl_dl_model.config import SLDLConfig
from sl_dl_model.encoder import state_original_token
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.losses import bag_loss, combine, distill_loss, sl_bce_loss
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


def _load_pert_vocab(checkpoint: Path) -> dict[str, np.ndarray] | None:
    """Load the sibling ``pert_onehot_map.pt`` for a STATE checkpoint.

    The file is expected at ``checkpoint.parent.parent / "pert_onehot_map.pt"``.
    Returns ``None`` if the file does not exist.

    Args:
        checkpoint: Path to the STATE checkpoint file.

    Returns:
        Dict mapping upper-case gene symbol to float32 one-hot ndarray, or
        ``None`` if the sibling file is missing.
    """
    vocab_path = checkpoint.parent.parent / "pert_onehot_map.pt"
    if not vocab_path.exists():
        logger.debug("pert_onehot_map.pt not found at %s; skipping distill", vocab_path)
        return None
    raw: dict[str, object] = torch.load(
        vocab_path, map_location="cpu", weights_only=True
    )
    return {str(k).upper(): np.asarray(v, dtype=np.float32) for k, v in raw.items()}


class StateDlProducer:
    """Train the DL model on a fold's train pairs; emit per-gene embeddings.

    Implements the :class:`~sl_dl_model.evaluate.EmbeddingProducer` protocol.
    After :meth:`produce` completes, the trained model and per-gene embedding
    table are cached on the instance for reuse by ``score_matrix``.

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
        # Cached per-gene embedding table (shape n_gene × emb_dim) and symbol order.
        self._e_table_cache: torch.Tensor | None = None
        self._e_table_symbols: np.ndarray | None = None
        # Bag-coverage mask aligned to the last produce() call.
        self._coverage_mask: np.ndarray | None = None
        # Perturbation vocab (one-hots) for distill supervision.
        self._pert_vocab: dict[str, np.ndarray] | None = None
        self._pert_vocab_loaded: bool = False

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

    def _ensure_pert_vocab(self) -> None:
        """Lazily load the distill pert-vocab once per instance.

        Sets ``self._pert_vocab`` to a dict or ``None`` if unavailable.
        Skips loading when using the linear_mock backend or when no
        ``state_checkpoint`` is configured.
        """
        if self._pert_vocab_loaded:
            return
        self._pert_vocab_loaded = True

        if self.config.state_backend == "linear_mock":
            logger.debug("linear_mock backend: skipping distill vocab load")
            return
        try:
            self._pert_vocab = _load_pert_vocab(self.config.state_checkpoint)
        except Exception:
            logger.debug("failed to load pert_vocab; distill loss will be skipped")
            self._pert_vocab = None

    def _distill_part(
        self,
        train_symbols_in_step: set[str],
    ) -> torch.Tensor | None:
        """Compute MSE between adapter tokens and STATE's original one-hot tokens.

        Only genes that are both in ``train_symbols_in_step`` and the loaded
        pert-vocab contribute. Returns ``None`` if no in-vocab genes are found
        or if ``_pert_vocab`` is ``None``.

        Args:
            train_symbols_in_step: Upper-case gene symbols present in this step.

        Returns:
            Scalar mean MSE tensor, or ``None`` if nothing to compute.
        """
        self._ensure_pert_vocab()
        if self._pert_vocab is None or self._model is None:
            return None

        inner = self._model
        state_model = inner.encoder.state.state_model
        if not hasattr(state_model, "pert_encoder"):
            return None

        device = next(inner.parameters()).device
        losses: list[torch.Tensor] = []
        for sym in train_symbols_in_step:
            onehot_arr = self._pert_vocab.get(sym.upper())
            if onehot_arr is None:
                continue
            esm_vec = self.esm.vectors_by_symbol.get(sym.upper())
            if esm_vec is None:
                continue
            onehot = torch.tensor(onehot_arr, device=device)
            esm_t = torch.tensor(esm_vec, device=device)
            adapter_tok = inner.encoder.adapter(esm_t.unsqueeze(0)).squeeze(0)
            target_tok = state_original_token(state_model, onehot)
            losses.append(
                distill_loss(adapter_tok.unsqueeze(0), target_tok.unsqueeze(0))
            )

        if not losses:
            return None
        return torch.stack(losses).mean()

    def produce(
        self,
        symbols: np.ndarray,
        train_symbols: set[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train on this fold then embed all universe genes.

        Genes without an ESM2 vector are zero-filled (ESM2 fallback) but still
        embedded with a zero vector for downstream consumers. The coverage mask
        reflects **gwps-bag coverage**: ``mask[i]=1`` iff ``symbols[i]`` is
        present in ``self.bags.bags_by_symbol``; ``0`` otherwise. This aligns
        with spec §7 honesty-check-1 (bag-coverage ~41%, not ESM2 ~100%).

        Args:
            symbols: Universe gene symbols in canonical order, shape
                ``(n_gene,)``.
            train_symbols: Set of upper-case gene symbols that appear in the
                training pairs for this fold.

        Returns:
            Tuple ``(embeddings, coverage_mask)`` where ``embeddings`` has
            shape ``(n_gene, emb_dim)`` (float32) and ``coverage_mask`` is an
            int array of shape ``(n_gene,)`` with ``1`` for bag-covered genes
            and ``0`` otherwise.
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
        n = len(symbols)
        embeddings = np.zeros((n, pooled_dim), dtype=np.float32)
        # Coverage mask: 1 iff gene has a gwps bag (bag coverage, not ESM2 coverage).
        coverage_mask = np.zeros(n, dtype=int)

        self._model.eval()
        e_table = torch.zeros((n, pooled_dim), device=device)
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
                e_table[row] = e_g
                embeddings[row] = e_g.cpu().numpy()
                # Bag coverage (not ESM2 coverage) — spec §7 honesty-check-1.
                if key in self.bags.bags_by_symbol:
                    coverage_mask[row] = 1

        # Cache the embedding table and symbol order for score_matrix reuse.
        self._e_table_cache = e_table
        self._e_table_symbols = symbols
        self._coverage_mask = coverage_mask

        return embeddings, coverage_mask

    def score_matrix(
        self,
        symbols: np.ndarray,
        gene_effects: np.ndarray,
    ) -> np.ndarray:
        """Score all candidate pairs with the trained pair head.

        Reuses the per-gene embedding table built during :meth:`produce` when
        ``symbols`` matches the cached order; otherwise recomputes it. If
        :meth:`produce` has never been called, triggers it first.

        Args:
            symbols: Universe gene symbols in canonical order, shape ``(n,)``.
            gene_effects: Per-gene GeneEffect scalar, shape ``(n,)``.

        Returns:
            Score matrix of shape ``(n, n)`` with values in [0, 1] (sigmoid
            output) and a zeroed diagonal.
        """
        if self._model is None:
            train_syms = {a.upper() for a, *_ in self.train_pairs} | {
                b.upper() for _, b, *_ in self.train_pairs
            }
            self.produce(symbols, train_syms)

        model = self._model
        assert model is not None
        device = next(model.parameters()).device

        # Reuse cached embedding table when symbols match, else recompute.
        if (
            self._e_table_cache is not None
            and self._e_table_symbols is not None
            and len(self._e_table_symbols) == len(symbols)
            and np.array_equal(self._e_table_symbols, symbols)
        ):
            e_table = self._e_table_cache.to(device)
            coverage_mask = self._coverage_mask
        else:
            control = torch.tensor(self.bags.control_template, device=device)
            n = len(symbols)
            e_table = torch.zeros((n, model.emb_dim), device=device)
            coverage_mask_list = []
            model.eval()
            with torch.no_grad():
                for i, symbol in enumerate(symbols):
                    key = str(symbol).upper()
                    vec = self.esm.vectors_by_symbol.get(key)
                    cov = 1 if key in self.bags.bags_by_symbol else 0
                    coverage_mask_list.append(cov)
                    if vec is not None:
                        e_table[i] = model.embed_gene(
                            torch.tensor(vec, device=device), control
                        )
            coverage_mask = np.array(coverage_mask_list, dtype=int)

        n = len(symbols)
        cov_tensor = torch.tensor(
            coverage_mask if coverage_mask is not None else np.zeros(n, dtype=int),
            device=device,
            dtype=torch.float32,
        )

        score = np.zeros((n, n), dtype=float)
        model.eval()
        with torch.no_grad():
            for i in range(n):
                ea = np.full(n, gene_effects[i])
                eb = gene_effects
                ge = torch.tensor(
                    build_pair_features(ea, eb),
                    device=device,
                    dtype=torch.float32,
                )
                e_a = e_table[i].unsqueeze(0).expand(n, -1)
                cov_a: torch.Tensor | None = None
                cov_b: torch.Tensor | None = None
                if self.config.include_coverage_flag:
                    cov_a = cov_tensor[i].expand(n)
                    cov_b = cov_tensor
                logits = model.score_pairs(e_a, e_table, ge, cov_a, cov_b)
                score[i] = torch.sigmoid(logits).cpu().numpy()

        np.fill_diagonal(score, 0.0)
        return score

    def _train(
        self,
        model: SlDlModel,
        optimizer: optim.Optimizer,
        accelerator: Accelerator,
        train_symbols: set[str],
    ) -> None:
        """Run the training loop over all epochs.

        Bag supervision is applied only for genes in
        ``train_symbols ∩ bags.bags_by_symbol`` (leakage rule). Distill loss
        is applied when weights["distill"] > 0 and pert_vocab is available.

        Args:
            model: Model prepared by Accelerate (possibly wrapped).
            optimizer: Optimizer prepared by Accelerate.
            accelerator: The Accelerator instance for backward + device.
            train_symbols: Upper-case gene symbols present in train pairs.

        Raises:
            RuntimeError: If all pairs in an epoch are skipped due to missing
                ESM2 vectors.
        """
        device = accelerator.device
        control = torch.tensor(self.bags.control_template, device=device)
        covered_train = {
            s.upper() for s in train_symbols if s.upper() in self.bags.bags_by_symbol
        }
        inner = accelerator.unwrap_model(model)
        self._model = inner  # Expose unwrapped model for _distill_part

        for epoch in range(self.config.max_epochs):
            weights = _epoch_weights(epoch, self.config)
            model.train()
            pbar = tqdm(
                self.train_pairs,
                desc=f"epoch {epoch}",
                disable=not accelerator.is_main_process,
            )
            skipped = 0
            trained = 0

            for a, b, label, ea, eb in pbar:
                key_a, key_b = a.upper(), b.upper()
                vec_a = self.esm.vectors_by_symbol.get(key_a)
                vec_b = self.esm.vectors_by_symbol.get(key_b)
                if vec_a is None or vec_b is None:
                    skipped += 1
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

                # Coverage flags for the pair head.
                cov_a: torch.Tensor | None = None
                cov_b: torch.Tensor | None = None
                if self.config.include_coverage_flag:
                    cov_a = torch.tensor(
                        [1.0 if key_a in self.bags.bags_by_symbol else 0.0],
                        device=device,
                    )
                    cov_b = torch.tensor(
                        [1.0 if key_b in self.bags.bags_by_symbol else 0.0],
                        device=device,
                    )

                logit = inner.score_pairs(
                    e_a.unsqueeze(0),
                    e_b.unsqueeze(0),
                    ge,
                    cov_a,
                    cov_b,
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

                # Distill supervision: adapter vs STATE original token.
                if weights["distill"] > 0:
                    distill_part = self._distill_part({key_a, key_b})
                    if distill_part is not None:
                        parts["distill"] = distill_part

                total = combine(parts, weights)
                optimizer.zero_grad()
                accelerator.backward(total)
                optimizer.step()
                trained += 1

            if skipped > 0:
                logger.warning(
                    "epoch %d: skipped %d pair(s) with missing ESM2 vector(s)",
                    epoch,
                    skipped,
                )
            if trained == 0:
                logger.error(
                    "epoch %d: no trainable pairs found; check ESM2 coverage", epoch
                )
                raise RuntimeError("no trainable pairs: check ESM2 coverage")


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
