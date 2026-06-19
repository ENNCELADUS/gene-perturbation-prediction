"""Per-fold training loop and the StateDlProducer.

StateDlProducer trains the model on one fold's pairs and then embeds every
gene in the universe through the frozen STATE backbone to produce a per-gene
embedding table.

Training runs on a single device per fold: each gene is forwarded one at a
time through the frozen STATE backbone, and gradients update only the trainable
adapter/pooling/pair-head. There is no DDP gradient all-reduce — fold-level
parallelism (one fold per rank) is orchestrated in
:func:`sl_dl_model.evaluate.run_cv`, which assigns disjoint folds to each rank
and gathers metric rows on the main process. ``PartialState`` supplies the
device and rank info; it does not wrap the model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from sklearn.metrics import roc_auc_score
from torch import optim
from tqdm.auto import tqdm

from sl_benchmark_baseline.features import Standardizer, build_pair_features
from sl_dl_model.bags import GwpsBags
from sl_dl_model.config import SLDLConfig
from sl_dl_model.encoder import state_encoded_token, state_original_token
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
        vocab_path, map_location="cpu", weights_only=False
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
        val_pairs: list[tuple[str, str, int, float, float]] | None = None,
    ) -> None:
        self.config = config
        self.esm = esm
        self.bags = bags
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Best-epoch tracking (set by _train).
        self.stopped_epoch: int | None = None
        self.epoch_metrics: list[dict[str, float]] = []
        self._model: SlDlModel | None = None
        # Cached per-gene embedding table (shape n_gene × emb_dim) and symbol order.
        self._e_table_cache: torch.Tensor | None = None
        self._e_table_symbols: np.ndarray | None = None
        # Bag-coverage mask aligned to the last produce() call.
        self._coverage_mask: np.ndarray | None = None
        # Perturbation vocab (one-hots) for distill supervision.
        self._pert_vocab: dict[str, np.ndarray] | None = None
        self._pert_vocab_loaded: bool = False
        # Train-fold GeneEffect standardizer (fit in produce, applied everywhere).
        self._ge_standardizer: Standardizer | None = None
        # Cached ESM2 fallback vector (zero or global_mean per fallback_strategy).
        self._esm_fallback_cache: np.ndarray | None = None

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

        Raises:
            RuntimeError: If distill is requested (``lambda_distill`` or
                ``lambda_distill_after_warmup`` > 0) on a real STATE backend but
                the sibling ``pert_onehot_map.pt`` is missing or unreadable. A
                config that claims a 3-part loss must not silently drop the
                distill anchor (spec §6.2).
        """
        if self._pert_vocab_loaded:
            return
        self._pert_vocab_loaded = True

        if self.config.state_backend == "linear_mock":
            logger.debug("linear_mock backend: skipping distill vocab load")
            return

        distill_requested = (
            self.config.lambda_distill > 0
            or self.config.lambda_distill_after_warmup > 0
        )
        try:
            self._pert_vocab = _load_pert_vocab(self.config.state_checkpoint)
        except Exception as exc:
            if distill_requested:
                logger.error(
                    "distill requested (lambda_distill=%s) but pert_onehot_map.pt "
                    "could not be loaded for %s: %s",
                    self.config.lambda_distill,
                    self.config.state_checkpoint,
                    exc,
                )
                raise RuntimeError(
                    "distill loss is configured (lambda_distill>0) but the STATE "
                    "pert_onehot_map.pt could not be loaded; refusing to train "
                    "silently without the distill anchor"
                ) from exc
            logger.debug("failed to load pert_vocab; distill loss will be skipped")
            self._pert_vocab = None
            return

        if self._pert_vocab is None and distill_requested:
            logger.error(
                "distill requested (lambda_distill=%s) but pert_onehot_map.pt is "
                "missing next to %s",
                self.config.lambda_distill,
                self.config.state_checkpoint,
            )
            raise RuntimeError(
                "distill loss is configured (lambda_distill>0) but the STATE "
                "pert_onehot_map.pt is missing; refusing to train silently "
                "without the distill anchor (expected at "
                "<checkpoint>.parent.parent/pert_onehot_map.pt)"
            )

    def _esm_fallback_vector(self) -> np.ndarray:
        """Compute the ESM2 fallback vector for genes lacking a real embedding.

        ``"zero"`` returns a zero vector; ``"global_mean"`` returns the mean of
        all resolved ESM2 vectors (matching
        :func:`sl_dl_model.gene_embeddings.align_esm2_to_universe`). The result
        is cached on the instance.

        Returns:
            Fallback embedding of shape ``(esm_dim,)``, float32.
        """
        if self._esm_fallback_cache is not None:
            return self._esm_fallback_cache
        vectors = list(self.esm.vectors_by_symbol.values())
        if self.config.fallback_strategy == "global_mean" and vectors:
            fallback = np.mean(np.vstack(vectors), axis=0).astype(np.float32)
        else:
            fallback = np.zeros(self.esm.dim, dtype=np.float32)
        self._esm_fallback_cache = fallback
        return fallback

    def _resolve_esm(self, symbol: str) -> tuple[np.ndarray, bool]:
        """Resolve a gene's ESM2 vector, applying the configured fallback.

        Args:
            symbol: Upper-case gene symbol.

        Returns:
            Tuple ``(vector, is_real)`` where ``vector`` is the gene's ESM2
            embedding (or the fallback) and ``is_real`` is ``True`` only when a
            genuine precomputed vector was found.
        """
        vec = self.esm.vectors_by_symbol.get(symbol.upper())
        if vec is not None:
            return np.asarray(vec, dtype=np.float32), True
        return self._esm_fallback_vector(), False

    def _fit_ge_standardizer(self) -> None:
        """Fit the GeneEffect-feature standardizer on this fold's train pairs.

        Mirrors the sklearn path's train-fold standardization (spec §5): the
        5-dim swap-invariant GeneEffect block is standardized using statistics
        from training pairs only, then applied during training and scoring.
        """
        if not self.train_pairs:
            self._ge_standardizer = None
            return
        ea = np.array([ea for *_, ea, _eb in self.train_pairs], dtype=float)
        eb = np.array([eb for *_, _ea, eb in self.train_pairs], dtype=float)
        raw = build_pair_features(ea, eb)
        self._ge_standardizer = Standardizer.fit(raw)

    def _ge_features(self, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
        """Build standardized GeneEffect pair features.

        Args:
            ea: GeneEffect values for gene A, shape ``(n,)``.
            eb: GeneEffect values for gene B, shape ``(n,)``.

        Returns:
            Standardized feature matrix, shape ``(n, 5)``; raw features if no
            standardizer has been fit (e.g. empty train set).
        """
        raw = build_pair_features(ea, eb)
        if self._ge_standardizer is None:
            return raw
        return self._ge_standardizer.transform(raw)

    def _distill_part(
        self,
        train_symbols_in_step: set[str],
    ) -> torch.Tensor | None:
        """Compute MSE between encoded adapter tokens and STATE's one-hot tokens.

        Both sides pass through the checkpoint's frozen ``pert_encoder``: the
        adapter's raw pert vector is encoded (grad-carrying) and compared to the
        encoded in-vocab one-hot (detached teacher). Only genes that are both in
        ``train_symbols_in_step`` and the loaded pert-vocab contribute. Returns
        ``None`` if no in-vocab genes are found or if ``_pert_vocab`` is ``None``.

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
            adapter_raw = inner.encoder.adapter(esm_t.unsqueeze(0)).squeeze(0)
            adapter_tok = state_encoded_token(state_model, adapter_raw)
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
        # Fit the train-fold GeneEffect standardizer before training so it is
        # applied consistently in training and full-matrix scoring (spec §5).
        self._fit_ge_standardizer()
        state = PartialState()
        model = self._build_model()
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=self.config.lr,
        )
        model = model.to(state.device)
        self._train(model, optimizer, state, train_symbols)
        self._model = model

        device = state.device
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
                    disable=not state.is_main_process,
                )
            ):
                key = str(symbol).upper()
                # FIX 4: resolve through the configured ESM2 fallback so missing
                # genes are embedded (deterministically), not silently zeroed.
                vec, _is_real = self._resolve_esm(key)
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
                    cov = 1 if key in self.bags.bags_by_symbol else 0
                    coverage_mask_list.append(cov)
                    # FIX 4: resolve through fallback so missing-ESM genes embed.
                    vec, _is_real = self._resolve_esm(key)
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
                    self._ge_features(ea, eb),
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

    def _validate_auroc(
        self,
        model: SlDlModel,
        device: torch.device | str,
        control: torch.Tensor,
    ) -> float | None:
        """Pair-AUROC over ``self.val_pairs`` (the fold's test split).

        Returns ``None`` when validation is impossible: no val pairs, fewer
        than two scorable pairs, or only one label class present.
        """
        if not self.val_pairs:
            return None
        model.eval()
        scores: list[float] = []
        labels: list[int] = []
        with torch.no_grad():
            for a, b, label, ea, eb in self.val_pairs:
                vec_a, _real_a = self._resolve_esm(a)
                vec_b, _real_b = self._resolve_esm(b)
                e_a = model.embed_gene(torch.tensor(vec_a, device=device), control)
                e_b = model.embed_gene(torch.tensor(vec_b, device=device), control)
                ge = torch.tensor(
                    self._ge_features(np.array([ea]), np.array([eb])),
                    device=device,
                    dtype=torch.float32,
                )
                cov_a: torch.Tensor | None = None
                cov_b: torch.Tensor | None = None
                if self.config.include_coverage_flag:
                    cov_a = torch.tensor(
                        [1.0 if a.upper() in self.bags.bags_by_symbol else 0.0],
                        device=device,
                    )
                    cov_b = torch.tensor(
                        [1.0 if b.upper() in self.bags.bags_by_symbol else 0.0],
                        device=device,
                    )
                logit = model.score_pairs(
                    e_a.unsqueeze(0), e_b.unsqueeze(0), ge, cov_a, cov_b
                )
                scores.append(float(torch.sigmoid(logit).item()))
                labels.append(int(label))
        model.train()
        if len(scores) < 2 or len(set(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))

    def _train(
        self,
        model: SlDlModel,
        optimizer: optim.Optimizer,
        state: PartialState,
        train_symbols: set[str],
    ) -> None:
        """Run the training loop over all epochs.

        Bag supervision is applied only for genes in
        ``train_symbols ∩ bags.bags_by_symbol`` (leakage rule). Distill loss
        is applied when weights["distill"] > 0 and pert_vocab is available.

        Args:
            model: Model on the target device (no DDP wrap).
            optimizer: Optimizer for the trainable parameters.
            state: PartialState for device and rank info.
            train_symbols: Upper-case gene symbols present in train pairs.

        Raises:
            RuntimeError: If all pairs in an epoch are skipped due to missing
                ESM2 vectors.
        """
        device = state.device
        control = torch.tensor(self.bags.control_template, device=device)
        covered_train = {
            s.upper() for s in train_symbols if s.upper() in self.bags.bags_by_symbol
        }
        self._model = model  # Expose model for _distill_part

        best_auroc: float | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch: int | None = None
        epochs_since_improve = 0

        def _snapshot_trainable() -> dict[str, torch.Tensor]:
            return {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

        def _restore_trainable(state_dict: dict[str, torch.Tensor]) -> None:
            params = dict(model.named_parameters())
            with torch.no_grad():
                for name, saved in state_dict.items():
                    params[name].copy_(saved.to(device=params[name].device))

        for epoch in range(self.config.max_epochs):
            weights = _epoch_weights(epoch, self.config)
            model.train()
            n_batches = max(
                1,
                (len(self.train_pairs) + self.config.batch_pairs - 1)
                // self.config.batch_pairs,
            )
            pbar = tqdm(
                total=n_batches,
                desc=f"epoch {epoch}",
                disable=not state.is_main_process,
            )
            skipped = 0
            trained = 0
            batch_losses: list[torch.Tensor] = []
            batch_loss_sum = 0.0
            batch_loss_count = 0

            def _flush() -> None:
                nonlocal batch_loss_count, batch_loss_sum, batch_losses
                if not batch_losses:
                    return
                batch_total = torch.stack(batch_losses).mean()
                batch_size = len(batch_losses)
                optimizer.zero_grad()
                batch_total.backward()
                optimizer.step()
                batch_loss_sum += float(batch_total.detach().cpu()) * batch_size
                batch_loss_count += batch_size
                batch_losses = []
                pbar.update(1)

            for a, b, label, ea, eb in self.train_pairs:
                key_a, key_b = a.upper(), b.upper()
                vec_a = self.esm.vectors_by_symbol.get(key_a)
                vec_b = self.esm.vectors_by_symbol.get(key_b)
                if vec_a is None or vec_b is None:
                    skipped += 1
                    continue

                esm_a = torch.tensor(vec_a, device=device)
                esm_b = torch.tensor(vec_b, device=device)
                e_a = model.embed_gene(esm_a, control)
                e_b = model.embed_gene(esm_b, control)

                ge = torch.tensor(
                    self._ge_features(np.array([ea]), np.array([eb])),
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

                logit = model.score_pairs(
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
                        model,
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
                batch_losses.append(total)
                trained += 1
                if len(batch_losses) >= self.config.batch_pairs:
                    _flush()

            _flush()
            pbar.close()

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

            mean_loss = (
                batch_loss_sum / batch_loss_count
                if batch_loss_count > 0
                else float("nan")
            )
            val_auroc = self._validate_auroc(model, device, control)
            peak_mb = (
                torch.cuda.max_memory_allocated() / 1e6
                if torch.cuda.is_available()
                else 0.0
            )
            self.epoch_metrics.append(
                {
                    "epoch": float(epoch),
                    "mean_train_loss": mean_loss,
                    "val_pair_auroc": float("nan") if val_auroc is None else val_auroc,
                    "peak_gpu_mem_mb": peak_mb,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Best-epoch selection only after warmup (val signal meaningful).
            if val_auroc is not None and epoch >= self.config.warmup_epochs:
                if best_auroc is None or val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_state = _snapshot_trainable()
                    best_epoch = epoch
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.config.early_stop_patience:
                        logger.info(
                            "early stop at epoch %d (best epoch %d, val_auroc=%.4f)",
                            epoch,
                            best_epoch,
                            best_auroc,
                        )
                        break

        if best_state is not None:
            _restore_trainable(best_state)
            self.stopped_epoch = best_epoch
        else:
            # No val signal (val_pairs None/unusable): keep final-epoch weights.
            self.stopped_epoch = self.config.max_epochs - 1


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
