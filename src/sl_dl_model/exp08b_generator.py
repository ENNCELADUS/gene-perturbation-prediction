"""Exp08b Step-1 generator: leakage-safe split and bag-loss scale helpers.

This module is used by Task 2 (split + scale) and will be extended by Task 3
(Step1GeneratorTrainer).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import PartialState
from torch import nn, optim

from sl_dl_model.bags import GwpsBags
from sl_dl_model.encoder import StateEncoder, state_encoded_token, state_original_token
from sl_dl_model.exp08b_artifacts import (
    embedding_cache_path,
    generator_monitor_path,
    generator_manifest_path,
    generator_weights_path,
    save_embedding_cache,
    write_generator_manifest,
)
from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.losses import _safe_energy_distance, bag_loss, distill_loss
from sl_dl_model.pert_vocab import load_pert_vocab
from sl_dl_model.pooling import MeanStdPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leakage-safe generator validation split
# ---------------------------------------------------------------------------


def select_generator_bag_sets(
    *,
    train_symbols: set[str],
    covered_symbols: set[str],
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split bag-covered train symbols into train-bag and val-bag sets.

    Only symbols that appear in **both** ``train_symbols`` and
    ``covered_symbols`` are eligible.  This prevents:

    - test-split leakage (symbols only in the test split of the SL fold
      cannot appear in the generator val set)
    - bag-coverage leakage (symbols without a real perturbation bag in the
      GWPS h5ad cannot receive bag supervision)

    The split is deterministic for a given ``seed``.

    Args:
        train_symbols: Upper-case gene symbols in this fold's train SL pairs.
        covered_symbols: Upper-case symbols that have a real bag in GwpsBags.
        val_fraction: Fraction of eligible symbols to allocate to validation.
            Rounded to the nearest integer; minimum 0.
        seed: Random seed for reproducibility.

    Returns:
        ``(train_bag, val_bag)`` — disjoint sets that together equal
        ``train_symbols & covered_symbols``.
    """
    eligible = sorted(
        {s.upper() for s in train_symbols} & {s.upper() for s in covered_symbols}
    )
    if not eligible:
        return set(), set()
    # Reserve at least 1 for train_bag.  For tiny folds, a positive validation
    # fraction should still hold out one validation bag when possible.
    n_val = max(0, math.floor((len(eligible) - 1) * val_fraction))
    if val_fraction > 0 and len(eligible) > 1:
        n_val = max(1, n_val)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eligible))
    val_indices = set(idx[:n_val].tolist())
    train_bag: set[str] = set()
    val_bag: set[str] = set()
    for i, sym in enumerate(eligible):
        if i in val_indices:
            val_bag.add(sym)
        else:
            train_bag.add(sym)
    return train_bag, val_bag


# ---------------------------------------------------------------------------
# Bag-loss scale normalizers
# ---------------------------------------------------------------------------


class FixedWarmupBagScale:
    """Median bag-loss scale chosen from detached warmup observations."""

    def __init__(self, *, min_scale: float) -> None:
        self.min_scale = float(min_scale)
        self._observed: list[float] = []
        self.value: float | None = None

    @property
    def ready(self) -> bool:
        """Return whether a fixed scale has been selected."""
        return self.value is not None

    def observe(self, loss: torch.Tensor) -> None:
        """Record one detached bag-loss value."""
        self._observed.append(float(loss.detach().cpu()))

    def finalize(self) -> float:
        """Select median observed scale, clamped to ``min_scale``."""
        if not self._observed:
            raise ValueError("no bag losses observed during warmup")
        finite = [x for x in self._observed if np.isfinite(x)]
        if not finite:
            raise ValueError("no finite bag losses observed during warmup")
        median = float(np.median(np.asarray(finite, dtype=float)))
        self.value = max(median, self.min_scale)
        logger.debug("FixedWarmupBagScale finalized to %.4g", self.value)
        return self.value

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Normalize a bag loss by the selected fixed scale."""
        if self.value is None:
            raise RuntimeError("bag scale has not been finalized")
        return loss / float(self.value)


class EmaBagScale:
    """EMA-normalized bag-loss scale for the normalization ablation."""

    def __init__(self, *, min_scale: float, decay: float) -> None:
        self.min_scale = float(min_scale)
        self.decay = float(decay)
        self.value: float | None = None

    @property
    def ready(self) -> bool:
        """Return whether at least one finite scale has been observed."""
        return self.value is not None

    def observe(self, loss: torch.Tensor) -> None:
        """Update the EMA scale from one detached bag-loss value."""
        current = max(float(loss.detach().cpu()), self.min_scale)
        if not np.isfinite(current):
            return
        if self.value is None:
            self.value = current
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * current
        self.value = max(float(self.value), self.min_scale)

    def finalize(self) -> float:
        """Return the current EMA scale."""
        if self.value is None:
            raise ValueError("no finite bag losses observed for EMA scale")
        return self.value

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Normalize a bag loss by the current EMA scale."""
        if self.value is None:
            raise RuntimeError("bag scale has not been initialized")
        return loss / float(self.value)


def build_bag_scale(config: Exp08bConfig) -> FixedWarmupBagScale | EmaBagScale:
    """Build the configured bag-loss normalizer."""
    if config.bag_scale_mode == "fixed_warmup":
        return FixedWarmupBagScale(min_scale=config.bag_scale_min)
    if config.bag_scale_mode == "ema":
        return EmaBagScale(
            min_scale=config.bag_scale_min,
            decay=config.bag_scale_ema_decay,
        )

    raise ValueError(f"unknown bag_scale_mode: {config.bag_scale_mode!r}")


def _mean_std_pool_np(bag: np.ndarray) -> np.ndarray:
    """Mean/std-pool a bag into the Step 1 embedding shape."""
    arr = np.asarray(bag, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)]).astype(np.float32)


def pooled_vector_metrics(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """Return direction and magnitude metrics for two pooled vectors."""
    pred_arr = np.asarray(pred, dtype=np.float32)
    real_arr = np.asarray(real, dtype=np.float32)
    denom = float(np.linalg.norm(pred_arr) * np.linalg.norm(real_arr))
    cosine = float(np.dot(pred_arr, real_arr) / denom) if denom > 0 else 0.0
    delta = pred_arr - real_arr
    return {
        "pooled_cosine": cosine,
        "pooled_mse": float(np.mean(delta * delta)),
        "pooled_l2": float(np.linalg.norm(delta)),
    }


def bag_energy_metric(pred_bag: np.ndarray, real_bag: np.ndarray) -> float:
    """Return the NaN-safe energy distance between two bags."""
    pred = torch.tensor(np.asarray(pred_bag, dtype=np.float32), dtype=torch.float32)
    real = torch.tensor(np.asarray(real_bag, dtype=np.float32), dtype=torch.float32)
    return float(_safe_energy_distance(pred, real).detach().cpu())


def nearest_neighbor_copy_predictions(
    *,
    val_symbols: set[str],
    train_covered_symbols: set[str],
    esm_vectors: dict[str, np.ndarray],
    real_bags: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Copy each validation gene's nearest train-covered real bag in ESM2 space."""
    candidates = [
        symbol
        for symbol in sorted({s.upper() for s in train_covered_symbols})
        if symbol in esm_vectors and symbol in real_bags
    ]
    if not candidates:
        return {}

    copied: dict[str, np.ndarray] = {}
    for symbol in sorted({s.upper() for s in val_symbols}):
        val_vec = esm_vectors.get(symbol)
        if val_vec is None:
            continue
        val_arr = np.asarray(val_vec, dtype=np.float32)
        best = min(
            candidates,
            key=lambda candidate: float(
                np.sum(
                    (val_arr - np.asarray(esm_vectors[candidate], dtype=np.float32))
                    ** 2
                )
            ),
        )
        copied[symbol] = np.asarray(real_bags[best], dtype=np.float32).copy()
    return copied


def compute_monitor_rows(
    *,
    epoch: int,
    split_type: str,
    fold_id: int,
    pred_bags: dict[str, np.ndarray],
    real_bags: dict[str, np.ndarray],
    nn_copy_bags: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """Compute per-epoch generator and ESM2 NN-copy monitor rows."""

    def summarize(predictor: str, bags: dict[str, np.ndarray]) -> dict[str, object]:
        symbols = sorted(set(bags) & set(real_bags))
        pooled = [
            pooled_vector_metrics(
                _mean_std_pool_np(bags[symbol]),
                _mean_std_pool_np(real_bags[symbol]),
            )
            for symbol in symbols
        ]
        energy = [
            bag_energy_metric(bags[symbol], real_bags[symbol]) for symbol in symbols
        ]
        return {
            "split_type": split_type,
            "fold_id": int(fold_id),
            "epoch": int(epoch),
            "predictor": predictor,
            "n_genes": len(symbols),
            "pooled_cosine": float(np.mean([row["pooled_cosine"] for row in pooled]))
            if pooled
            else np.nan,
            "pooled_mse": float(np.mean([row["pooled_mse"] for row in pooled]))
            if pooled
            else np.nan,
            "pooled_l2": float(np.mean([row["pooled_l2"] for row in pooled]))
            if pooled
            else np.nan,
            "bag_energy": float(np.mean(energy)) if energy else np.nan,
        }

    return [
        summarize("generator", pred_bags),
        summarize("esm2_nn_copy", nn_copy_bags),
    ]


@dataclass(frozen=True)
class Step1TrainResult:
    """Paths and summary counts emitted by one Step 1 generator fold."""

    embedding_path: Path
    manifest_path: Path
    weights_path: Path
    bag_scale: float
    train_bag_gene_count: int
    val_bag_gene_count: int


class StateAdapterBagGenerator(nn.Module):
    """Step 1 generator: ESM2 adapter through STATE, pooled per gene."""

    def __init__(
        self,
        config: Exp08bConfig,
        *,
        esm_dim: int,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        checkpoint = (
            None if config.state_backend == "linear_mock" else config.state_checkpoint
        )
        self.encoder = StateEncoder(
            backend=config.state_backend,
            checkpoint=checkpoint,
            esm_dim=esm_dim,
            adapter_hidden=config.adapter_hidden,
            pert_dim=config.pert_dim,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.pool = MeanStdPool()

    def forward(self, esm_vec: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict a response bag for one perturbation gene."""
        return self.encoder(esm_vec, control)

    def pooled(self, esm_vec: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict and mean/std-pool one perturbation gene."""
        return self.pool(self.forward(esm_vec, control))


class DirectMlpBagGenerator(nn.Module):
    """Step 1 control generator: direct ESM2 MLP delta over control cells."""

    def __init__(self, *, esm_dim: int, hidden: int, output_dim: int) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )
        self.pool = MeanStdPool()

    def forward(self, esm_vec: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict a response bag by broadcasting one ESM2-conditioned delta."""
        return control + self.delta(esm_vec)

    def pooled(self, esm_vec: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict and mean/std-pool one perturbation gene."""
        return self.pool(self.forward(esm_vec, control))


class Step1GeneratorTrainer:
    """Train and cache fold-local exp08b Step 1 generator embeddings."""

    def __init__(
        self,
        config: Exp08bConfig,
        *,
        esm: Esm2EmbeddingTable,
        bags: GwpsBags,
        input_dim: int,
        output_dim: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.esm = esm
        self.bags = bags
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self._device = torch.device(device) if device is not None else None
        self._pert_vocab: dict[str, np.ndarray] | None = None
        self._pert_vocab_loaded = False
        self._esm_fallback_cache: np.ndarray | None = None

    def train_fold(
        self,
        *,
        split_type: str,
        fold_id: int,
        symbols: np.ndarray,
        train_symbols: set[str],
    ) -> Step1TrainResult:
        """Train the Step 1 generator and write fold-local artifacts."""
        monitor_path = generator_monitor_path(self.config, split_type, fold_id)
        if monitor_path.exists():
            monitor_path.unlink()

        fold_train = {str(symbol).upper() for symbol in train_symbols}
        train_bag, val_bag = select_generator_bag_sets(
            train_symbols=fold_train,
            covered_symbols=set(self.bags.bags_by_symbol),
            val_fraction=self.config.generator_val_fraction,
            seed=self.config.generator_val_seed + int(fold_id),
        )
        if self.config.generator_kind == "nn_copy":
            return self._produce_nn_copy_fold(
                split_type=split_type,
                fold_id=fold_id,
                symbols=symbols,
                train_covered=train_bag | val_bag,
                val_bag=val_bag,
            )

        distill_symbols = self.distill_symbols_for_fold(fold_train)

        torch.manual_seed(int(self.config.seed) + int(fold_id))
        device = self._device if self._device is not None else PartialState().device
        generator = self._build_generator().to(device)
        params = [p for p in generator.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=float(self.config.lr))
        control = torch.tensor(
            self.bags.control_template,
            dtype=torch.float32,
            device=device,
        )
        scale = build_bag_scale(self.config)
        scale_observed = False
        produced_loss = False

        for epoch in range(int(self.config.max_epochs)):
            generator.train()
            distill_weight = (
                float(self.config.lambda_distill)
                if epoch < int(self.config.warmup_epochs)
                else float(self.config.lambda_distill_after_warmup)
            )
            for symbol in sorted(train_bag | distill_symbols):
                parts: list[torch.Tensor] = []
                esm_vec = self._esm_tensor(symbol, device)

                if float(self.config.lambda_bag) > 0 and symbol in train_bag:
                    pred = generator(esm_vec, control)
                    real = torch.tensor(
                        self.bags.bags_by_symbol[symbol],
                        dtype=torch.float32,
                        device=device,
                    )
                    raw_bag = bag_loss(pred, real)
                    scale.observe(raw_bag)
                    scale_observed = True
                    if scale.ready:
                        raw_bag = scale.normalize(raw_bag)
                    parts.append(float(self.config.lambda_bag) * raw_bag)

                if distill_weight > 0 and symbol in distill_symbols:
                    term = self._distill_term(generator, symbol, device)
                    if term is None:
                        raise RuntimeError(
                            "distill loss is configured but no distill term "
                            f"could be constructed for {symbol}"
                        )
                    parts.append(distill_weight * term)

                if not parts:
                    continue

                total = torch.stack(parts).sum()
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if float(self.config.max_grad_norm) > 0:
                    nn.utils.clip_grad_norm_(params, float(self.config.max_grad_norm))
                optimizer.step()
                produced_loss = True

            if val_bag:
                pred_bags = self._predict_bags(generator, control, val_bag, device)
                real_bags = {
                    symbol: np.asarray(
                        self.bags.bags_by_symbol[symbol],
                        dtype=np.float32,
                    )
                    for symbol in sorted(val_bag)
                }
                esm_vectors = self._esm_vectors_for(train_bag | val_bag)
                nn_copy_bags = nearest_neighbor_copy_predictions(
                    val_symbols=val_bag,
                    train_covered_symbols=train_bag,
                    esm_vectors=esm_vectors,
                    real_bags=self.bags.bags_by_symbol,
                )
                rows = compute_monitor_rows(
                    epoch=epoch,
                    split_type=split_type,
                    fold_id=fold_id,
                    pred_bags=pred_bags,
                    real_bags=real_bags,
                    nn_copy_bags=nn_copy_bags,
                )
                self._append_monitor_rows(
                    split_type=split_type,
                    fold_id=fold_id,
                    rows=rows,
                )

            if (
                isinstance(scale, FixedWarmupBagScale)
                and scale_observed
                and not scale.ready
                and epoch + 1 >= int(self.config.warmup_epochs)
            ):
                scale.finalize()

        if not produced_loss and (train_bag or distill_symbols):
            raise RuntimeError("Step 1 produced no trainable generator losses")
        if scale_observed and not scale.ready:
            scale.finalize()
        final_scale = (
            float(scale.value) if scale.ready and scale.value is not None else 1.0
        )

        embedding_path = embedding_cache_path(self.config, split_type, fold_id)
        manifest_path = generator_manifest_path(self.config, split_type, fold_id)
        weights_path = generator_weights_path(self.config, split_type, fold_id)

        embeddings, coverage = self._embed_universe(generator, symbols, control)
        save_embedding_cache(
            embedding_path,
            symbols=np.asarray(symbols, dtype=object),
            embeddings=embeddings,
            coverage_mask=coverage,
            embedding_method=self.config.embedding_method,
        )
        self._save_weights(generator, weights_path)
        write_generator_manifest(
            manifest_path,
            {
                "split_type": split_type,
                "fold_id": int(fold_id),
                "generator_kind": self.config.generator_kind,
                "embedding_method": self.config.embedding_method,
                "bag_scale": final_scale,
                "bag_scale_mode": self.config.bag_scale_mode,
                "generator_weights_path": str(weights_path),
                "train_bag_gene_count": len(train_bag),
                "val_bag_gene_count": len(val_bag),
                "distill_gene_count": len(distill_symbols),
                "universe_gene_count": int(len(symbols)),
            },
        )
        return Step1TrainResult(
            embedding_path=embedding_path,
            manifest_path=manifest_path,
            weights_path=weights_path,
            bag_scale=final_scale,
            train_bag_gene_count=len(train_bag),
            val_bag_gene_count=len(val_bag),
        )

    def distill_symbols_for_fold(self, train_symbols: set[str]) -> set[str]:
        """Return fold-train symbols that exist in the STATE pert vocab."""
        vocab = self._load_distill_vocab()
        return {
            str(symbol).upper()
            for symbol in train_symbols
            if str(symbol).upper() in vocab
        }

    def _load_distill_vocab(self) -> dict[str, np.ndarray]:
        if self._pert_vocab is not None:
            return self._pert_vocab
        if self._pert_vocab_loaded:
            return {}

        self._pert_vocab_loaded = True
        distill_requested = (
            float(self.config.lambda_distill) > 0
            or float(self.config.lambda_distill_after_warmup) > 0
        )
        if self.config.state_backend == "linear_mock" or not distill_requested:
            self._pert_vocab = {}
            return self._pert_vocab

        try:
            vocab = load_pert_vocab(self.config.state_checkpoint)
        except Exception as exc:
            raise RuntimeError(
                "distill loss is configured but pert_onehot_map.pt could not be "
                "loaded for the STATE checkpoint"
            ) from exc
        if vocab is None:
            raise RuntimeError(
                "distill loss is configured but pert_onehot_map.pt is missing "
                "next to the STATE checkpoint"
            )
        self._pert_vocab = vocab
        return self._pert_vocab

    def _build_generator(self) -> nn.Module:
        if self.config.generator_kind == "state_adapter":
            return StateAdapterBagGenerator(
                self.config,
                esm_dim=self.esm.dim,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )
        if self.config.generator_kind == "direct_mlp":
            return DirectMlpBagGenerator(
                esm_dim=self.esm.dim,
                hidden=int(self.config.direct_mlp_hidden),
                output_dim=self.output_dim,
            )

        raise ValueError(f"unknown generator_kind: {self.config.generator_kind!r}")

    def _produce_nn_copy_fold(
        self,
        *,
        split_type: str,
        fold_id: int,
        symbols: np.ndarray,
        train_covered: set[str],
        val_bag: set[str],
    ) -> Step1TrainResult:
        embedding_path = embedding_cache_path(self.config, split_type, fold_id)
        manifest_path = generator_manifest_path(self.config, split_type, fold_id)
        weights_path = generator_weights_path(self.config, split_type, fold_id)
        universe = [str(symbol).upper() for symbol in symbols]
        esm_vectors = self._esm_vectors_for(set(universe) | train_covered)
        copied = nearest_neighbor_copy_predictions(
            val_symbols=set(universe),
            train_covered_symbols=train_covered,
            esm_vectors=esm_vectors,
            real_bags=self.bags.bags_by_symbol,
        )

        embeddings = np.zeros((len(symbols), 2 * self.output_dim), dtype=np.float32)
        coverage = np.zeros(len(symbols), dtype=np.int64)
        for row, symbol in enumerate(universe):
            copied_bag = copied.get(symbol)
            if copied_bag is not None:
                embeddings[row] = _mean_std_pool_np(copied_bag)
            if symbol in self.bags.bags_by_symbol:
                coverage[row] = 1

        save_embedding_cache(
            embedding_path,
            symbols=np.asarray(symbols, dtype=object),
            embeddings=embeddings,
            coverage_mask=coverage,
            embedding_method=self.config.embedding_method,
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = weights_path.with_suffix(weights_path.suffix + f".tmp.{os.getpid()}")
        torch.save({"generator_kind": "nn_copy", "state_dict": {}}, tmp)
        os.replace(tmp, weights_path)
        write_generator_manifest(
            manifest_path,
            {
                "split_type": split_type,
                "fold_id": int(fold_id),
                "generator_kind": self.config.generator_kind,
                "embedding_method": self.config.embedding_method,
                "bag_scale": 1.0,
                "bag_scale_mode": self.config.bag_scale_mode,
                "generator_weights_path": str(weights_path),
                "train_bag_gene_count": len(train_covered),
                "val_bag_gene_count": len(val_bag),
                "distill_gene_count": 0,
                "universe_gene_count": int(len(symbols)),
            },
        )
        return Step1TrainResult(
            embedding_path=embedding_path,
            manifest_path=manifest_path,
            weights_path=weights_path,
            bag_scale=1.0,
            train_bag_gene_count=len(train_covered),
            val_bag_gene_count=len(val_bag),
        )

    def _distill_term(
        self,
        generator: StateAdapterBagGenerator,
        symbol: str,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        vocab = self._load_distill_vocab()
        onehot_arr = vocab.get(symbol.upper())
        if onehot_arr is None:
            return None
        state_model = generator.encoder.state.state_model
        if not hasattr(state_model, "pert_encoder"):
            raise RuntimeError(
                "distill loss is configured but the STATE model has no pert_encoder"
            )
        onehot = torch.tensor(onehot_arr, dtype=torch.float32, device=device)
        esm_vec = self._esm_tensor(symbol, device)
        adapter_raw = generator.encoder.adapter(esm_vec.unsqueeze(0)).squeeze(0)
        adapter_tok = state_encoded_token(state_model, adapter_raw)
        target_tok = state_original_token(state_model, onehot)
        return distill_loss(adapter_tok.unsqueeze(0), target_tok.unsqueeze(0))

    def _embed_universe(
        self,
        generator: nn.Module,
        symbols: np.ndarray,
        control: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        generator.eval()
        embeddings = np.zeros((len(symbols), 2 * self.output_dim), dtype=np.float32)
        coverage = np.zeros(len(symbols), dtype=np.int64)
        device = control.device
        with torch.no_grad():
            for row, symbol in enumerate(symbols):
                key = str(symbol).upper()
                embeddings[row] = (
                    generator.pooled(self._esm_tensor(key, device), control)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                if key in self.bags.bags_by_symbol:
                    coverage[row] = 1
        return embeddings, coverage

    def _esm_vectors_for(self, symbols: set[str]) -> dict[str, np.ndarray]:
        """Resolve ESM2 vectors for a set of symbols."""
        return {
            str(symbol).upper(): self._resolve_esm(str(symbol).upper())
            for symbol in symbols
        }

    def _predict_bags(
        self,
        generator: nn.Module,
        control: torch.Tensor,
        symbols: set[str],
        device: torch.device,
    ) -> dict[str, np.ndarray]:
        """Predict response bags for a set of symbols."""
        generator.eval()
        pred_bags: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for symbol in sorted(symbols):
                key = str(symbol).upper()
                pred_bags[key] = (
                    generator(self._esm_tensor(key, device), control)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        return pred_bags

    def _append_monitor_rows(
        self,
        *,
        split_type: str,
        fold_id: int,
        rows: list[dict[str, object]],
    ) -> None:
        """Append monitor rows to the fold-local Step 1 CSV."""
        if not rows:
            return
        path = generator_monitor_path(self.config, split_type, fold_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)

    def _esm_tensor(self, symbol: str, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(
            self._resolve_esm(symbol),
            dtype=torch.float32,
            device=device,
        )

    def _resolve_esm(self, symbol: str) -> np.ndarray:
        vec = self.esm.vectors_by_symbol.get(symbol.upper())
        if vec is not None:
            return np.asarray(vec, dtype=np.float32)
        if self._esm_fallback_cache is None:
            vectors = list(self.esm.vectors_by_symbol.values())
            if self.config.fallback_strategy == "global_mean" and vectors:
                self._esm_fallback_cache = np.mean(
                    np.vstack(vectors),
                    axis=0,
                ).astype(np.float32)
            else:
                self._esm_fallback_cache = np.zeros(self.esm.dim, dtype=np.float32)
        return self._esm_fallback_cache

    def _save_weights(
        self,
        generator: nn.Module,
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(
            {
                "generator_kind": self.config.generator_kind,
                "state_dict": generator.state_dict(),
            },
            tmp,
        )
        os.replace(tmp, path)
