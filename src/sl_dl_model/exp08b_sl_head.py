"""Step 2 SL-head trainer for cached embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from torch import optim

from sl_benchmark_baseline.features import Standardizer, build_pair_features
from sl_dl_model.exp08b_artifacts import load_embedding_cache
from sl_dl_model.exp08b_config import SlHeadConfig
from sl_dl_model.losses import sl_bce_loss
from sl_dl_model.pair_head import SymmetricPairHead


TrainPair = tuple[str, str, int, float, float]


class CachedEmbeddingPairHeadProducer:
    """Train a symmetric pair head over a fold-local cached embedding table."""

    def __init__(
        self,
        config: SlHeadConfig,
        *,
        cache_path: Path,
        train_pairs: list[TrainPair],
        metric_model_name: str,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.cache_path = Path(cache_path)
        self.train_pairs = list(train_pairs)
        self.metric_model_name = metric_model_name
        self._device = device
        self._symbols: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        self._coverage_mask: np.ndarray | None = None
        self._index_by_symbol: dict[str, int] = {}
        self._head: SymmetricPairHead | None = None
        self._ge_standardizer: Standardizer | None = None

    def produce(
        self, symbols: np.ndarray, train_symbols: set[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load cached embeddings aligned to ``symbols`` and fit the pair head."""
        _ = train_symbols
        cache = load_embedding_cache(self.cache_path)
        cache_symbols = [str(symbol).upper() for symbol in cache["symbols"]]
        cache_index = {symbol: idx for idx, symbol in enumerate(cache_symbols)}
        wanted = [str(symbol).upper() for symbol in symbols]
        missing = [symbol for symbol in wanted if symbol not in cache_index]
        if missing:
            raise ValueError(f"missing cached genes: {missing}")

        row_idx = np.array([cache_index[symbol] for symbol in wanted], dtype=int)
        self._symbols = np.asarray(symbols, dtype=object)
        self._embeddings = np.asarray(cache["embeddings"][row_idx], dtype=np.float32)
        self._coverage_mask = np.asarray(
            cache["coverage_mask"][row_idx], dtype=np.int64
        )
        self._index_by_symbol = {symbol: idx for idx, symbol in enumerate(wanted)}
        self._train_head()
        return self._embeddings, self._coverage_mask

    def score_matrix(self, symbols: np.ndarray, gene_effects: np.ndarray) -> np.ndarray:
        """Score every pair in ``symbols`` with the trained pair head."""
        if self._head is None:
            self.produce(symbols, set())
        if (
            self._head is None
            or self._embeddings is None
            or self._coverage_mask is None
        ):
            raise RuntimeError("pair head is not trained")

        wanted = [str(symbol).upper() for symbol in symbols]
        missing = [symbol for symbol in wanted if symbol not in self._index_by_symbol]
        if missing:
            raise ValueError(f"missing cached genes: {missing}")

        row_idx = np.array(
            [self._index_by_symbol[symbol] for symbol in wanted], dtype=int
        )
        embeddings = self._embeddings[row_idx]
        coverage = self._coverage_mask[row_idx]
        effects = np.asarray(gene_effects, dtype=float)
        device = self._model_device()
        n = len(wanted)
        scores = np.zeros((n, n), dtype=np.float32)
        self._head.eval()
        with torch.no_grad():
            for start in range(n):
                a_idx = np.full(n, start, dtype=int)
                b_idx = np.arange(n, dtype=int)
                logits = self._logits_for_indices(
                    embeddings,
                    coverage,
                    effects,
                    a_idx,
                    b_idx,
                    device,
                )
                scores[start] = torch.sigmoid(logits).cpu().numpy()
        scores = (scores + scores.T) / 2.0
        np.fill_diagonal(scores, 0.0)
        return scores

    def _train_head(self) -> None:
        if self._embeddings is None or self._coverage_mask is None:
            raise RuntimeError("cached embeddings are not loaded")
        device = self._model_device()
        emb_dim = int(self._embeddings.shape[1])
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(int(self.config.seed))
            self._head = SymmetricPairHead(
                emb_dim=emb_dim,
                hidden=tuple(self.config.pair_hidden),
                include_coverage_flag=self.config.include_coverage_flag,
            ).to(device)
        finally:
            torch.random.set_rng_state(rng_state)
        self._fit_ge_standardizer()
        optimizer = optim.AdamW(self._head.parameters(), lr=float(self.config.lr))
        labels = torch.tensor(
            [label for _a, _b, label, _ea, _eb in self.train_pairs],
            dtype=torch.float32,
            device=device,
        )
        pair_count = len(self.train_pairs)
        batch_size = max(1, int(self.config.batch_pairs))
        for _epoch in range(int(self.config.max_epochs)):
            self._head.train()
            for start in range(0, pair_count, batch_size):
                stop = min(pair_count, start + batch_size)
                batch = self.train_pairs[start:stop]
                a_idx, b_idx = self._pair_indices(batch)
                ea, eb = self._pair_effects(batch)
                logits = self._logits_for_arrays(
                    self._embeddings,
                    self._coverage_mask,
                    ea,
                    eb,
                    a_idx,
                    b_idx,
                    device,
                )
                loss = sl_bce_loss(logits, labels[start:stop])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._head.parameters(), float(self.config.max_grad_norm)
                )
                optimizer.step()

    def _fit_ge_standardizer(self) -> None:
        if not self.train_pairs:
            self._ge_standardizer = None
            return
        ea = np.array(
            [gene_effect_a for *_prefix, gene_effect_a, _b in self.train_pairs]
        )
        eb = np.array(
            [gene_effect_b for *_prefix, _a, gene_effect_b in self.train_pairs]
        )
        self._ge_standardizer = Standardizer.fit(build_pair_features(ea, eb))

    def _pair_indices(self, pairs: list[TrainPair]) -> tuple[np.ndarray, np.ndarray]:
        a_idx = np.array(
            [self._index_by_symbol[gene_a.upper()] for gene_a, *_rest in pairs],
            dtype=int,
        )
        b_idx = np.array(
            [
                self._index_by_symbol[gene_b.upper()]
                for _gene_a, gene_b, *_rest in pairs
            ],
            dtype=int,
        )
        return a_idx, b_idx

    def _pair_effects(self, pairs: list[TrainPair]) -> tuple[np.ndarray, np.ndarray]:
        ea = np.array(
            [gene_effect_a for *_prefix, gene_effect_a, _b in pairs],
            dtype=float,
        )
        eb = np.array(
            [gene_effect_b for *_prefix, _a, gene_effect_b in pairs],
            dtype=float,
        )
        return ea, eb

    def _logits_for_indices(
        self,
        embeddings: np.ndarray,
        coverage: np.ndarray,
        effects: np.ndarray,
        a_idx: np.ndarray,
        b_idx: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        ea = effects[a_idx]
        eb = effects[b_idx]
        return self._logits_for_arrays(
            embeddings, coverage, ea, eb, a_idx, b_idx, device
        )

    def _logits_for_arrays(
        self,
        embeddings: np.ndarray,
        coverage: np.ndarray,
        ea: np.ndarray,
        eb: np.ndarray,
        a_idx: np.ndarray,
        b_idx: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        if self._head is None:
            raise RuntimeError("pair head is not trained")
        ge = build_pair_features(ea, eb)
        if self._ge_standardizer is not None:
            ge = self._ge_standardizer.transform(ge)
        e_a = torch.as_tensor(embeddings[a_idx], dtype=torch.float32, device=device)
        e_b = torch.as_tensor(embeddings[b_idx], dtype=torch.float32, device=device)
        ge_features = torch.as_tensor(ge, dtype=torch.float32, device=device)
        if not self.config.include_coverage_flag:
            return self._head(e_a, e_b, ge_features)
        cov_a = torch.as_tensor(coverage[a_idx], dtype=torch.float32, device=device)
        cov_b = torch.as_tensor(coverage[b_idx], dtype=torch.float32, device=device)
        return self._head(e_a, e_b, ge_features, cov_a, cov_b)

    def _model_device(self) -> torch.device:
        if self._device is not None:
            return torch.device(self._device)
        return PartialState().device
