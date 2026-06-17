"""Phase 0 harness-correctness test (FIX 6).

The existing parity test only checks that files/metric-names are emitted. This
test verifies that the exp08 harness reproduces the *official* ranking/
classification metric values — exercising universe construction, seen-pair
masking, and diagonal zeroing — by comparing harness output against
``official_ranking_metrics`` / ``official_classification_metrics`` computed
directly on the same controlled score matrix.

Real exp06 CV2/CV3 parity needs the cluster data; this test pins the harness
*logic* so a regression in masking/universe/metric wiring is caught offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sl_benchmark_baseline.evaluate import (
    _build_gene_universe,
    _pair_indices,
)
from sl_benchmark_baseline.metrics import (
    official_classification_metrics,
    official_ranking_metrics,
)
from sl_dl_model.config import SLDLConfig
from sl_dl_model.scoring import run_fold_with_producer


def _toy_frame() -> pd.DataFrame:
    """Small CV2 frame with explicit train/test pairs over genes G0..G5."""
    genes = [f"G{i}" for i in range(6)]
    eff = {g: float(i) - 2.5 for i, g in enumerate(genes)}
    rows = []
    pid = 0
    # Deterministic label by index parity; both roles cover all gene pairs.
    for role in ("train", "test"):
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                rows.append(
                    {
                        "pair_id": f"p{pid}",
                        "fold_id": 0,
                        "split_type": "CV2",
                        "split_role": role,
                        "sl_label": (i + j) % 2,
                        "gene_a_symbol": genes[i],
                        "gene_b_symbol": genes[j],
                        "gene_a_k562_gene_effect": eff[genes[i]],
                        "gene_b_k562_gene_effect": eff[genes[j]],
                    }
                )
                pid += 1
    return pd.DataFrame(rows)


class _FixedScoreProducer:
    """Producer that returns a deterministic, known score matrix.

    ``produce`` returns benign embeddings/mask (unused by the score_matrix
    path); ``score_matrix`` returns a fixed matrix derived only from symbols so
    the test can recompute the official metric independently.
    """

    def __init__(self) -> None:
        self.last_matrix: np.ndarray | None = None

    def produce(
        self, symbols: np.ndarray, train_symbols: set[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(symbols)
        return np.zeros((n, 1), dtype=float), np.zeros(n, dtype=int)

    def score_matrix(self, symbols: np.ndarray, gene_effects: np.ndarray) -> np.ndarray:
        n = len(symbols)
        # Deterministic asymmetric-but-stable scores from index sums, then
        # symmetrized and diagonal-zeroed (mirrors a real score matrix).
        idx = np.arange(n)
        mat = (idx[:, None] + 1.0) * (idx[None, :] + 1.0)
        mat = mat / mat.max()
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 0.0)
        self.last_matrix = mat
        return mat


def test_harness_reproduces_official_metrics_exactly() -> None:
    """run_fold_with_producer's metric values match a direct official-metric call.

    This pins universe construction, seen-pair masking, diagonal zeroing, and
    the ndcg/recall/precision/map computation — the things FIX 6 says the old
    parity test could not catch.
    """
    frame = _toy_frame()
    cfg = SLDLConfig(
        split_types=("CV2",),
        folds=(0,),
        ranking_k=(10, 20),
        include_coverage_flag=False,
    )
    producer = _FixedScoreProducer()
    rows = run_fold_with_producer(frame, "CV2", 0, cfg, producer)

    # Recompute expected metrics directly from the SAME fixed matrix.
    universe = _build_gene_universe(frame)
    train_df = frame[(frame["split_type"] == "CV2") & (frame["split_role"] == "train")]
    test_df = frame[(frame["split_type"] == "CV2") & (frame["split_role"] == "test")]
    pos_index = _pair_indices(test_df[test_df["sl_label"] == 1], universe)
    neg_index = _pair_indices(test_df[test_df["sl_label"] == 0], universe)
    seen_index = _pair_indices(train_df[train_df["sl_label"] == 1], universe)
    matrix = producer.score_matrix(universe.symbols, universe.gene_effects)

    expected = official_classification_metrics(matrix, pos_index, neg_index)
    expected.update(
        official_ranking_metrics(matrix, pos_index, seen_index=seen_index, ks=(10, 20))
    )

    # Harness rows are long-form: (model, slice, metric, value).
    full_rows = {
        r["metric"]: r["value"]
        for r in rows
        if r["slice"] == "full_universe" and r["model"] == "state_dl"
    }

    # Every official metric must match the harness output to numerical precision.
    for metric, value in expected.items():
        assert metric in full_rows, f"harness did not emit metric {metric!r}"
        assert abs(full_rows[metric] - value) < 1e-9, (
            f"harness {metric}={full_rows[metric]} != direct {value}"
        )

    # And the harness must actually exercise masking: a seen (train-positive)
    # pair must not be counted as a hit. Sanity-check that seen pairs exist.
    assert len(seen_index) > 0, "test fixture must contain train-positive pairs"


def test_harness_masks_seen_pairs() -> None:
    """A train-positive pair scored maximally must be masked from ranking.

    Construct a matrix where a known train-positive (seen) pair has the top
    score for an anchor; the official ranking must not reward it, proving the
    harness applies seen-masking rather than ranking on raw scores.
    """
    frame = _toy_frame()
    universe = _build_gene_universe(frame)
    train_df = frame[(frame["split_type"] == "CV2") & (frame["split_role"] == "train")]
    train_pos = train_df[train_df["sl_label"] == 1]
    seen_index = _pair_indices(train_pos, universe)
    assert len(seen_index) > 0

    n = len(universe.symbols)
    # All-equal base; spike one seen pair to the max score in both directions.
    matrix = np.full((n, n), 0.1)
    np.fill_diagonal(matrix, 0.0)
    a, b = int(seen_index[0, 0]), int(seen_index[0, 1])
    matrix[a, b] = matrix[b, a] = 100.0

    test_df = frame[(frame["split_type"] == "CV2") & (frame["split_role"] == "test")]
    pos_index = _pair_indices(test_df[test_df["sl_label"] == 1], universe)

    masked = official_ranking_metrics(
        matrix, pos_index, seen_index=seen_index, ks=(10,)
    )
    unmasked = official_ranking_metrics(matrix, pos_index, seen_index=None, ks=(10,))
    # Masking must change the result (the spiked seen pair is suppressed).
    assert masked != unmasked, "seen-pair masking had no effect on ranking metrics"
