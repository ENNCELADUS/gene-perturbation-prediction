from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from sl_dl_model.exp08b_artifacts import save_embedding_cache
from sl_dl_model.config import SLDLConfig
from sl_dl_model.exp08b_config import SlHeadConfig
from sl_dl_model.exp08b_sl_head import CachedEmbeddingPairHeadProducer
from sl_dl_model.scoring import run_fold_with_producer


def _write_cache(path: Path) -> None:
    save_embedding_cache(
        path,
        symbols=np.array(["A", "B", "C"], dtype=object),
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        coverage_mask=np.array([1, 1, 0], dtype=np.int64),
        embedding_method="test",
    )


def test_cached_embedding_producer_returns_frozen_table(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)
    config = SlHeadConfig(
        max_epochs=1,
        batch_pairs=2,
        pair_hidden=(8,),
        include_coverage_flag=True,
    )
    producer = CachedEmbeddingPairHeadProducer(
        config,
        cache_path=cache_path,
        train_pairs=[
            ("A", "B", 1, -1.0, -0.5),
            ("A", "C", 0, -1.0, 0.2),
            ("B", "C", 0, -0.5, 0.2),
        ],
        metric_model_name="exp08b",
    )

    embeddings, coverage_mask = producer.produce(
        np.array(["A", "B", "C"], dtype=object), {"A", "B"}
    )

    assert embeddings.shape == (3, 2)
    np.testing.assert_allclose(
        embeddings, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    )
    assert coverage_mask.tolist() == [1, 1, 0]
    assert producer.metric_model_name == "exp08b"


def test_cached_pair_head_scores_full_matrix(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)
    config = SlHeadConfig(
        max_epochs=2,
        batch_pairs=2,
        pair_hidden=(8,),
        include_coverage_flag=False,
    )
    producer = CachedEmbeddingPairHeadProducer(
        config,
        cache_path=cache_path,
        train_pairs=[
            ("A", "B", 1, -1.0, -0.5),
            ("A", "C", 0, -1.0, 0.2),
            ("B", "C", 0, -0.5, 0.2),
        ],
        metric_model_name="direct_esm2_mlp",
    )
    symbols = np.array(["A", "B", "C"], dtype=object)
    gene_effects = np.array([-1.0, -0.5, 0.2], dtype=float)

    producer.produce(symbols, {"A", "B"})
    scores = producer.score_matrix(symbols, gene_effects)

    assert scores.shape == (3, 3)
    np.testing.assert_allclose(np.diag(scores), np.zeros(3))
    assert np.isfinite(scores).all()


def test_cached_pair_head_same_seed_is_deterministic(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)
    config = SlHeadConfig(
        seed=123,
        max_epochs=3,
        batch_pairs=2,
        pair_hidden=(8,),
        include_coverage_flag=False,
    )
    train_pairs = [
        ("A", "B", 1, -1.0, -0.5),
        ("A", "C", 0, -1.0, 0.2),
        ("B", "C", 0, -0.5, 0.2),
    ]
    symbols = np.array(["A", "B", "C"], dtype=object)
    gene_effects = np.array([-1.0, -0.5, 0.2], dtype=float)

    first = CachedEmbeddingPairHeadProducer(
        config,
        cache_path=cache_path,
        train_pairs=train_pairs,
        metric_model_name="exp08b",
        device="cpu",
    )
    second = CachedEmbeddingPairHeadProducer(
        config,
        cache_path=cache_path,
        train_pairs=train_pairs,
        metric_model_name="exp08b",
        device="cpu",
    )

    first.produce(symbols, {"A", "B"})
    second.produce(symbols, {"A", "B"})

    np.testing.assert_allclose(
        first.score_matrix(symbols, gene_effects),
        second.score_matrix(symbols, gene_effects),
    )


def test_exp08b_sl_head_import_separation_guard() -> None:
    path = Path("src/sl_dl_model/exp08b_sl_head.py")
    source = path.read_text()
    tree = ast.parse(source)
    forbidden = {
        "StateEncoder",
        "PertAdapter",
        "SlDlModel",
        "StateAdapterBagGenerator",
        "Step1GeneratorTrainer",
        "Exp08bConfig",
        "sl_dl_model.train",
        "state_checkpoint",
        "esm2_npz",
        "gwps_h5ad",
    }

    imported_names: set[str] = set()
    literal_strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_names.add(node.module)
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal_strings.add(node.value)

    assert not (forbidden & imported_names)
    assert not (forbidden & literal_strings)
    for value in forbidden:
        assert value not in source


def test_cached_pair_head_uses_partialstate_device_or_injected_device() -> None:
    source = Path("src/sl_dl_model/exp08b_sl_head.py").read_text()

    assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in source
    assert "PartialState().device" in source or "device=" in source


def test_run_fold_with_producer_labels_rows_by_metric_model_name(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.npz"
    save_embedding_cache(
        cache_path,
        symbols=np.array(["A", "B", "C", "D"], dtype=object),
        embeddings=np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 1.0]],
            dtype=np.float32,
        ),
        coverage_mask=np.array([1, 1, 1, 1], dtype=np.int64),
        embedding_method="test",
    )
    effects = {"A": -1.0, "B": -0.5, "C": 0.2, "D": 0.7}
    frame = pd.DataFrame(
        [
            {
                "pair_id": "p0",
                "fold_id": 0,
                "split_type": "CV2",
                "split_role": "train",
                "sl_label": 1,
                "gene_a_symbol": "A",
                "gene_b_symbol": "B",
                "gene_a_k562_gene_effect": effects["A"],
                "gene_b_k562_gene_effect": effects["B"],
            },
            {
                "pair_id": "p1",
                "fold_id": 0,
                "split_type": "CV2",
                "split_role": "train",
                "sl_label": 0,
                "gene_a_symbol": "A",
                "gene_b_symbol": "C",
                "gene_a_k562_gene_effect": effects["A"],
                "gene_b_k562_gene_effect": effects["C"],
            },
            {
                "pair_id": "p2",
                "fold_id": 0,
                "split_type": "CV2",
                "split_role": "train",
                "sl_label": 0,
                "gene_a_symbol": "B",
                "gene_b_symbol": "C",
                "gene_a_k562_gene_effect": effects["B"],
                "gene_b_k562_gene_effect": effects["C"],
            },
            {
                "pair_id": "p3",
                "fold_id": 0,
                "split_type": "CV2",
                "split_role": "test",
                "sl_label": 1,
                "gene_a_symbol": "A",
                "gene_b_symbol": "D",
                "gene_a_k562_gene_effect": effects["A"],
                "gene_b_k562_gene_effect": effects["D"],
            },
            {
                "pair_id": "p4",
                "fold_id": 0,
                "split_type": "CV2",
                "split_role": "test",
                "sl_label": 0,
                "gene_a_symbol": "B",
                "gene_b_symbol": "D",
                "gene_a_k562_gene_effect": effects["B"],
                "gene_b_k562_gene_effect": effects["D"],
            },
            {
                "pair_id": "p5",
                "fold_id": 0,
                "split_type": "CV2",
                "split_role": "test",
                "sl_label": 0,
                "gene_a_symbol": "C",
                "gene_b_symbol": "D",
                "gene_a_k562_gene_effect": effects["C"],
                "gene_b_k562_gene_effect": effects["D"],
            },
        ]
    )
    train_pairs = [
        ("A", "B", 1, effects["A"], effects["B"]),
        ("A", "C", 0, effects["A"], effects["C"]),
        ("B", "C", 0, effects["B"], effects["C"]),
    ]

    for model_name in ("exp08b", "direct_esm2_mlp", "nn_copy"):
        producer = CachedEmbeddingPairHeadProducer(
            SlHeadConfig(
                max_epochs=1,
                batch_pairs=2,
                pair_hidden=(8,),
                include_coverage_flag=False,
            ),
            cache_path=cache_path,
            train_pairs=train_pairs,
            metric_model_name=model_name,
        )

        rows = run_fold_with_producer(
            frame,
            "CV2",
            0,
            SLDLConfig(include_coverage_flag=False, ranking_k=(2,)),
            producer,
        )

        labels = {row["model"] for row in rows}
        assert labels == {model_name}
        assert "state_dl" not in labels
