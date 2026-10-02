"""Preparation writes every expression quantity once, in STATE's log space."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.sparse import csr_matrix

from src.data.expression import library_sizes, log_normalize
from src.data.prepared import load_inputs, select_context_cells
from src.data.tx1_cache import write_line_cache
from src.experiments.prepare import prepare_inputs

K562, HEPG2, HCT116, JURKAT = "ACH-000551", "ACH-000739", "ACH-000971", "ACH-000995"
ANCHORS = (K562, HEPG2, HCT116, JURKAT)
TRAIN = (*ANCHORS, "ACH-T1", "ACH-T2", "ACH-U1")
VAL = ("ACH-V1", "ACH-V2", "ACH-V3")
TEST = ("ACH-S1", "ACH-S2", "ACH-S3")
UNLABELED = ("ACH-U1",)
HVG = ("H1", "H2", "H3", "H4", "H5", "H6")
PANEL_CANDIDATES = ("GA", "GB", "GC", "GD", "GE")
SOURCE_GENES = (*HVG, *PANEL_CANDIDATES, "X1", "X2")
CELLS_PER_LINE = 10
CACHED_CELLS = (9, 7, 5, 3, 1, 0)
CELLS_PER_CONTEXT = 4
RESPONSE_GENES = {"GA": 3, "GB": 3, "ZZ": 2}
N_CONTROLS = 6


def _counts(rng: np.random.Generator, n_cells: int, n_genes: int) -> np.ndarray:
    counts = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    counts[:, -2:] += rng.integers(20, 60, size=(n_cells, 2))  # extra genes
    return counts


def _write_basal_source(path: Path, model_id: str, seed: int, *, scale=1.0) -> None:
    rng = np.random.default_rng(seed)
    counts = _counts(rng, CELLS_PER_LINE, len(SOURCE_GENES)) * scale
    obs = pd.DataFrame(
        {"model_id": [model_id] * CELLS_PER_LINE},
        index=[f"{model_id}-c{i}" for i in range(CELLS_PER_LINE)],
    )
    var = pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{i:011d}" for i in range(len(SOURCE_GENES))],
            "gene_symbol": list(SOURCE_GENES),
        },
        index=[f"v{i}" for i in range(len(SOURCE_GENES))],
    )
    ad.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)


def _write_response_source(path: Path, seed: int, genes: tuple[str, ...]) -> None:
    rng = np.random.default_rng(seed)
    labels = ["non-targeting"] * N_CONTROLS
    for gene, count in RESPONSE_GENES.items():
        labels += [gene] * count
    counts = _counts(rng, len(labels), len(genes))
    obs = pd.DataFrame(
        {"gene": labels}, index=[f"r{seed}-{i}" for i in range(len(labels))]
    )
    var = pd.DataFrame(
        {
            "gene_id": [f"ENSG{SOURCE_GENES.index(g):011d}" for g in genes],
            "gene_name": list(genes),
        },
        index=[f"v{i}" for i in range(len(genes))],
    )
    ad.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)


def build_world(root: Path) -> dict:
    """A complete raw world: split, labels, sources, Tx1 cache, STATE, ESM2."""
    root.mkdir(parents=True, exist_ok=True)
    lines = (*TRAIN, *VAL, *TEST)
    (root / "split.json").write_text(
        json.dumps(
            {
                "train": list(TRAIN),
                "val": list(VAL),
                "test": list(TEST),
                "unlabeled_train": list(UNLABELED),
            }
        )
    )
    rng = np.random.default_rng(7)
    labeled = [line for line in lines if line not in UNLABELED]
    effects = rng.normal(size=(len(labeled), len(PANEL_CANDIDATES)))
    for row, line in enumerate(labeled):
        if line in VAL or line in TEST:
            effects[row] += 100.0  # would visibly shift any leaked fit
    frame = pd.DataFrame(
        effects,
        index=labeled,
        columns=[f"{g} ({i + 1})" for i, g in enumerate(PANEL_CANDIDATES)],
    )
    frame.loc[K562, "GE (5)"] = np.nan  # no copy-prior donor value -> off panel
    frame.to_csv(root / "gene_effect.csv")

    sources = root / "basal"
    sources.mkdir()
    registry = []
    for index, line in enumerate(lines):
        path = sources / f"{line}.h5ad"
        _write_basal_source(path, line, seed=index)
        registry.append(
            {
                "model_id": line,
                "source_path": str(path),
                "source_kind": "h5ad",
                "matrix_semantics": "raw_umi_counts",
            }
        )
        cell_ids = [f"{line}-c{i}" for i in CACHED_CELLS]
        write_line_cache(
            root / "tx1",
            line,
            rng.normal(size=(len(cell_ids), 2560)).astype(np.float32),
            np.zeros((len(cell_ids), len(HVG)), dtype=np.float32),
            pd.DataFrame({"model_id": line}, index=cell_ids),
        )
    pd.DataFrame(registry).to_csv(root / "registry.csv", index=False)

    response = {}
    for index, anchor in enumerate(ANCHORS):
        genes = SOURCE_GENES if anchor != HCT116 else SOURCE_GENES[1:]  # no H1
        path = root / f"response_{anchor}.h5ad"
        _write_response_source(path, seed=100 + index, genes=genes)
        response[anchor] = {
            "source_type": "h5ad",
            "h5ad_path": str(path),
            "perturbation_col": "gene",
            "control_label": "non-targeting",
            "var_ensembl_col": "gene_id",
            "target_gene_symbol_col": "gene_name",
        }
    (root / "perturbseq_sources.json").write_text(json.dumps(response))

    (root / "state").mkdir()
    (root / "state" / "var_dims.pkl").write_bytes(
        pickle.dumps({"gene_names": list(HVG)})
    )
    esm2 = (*PANEL_CANDIDATES, "GF")
    np.savez(
        root / "esm2.npz",
        symbols=np.array(esm2),
        vectors=rng.normal(size=(len(esm2), 4)).astype(np.float32),
        resolved=np.ones(len(esm2), dtype=bool),
    )

    config = yaml.safe_load(Path("configs/geneeffect_joint.yaml").read_text())
    config["prepared_root"] = str(root / "prepared")
    config["features"]["cells_per_context"] = CELLS_PER_CONTEXT
    config["paths"].update(
        split=str(root / "split.json"),
        gene_effect=str(root / "gene_effect.csv"),
        source_registry=str(root / "registry.csv"),
        tx1_cache=str(root / "tx1"),
        esm2_embeddings=str(root / "esm2.npz"),
        state_model_dir=str(root / "state"),
        perturbseq_sources=str(root / "perturbseq_sources.json"),
    )
    return config


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> dict:
    config = build_world(tmp_path_factory.mktemp("world"))
    prepare_inputs(config)
    return config


def _raw(path: str | Path) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    adata.X = adata.X.toarray()
    return adata


def _manifest(config: dict) -> dict:
    return json.loads(
        (Path(config["prepared_root"]) / "prepared_inputs.json").read_text()
    )


def _target_sum(config: dict) -> float:
    sources = json.loads(Path(config["paths"]["perturbseq_sources"]).read_text())
    sizes = []
    for anchor in (JURKAT, HEPG2):
        adata = _raw(sources[anchor]["h5ad_path"])
        controls = adata.obs["gene"].to_numpy() == "non-targeting"
        sizes.append(adata.X[controls].sum(axis=1))
    return float(np.median(np.concatenate(sizes)))


def _line(config: dict, model_id: str) -> dict[str, np.ndarray]:
    path = Path(config["prepared_root"]) / "lines" / f"{model_id}.npz"
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def test_target_sum_is_median_of_jurkat_hepg2_controls(prepared):
    space = _manifest(prepared)["expression_space"]
    assert space == {
        "transform": "log1p_normalize_total",
        "target_sum": pytest.approx(_target_sum(prepared)),
        "library_size": "all_genes",
        "target_sum_sources": [JURKAT, HEPG2],
    }
    assert load_inputs(prepared).target_sum == pytest.approx(_target_sum(prepared))


def test_basal_hvg_is_log_space_over_all_genes(prepared):
    target_sum = _target_sum(prepared)
    for model_id in ("ACH-T1", "ACH-V2", K562):
        raw = _raw(
            Path(prepared["paths"]["source_registry"]).parent
            / "basal"
            / f"{model_id}.h5ad"
        )
        cached = [f"{model_id}-c{i}" for i in CACHED_CELLS]
        order = select_context_cells(model_id, cached, CELLS_PER_CONTEXT)
        rows = raw.obs_names.get_indexer([cached[i] for i in order])
        hvg = [SOURCE_GENES.index(gene) for gene in HVG]
        expected = log_normalize(
            raw.X[rows][:, hvg], library_sizes(raw.X)[rows], target_sum
        )
        line = _line(prepared, model_id)
        np.testing.assert_allclose(line["basal_hvg"], expected, rtol=1e-6)
        assert not np.allclose(
            line["basal_hvg"],
            log_normalize(
                raw.X[rows][:, hvg], library_sizes(raw.X[rows][:, hvg]), target_sum
            ),
        )
        cache = np.load(
            Path(prepared["paths"]["tx1_cache"]) / model_id / "embeddings.npy"
        )
        np.testing.assert_array_equal(line["controls_tx1"], cache[order])
    inputs = load_inputs(prepared)
    np.testing.assert_array_equal(
        inputs.lines["ACH-T1"].basal_hvg, _line(prepared, "ACH-T1")["basal_hvg"]
    )


def test_response_targets_are_log_space(prepared):
    target_sum = _target_sum(prepared)
    sources = json.loads(Path(prepared["paths"]["perturbseq_sources"]).read_text())
    inputs = load_inputs(prepared)
    keys = inputs.response_targets.keys
    for anchor in (JURKAT, HCT116):
        raw = _raw(sources[anchor]["h5ad_path"])
        symbols = list(raw.var["gene_name"])
        rows = np.flatnonzero(raw.obs["gene"].to_numpy() == "GB")
        expected = np.zeros((len(rows), len(HVG)), dtype=np.float32)
        present = [i for i, gene in enumerate(HVG) if gene in symbols]
        expected[:, present] = log_normalize(
            raw.X[rows][:, [symbols.index(HVG[i]) for i in present]],
            library_sizes(raw.X)[rows],
            target_sum,
        )
        bag = inputs.response_targets.target_bag(keys.index((anchor, "GB")))
        np.testing.assert_allclose(bag, expected, rtol=1e-6)
    np.testing.assert_array_equal(
        inputs.response_targets.target_bag(keys.index((HCT116, "GA")))[:, 0], 0.0
    )


def test_q_sc_mean_variance_in_log_space(prepared):
    target_sum = _target_sum(prepared)
    inputs = load_inputs(prepared, include_test=True)
    for model_id in ("ACH-T2", "ACH-S1"):
        raw = _raw(
            Path(prepared["paths"]["source_registry"]).parent
            / "basal"
            / f"{model_id}.h5ad"
        )
        q_sc = inputs.lines[model_id].q_sc
        assert q_sc.symbols == inputs.genes
        assert q_sc.available.all()
        columns = [SOURCE_GENES.index(gene) for gene in inputs.genes]
        logged = log_normalize(raw.X[:, columns], library_sizes(raw.X), target_sum)
        np.testing.assert_allclose(q_sc.values[:, 0], logged.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(q_sc.values[:, 2], logged.var(axis=0), rtol=1e-4)
        np.testing.assert_array_equal(
            q_sc.values[:, 1], (raw.X[:, columns] > 0).mean(axis=0).astype(np.float32)
        )


def test_no_condition_holdout_in_manifest(prepared):
    manifest = _manifest(prepared)
    assert not [key for key in manifest if "holdout" in key]
    inputs = load_inputs(prepared)
    assert inputs.response_anchors == tuple(sorted(ANCHORS))
    assert set(inputs.response_targets.keys) == {
        (anchor, gene) for anchor in ANCHORS for gene in ("GA", "GB")
    }
    assert not hasattr(inputs, "response_holdout")


def test_load_inputs_refuses_v1_manifest(prepared, tmp_path):
    config = json.loads(json.dumps(prepared))
    root = tmp_path / "v1"
    root.mkdir()
    manifest = _manifest(prepared)
    del manifest["expression_space"]
    manifest["schema_version"] = "geneeffect-joint-prepared-v1"
    (root / "prepared_inputs.json").write_text(json.dumps(manifest))
    config["prepared_root"] = str(root)
    with pytest.raises(ValueError, match="expression_space"):
        load_inputs(config)


def test_load_inputs_refuses_checkpoint_from_raw_count_space(prepared):
    state = load_inputs(prepared).preprocessing_state()
    del state["target_sum"]
    with pytest.raises(ValueError, match="expression_space"):
        load_inputs(prepared, preprocessing=state)


def test_checkpoint_preprocessing_restores_fit(prepared):
    fitted = load_inputs(prepared)
    restored = load_inputs(prepared, preprocessing=fitted.preprocessing_state())
    pd.testing.assert_series_equal(restored.train_gene_means, fitted.train_gene_means)
    assert restored.variable_genes == fitted.variable_genes
    assert restored.esm2_symbols == fitted.esm2_symbols


def test_prepare_rejects_non_integer_source(tmp_path):
    config = build_world(tmp_path)
    source = tmp_path / "basal" / "ACH-V1.h5ad"
    _write_basal_source(source, "ACH-V1", seed=0, scale=0.37)
    with pytest.raises(ValueError, match="integer"):
        prepare_inputs(config)
    assert not (tmp_path / "prepared" / "prepared_inputs.json").exists()


def test_prepare_skips_when_manifest_exists(tmp_path):
    config = build_world(tmp_path)
    root = Path(config["prepared_root"])
    root.mkdir()
    (root / "prepared_inputs.json").write_text("{}")
    config["paths"]["source_registry"] = str(tmp_path / "missing.csv")
    assert prepare_inputs(config) == root / "prepared_inputs.json"
    assert (root / "prepared_inputs.json").read_text() == "{}"
    assert not (root / "lines").exists()


def test_train_means_fit_on_supervised_train_only(prepared, monkeypatch):
    import src.data.prepared as module

    checked = []
    original = module.assert_fit_eligible

    def spy(model_id, split):
        checked.append(model_id)
        original(model_id, split)

    monkeypatch.setattr(module, "assert_fit_eligible", spy)
    inputs = load_inputs(prepared)
    supervised = [line for line in TRAIN if line not in UNLABELED]
    assert sorted(checked) == sorted(supervised)
    labels = pd.read_csv(prepared["paths"]["gene_effect"], index_col=0)
    for gene in inputs.genes:
        column = next(c for c in labels.columns if c.startswith(f"{gene} "))
        assert inputs.train_gene_means[gene] == pytest.approx(
            labels.loc[supervised, column].mean()
        )
    assert "GE" not in inputs.genes  # the copy-prior donor has no value
    assert set(inputs.labels["model_id"]) == {*supervised, *VAL}
    np.testing.assert_allclose(
        inputs.labels["residual"],
        inputs.labels["gene_effect"]
        - inputs.labels["gene_symbol"].map(inputs.train_gene_means),
    )
    assert set(inputs.lines) == {*supervised, *VAL}
    assert set(load_inputs(prepared, include_test=True).lines) == {
        *supervised,
        *VAL,
        *TEST,
    }
