import numpy as np
import pytest

from src.experiments.sl import gene_in_line


def test_components_span_the_training_vectors():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 9)) + 3.0
    centre, basis = gene_in_line.fit_components(vectors, 2)
    assert centre.shape == (9,) and basis.shape == (9, 2)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-4)
    assert np.allclose(
        (vectors - centre) @ basis @ basis.T, vectors - centre, atol=1e-3
    )


def test_the_config_loads_and_an_unknown_step_is_refused():
    from pathlib import Path

    config = gene_in_line.load_config(Path("configs/sl/gene_in_line.yaml"))
    assert config["kind"] == "sl_gene_in_line" and config["pie"]["components"] == 64
    with pytest.raises(SystemExit):
        gene_in_line.main(["configs/sl/gene_in_line.yaml", "--steps", "cellnet"])


def test_pie_change_writes_each_pie_line_in_components(tmp_path):
    import json

    import pandas as pd

    root = tmp_path / "pie_root"
    (root / "pie" / "change").mkdir(parents=True)
    lines = pd.DataFrame(
        {"role": ["train", "train", "val"]},
        index=pd.Index(["ACH-1", "ACH-2", "ACH-3"], name="model_id"),
    )
    lines.to_parquet(root / "lines.parquet")
    rng = np.random.default_rng(0)
    for gene in ("ATM", "C1ORF1"):
        np.save(
            root / "pie" / "change" / f"{gene}.npy",
            rng.normal(size=(3, 5)).astype(np.float16),
        )
    config = {
        "seed": 0,
        "out": str(tmp_path / "out"),
        "pie": {"root": str(root), "components": 2, "sample": 3},
    }
    genes = ["ATM", "BRCA1", "C1orf1"]
    gene_in_line.pie_change(config, genes, ["ACH-1", "ACH-9"])
    cache = tmp_path / "out" / "pie_change"
    assert json.loads((cache / "genes.json").read_text()) == genes
    values = np.load(cache / "ACH-3.npy")
    assert values.shape == (3, 2) and values.dtype == np.float16
    assert np.isfinite(values[[0, 2]]).all() and np.isnan(values[1]).all()
    report = json.loads((cache / "report.json").read_text())
    assert report["sample"] == 3 and report["labelled_lines_without_pie"] == ["ACH-9"]
    with pytest.raises(ValueError, match="gene axis"):
        gene_in_line.pie_change(config, ["ATM"], [])
