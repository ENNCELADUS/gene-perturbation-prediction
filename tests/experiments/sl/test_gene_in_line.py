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


def pie_root(path, held_out_value=None):
    """A PIE root of two training lines and one validation line; two lost genes
    with change files, one gene that is not lost."""
    import pandas as pd

    (path / "pie" / "change").mkdir(parents=True)
    lines = pd.DataFrame(
        {"role": ["train", "train", "val"]},
        index=pd.Index(["ACH-1", "ACH-2", "ACH-3"], name="model_id"),
    )
    lines.to_parquet(path / "lines.parquet")
    genes = pd.DataFrame(
        {"lost": [True, False, True]},
        index=pd.Index(["ATM", "BRCA1", "C1ORF1"], name="gene"),
    )
    genes.to_parquet(path / "genes.parquet")
    rng = np.random.default_rng(0)
    for gene in ("ATM", "C1ORF1"):
        change = rng.normal(size=(3, 5))
        if held_out_value is not None:
            change[2] = held_out_value
        np.save(path / "pie" / "change" / f"{gene}.npy", change.astype(np.float16))
    return {
        "seed": 0,
        "out": str(path / "out"),
        "pie": {"root": str(path), "components": 2, "sample": 3},
    }


GENES = ["ATM", "BRCA1", "C1orf1"]


def test_pie_change_writes_each_pie_line_in_components(tmp_path):
    import json

    config = pie_root(tmp_path)
    gene_in_line.pie_change(config, GENES, ["ACH-1", "ACH-9"])
    cache = tmp_path / "out" / "pie_change"
    assert json.loads((cache / "genes.json").read_text()) == GENES
    values = np.load(cache / "ACH-3.npy")
    assert values.shape == (3, 2) and values.dtype == np.float16
    assert np.isfinite(values[[0, 2]]).all() and np.isnan(values[1]).all()
    report = json.loads((cache / "report.json").read_text())
    assert report["sample"] == 3 and report["labelled_lines_without_pie"] == ["ACH-9"]
    with pytest.raises(ValueError, match="gene axis"):
        gene_in_line.pie_change(config, ["ATM"], [])


def test_pie_change_writes_every_line_again_when_one_line_is_missing(tmp_path):
    config = pie_root(tmp_path)
    gene_in_line.pie_change(config, GENES, [])
    cache = tmp_path / "out" / "pie_change"
    first = np.load(cache / "ACH-1.npy")
    np.save(cache / "ACH-1.npy", np.zeros_like(first))
    gene_in_line.pie_change(config, GENES, [])  # every file exists: nothing is written
    assert not np.load(cache / "ACH-1.npy").any()
    (cache / "ACH-2.npy").unlink()
    gene_in_line.pie_change(config, GENES, [])  # one basis for all lines
    np.testing.assert_array_equal(np.load(cache / "ACH-1.npy"), first)
    assert (cache / "ACH-2.npy").is_file()


def test_pie_change_fits_its_components_on_the_training_lines_only(tmp_path):
    for name, value in (("plain", None), ("extreme", 1000.0)):
        gene_in_line.pie_change(pie_root(tmp_path / name, value), GENES, [])
    for model_id in ("ACH-1", "ACH-2"):
        plain, extreme = (
            np.load(tmp_path / name / "out" / "pie_change" / f"{model_id}.npy")
            for name in ("plain", "extreme")
        )
        np.testing.assert_array_equal(plain, extreme)


def test_pie_change_refuses_an_unfinished_pie_pass(tmp_path):
    config = pie_root(tmp_path)
    (tmp_path / "pie" / "change" / "C1ORF1.npy").unlink()
    with pytest.raises(FileNotFoundError, match="1 lost genes"):
        gene_in_line.pie_change(config, GENES, [])


def test_a_value_outside_the_float16_range_is_refused():
    assert gene_in_line.as_float16(np.array([1.0, np.nan]), "ACH-1").dtype == np.float16
    with pytest.raises(ValueError, match="ACH-1"):
        gene_in_line.as_float16(np.array([1.0e6]), "ACH-1")


def test_an_axis_gene_takes_the_first_var_row_of_its_upper_cased_symbol():
    import pandas as pd

    symbols = pd.Series(["ATM", "C1orf112", "ATM", "TP53"])
    rows = gene_in_line.var_rows(symbols, ["ATM", "BRCA1", "C1ORF112"])
    np.testing.assert_array_equal(rows, [0.0, np.nan, 1.0])
