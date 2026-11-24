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
    assert config["ladder"] == "data/geneeffect_joint/v2/ladder"
    assert set(config["depmap"]) == {"damaging", "hotspot", "copy_number"}
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


# The ladder's source genes in their own case; the axis holds upper-cased symbols, and
# KRAS is not a source gene.
SOURCE = ["ATM", "Brca1", "C1orf1", "TP53"]
AXIS = ["ATM", "BRCA1", "C1ORF1", "KRAS", "TP53"]
# ACH-1..3 train, ACH-4 validation (extreme ATM), ACH-5 test without a cell file
PSEUDOBULK = {
    "ACH-1": [1.0, 4.0, 0.5, 2.0],
    "ACH-2": [2.0, 5.0, 1.5, 3.0],
    "ACH-3": [3.0, 6.0, 2.5, 4.0],
    "ACH-4": [-100.0, 7.0, 0.1, 9.0],
}
LINES = ["ACH-1", "ACH-2", "ACH-3", "ACH-4", "ACH-5"]


def cell_world(path):
    """A ladder with cell files of four lines and a split of three training lines,
    one validation line and one test line."""
    import json

    cells = path / "ladder" / "cells"
    cells.mkdir(parents=True)
    (cells / "genes.json").write_text(json.dumps({"genes": SOURCE}))
    for k, (model_id, values) in enumerate(PSEUDOBULK.items()):
        np.savez(
            cells / f"{model_id}.npz",
            pseudobulk=np.array(values, np.float32),
            detected=np.array([0.1, 0.2, 0.3, 0.4], np.float32) + k,
            sd=np.array([1.0, 2.0, 3.0, 4.0], np.float32) * (k + 1),
            detected_expected=np.full(4, 0.5, np.float32),
            sd_expected=np.full(4, 0.5, np.float32),
        )
    split = {
        "train": ["ACH-1", "ACH-2", "ACH-3"],
        "val": ["ACH-4"],
        "test": ["ACH-5"],
    }
    (path / "split.json").write_text(json.dumps(split))
    return {
        "seed": 0,
        "out": str(path / "out"),
        "ladder": str(path / "ladder"),
        "split": str(path / "split.json"),
    }


def cache_of(config, step, model_id):
    from pathlib import Path

    return np.load(Path(config["out"]) / step / f"{model_id}.npy").astype(np.float32)


def test_on_axis_matches_upper_cased_symbols_and_leaves_nan_rows():
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    out = gene_in_line.on_axis(values, ["Atm", "BRCA1", "ATM", "c1orf1"], AXIS)
    np.testing.assert_array_equal(out[[0, 1, 2]], [[0, 1], [2, 3], [6, 7]])
    assert np.isnan(out[[3, 4]]).all()


def test_wide_expression_holds_the_value_and_the_silenced_flag(tmp_path):
    config = cell_world(tmp_path)
    gene_in_line.wide_expression(config, AXIS, LINES)
    # the training 10th percentiles: ATM 1.2, BRCA1 4.2, C1ORF1 0.7, TP53 2.2
    expected = {
        "ACH-1": [[1.0, 1], [4.0, 1], [0.5, 1], [np.nan] * 2, [2.0, 1]],
        "ACH-2": [[2.0, 0], [5.0, 0], [1.5, 0], [np.nan] * 2, [3.0, 0]],
        # had the validation line entered the percentile, ATM's would be -69.7
        "ACH-4": [[-100.0, 1], [7.0, 0], [0.1, 1], [np.nan] * 2, [9.0, 0]],
    }
    for model_id, rows in expected.items():
        np.testing.assert_allclose(
            cache_of(config, "wide_expression", model_id), rows, rtol=1e-3
        )


def test_cell_spread_holds_the_detected_share_and_the_excess_sd(tmp_path):
    from pathlib import Path

    config = cell_world(tmp_path)
    gene_in_line.cell_spread(config, AXIS, LINES)
    cells = np.load(Path(config["ladder"]) / "cells" / "ACH-2.npz")
    values = cache_of(config, "cell_spread", "ACH-2")
    np.testing.assert_allclose(values[[0, 1, 2, 4], 0], cells["detected"], rtol=1e-3)
    np.testing.assert_allclose(
        values[[0, 1, 2, 4], 1], cells["sd"] - cells["sd_expected"], rtol=1e-3
    )
    assert np.isnan(values[3]).all()


def test_a_line_without_a_cell_file_gets_nan_rows(tmp_path):
    import json

    config = cell_world(tmp_path)
    for step in (gene_in_line.wide_expression, gene_in_line.cell_spread):
        step(config, AXIS, LINES)
    for step in ("wide_expression", "cell_spread"):
        values = cache_of(config, step, "ACH-5")
        assert values.shape == (len(AXIS), 2) and np.isnan(values).all()
        report = json.loads((tmp_path / "out" / step / "report.json").read_text())
        assert report == {"lines": 5, "lines_with_source": 4, "genes_with_value": 4}
        assert json.loads((tmp_path / "out" / step / "genes.json").read_text()) == AXIS


def buffer_world(path):
    """``cell_world`` with a linear-model config whose reference tables say: ATM
    has the paralogs BRCA1 (closest) and C1ORF1; ATM and TP53 share a complex."""
    import pandas as pd

    config = cell_world(path)
    reference = path / "reference"
    reference.mkdir()
    tables = {
        "paralogs.tsv.gz": {
            "gene": ["ATM", "ATM"],
            "paralog": ["BRCA1", "C1ORF1"],
            "identity": [60.0, 40.0],
        },
        "complexes.tsv": {"complex_id": [1, 1], "gene": ["ATM", "TP53"]},
        "hallmark.tsv": {"gene_set": [], "gene": []},
        "progeny.tsv": {"pathway": [], "gene": [], "weight": []},
        "drivers.tsv": {"model_id": ["ACH-4"], "KRAS": [1]},
        "msi.tsv": {"model_id": ["ACH-4"], "msi_score": [0.0]},
    }
    for name, table in tables.items():
        pd.DataFrame(table).to_csv(reference / name, sep="\t", index=False)
    (path / "linear_model.yaml").write_text(f"paths: {{reference: {reference}}}\n")
    return {**config, "linear_model": str(path / "linear_model.yaml")}


def test_the_buffer_caches_hold_the_partner_columns(tmp_path):
    config = buffer_world(tmp_path)
    for step in ("paralog_buffer", "complex_buffer", "partner_low"):
        gene_in_line.buffer_cache(config, AXIS, LINES, step)
    # ACH-1: BRCA1 4.0, C1ORF1 0.5, TP53 2.0, every one at or below its percentile
    np.testing.assert_allclose(
        cache_of(config, "paralog_buffer", "ACH-1")[0], [0.5, 4.5, 4.0], rtol=1e-3
    )
    np.testing.assert_allclose(cache_of(config, "complex_buffer", "ACH-1")[0], [2.0])
    np.testing.assert_allclose(cache_of(config, "partner_low", "ACH-1")[0], [1.0])
    # ACH-4 (validation): of BRCA1 7.0, C1ORF1 0.1 and TP53 9.0 only C1ORF1 is low
    np.testing.assert_allclose(
        cache_of(config, "partner_low", "ACH-4")[0], [1 / 3], rtol=1e-3
    )
    for step in ("paralog_buffer", "complex_buffer", "partner_low"):
        assert np.isnan(cache_of(config, step, "ACH-5")).all()


def test_a_gene_without_a_paralog_is_absent_from_the_paralog_cache_only(tmp_path):
    config = buffer_world(tmp_path)
    for step in ("paralog_buffer", "complex_buffer"):
        gene_in_line.buffer_cache(config, AXIS, LINES, step)
    tp53 = AXIS.index("TP53")
    assert np.isnan(cache_of(config, "paralog_buffer", "ACH-2")[tp53]).all()
    np.testing.assert_allclose(cache_of(config, "complex_buffer", "ACH-2")[tp53], [2.0])


LEADING = (
    "SequencingID,ModelID,ModelConditionID,IsDefaultEntryForModel,IsDefaultEntryForMC"
)


def depmap_world(path):
    """Mutation matrices of the DepMap layout (two sequencings of ACH-1, one the
    default; TP53 twice in the damaging matrix, the first column kept) and a
    copy-number matrix with the ModelID as its first column."""
    damaging = path / "damaging.csv"
    damaging.write_text(
        f"{LEADING},ATM (472),Brca1 (672),TP53 (7157),TP53 (999)\n"
        "CDS-1,ACH-1,MC-1,Yes,Yes,1,0,2,0\n"
        "CDS-2,ACH-1,MC-2,No,Yes,0,2,0,0\n"
        "CDS-3,ACH-2,MC-3,Yes,Yes,0,1,0,2\n"
    )
    hotspot = path / "hotspot.csv"
    hotspot.write_text(
        f"{LEADING},TP53 (7157)\n"
        "CDS-1,ACH-1,MC-1,Yes,Yes,1\n"
        "CDS-2,ACH-1,MC-2,No,Yes,0\n"
        "CDS-3,ACH-2,MC-3,Yes,Yes,0\n"
    )
    copy_number = path / "copy_number.csv"
    copy_number.write_text(",ATM (472),TP53 (7157)\nACH-1,1.0,0.1\nACH-2,,0.7\n")
    return {
        "out": str(path / "out"),
        "depmap": {
            "damaging": str(damaging),
            "hotspot": str(hotspot),
            "copy_number": str(copy_number),
        },
    }


DEPMAP_LINES = ["ACH-1", "ACH-2", "CVCL_0058"]


def test_the_mutation_matrix_keeps_the_default_entry_of_each_line(tmp_path):
    from pathlib import Path

    config = depmap_world(tmp_path)
    matrix = gene_in_line.read_mutation_matrix(Path(config["depmap"]["damaging"]))
    assert list(matrix.index) == ["ACH-1", "ACH-2"]
    assert list(matrix.columns) == ["ATM", "BRCA1", "TP53"]
    np.testing.assert_array_equal(matrix.loc["ACH-1"], [1, 0, 2])
    np.testing.assert_array_equal(matrix.loc["ACH-2"], [0, 1, 0])


def test_a_gene_outside_the_hotspot_matrix_has_hotspot_zero(tmp_path):
    config = depmap_world(tmp_path)
    gene_in_line.depmap_mutation(config, AXIS, DEPMAP_LINES)
    values = cache_of(config, "depmap_mutation", "ACH-1")
    np.testing.assert_array_equal(values[[0, 1, 4]], [[1, 0], [0, 0], [2, 1]])
    assert np.isnan(values[[2, 3]]).all()  # C1ORF1 and KRAS: in neither matrix


def test_copy_number_holds_the_value_and_the_deep_loss_flag(tmp_path):
    config = depmap_world(tmp_path)
    gene_in_line.depmap_copy_number(config, AXIS, DEPMAP_LINES)
    # 2 ** 0.1 - 1 = 0.07 is below 2 ** -0.8 = 0.57; 2 ** 0.7 - 1 = 0.62 is not
    np.testing.assert_allclose(
        cache_of(config, "depmap_copy_number", "ACH-1")[[0, 4]],
        [[1.0, 0.0], [0.1, 1.0]],
        rtol=1e-3,
    )
    values = cache_of(config, "depmap_copy_number", "ACH-2")
    assert np.isnan(values[0]).all()  # ATM: no value in the matrix
    np.testing.assert_allclose(values[4], [0.7, 0.0], rtol=1e-3)


def test_a_line_without_depmap_data_gets_nan_rows(tmp_path):
    import json

    config = depmap_world(tmp_path)
    gene_in_line.depmap_mutation(config, AXIS, DEPMAP_LINES)
    gene_in_line.depmap_copy_number(config, AXIS, DEPMAP_LINES)
    for step in ("depmap_mutation", "depmap_copy_number"):
        assert np.isnan(cache_of(config, step, "CVCL_0058")).all()
        report = json.loads((tmp_path / "out" / step / "report.json").read_text())
        assert report["lines"] == 3 and report["lines_with_source"] == 2
