"""Ordinary raw-UMI q_sc command keeps panel/source correctness boundaries."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import anndata as ad

from src.data import geneeffect
from src.data.geneeffect import Exp13Split
from src.data.prepare import build_exp13_q_sc_cache as cli
from src.data.q_sc import load_q_sc_line


def test_build_from_raw_sources_and_reuse_without_source_read(tmp_path, monkeypatch):
    split = Exp13Split(("A",), ("V",), ("T",), ())
    monkeypatch.setattr(geneeffect, "load_exp13_split", lambda _: split)
    rows = []
    for model_id in split.all_model_ids:
        source = tmp_path / f"{model_id}.h5ad"
        ad.AnnData(
            X=np.array([[0, 1], [2, 3]], dtype=np.float32),
            obs=pd.DataFrame({"model_id": [model_id] * 2}, index=["c1", "c2"]),
            var=pd.DataFrame({"gene_symbol": ["G1", "G2"]}, index=["g1", "g2"]),
        ).write_h5ad(source)
        rows.append(
            dict(
                model_id=model_id,
                source_path=str(source),
                source_kind="h5ad",
                matrix_semantics="raw_umi_counts",
            )
        )
    registry = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)
    panel = tmp_path / "panel.csv"
    panel.write_text("gene_symbol\nG2\nG1\nMISSING\n")
    output = tmp_path / "q_sc"
    args = dict(
        split_path=tmp_path / "split.json",
        registry_path=registry,
        panel_path=panel,
        output_dir=output,
    )
    report = cli.build_q_sc_cache(**args)
    assert report["line_count"] == 3
    value = load_q_sc_line(output, "A", ("G2", "G1", "MISSING"))
    np.testing.assert_allclose(value.values[:2], [[2, 1, 1], [1, 0.5, 1]])
    assert not value.available[-1]

    def forbidden(*a):
        pytest.fail("ready shard rebuilt raw data")

    cli.build_q_sc_cache(**args, reader=forbidden)
    malformed = pd.DataFrame(rows)
    malformed.loc[0, "matrix_semantics"] = "log1p"
    malformed.to_csv(registry, index=False)
    with pytest.raises(ValueError, match="non-raw-UMI"):
        cli.build_q_sc_cache(**args)


def test_cli_argument_dispatch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "build_q_sc_cache", lambda **kw: calls.append(kw))
    assert (
        cli.main(
            ["--split", "s", "--registry", "r", "--panel", "p", "--output-dir", "o"]
        )
        == 0
    )
    assert calls == [
        dict(
            split_path=Path("s"),
            registry_path=Path("r"),
            panel_path=Path("p"),
            output_dir=Path("o"),
        )
    ]
