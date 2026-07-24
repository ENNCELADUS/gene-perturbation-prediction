"""Regression tests for the ``evaluate_tx1_backbone`` CLI script.

``scripts/evaluate_tx1_backbone.py`` is not part of an installed package, so
it is loaded directly by file path (mirroring the pattern already used by
``tests/test_make_tables.py`` for another ``scripts/``-adjacent module).
``main()`` is driven end to end against a tiny, self-contained,
frozen-contract-valid fixture written to a tmp directory; nothing here
touches real prediction files or ``results/phase_a_tx1_20260724``.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts"
    / "evaluate_tx1_backbone.py"
)
_SPEC = importlib.util.spec_from_file_location("evaluate_tx1_backbone", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
evaluate_tx1_backbone = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_tx1_backbone)

N_GENES = 150
N_PANELS = 20
N_LABELS = 50
SLICE_GENES = [f"GENE{i} ({i})" for i in range(N_GENES)]


def _build_panels(model_ids: list[str], seed: int) -> pd.DataFrame:
    """Build a frozen-contract-valid k_label_panels.csv-shaped DataFrame."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model_id in model_ids:
        for panel in range(N_PANELS):
            panel_seed = int(rng.integers(0, 2**31))
            order = rng.permutation(N_GENES)[:N_LABELS]
            for label_order, gene_idx in enumerate(order, start=1):
                rows.append(
                    {
                        "model_id": model_id,
                        "panel": panel,
                        "panel_seed": panel_seed,
                        "label_order": label_order,
                        "depmap_column": SLICE_GENES[gene_idx],
                    }
                )
    return pd.DataFrame(rows)


def _write_fixture(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Write a tiny frozen-contract-valid Phase-A dir + predictions file.

    Returns:
        ``(phase_a_dir, predictions_path)``.
    """
    rng = np.random.default_rng(0)
    test_lines = [f"TEST_{i}" for i in range(9)]
    manifest = pd.DataFrame({"model_id": test_lines, "role": ["test"] * 9})
    slice_df = pd.DataFrame({"depmap_column": SLICE_GENES})
    panels = _build_panels(test_lines, seed=1)

    pred_rows: list[dict[str, object]] = []
    for line in test_lines:
        y_true = rng.normal(size=N_GENES)
        tx1_pred = y_true + 0.1 * rng.normal(size=N_GENES)
        copy_pred = rng.normal(size=N_GENES)
        for gene, yt, tp, cp in zip(SLICE_GENES, y_true, tx1_pred, copy_pred):
            pred_rows.append(
                {
                    "model_id": line,
                    "depmap_column": gene,
                    "method": "tx1_3b_st",
                    "base_pred": tp,
                    "y_true": yt,
                }
            )
            pred_rows.append(
                {
                    "model_id": line,
                    "depmap_column": gene,
                    "method": "copy_k562",
                    "base_pred": cp,
                    "y_true": yt,
                }
            )
    predictions = pd.DataFrame(pred_rows)

    phase_a_dir = tmp_path / "phase_a"
    phase_a_dir.mkdir()
    manifest.to_csv(phase_a_dir / "cell_line_manifest.csv", index=False)
    slice_df.to_csv(phase_a_dir / "differentially_essential_slice.csv", index=False)
    panels.to_csv(phase_a_dir / "k_label_panels.csv", index=False)

    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    return phase_a_dir, predictions_path


def test_strict_run_writes_formal_verdict_json(tmp_path: pathlib.Path) -> None:
    """Default (strict) run writes verdict.json with formal=true."""
    phase_a_dir, predictions_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out_strict"

    exit_code = evaluate_tx1_backbone.main(
        [
            "--predictions",
            str(predictions_path),
            "--phase-a-dir",
            str(phase_a_dir),
            "--out-dir",
            str(out_dir),
        ]
    )

    verdict_path = out_dir / "verdict.json"
    assert verdict_path.exists()
    assert not (out_dir / "verdict_diagnostic.json").exists()
    verdict = json.loads(verdict_path.read_text())
    assert verdict["formal"] is True
    assert "reason" not in verdict
    assert exit_code in (0, 1)  # a formal gate verdict, never the diagnostic 2


def test_allow_partial_writes_diagnostic_verdict_not_formal(
    tmp_path: pathlib.Path, caplog
) -> None:
    """--allow-partial writes verdict_diagnostic.json, never verdict.json."""
    phase_a_dir, predictions_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out_partial"

    with caplog.at_level("WARNING", logger=evaluate_tx1_backbone._LOGGER.name):
        exit_code = evaluate_tx1_backbone.main(
            [
                "--predictions",
                str(predictions_path),
                "--phase-a-dir",
                str(phase_a_dir),
                "--out-dir",
                str(out_dir),
                "--allow-partial",
            ]
        )

    diagnostic_path = out_dir / "verdict_diagnostic.json"
    assert diagnostic_path.exists()
    assert not (out_dir / "verdict.json").exists()
    verdict = json.loads(diagnostic_path.read_text())
    assert verdict["formal"] is False
    assert verdict["reason"] == "partial/diagnostic run (contract validation bypassed)"
    assert exit_code == 2  # distinct from the 0/1 formal gate exit codes
    assert any(
        "PARTIAL" in record.message or "diagnostic" in record.message.lower()
        for record in caplog.records
        if record.levelname == "WARNING"
    )
