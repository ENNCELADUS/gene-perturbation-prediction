"""Regression tests for the ``evaluate_tx1_backbone`` CLI script.

``scripts/evaluate_tx1_backbone.py`` is not part of an installed package, so
it is loaded directly by file path (mirroring the pattern already used by
``tests/test_make_tables.py`` for another ``scripts/``-adjacent module).
``main()`` is driven end to end against a tiny, self-contained,
frozen-contract-valid fixture written to a tmp directory; nothing here
touches real prediction files or ``results/phase_a_tx1_20260724``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from aivc_model.tx1_geneeffect_eval import EvaluationContractError
from aivc_model.tx1_geneeffect_eval import (
    verify_artifact_hashes as _real_verify_artifact_hashes,
)

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
    manifest_path = phase_a_dir / "cell_line_manifest.csv"
    slice_path = phase_a_dir / "differentially_essential_slice.csv"
    panels_path = phase_a_dir / "k_label_panels.csv"
    manifest.to_csv(manifest_path, index=False)
    slice_df.to_csv(slice_path, index=False)
    panels.to_csv(panels_path, index=False)
    _write_matching_registration(phase_a_dir, manifest_path, slice_path, panels_path)

    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    return phase_a_dir, predictions_path


def _write_matching_registration(
    phase_a_dir: pathlib.Path,
    manifest_path: pathlib.Path,
    slice_path: pathlib.Path,
    panels_path: pathlib.Path,
) -> None:
    """Write a ``phase_a_registration.json`` whose hashes match the files.

    Finding 3 regression: ``main()`` now calls ``verify_artifact_hashes``
    before scoring in strict mode, so every strict-mode fixture needs a
    registration file whose recorded SHA-256s actually match the bytes on
    disk (mirroring how the real Phase-A freeze script would produce it).
    """
    registration = {
        "artifacts": {
            "cell_line_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "differential_slice_sha256": hashlib.sha256(
                slice_path.read_bytes()
            ).hexdigest(),
            "k_label_panels_sha256": hashlib.sha256(
                panels_path.read_bytes()
            ).hexdigest(),
        }
    }
    (phase_a_dir / "phase_a_registration.json").write_text(json.dumps(registration))


def _install_lenient_hash_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decouple these CLI tests from the real frozen-registration anchor.

    ``verify_artifact_hashes`` pins its registration-identity check to
    ``FROZEN_REGISTRATION_SHA256``, the SHA-256 of the real, committed
    ``results/phase_a_tx1_20260724/phase_a_registration.json`` (T2 review
    Finding 2). These CLI tests build tiny synthetic fixtures instead, to
    stay self-contained, so this patches
    ``evaluate_tx1_backbone.verify_artifact_hashes`` to trust whatever
    registration file is actually on disk in the fixture at call time --
    the per-artifact hash comparisons (what these tests actually exercise)
    still run for real. Finding 2's real trust anchor is tested directly,
    against the real frozen file, in ``test_tx1_geneeffect_eval.py``.
    """

    def _lenient_verify(phase_a_dir: pathlib.Path) -> None:
        registration_path = pathlib.Path(phase_a_dir) / "phase_a_registration.json"
        expected = hashlib.sha256(registration_path.read_bytes()).hexdigest()
        _real_verify_artifact_hashes(phase_a_dir, expected_registration_sha256=expected)

    monkeypatch.setattr(
        evaluate_tx1_backbone, "verify_artifact_hashes", _lenient_verify
    )


def _write_prediction_manifest(
    path: pathlib.Path,
    predictions_path: pathlib.Path,
) -> pathlib.Path:
    path.write_text(
        json.dumps(
            {
                "predictions_sha256": hashlib.sha256(
                    predictions_path.read_bytes()
                ).hexdigest(),
            }
        )
    )
    return path


def test_prediction_manifest_hash_mismatch_fails_closed(
    tmp_path: pathlib.Path,
) -> None:
    phase_a_dir, predictions_path = _write_fixture(tmp_path)
    prediction_manifest = tmp_path / "prediction_manifest.json"
    prediction_manifest.write_text(
        json.dumps(
            {
                "predictions_sha256": "wrong",
            }
        )
    )
    with pytest.raises(ValueError, match="prediction manifest SHA-256 mismatch"):
        evaluate_tx1_backbone.main(
            [
                "--predictions",
                str(predictions_path),
                "--prediction-manifest",
                str(prediction_manifest),
                "--phase-a-dir",
                str(phase_a_dir),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )


def test_prediction_manifest_hash_match_is_accepted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_lenient_hash_check(monkeypatch)
    phase_a_dir, predictions_path = _write_fixture(tmp_path)
    prediction_manifest = _write_prediction_manifest(
        tmp_path / "prediction_manifest.json",
        predictions_path,
    )
    out_dir = tmp_path / "out_manifest"

    exit_code = evaluate_tx1_backbone.main(
        [
            "--predictions",
            str(predictions_path),
            "--prediction-manifest",
            str(prediction_manifest),
            "--phase-a-dir",
            str(phase_a_dir),
            "--out-dir",
            str(out_dir),
        ]
    )

    verdict = json.loads((out_dir / "verdict.json").read_text())
    assert "formal" not in verdict
    assert "claim_status" not in verdict
    assert exit_code in (0, 1)


def test_strict_run_writes_verdict_json(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default strict run writes verdict.json."""
    _install_lenient_hash_check(monkeypatch)
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
    verdict = json.loads(verdict_path.read_text())
    assert "formal" not in verdict
    assert "reason" not in verdict
    assert exit_code in (0, 1)


def test_strict_run_fails_closed_on_tampered_artifact(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 3 regression: a strict run must verify frozen artifact hashes.

    A same-shaped but modified manifest file must never reach a verdict;
    ``main()`` must raise before any scoring. Uses
    the lenient registration-identity patch (Finding 2 is tested directly
    elsewhere) so this isolates the per-artifact hash comparison the
    tampered manifest is meant to trip.
    """
    _install_lenient_hash_check(monkeypatch)
    phase_a_dir, predictions_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out_tampered"
    manifest_path = phase_a_dir / "cell_line_manifest.csv"
    manifest_path.write_text(manifest_path.read_text() + "\n# tampered\n")

    with pytest.raises(EvaluationContractError, match="SHA-256 mismatch"):
        evaluate_tx1_backbone.main(
            [
                "--predictions",
                str(predictions_path),
                "--phase-a-dir",
                str(phase_a_dir),
                "--out-dir",
                str(out_dir),
            ]
        )
    assert not out_dir.exists()  # failed before any output was written


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--allow-partial"],
        ["--skip-hash-check"],
        ["--rho-min", "0.1"],
        ["--gate-k", "5"],
        ["--k-schedule", "0", "5"],
        ["--primary-method", "candidate"],
        ["--baseline-method", "baseline"],
        ["--expected-test-lines", "0"],
    ],
)
def test_removed_contract_bypasses_are_rejected(removed_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        evaluate_tx1_backbone.parse_args(
            ["--predictions", "predictions.csv", "--out-dir", "out", *removed_args]
        )
