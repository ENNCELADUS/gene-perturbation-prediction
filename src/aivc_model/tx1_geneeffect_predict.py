"""Phase D Task 4: emit the Phase F artifact contract for the 9 held-out lines.

Discovery A (`.superpowers/sdd/phase-d/discovery-a-components.md` Sec.8)
found the head/loss tests and the calibration/evaluator tests are two
disjoint universes: no test anywhere wires a real ``Tx1GeneEffectHead``
forward pass through ``tx1_fewshot_calibration.make_predictions_long`` into
``tx1_geneeffect_eval``'s validators. This module is that seam, for
inference over the 9 frozen held-out (``role="test"``) lines specifically
-- Task 3's ``tx1_geneeffect_train``/``tx1_geneeffect_train_io`` build and
select the head over the 28/5/9 split's TRAIN and VALIDATION lines only.

Two things a training-time assembler must not do, that this INFERENCE
assembler must:

1. **Cover every requested gene, not just the ones with a finite target.**
   ``tx1_geneeffect_train_io.assemble_line_features`` intersects the
   requested genes with ``gene_effect.notna()`` -- correct for
   training (a gene with no signal is simply not trained on), wrong here:
   ``tx1_geneeffect_eval._validate_panel_aware_coverage`` requires a
   prediction ROW for every scored slice gene regardless of whether
   ``y_true`` is finite (held-out lines are not completeness-guaranteed --
   see ``tx1_fewshot_calibration.fit_ridge_calibration``'s identical
   caveat). :func:`assemble_test_line_features` therefore assembles every
   gene in the caller's slice, filling a missing DepMap target with NaN
   rather than dropping the gene.
2. **Guard the held-out-only path deliberately.**
   :func:`assert_test_role` is the mirror image of
   ``tx1_geneeffect_data.assert_training_role``:
   it raises unless ``role`` IS the frozen test role, so this module can
   never be accidentally pointed at a train/validation line and silently
   produce an inference row for a line that was never supposed to need one.

Producing the D1 predictions table is wired through
``tx1_fewshot_calibration.make_predictions_long`` (:func:`build_test_line_predictions`),
never hand-rolled: that function already knows the panel-aware
``[model_id, panel, depmap_column, method, k, base_pred, y_true]`` schema,
and hand-rolling it here would create a second definition of the contract
that can drift from the one ``tx1_geneeffect_eval.py`` actually enforces.
:func:`build_tx1_3b_st_predictions` concatenates one such frame per held-out
line into the single ``tx1_3b_st`` table
``scripts/evaluate_tx1_backbone.py`` expects.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from aivc_model.tx1_embed_cache import load_line_cache
from aivc_model.tx1_fewshot_calibration import DEFAULT_ALPHA, make_predictions_long
from aivc_model.tx1_geneeffect_data import TEST_ROLE
from aivc_model.tx1_geneeffect_eval import DEFAULT_PRIMARY_METHOD, K_SCHEDULE
from aivc_model.tx1_geneeffect_head import Tx1GeneEffectHead, moment_pool
from aivc_model.tx1_geneeffect_train import (
    LineData,
    PooledLineExamples,
    compute_line_features_and_predictions,
)
from aivc_model.tx1_predicted_response import (
    ARM_HVG,
    ARM_TX1,
    ForwardOnlyStateModel,
    generate_predicted_response,
    resolve_genes_against_vocabulary,
)

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "assemble_test_line_features",
    "assert_test_role",
    "build_test_line_predictions",
    "build_tx1_3b_st_predictions",
]


def assert_test_role(role: str, model_id: str) -> None:
    """Hard guard: raise unless ``role`` is the frozen held-out test role.

    The mirror image of ``tx1_geneeffect_data.assert_training_role``: that
    guard raises on ``TEST_ROLE`` (refusing a held-out line into training);
    this one raises on anything ELSE (refusing a train/validation line into
    held-out inference). Both directions matter -- this module's whole
    contract is "the 9 test lines and nothing else", so a caller passing it
    a ``train_head`` line by mistake (e.g. a copy-pasted ``model_id`` list)
    would otherwise silently produce a Phase F prediction row for a line
    that was never supposed to need one.

    Args:
        role: The candidate line's manifest ``role`` value.
        model_id: The line's ACH model id, included only for the error
            message.

    Raises:
        ValueError: ``role != TEST_ROLE``.
    """
    if role != TEST_ROLE:
        raise ValueError(
            f"line {model_id}: held-out inference is for role={TEST_ROLE!r} "
            f"lines only, got role={role!r}; refusing to route a training/"
            "validation line through the Phase F held-out-inference path"
        )


def assemble_test_line_features(
    model_id: str,
    role: str,
    model: ForwardOnlyStateModel,
    arm: str,
    tx1_cache_dir: Path,
    slice_df: pd.DataFrame,
    gene_effect: pd.Series,
    *,
    moments: int,
    cell_set_len: int,
    seed: int,
    batch_lookup: Mapping[str, int] | None = None,
    device: torch.device | str = "cpu",
) -> PooledLineExamples:
    """Stream and pool one held-out line without retaining raw responses.

    Forwards ST live, gene by gene, via
    ``tx1_predicted_response.generate_predicted_response_for_line(...,
    require_training_role=False)`` -- Task 2's explicit held-out-line
    opt-out -- rather than reading Task 2/3's predicted-response CACHE
    (populated only for the 28/5 training-role lines). ``require_training_role
    =False`` is passed explicitly and only after :func:`assert_test_role`
    passes, so this is never a silent bypass of D6.

    Reloading this line's basal cache once per gene (one call per gene to
    ``generate_predicted_response_for_line``, which itself calls
    ``load_line_cache``) is a deliberate simplicity-over-micro-efficiency
    choice: it keeps every gene's forward call routed through Task 2's own
    tested, role-guarded entry point rather than partially reimplementing
    it here to save a handful of memory-mapped reads.

    Args:
        model_id: ACH model id (one of the 9 frozen held-out test lines).
        role: This line's manifest ``role`` value; must equal
            ``tx1_geneeffect_data.TEST_ROLE`` (checked by
            :func:`assert_test_role` before anything else runs).
        model: A loaded ``ForwardOnlyStateModel`` (Task 2).
        arm: ``"tx1_arm"`` or ``"hvg_arm"``.
        tx1_cache_dir: Root of the Phase B basal-embedding cache.
        slice_df: The differentially-essential slice (as amended, read from
            disk by the caller -- never a hardcoded gene count here), with
            columns ``[depmap_column, gene_symbol]``. ``gene_symbol`` is
            resolved against the ST vocabulary (with alias correction, e.g.
            CRIPTO/HEMK2); ``depmap_column`` becomes this line's
            :attr:`LineExamples.genes`, matching the frozen panels table and
            Phase F's output schema.
        gene_effect: DepMap GeneEffect targets for this line, indexed by
            ``depmap_column``. A slice gene absent from the index gets a NaN
            target (held-out lines are not completeness-guaranteed) rather
            than being dropped from the returned examples.
        cell_set_len: ST cell-set window size (forwarded).
        seed: Deterministic padding-resample seed (forwarded).
        batch_lookup: Optional batch-label lookup (forwarded).
        device: Torch device for the forward pass.

    Returns:
        A :class:`LineExamples` covering every gene in ``slice_df``, with
        ``role="test"``.

    Raises:
        ValueError: ``role != TEST_ROLE``.
        UnknownPerturbationGeneError: A ``gene_symbol`` (after alias
            resolution) is outside the ST vocabulary -- unexpected for the
            already vocabulary-amended frozen slice; a raise here means the
            slice or the vocabulary has drifted since amendment.
    """
    assert_test_role(role, model_id)
    depmap_columns = [str(value) for value in slice_df["depmap_column"]]
    gene_symbols = [str(value) for value in slice_df["gene_symbol"]]
    resolved_symbols = resolve_genes_against_vocabulary(
        gene_symbols, model.perturbations
    )

    embeddings, hvg_matrix, _obs = load_line_cache(tx1_cache_dir, model_id)
    if arm not in {ARM_TX1, ARM_HVG}:
        raise ValueError(f"unknown arm {arm!r}")
    basal_view = np.asarray(
        embeddings if arm == ARM_TX1 else hvg_matrix, dtype=np.float32
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        basal_tensor = torch.from_numpy(basal_view)
    pooled_basal = moment_pool(basal_tensor, moments=moments)
    batch_labels = np.full(basal_view.shape[0], model_id, dtype=object)
    pooled_responses: list[torch.Tensor] = []
    response_dim: int | None = None
    for gene_index, gene in enumerate(resolved_symbols, start=1):
        response = generate_predicted_response(
            model,
            basal_view,
            gene,
            cell_set_len=cell_set_len,
            seed=seed,
            batch_labels=batch_labels,
            batch_lookup=batch_lookup,
            device=device,
        )
        response_dim = int(response.shape[-1])
        pooled_responses.append(moment_pool(response, moments=moments))
        del response
        if gene_index % 50 == 0 or gene_index == len(resolved_symbols):
            _LOGGER.info(
                "streamed held-out %s/%s: %d/%d genes",
                model_id,
                arm,
                gene_index,
                len(resolved_symbols),
            )
    pooled_response_matrix = torch.stack(pooled_responses, dim=0)
    features = torch.cat(
        [
            pooled_response_matrix,
            pooled_basal.unsqueeze(0).expand(len(depmap_columns), -1),
        ],
        dim=1,
    )
    targets = torch.as_tensor(
        [float(gene_effect.get(column, float("nan"))) for column in depmap_columns],
        dtype=torch.float32,
    )
    _LOGGER.info(
        "assembled held-out line %s (arm=%s): %d slice genes, %d finite "
        "GeneEffect targets",
        model_id,
        arm,
        len(depmap_columns),
        int(torch.isfinite(targets).sum()),
    )
    return PooledLineExamples(
        model_id=str(model_id),
        role=str(role),
        arm=str(arm),
        genes=tuple(depmap_columns),
        features=features,
        targets=targets,
        response_dim=int(response_dim),
        basal_dim=int(basal_view.shape[-1]),
        moments=int(moments),
    )


def build_test_line_predictions(
    head: Tx1GeneEffectHead,
    line: LineData,
    panels_for_line: pd.DataFrame,
    k_schedule: Sequence[int],
    *,
    method: str = DEFAULT_PRIMARY_METHOD,
    alpha: float = DEFAULT_ALPHA,
    residual: bool = True,
) -> pd.DataFrame:
    """One held-out line's panel-aware ``tx1_3b_st`` predictions frame.

    D2: computes BOTH the pooled ``features`` matrix and the k=0 scalar
    ``base_pred`` via ``tx1_geneeffect_train.compute_line_features_and_predictions``
    (the same code path Task 3's validation loop uses), then hands both,
    plus ``y_true`` and ``genes``, to
    ``tx1_fewshot_calibration.make_predictions_long`` -- never a hand-rolled
    frame -- so the emitted schema is bit-for-bit what that function
    defines.

    Args:
        head: A trained (or, for tests, untrained) ``Tx1GeneEffectHead``.
        line: This held-out line's :class:`LineExamples`; ``line.role`` must
            equal ``tx1_geneeffect_data.TEST_ROLE`` (:func:`assert_test_role`).
        panels_for_line: This line's rows of the frozen ``k_label_panels.csv``
            (columns ``[panel, label_order, depmap_column]``).
        k_schedule: k values to emit rows for (the frozen contract requires
            the full ``tx1_geneeffect_eval.K_SCHEDULE``, not a subset).
        method: Method identifier to stamp (frozen contract: ``"tx1_3b_st"``).
        alpha: Ridge regularization strength, forwarded to
            ``make_predictions_long``.
        residual: Forwarded to ``make_predictions_long``.

    Returns:
        A tidy DataFrame with columns ``[model_id, panel, depmap_column,
        method, k, base_pred, y_true]`` -- one row per (panel, gene, k).

    Raises:
        ValueError: ``line.role != TEST_ROLE``.
    """
    assert_test_role(line.role, line.model_id)
    features, predictions = compute_line_features_and_predictions(head, line)
    return make_predictions_long(
        model_id=line.model_id,
        genes=np.array(line.genes),
        features=features.numpy(),
        base_pred=predictions.numpy(),
        y_true=line.targets.numpy(),
        panels_for_line=panels_for_line,
        k_schedule=list(k_schedule),
        method=method,
        alpha=alpha,
        residual=residual,
    )


def build_tx1_3b_st_predictions(
    head: Tx1GeneEffectHead,
    lines: Sequence[LineData],
    panels: pd.DataFrame,
    *,
    k_schedule: Sequence[int] = K_SCHEDULE,
    method: str = DEFAULT_PRIMARY_METHOD,
    alpha: float = DEFAULT_ALPHA,
    residual: bool = True,
) -> pd.DataFrame:
    """Assemble the full D1 panel-aware predictions table for every held-out line.

    One ``tx1_3b_st`` predictions frame per line (:func:`build_test_line_predictions`),
    concatenated. Coverage of "all 9 frozen held-out lines" is the CALLER's
    responsibility (pass all 9 lines' :class:`LineExamples`); this function
    only guarantees no line is silently dropped or duplicated.

    Args:
        head: A trained ``Tx1GeneEffectHead``.
        lines: Every held-out line's :class:`LineExamples`, each with
            ``role == TEST_ROLE``.
        panels: The full frozen ``k_label_panels.csv`` (all held-out lines'
            rows; restricted per-line internally).
        k_schedule: k values every line is scored at (frozen contract:
            ``tx1_geneeffect_eval.K_SCHEDULE``).
        method: Method identifier to stamp (frozen contract: ``"tx1_3b_st"``).
        alpha: Ridge regularization strength, forwarded.
        residual: Forwarded.

    Returns:
        The concatenated ``tx1_3b_st`` predictions table, ready to be
        combined with a baseline method's rows and handed to
        ``scripts/evaluate_tx1_backbone.py``.

    Raises:
        ValueError: ``lines`` is empty, some line's role is not ``TEST_ROLE``,
            or a ``model_id`` repeats across ``lines``.
    """
    if not lines:
        raise ValueError("lines must be non-empty")
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    for line in lines:
        assert_test_role(line.role, line.model_id)
        if line.model_id in seen:
            raise ValueError(
                f"duplicate held-out line model_id in `lines`: {line.model_id}"
            )
        seen.add(line.model_id)
        panels_for_line = panels[panels["model_id"] == line.model_id]
        frames.append(
            build_test_line_predictions(
                head,
                line,
                panels_for_line,
                k_schedule,
                method=method,
                alpha=alpha,
                residual=residual,
            )
        )
    _LOGGER.info(
        "assembled tx1_3b_st predictions for %d held-out line(s): %d total rows",
        len(frames),
        sum(len(frame) for frame in frames),
    )
    return pd.concat(frames, ignore_index=True)
