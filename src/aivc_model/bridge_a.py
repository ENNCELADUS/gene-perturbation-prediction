"""Bridge A forward-inference wiring on the frozen exp05 checkpoint.

Bridge A (counterfactual co-dependency): in K562, feed an OBSERVED basal cell
panel into the frozen ``AivcModel`` and query another gene's predicted DepMap
GeneEffect. Two basal states are compared per query gene ``b``:

* ``c_hat[b | control]``     -- basal = the pooled non-targeting K562 control
  cells.
* ``c_hat[b | a-perturbed]`` -- basal = the OBSERVED ``a``-perturbed K562
  cell bag (Replogle GWPS Perturb-seq cells for gene ``a``).

Co-dependency spike (GeneEffect is more negative when more essential)::

    delta(a -> b) = c_hat[b | control] - c_hat[b | a-perturbed]

Symmetrized score::

    s_A(a, b) = 0.5 * (delta(a -> b) + delta(b -> a))

This module only wires and validates the forward path; it makes no claim
about the correspondence between ``s_A`` and measured Horlbeck GI (see
``scripts/bridge_a_forward.py`` for the smoke gate and a tiny sanity-only
pilot correlation).

Pooler feature composition (grounds Finding 1)
-----------------------------------------------
``TrainableDiagonalGMM.forward(bag, control_bag)`` (``response.py:68-77``)
concatenates five blocks, computed from ``bag`` (the query gene's predicted
response latent) and ``control_bag`` (the reference latent, named
``control_latent`` at every call site)::

    cat([occupancy(bag), occupancy(bag) - occupancy(control_bag),
         bag.mean(0), bag.var(0), entropy(occupancy(bag))])

Only the second block, ``occupancy(bag) - occupancy(control_bag)``, reads
``control_bag`` (REFERENCE-RELATIVE); the other four -- ``occupancy(bag)``,
``bag.mean(0)``, ``bag.var(0)``, and the entropy of ``occupancy(bag)`` -- are
ABSOLUTE functions of ``bag`` alone. So only ``n_components`` of the
``2 * n_components + 2 * latent_dim + 1`` output dims are reference-relative.
At training/eval time (``train.py:_control_panel_latent``,
``_final_prediction_tensor``, ~2635-2720) ``control_bag`` is always
``response_encoder(pooled_control_panel)`` -- an in-distribution reference the
head was calibrated against.

Reference-latent convention for the a-perturbed arm (Finding 1)
-----------------------------------------------------------------
Two conventions for ``control_latent`` in ``c_hat[b | a-perturbed]``, chosen
by ``reference_convention`` (CLI: ``--reference-convention {self,control}``).
Neither is asserted correct a priori; both are computed and reported by the
pilot, leaving adjudication to the future Horlbeck sweep.

* ``"self"``    -- ``control_latent = response_encoder(a-perturbed panel)``,
  mirroring the control arm's self-reference. The occupancy-difference block
  becomes "b's response relative to a's own basal distribution" -- a
  combination the head never saw in training, where the reference was always
  the in-distribution pooled control.
* ``"control"`` -- ``control_latent`` is always
  ``response_encoder(pooled control panel)``; only ``control_chunks``
  (the STATE basal input) becomes the a-perturbed cells. In-distribution
  reference, matching training, at the cost of no longer self-referencing
  ``a`` in that one block.

Panel construction: two SEPARATE mechanisms
----------------------------------------------
``control_panel``/``PanelSpec`` in this module build a FIXED nominal-size
pooled-control panel and are used ONLY by the CONTROL smoke gate's bit-exact
reproduction of ``predictions.csv`` (``scripts/bridge_a_forward.py``). They
never build an a-perturbed panel.

Every a-perturbed panel -- for the perturbed-arm gate and the pilot -- is
built by ``aivc_model.bridge_a_independent`` at an INDEPENDENT-cell-count
window budget (Finding A), not a flat nominal cell target: see that
module's docstring and ``bridge_a_panels.compute_window_budget`` for the
windowed / capped / sub-window-bootstrap-and-flag policy. A flat target
(e.g. bootstrapping an 8-cell bag WITH replacement up to a nominal 1024)
equalizes window COUNT but not independent SAMPLE count, and was found to
mask a small bag's true Monte Carlo variance -- that mechanism has been
removed, not merely tuned.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aivc_model.cross_validate import (
    _assert_canonical_universe,
    _file_sha256,
    _fixed_manifest_authority,
    _load_primary_bags,
    _manifest_authority,
)
from aivc_model.gene_splits import (
    fixed_fold_spec,
    load_canonical_outer_manifest,
    load_fixed_split_manifest,
    sha256_file,
)
from aivc_model.model import AivcModel
from aivc_model.prepare import (
    AivcConfig,
    GeneBags,
    load_config,
    load_external_gene_bags,
    load_state_batch_lookup,
    merge_gene_bag_pool,
)
from aivc_model.train import (
    _EvaluationControlPanel,
    _build_e2e_model,
    _build_evaluation_control_panel,
    _configure_float32_matmul_precision,
    _control_panel_latent,
)

_LOGGER = logging.getLogger(__name__)

CONTROL_BASAL_STATE = "control"
REFERENCE_CONVENTIONS = ("self", "control")


@dataclass(frozen=True)
class BridgeAContext:
    """A frozen ``AivcModel`` plus the exact fixed-pool data it was fit on."""

    model: AivcModel
    data: GeneBags
    batch_lookup: dict[str, int]
    config: AivcConfig
    eval_seed: int
    device: torch.device
    gene_to_index: dict[str, int]


@dataclass(frozen=True)
class PanelSpec:
    """Fixed nominal-size STATE panel sizing for the pooled control panel.

    Used ONLY by ``control_panel`` (in turn used only by the CONTROL smoke
    gate's bit-exact reproduction of ``predictions.csv``, where
    ``target_cells`` matches ``config.train.eval_control_panel_size``
    exactly). NOT used for a-perturbed panels -- see
    ``aivc_model.bridge_a_independent.IndependentPanelSpec`` and this
    module's docstring.
    """

    cell_set_len: int
    target_cells: int
    macro_batch_windows: int
    seed: int


def verify_checkpoint_sha256(checkpoint_path: Path, expected_sha256: str) -> None:
    """Raise if the checkpoint file's SHA-256 does not match the frozen digest.

    Args:
        checkpoint_path: Path to the ``pytorch_model.bin`` state dict.
        expected_sha256: The frozen, expected hex digest.

    Raises:
        ValueError: If the observed digest differs from ``expected_sha256``.
    """
    observed = sha256_file(checkpoint_path)
    if observed != expected_sha256:
        raise ValueError(
            "Bridge A checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed} at {checkpoint_path}"
        )


def load_bridge_a_context(
    config_path: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    device: torch.device,
) -> BridgeAContext:
    """Reconstruct the exact fixed-pool ``AivcModel`` and load the frozen checkpoint.

    Rebuilds the model architecture through the same code path as
    ``cross_validate.run_cross_validation`` / ``cross_validate._run_fixed_split``
    (canonical + fixed-pool manifests, external-supplement pool merge, ESM-2
    perturbation vocabulary), so the checkpoint's state dict loads with exact
    shape agreement. Training-only orchestration (``accelerate``, distributed
    preflight, fold-role gene-access authorization) is intentionally skipped;
    this is a read-only forward-inference reconstruction, not a training run.

    Args:
        config_path: Path to the exp05 fixed-pool training YAML config.
        checkpoint_path: Path to the frozen ``pytorch_model.bin`` state dict.
        expected_checkpoint_sha256: Frozen hex digest the checkpoint must match.
        device: Torch device to move the model to.

    Returns:
        A ``BridgeAContext`` with the loaded model in ``eval()`` mode and
        every parameter frozen (``requires_grad_(False)``).
    """
    config = load_config(config_path)
    _configure_float32_matmul_precision(config)

    data = _load_primary_bags(config)
    labels = pd.DataFrame(
        {
            "perturbation_gene": [str(gene).upper() for gene in data.genes],
            "depmap_gene_effect": np.asarray(data.y, dtype=float),
        }
    )
    manifest_path, expected_outer_sha256 = _manifest_authority(config)
    if _file_sha256(manifest_path) != expected_outer_sha256:
        raise ValueError("canonical outer manifest SHA-256 mismatch")
    canonical_manifest = load_canonical_outer_manifest(
        manifest_path,
        labels,
        expected_outer_sha256,
    )
    _assert_canonical_universe(canonical_manifest)

    supplement = load_external_gene_bags(
        config,
        data,
        checkpoint_path.parent,  # unused: only read when projector.teacher=="scvi"
        project_scvi=False,
        source_as_batch=True,
    )
    if supplement is not None:
        data = merge_gene_bag_pool(data, supplement.data, config.data.depmap_label_col)

    fixed_labels = pd.DataFrame(
        {
            "perturbation_gene": [str(gene).upper() for gene in data.genes],
            "depmap_gene_effect": np.asarray(data.y, dtype=float),
        }
    )
    fixed_manifest_path, expected_fixed_sha256 = _fixed_manifest_authority(config)
    fixed_manifest = load_fixed_split_manifest(
        fixed_manifest_path,
        fixed_labels,
        expected_fixed_sha256,
    )
    fold = fixed_fold_spec(fixed_manifest)

    canonical_genes = tuple(canonical_manifest["perturbation_gene"].astype(str))
    canonical_set = set(canonical_genes)
    added_genes = tuple(
        sorted(set(fixed_labels["perturbation_gene"]).difference(canonical_set))
    )
    canonical_gene_order = canonical_genes + added_genes

    model = _build_e2e_model(
        config,
        data,
        extra_genes=(),
        canonical_gene_order=canonical_gene_order,
        emit_checkpoint_output=False,
    )
    verify_checkpoint_sha256(checkpoint_path, expected_checkpoint_sha256)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    batch_lookup = load_state_batch_lookup(config.state.model_dir)
    eval_seed = int(config.train.seed) + int(fold.outer_fold)
    gene_to_index = {str(gene).upper(): index for index, gene in enumerate(data.genes)}
    _LOGGER.info(
        "Bridge A context loaded: %d fixed-pool genes, eval_seed=%d, device=%s",
        len(data.genes),
        eval_seed,
        device,
    )
    return BridgeAContext(
        model=model,
        data=data,
        batch_lookup=batch_lookup,
        config=config,
        eval_seed=eval_seed,
        device=device,
        gene_to_index=gene_to_index,
    )


def control_panel(context: BridgeAContext, spec: PanelSpec) -> _EvaluationControlPanel:
    """Build the pooled non-targeting control panel at a FIXED nominal size.

    Used only by the CONTROL smoke gate (see module docstring). Every
    a-perturbed panel, and every independent-N-matched control panel used to
    score it, is built by ``aivc_model.bridge_a_independent`` instead.
    """
    return _build_evaluation_control_panel(
        context.data,
        spec.cell_set_len,
        spec.target_cells,
        spec.macro_batch_windows,
        spec.seed,
        context.device,
        context.batch_lookup,
    )


def panel_latent(model: AivcModel, panel: _EvaluationControlPanel) -> torch.Tensor:
    """Encode a basal panel with the shared response encoder."""
    return _control_panel_latent(model, panel)


def predict_c_hat(
    model: AivcModel,
    panel: _EvaluationControlPanel,
    control_latent: torch.Tensor,
    gene: str,
) -> float:
    """Predict ``c_hat[gene | panel's basal state]``.

    Mirrors ``train._final_prediction_tensor``'s per-gene body: concatenates
    the predicted response latent across every macro-batch window of
    ``panel`` before pooling, matching exactly how ``predictions.csv`` was
    produced for the control arm.

    Args:
        model: A loaded, ``eval()``-mode ``AivcModel``.
        panel: The basal panel to feed as ``predict_response_chunks``' basal
            cells.
        control_latent: The response-encoder latent used as the pooler's
            reference bag (see module docstring for the a-perturbed-arm
            convention).
        gene: The query gene symbol.

    Returns:
        The scalar predicted GeneEffect.
    """
    predicted_latents: list[torch.Tensor] = []
    for control_chunks, batch_chunks in panel.macro_batches():
        _expression_chunks, predicted_latent = model.predict_response_chunks(
            control_chunks,
            gene,
            batch_chunks,
        )
        predicted_latents.append(predicted_latent)
    c_hat = model.predict_c_from_latents(
        torch.cat(predicted_latents, dim=0),
        control_latent,
    )
    return float(c_hat.reshape(()).item())


def delta_a_to_b(
    c_hat_b_given_control: float,
    c_hat_b_given_a_perturbed: float,
) -> float:
    """Co-dependency spike delta(a -> b) = c_hat[b|control] - c_hat[b|a-perturbed].

    GeneEffect is more negative when a gene is more essential, so a positive
    delta means ``b`` looks more essential once ``a`` is lost -- a
    co-dependency spike.
    """
    return float(c_hat_b_given_control) - float(c_hat_b_given_a_perturbed)


def symmetrized_codependency(delta_ab: float, delta_ba: float) -> float:
    """s_A(a, b) = 0.5 * (delta(a -> b) + delta(b -> a))."""
    return 0.5 * (float(delta_ab) + float(delta_ba))


def reference_latent(
    convention: str,
    shared_control_latent: torch.Tensor,
    basal_latent: torch.Tensor,
) -> torch.Tensor:
    """Select the pooler reference latent for the a-perturbed arm (Finding 1).

    See the module docstring's "Reference-latent convention" section: neither
    convention is asserted correct a priori. Public (used by
    ``aivc_model.bridge_a_independent``, the independent-N-matched forward
    orchestration -- see that module for ``compute_independent_c_hat_table``,
    the Finding-A replacement for the old flat-panel ``compute_c_hat_table``).

    Raises:
        ValueError: If ``convention`` is not one of ``REFERENCE_CONVENTIONS``.
    """
    if convention == "self":
        return basal_latent
    if convention == "control":
        return shared_control_latent
    raise ValueError(
        f"unknown reference_convention {convention!r}; expected one of "
        f"{REFERENCE_CONVENTIONS}"
    )


def sample_genes(universe: list[str], n: int, seed: int) -> list[str]:
    """Deterministically sample ``n`` distinct gene symbols from ``universe``."""
    rng = np.random.default_rng(seed)
    if n > len(universe):
        raise ValueError(
            f"requested {n} genes but the universe only has {len(universe)}"
        )
    indices = rng.choice(len(universe), size=n, replace=False)
    return sorted(universe[index] for index in indices)


def run_smoke_gate(
    context: BridgeAContext,
    reference_csv: Path,
    n_genes: int,
    seed: int,
    spec: PanelSpec,
    output_dir: Path,
) -> tuple[dict[str, float], list[float]]:
    """Reproduce c_hat[gene | control] against the frozen run's predictions.csv.

    Gates on MAX absolute error, not mean -- a single outlier gene hiding
    behind a low mean must still fail.

    Args:
        context: A loaded ``BridgeAContext``.
        reference_csv: Path to the frozen run's ``predictions.csv``.
        n_genes: How many ``internal_outer_test`` genes to reproduce.
        seed: Deterministic gene-sampling seed.
        spec: The FIXED nominal-size control panel spec to reproduce against
            (matches ``config.train.eval_control_panel_size``).
        output_dir: Directory to write the per-gene comparison CSV to.

    Returns:
        A tuple of the summary dict (``mae``, ``max_abs_error``, gene count)
        and the list of per-gene forward wall-clock seconds.

    Raises:
        ValueError: If ``reference_csv`` has no ``internal_outer_test`` rows.
        RuntimeError: If any reproduced ``c_hat`` is non-finite.
    """
    reference = pd.read_csv(reference_csv)
    reference = reference.loc[reference["evaluation_scope"] == "internal_outer_test"]
    if reference.empty:
        raise ValueError(f"no internal_outer_test rows found in {reference_csv}")
    genes = sample_genes(sorted(reference["perturbation_gene"].unique()), n_genes, seed)
    _LOGGER.info(
        "control smoke gate: reproducing %d internal_test genes: %s", len(genes), genes
    )

    forward_seconds: list[float] = []
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        panel = control_panel(context, spec)
        latent = panel_latent(context.model, panel)
        for gene in genes:
            start = time.perf_counter()
            c_hat = predict_c_hat(context.model, panel, latent, gene)
            forward_seconds.append(time.perf_counter() - start)
            reference_row = reference.loc[reference["perturbation_gene"] == gene].iloc[
                0
            ]
            rows.append(
                {
                    "perturbation_gene": gene,
                    "reference_y_pred": float(reference_row["y_pred"]),
                    "bridge_a_c_hat": c_hat,
                    "absolute_error": abs(c_hat - float(reference_row["y_pred"])),
                }
            )
    comparison = pd.DataFrame(rows)
    if not comparison["bridge_a_c_hat"].apply(math.isfinite).all():
        raise RuntimeError("smoke gate produced a non-finite c_hat value")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "smoke_reproduction.csv", index=False)

    summary = {
        "n_genes": int(len(comparison)),
        "mae": float(comparison["absolute_error"].mean()),
        "max_abs_error": float(comparison["absolute_error"].max()),
    }
    _LOGGER.info(
        "control smoke gate: n=%d mae=%.3e max_abs_error=%.3e",
        summary["n_genes"],
        summary["mae"],
        summary["max_abs_error"],
    )
    return summary, forward_seconds
