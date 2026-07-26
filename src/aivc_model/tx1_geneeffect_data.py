"""Phase D Task 1: line-role selection and DepMap GeneEffect label loading.

Phase D trains a context-conditioned GeneEffect head on top of the frozen
Phase-A/B Tx1 basal-embedding contract. Two pieces of plumbing that no
existing module provides:

1. A role-aware line selector giving the 28/5/9 train/validation/test split
   Phase D trains and evaluates over. Training roles (``train_head``,
   ``train_response_and_head``) come straight from the frozen manifest via
   ``tx1_basal.load_line_manifest``; the 5-line validation carve-out is a
   *derived*, one-time artifact
   (``configs/experiments/12_tx1_st_geneeffect/phase_d/validation_lines.json``)
   produced once by :func:`generate_validation_line_registration` and
   thereafter loaded read-only by :func:`load_validation_line_registration`,
   which asserts the frozen manifest has not changed underneath it before
   trusting the file.
2. An importable DepMap ``CRISPRGeneEffect.csv`` loader
   (:func:`load_depmap_gene_effect`) producing the raw per-``(gene,
   model_id)`` matrix that the frozen differentially-essential slice does not
   carry (it has only aggregate per-gene statistics). Resolved by CONTENT
   HASH against ``phase_a_registration.json``'s recorded
   ``sources.depmap_gene_effect`` entry -- never by trusting either the
   registration's or the local machine's path -- and joined purely on the
   file's own row index (ACH ``model_id``); the loader never reads any
   free-text line-name column, so a name-keyed join is impossible by
   construction (DepMap's own line-name field is hyphenated where this
   codebase's convention is not -- see ``repo-hazards.md`` -- so a name join
   would silently corrupt every row; only ACH ids are compared here).

:func:`assert_training_role` is the hard guard every consumer must call
before admitting a line to the training path: ``tx1_embed_cache.verify_cache``
and ``load_line_cache`` are role-agnostic (they serve every line, including
the 9 held-out ``test`` lines, since those also need cached basal embeddings
for Phase F inference), so nothing else in the codebase stops a ``test``-role
line from reaching training.

The one-shot ``train_head``/``test`` hold-out Phase A itself drew
(``scripts/freeze_tx1_phase_a.py: _allocate_test_counts`` +
``stratified_holdout``) is mirrored here by *reimplementation*, not import:
``src/`` does not depend on ``scripts/`` (not a wheel package, and that
script's own ``freeze()`` entry point must never be re-run -- it would
regenerate and thus void the hash-pinned Phase-A contract).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from aivc_model.gene_splits import sha256_file

_LOGGER = logging.getLogger(__name__)

TRAIN_HEAD_ROLE: Final[str] = "train_head"
TRAIN_RESPONSE_ROLE: Final[str] = "train_response_and_head"
TEST_ROLE: Final[str] = "test"
_TRAINING_ROLES: Final[frozenset[str]] = frozenset(
    {TRAIN_HEAD_ROLE, TRAIN_RESPONSE_ROLE}
)

_DEFAULT_VALIDATION_COUNT: Final[int] = 5
_DEFAULT_VALIDATION_SEED: Final[int] = 20260726

_STRATIFICATION_RULE: Final[str] = (
    "lineage-stratified hold-out drawn only from train_head lines, mirroring "
    "scripts/freeze_tx1_phase_a.py's stratified_holdout/_allocate_test_counts: "
    "per-lineage allocation proportional to the train_head pool size (floor "
    "of lineage_count * validation_count / pool_size), remainder slots "
    "assigned one at a time to the lineage with the largest fractional "
    "shortfall (ties broken by lineage pool size, then lineage name), never "
    "leaving a lineage with zero train_head lines remaining; then "
    "numpy.random.default_rng(seed).choice per allocated lineage (visited in "
    "sorted alphabetical order) without replacement from that lineage's own "
    "sorted model_id list"
)

__all__ = [
    "LineRoleSplit",
    "assert_training_role",
    "select_line_roles",
    "build_line_role_split",
    "generate_validation_line_registration",
    "load_validation_line_registration",
    "load_depmap_gene_effect",
]


def assert_training_role(role: str, model_id: str) -> None:
    """Hard guard (D6): raise unless ``role`` is admissible for training.

    Every consumer that is about to admit a line into head training,
    GeneEffect label loading, or validation-line selection MUST call this
    first. This is an assertion, not a comment or a docstring note -- a
    ``test``-role line reaching the training path is a Critical defect.

    Args:
        role: The candidate line's ``role`` value from the frozen manifest.
        model_id: The line's ACH model id, included only for the error
            message.

    Raises:
        ValueError: ``role`` is not one of ``{"train_head",
            "train_response_and_head"}``. In particular, ``"test"`` raises.
    """
    if role not in _TRAINING_ROLES:
        raise ValueError(
            f"line {model_id}: refusing to admit role={role!r} into the "
            f"Phase D training path; only {sorted(_TRAINING_ROLES)} are "
            "allowed (D6)"
        )


@dataclass(frozen=True)
class LineRoleSplit:
    """The 28/5/9 train/validation/test partition Phase D trains and evaluates over.

    Attributes:
        train_model_ids: Combined training-role lines minus the 5 validation
            lines (28 on the frozen Phase-A manifest; never hardcoded).
        validation_model_ids: The derived 5-line, lineage-stratified
            model-selection hold-out, drawn only from ``train_head``.
        test_model_ids: The frozen manifest's 9 ``test``-role lines, never
            touched by training or validation selection.
        anchor_model_ids: The 4 ``train_response_and_head`` lines (the only
            lines carrying observed Perturb-seq response supervision); a
            subset of ``train_model_ids`` by construction.
    """

    train_model_ids: tuple[str, ...]
    validation_model_ids: tuple[str, ...]
    test_model_ids: tuple[str, ...]
    anchor_model_ids: tuple[str, ...]


def select_line_roles(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the frozen manifest into the combined training pool and test lines.

    Args:
        manifest: A manifest as returned by ``tx1_basal.load_line_manifest``
            (already validated: ``role`` is a closed enumeration of
            ``{"test", "train_head", "train_response_and_head"}``).

    Returns:
        ``(training, test)``. ``training`` is every row whose role is
        ``train_head`` or ``train_response_and_head`` (the combined 33-line
        pool on the frozen Phase-A manifest); ``test`` is every ``test``-role
        row (9 lines). Neither count is asserted or hardcoded here -- the
        manifest is the sole source of truth for how many lines exist in
        each role.

    Raises:
        ValueError: Either partition is empty, or some row's role is outside
            the training/test partition (defensive: this should be
            unreachable for a manifest that passed ``load_line_manifest``,
            but this function does not assume its caller used that gate).
    """
    training = manifest.loc[manifest["role"].isin(_TRAINING_ROLES)].reset_index(
        drop=True
    )
    test = manifest.loc[manifest["role"] == TEST_ROLE].reset_index(drop=True)
    if training.empty:
        raise ValueError("cell line manifest has no training-role rows")
    if test.empty:
        raise ValueError("cell line manifest has no test-role rows")
    if len(training) + len(test) != len(manifest):
        unrecognized = sorted(
            set(manifest["role"].astype(str)) - _TRAINING_ROLES - {TEST_ROLE}
        )
        raise ValueError(
            f"cell line manifest has unrecognized role values: {unrecognized}"
        )
    for row in training.itertuples(index=False):
        assert_training_role(str(row.role), str(row.model_id))
    return training, test


def _allocate_stratified_counts(counts: pd.Series, total: int) -> dict[str, int]:
    """Proportional-to-pool-size floor allocation across lineages.

    Reimplements ``scripts/freeze_tx1_phase_a.py``'s ``_allocate_test_counts``
    exactly (same floor-then-largest-remainder rule), so the derived
    validation hold-out mirrors Phase A's own ``train_head``/``test`` split
    algorithm. Not imported from ``scripts/`` -- see the module docstring.

    Args:
        counts: Per-lineage line counts within the pool being stratified.
        total: Number of lines to allocate across lineages in total.

    Returns:
        ``{lineage: allocated_count}`` for every lineage that received at
        least one slot. A lineage with fewer than 2 pool lines is never
        allocated a slot (holding one out would leave none of that lineage
        behind for training).

    Raises:
        ValueError: No lineage has at least 2 pool lines, or ``total`` cannot
            be satisfied without holding out every line of some lineage.
    """
    eligible = counts[counts >= 2].sort_index()
    if eligible.empty:
        raise ValueError("no lineage has >=2 pool lines; cannot stratify a hold-out")
    raw = eligible * total / int(counts.sum())
    allocation = np.floor(raw).astype(int).clip(upper=eligible - 1)
    while int(allocation.sum()) < total:
        candidates = [
            lineage
            for lineage in eligible.index
            if allocation[lineage] < eligible[lineage] - 1
        ]
        if not candidates:
            raise ValueError(f"cannot allocate {total} lines across available lineages")
        lineage = max(
            candidates,
            key=lambda value: (raw[value] - allocation[value], counts[value], value),
        )
        allocation[lineage] += 1
    return {str(key): int(value) for key, value in allocation.items() if value}


def _stratified_sample(
    pool: pd.DataFrame, allocation: Mapping[str, int], seed: int
) -> list[str]:
    """Deterministically draw ``allocation[lineage]`` ids per lineage.

    Mirrors ``freeze_tx1_phase_a.stratified_holdout``'s sampling loop: one
    ``numpy.random.default_rng(seed)`` draws, per lineage in sorted
    (alphabetical) order, ``allocation[lineage]`` ids without replacement
    from that lineage's own sorted ``model_id`` list.
    """
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for lineage in sorted(allocation):
        ids = sorted(pool.loc[pool["lineage"] == lineage, "model_id"].astype(str))
        chosen = rng.choice(ids, size=allocation[lineage], replace=False)
        selected.extend(str(value) for value in chosen)
    return selected


def generate_validation_line_registration(
    manifest: pd.DataFrame,
    manifest_path: Path,
    *,
    validation_count: int = _DEFAULT_VALIDATION_COUNT,
    seed: int = _DEFAULT_VALIDATION_SEED,
) -> dict[str, object]:
    """Pick the one-time, lineage-stratified validation hold-out.

    Mirrors Phase A's own ``train_head``/``test`` hold-out rule
    (:func:`_allocate_stratified_counts` + :func:`_stratified_sample`,
    reimplementing ``scripts/freeze_tx1_phase_a.py``'s
    ``_allocate_test_counts``/``stratified_holdout``), applied to the
    ``train_head`` lines only -- so the ``train_response_and_head`` anchors
    and the frozen ``test`` lines can never be selected, by construction.

    This function has **no side effect**: it does not write
    ``validation_lines.json``. Generate the registration once, persist the
    result yourself, and treat the file as read-only from then on; every
    subsequent Phase D run must call :func:`load_validation_line_registration`
    instead, which enforces that the manifest has not changed since this was
    derived.

    Args:
        manifest: The frozen manifest, as returned by
            ``tx1_basal.load_line_manifest``.
        manifest_path: Path to the manifest file this was derived from, used
            only to compute ``derived_from_sha256``.
        validation_count: Number of validation lines to select.
        seed: Fixed seed for the deterministic stratified draw.

    Returns:
        A JSON-serializable registration dict: ``validation_model_ids``
        (sorted), ``validation_count``, ``seed``, ``rule`` (a human-readable
        description of the stratification algorithm),
        ``validation_lineage_allocation`` (per-lineage counts, for audit),
        and ``derived_from_sha256`` (the manifest file's sha256 at the time
        of derivation).

    Raises:
        ValueError: No ``train_head`` rows exist, or the stratified draw does
            not produce exactly ``validation_count`` lines.
    """
    train_head = manifest.loc[manifest["role"] == TRAIN_HEAD_ROLE]
    if train_head.empty:
        raise ValueError(
            "manifest has no train_head rows to draw validation lines from"
        )
    counts = train_head["lineage"].value_counts()
    allocation = _allocate_stratified_counts(counts, validation_count)
    selected = sorted(_stratified_sample(train_head, allocation, seed))
    if len(selected) != validation_count:
        raise ValueError(
            f"stratified allocation produced {len(selected)} lines, expected "
            f"{validation_count}"
        )
    return {
        "validation_model_ids": selected,
        "validation_count": validation_count,
        "seed": seed,
        "rule": _STRATIFICATION_RULE,
        "validation_lineage_allocation": allocation,
        "derived_from_sha256": sha256_file(Path(manifest_path)),
    }


def _assert_validation_disjoint(
    manifest: pd.DataFrame, validation_model_ids: Sequence[str]
) -> None:
    """Structural invariant: validation lines subset train_head, disjoint from
    the ``train_response_and_head`` anchors and the frozen ``test`` lines.

    Raises:
        ValueError: Any validation line is not a ``train_head`` row, or
            overlaps the anchors or the test lines.
    """
    validation_ids = {str(value) for value in validation_model_ids}
    train_head_ids = set(
        manifest.loc[manifest["role"] == TRAIN_HEAD_ROLE, "model_id"].astype(str)
    )
    anchor_ids = set(
        manifest.loc[manifest["role"] == TRAIN_RESPONSE_ROLE, "model_id"].astype(str)
    )
    test_ids = set(manifest.loc[manifest["role"] == TEST_ROLE, "model_id"].astype(str))

    not_train_head = validation_ids - train_head_ids
    if not not_train_head:
        return
    # Every id in not_train_head is, by role exclusivity, in exactly one of
    # anchor_ids/test_ids/neither -- report the specific reason for each
    # rather than always stopping at the generic "not train_head" case, so
    # an anchor overlap and a test overlap are both directly diagnosable.
    problems: list[str] = []
    anchor_overlap = not_train_head & anchor_ids
    if anchor_overlap:
        problems.append(
            f"{sorted(anchor_overlap)} overlap the {TRAIN_RESPONSE_ROLE!r} "
            "anchors; anchors must stay in training"
        )
    test_overlap = not_train_head & test_ids
    if test_overlap:
        problems.append(f"{sorted(test_overlap)} overlap the frozen test lines")
    unrecognized = not_train_head - anchor_ids - test_ids
    if unrecognized:
        problems.append(f"{sorted(unrecognized)} are not train_head lines")
    raise ValueError("validation line(s) " + "; ".join(problems))


def load_validation_line_registration(
    path: Path, manifest: pd.DataFrame, manifest_path: Path
) -> dict[str, object]:
    """Load the derived validation-line registration, read-only.

    Asserts the frozen manifest's *current* sha256 still equals the
    registration's recorded ``derived_from_sha256`` and fails loudly
    otherwise -- the registration must not silently outlive the contract it
    was derived from -- then re-checks the disjointness invariants
    (validation subset of ``train_head``, disjoint from the anchors and test
    lines) against the manifest actually passed in.

    Args:
        path: Path to ``validation_lines.json``.
        manifest: The frozen manifest, as returned by
            ``tx1_basal.load_line_manifest``.
        manifest_path: Path to the manifest file on disk, hashed and compared
            against the registration's own ``derived_from_sha256``.

    Returns:
        The parsed registration dict.

    Raises:
        ValueError: The manifest's current sha256 no longer matches
            ``derived_from_sha256``, or any disjointness invariant fails.
    """
    registration = json.loads(Path(path).read_text())
    current_sha256 = sha256_file(Path(manifest_path))
    recorded_sha256 = str(registration["derived_from_sha256"])
    if current_sha256 != recorded_sha256:
        raise ValueError(
            f"{path} is stale: the frozen manifest's current sha256 is "
            f"{current_sha256}, but this registration was derived from "
            f"{recorded_sha256}; regenerate it against the current contract"
        )
    _assert_validation_registration_count(path, registration)
    _assert_validation_disjoint(manifest, registration["validation_model_ids"])
    return registration


def _assert_validation_registration_count(
    path: Path, registration: Mapping[str, object]
) -> None:
    """Wave 3 Codex gate P2-2: reject a truncated or duplicated registration.

    ``_assert_validation_disjoint`` (called right after this) converts
    ``validation_model_ids`` to a ``set`` before checking disjointness, so a
    DUPLICATED id collapses invisibly there without ever being noticed. A
    TRUNCATED list (fewer ids than this registration's own recorded
    ``validation_count``) has no check anywhere else in this module. Both
    would otherwise silently train with fewer validation lines than the
    registration claims -- the manifest-freshness check above cannot catch
    either, since neither requires the frozen manifest itself to have
    changed.

    This does not, and cannot, catch a registration where BOTH
    ``validation_model_ids`` and ``validation_count`` were edited together
    to agree on a smaller, self-consistent number -- that is a genuinely
    different (re-registered) contract, not a truncation/duplication bug,
    and is out of scope for a generic loader that legitimately serves
    smaller synthetic registrations in tests. Guarding the specific,
    frozen, real ``validation_lines.json`` contract's line count of 5 is a
    dedicated test on that file (not a runtime assertion here), since this
    function is also exercised by tests with deliberately smaller
    registrations.

    Raises:
        ValueError: ``validation_model_ids`` contains a duplicate id, or its
            length disagrees with the registration's own ``validation_count``.
    """
    ids = [str(value) for value in registration["validation_model_ids"]]
    if len(set(ids)) != len(ids):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise ValueError(
            f"{path}: validation_model_ids contains duplicate id(s) "
            f"{duplicates} -- a duplicated id silently trains with fewer "
            "than the registered validation line count"
        )
    recorded_count = int(registration["validation_count"])
    if len(ids) != recorded_count:
        raise ValueError(
            f"{path}: validation_model_ids has {len(ids)} entries but this "
            f"registration's own validation_count is {recorded_count}; "
            "refusing to trust a registration whose recorded count "
            "disagrees with its own id list (a truncated or otherwise "
            "edited validation_lines.json)"
        )


def build_line_role_split(
    manifest: pd.DataFrame, validation_registration: Mapping[str, object]
) -> LineRoleSplit:
    """Combine the frozen manifest and the derived validation registration
    into the final 28/5/9 train/validation/test split Phase D trains over.

    Args:
        manifest: The frozen manifest, as returned by
            ``tx1_basal.load_line_manifest``.
        validation_registration: A registration dict shaped like
            :func:`generate_validation_line_registration`'s return value
            (normally obtained via :func:`load_validation_line_registration`,
            which has already checked freshness and disjointness; re-checked
            here too, defensively, since this function may also be called
            with a hand-built dict, e.g. in tests).

    Returns:
        A :class:`LineRoleSplit` with ``train_model_ids`` = the combined
        training-role lines minus the 5 validation lines (28 on the frozen
        manifest), ``validation_model_ids`` = the 5 lines,
        ``test_model_ids`` = the 9 frozen test lines, and
        ``anchor_model_ids`` = the ``train_response_and_head`` lines (a
        subset of ``train_model_ids``).

    Raises:
        ValueError: Any structural invariant from
            :func:`_assert_validation_disjoint` is violated, or a role value
            in the manifest is outside the training/test partition.
    """
    training, test = select_line_roles(manifest)
    validation_ids = [
        str(value) for value in validation_registration["validation_model_ids"]
    ]
    _assert_validation_disjoint(manifest, validation_ids)
    validation_set = set(validation_ids)
    anchor_ids = tuple(
        sorted(
            training.loc[training["role"] == TRAIN_RESPONSE_ROLE, "model_id"].astype(
                str
            )
        )
    )
    all_training_ids = set(training["model_id"].astype(str))
    train_only_ids = tuple(sorted(all_training_ids - validation_set))
    return LineRoleSplit(
        train_model_ids=train_only_ids,
        validation_model_ids=tuple(sorted(validation_set)),
        test_model_ids=tuple(sorted(test["model_id"].astype(str))),
        anchor_model_ids=anchor_ids,
    )


def load_depmap_gene_effect(
    depmap_gene_effect_path: Path,
    phase_a_registration: Mapping[str, object],
    model_ids: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the raw per-``(gene, model_id)`` DepMap CRISPRGeneEffect matrix.

    The frozen ``differentially_essential_slice.csv`` carries only aggregate
    per-gene statistics (``training_std``, ``training_dependency_prevalence``,
    ``training_line_count``) -- this function derives the raw matrix Phase D
    training actually needs, directly from the source DepMap file.

    Resolves ``depmap_gene_effect_path`` by CONTENT HASH against
    ``phase_a_registration["sources"]["depmap_gene_effect"]["sha256"]`` (D5)
    -- never by trusting either the registration's recorded path (which
    points at a ``depmap_26q1/`` directory that does not exist on every
    machine) or the caller's path. Joins purely on the file's own row index
    (ACH ``model_id``); no free-text line-name column is ever read here, so a
    name-keyed join is impossible by construction (DepMap's own line-name
    field is spelled differently from this codebase's convention -- see
    ``repo-hazards.md`` -- so a name join would silently corrupt every row).

    Args:
        depmap_gene_effect_path: Local path to ``CRISPRGeneEffect.csv``.
        phase_a_registration: The parsed, frozen ``phase_a_registration.json``
            (read-only; this function never writes it).
        model_ids: ACH model ids to select rows for, in the order returned.
        columns: Optional restriction to these ``depmap_column`` labels (the
            same ``"GENE_SYMBOL (entrez_id)"`` strings
            ``differentially_essential_slice.csv["depmap_column"]`` carries).
            ``None`` returns every gene column in the file.

    Returns:
        A numeric ``DataFrame`` indexed by ``model_id`` (row order matching
        ``model_ids``); non-numeric cells are coerced to ``NaN``.

    Raises:
        ValueError: The file's sha256 does not match the registered hash, a
            requested ``model_id`` is absent from the file, or a requested
            column is absent.
    """
    expected_sha256 = str(
        phase_a_registration["sources"]["depmap_gene_effect"]["sha256"]
    )
    observed_sha256 = sha256_file(Path(depmap_gene_effect_path))
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "DepMap GeneEffect source failed content-hash resolution (D5): "
            f"expected sha256 {expected_sha256}, got {observed_sha256} at "
            f"{depmap_gene_effect_path}"
        )
    frame = pd.read_csv(depmap_gene_effect_path, index_col=0)
    frame.index = frame.index.astype(str)
    ids = [str(value) for value in model_ids]
    missing_rows = sorted(set(ids) - set(frame.index))
    if missing_rows:
        raise ValueError(
            f"DepMap GeneEffect is missing rows for model_ids: {missing_rows}"
        )
    selected = frame.loc[ids]
    if columns is not None:
        wanted = [str(value) for value in columns]
        missing_columns = sorted(set(wanted) - set(selected.columns))
        if missing_columns:
            raise ValueError(f"DepMap GeneEffect is missing columns: {missing_columns}")
        selected = selected[wanted]
    return selected.apply(pd.to_numeric, errors="coerce")
