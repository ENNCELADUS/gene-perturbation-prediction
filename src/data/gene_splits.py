"""Canonical exp05 gene-universe outer-fold manifests."""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

CANONICAL_GENE_COUNT = 9338
CANONICAL_OUTER_FOLDS = frozenset(range(5))
FIXED_SPLIT_ROLES = frozenset({"train", "validation", "internal_test"})

FIT_STAGES = frozenset(
    {
        "adapter_fit",
        "state_fit",
        "response_encoder_fit",
        "gmm_fit",
        "c_head_fit",
        "transition_supervision",
        "gene_prompt_fit",
        "fine_tuning",
    }
)
SELECTION_STAGES = frozenset({"early_stopping_prediction_only", "layer_selection"})
FINAL_RESPONSE_STAGES = frozenset({"generation_loss_outer_test"})
FINAL_LABEL_STAGES = frozenset({"internal_outer_test"})


@dataclass(frozen=True)
class FoldSpec:
    """One frozen outer fold plus its derived inner train/validation roles."""

    outer_fold: int
    train_genes: tuple[str, ...]
    val_genes: tuple[str, ...]
    test_genes: tuple[str, ...]


@dataclass(frozen=True)
class GeneAccessEvent:
    """One authorized, actually executed gene access."""

    stage: str
    outer_fold: int
    gene_count: int
    gene_set_sha256: str
    checkpoint_frozen: bool


class GeneAccessRecorder:
    """Capability that authorizes and records actual fold-scoped accesses."""

    def __init__(self, fold: FoldSpec) -> None:
        self.fold = fold
        self._events: list[GeneAccessEvent] = []

    def record(
        self,
        stage: str,
        genes: Collection[str],
        *,
        checkpoint_frozen: bool = False,
    ) -> None:
        """Authorize the request, then append evidence only on success."""
        normalized = tuple(sorted({str(gene).upper() for gene in genes}))
        self.authorize(
            stage,
            normalized,
            checkpoint_frozen=checkpoint_frozen,
        )
        self._events.append(
            GeneAccessEvent(
                stage=stage,
                outer_fold=self.fold.outer_fold,
                gene_count=len(normalized),
                gene_set_sha256=_gene_set_sha256(normalized),
                checkpoint_frozen=checkpoint_frozen,
            )
        )

    def authorize(
        self,
        stage: str,
        genes: Collection[str],
        *,
        checkpoint_frozen: bool = False,
    ) -> None:
        """Reject an unauthorized operation before it touches data or fits."""
        assert_gene_access(stage, genes, self.fold, checkpoint_frozen)

    def to_frame(self) -> pd.DataFrame:
        """Return immutable audit evidence for emitted accesses."""
        columns = [
            "stage",
            "outer_fold",
            "gene_count",
            "gene_set_sha256",
            "checkpoint_frozen",
        ]
        return pd.DataFrame(
            [event.__dict__ for event in self._events],
            columns=columns,
        )


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_strata(values: np.ndarray, n_splits: int) -> np.ndarray:
    """Assign deterministic, balanced quantile strata for stratified folding."""
    numeric = np.asarray(values, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("depmap_gene_effect must contain only finite values")
    ranks = pd.Series(numeric).rank(method="first")
    return pd.qcut(ranks, q=n_splits, labels=False).to_numpy(dtype=np.int64)


def make_inner_fold_spec(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    outer_fold: int,
    inner_val_fraction: float,
    seed: int,
) -> FoldSpec:
    """Derive an inner split while consuming the frozen outer assignment verbatim."""
    if not 0.0 < inner_val_fraction < 1.0:
        raise ValueError("inner_val_fraction must be between zero and one")
    canonical = _canonical_gene_fold_view(manifest)
    label_frame = labels[["perturbation_gene", "depmap_gene_effect"]].copy()
    label_frame["perturbation_gene"] = (
        label_frame["perturbation_gene"].astype(str).str.upper()
    )
    if label_frame["perturbation_gene"].duplicated().any():
        raise ValueError("labels must contain one row per perturbation_gene")
    label_by_gene = label_frame.set_index("perturbation_gene")["depmap_gene_effect"]
    missing = sorted(set(canonical["perturbation_gene"]) - set(label_by_gene.index))
    if missing:
        raise ValueError(f"labels are missing canonical genes: {missing[:5]}")

    test_genes = sorted(
        canonical.loc[
            canonical["outer_fold"] == outer_fold,
            "perturbation_gene",
        ]
    )
    if not test_genes:
        raise ValueError(f"outer_fold {outer_fold} is absent from the manifest")
    outer_train_genes = sorted(
        canonical.loc[
            canonical["outer_fold"] != outer_fold,
            "perturbation_gene",
        ]
    )
    validation_count = max(1, int(np.ceil(len(outer_train_genes) * inner_val_fraction)))
    strata_count = min(5, validation_count, len(outer_train_genes) - validation_count)
    stratify = None
    if strata_count >= 2:
        y = label_by_gene.loc[outer_train_genes].to_numpy()
        stratify = quantile_strata(y, strata_count)
    train_genes, val_genes = train_test_split(
        outer_train_genes,
        test_size=inner_val_fraction,
        random_state=seed + outer_fold + 1,
        stratify=stratify,
    )
    return FoldSpec(
        outer_fold=outer_fold,
        train_genes=tuple(sorted(train_genes)),
        val_genes=tuple(sorted(val_genes)),
        test_genes=tuple(test_genes),
    )


def attach_gene_provenance(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    gene_col: str,
    source_kind: str,
) -> pd.DataFrame:
    """Attach canonical perturbation gene/fold provenance to gene-derived rows."""
    if gene_col not in frame:
        raise ValueError(f"missing gene column {gene_col!r}")
    canonical = _canonical_gene_fold_view(manifest)
    result = frame.copy()
    result[gene_col] = result[gene_col].astype(str).str.upper()
    existing_fold = result.pop("outer_fold") if "outer_fold" in result else None
    if "perturbation_gene" in result and gene_col != "perturbation_gene":
        existing_gene = result.pop("perturbation_gene").astype(str).str.upper()
        if not existing_gene.equals(result[gene_col]):
            raise ValueError("conflicting perturbation_gene provenance")
    result["perturbation_gene"] = result[gene_col]
    result = result.merge(
        canonical,
        on="perturbation_gene",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if result["outer_fold"].isna().any():
        missing = sorted(
            result.loc[result["outer_fold"].isna(), "perturbation_gene"].unique()
        )
        raise ValueError(f"genes are missing from canonical manifest: {missing[:5]}")
    result["outer_fold"] = result["outer_fold"].astype(np.int64)
    if existing_fold is not None:
        supplied = pd.to_numeric(existing_fold, errors="coerce").to_numpy()
        if not np.array_equal(supplied, result["outer_fold"].to_numpy()):
            raise ValueError("conflicting outer_fold provenance")
    result["source_kind"] = str(source_kind)
    return result


def assert_gene_access(
    stage: str,
    genes: Collection[str],
    fold: FoldSpec,
    checkpoint_frozen: bool,
) -> None:
    """Reject gene access outside the role authorized for a protocol stage."""
    requested = {str(gene).upper() for gene in genes}
    train = {gene.upper() for gene in fold.train_genes}
    validation = {gene.upper() for gene in fold.val_genes}
    test = {gene.upper() for gene in fold.test_genes}
    if stage in FIT_STAGES:
        if not requested <= train:
            raise ValueError(f"{stage} attempted outer-test or validation gene access")
        return
    if stage in SELECTION_STAGES:
        if not requested <= validation:
            raise ValueError(f"{stage} must use inner-validation genes only")
        return
    if stage in FINAL_RESPONSE_STAGES:
        if not checkpoint_frozen:
            raise PermissionError(
                "selected checkpoint is frozen before outer-test response access"
            )
        if not requested <= test:
            raise ValueError(f"{stage} accepts outer-test genes only")
        return
    if stage in FINAL_LABEL_STAGES:
        if not checkpoint_frozen:
            raise ValueError(
                "selected checkpoint is frozen before outer-test label access"
            )
        if not requested <= test:
            raise ValueError(f"{stage} accepts outer-test genes only")
        return
    raise ValueError(f"unknown gene-access stage {stage!r}")


def _gene_set_sha256(genes: Collection[str]) -> str:
    value = "\n".join(sorted(str(gene).upper() for gene in genes))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_gene_fold_view(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"perturbation_gene", "outer_fold"}
    if not required <= set(manifest.columns):
        raise ValueError(
            "canonical manifest must contain perturbation_gene and outer_fold"
        )
    canonical = manifest[["perturbation_gene", "outer_fold"]].copy()
    canonical["perturbation_gene"] = (
        canonical["perturbation_gene"].astype(str).str.upper()
    )
    if canonical["perturbation_gene"].duplicated().any():
        raise ValueError("canonical manifest contains duplicate genes")
    if canonical["outer_fold"].isna().any():
        raise ValueError("canonical manifest contains missing outer_fold values")
    canonical["outer_fold"] = canonical["outer_fold"].astype(np.int64)
    return canonical


def build_canonical_outer_manifest(
    labels: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """Assign every canonical gene to exactly one deterministic outer fold."""
    frame = labels[["perturbation_gene", "depmap_gene_effect"]].copy()
    frame["perturbation_gene"] = frame["perturbation_gene"].astype(str).str.upper()
    if (
        len(frame) != CANONICAL_GENE_COUNT
        or frame["perturbation_gene"].nunique() != CANONICAL_GENE_COUNT
    ):
        raise ValueError(
            "canonical gene universe must contain exactly 9338 unique genes"
        )
    frame = frame.sort_values("perturbation_gene").reset_index(drop=True)
    strata = quantile_strata(frame["depmap_gene_effect"].to_numpy(), n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outer_fold = np.full(len(frame), -1, dtype=np.int64)
    for fold, (_, test_index) in enumerate(splitter.split(frame.index, strata)):
        outer_fold[test_index] = fold
    if np.any(outer_fold < 0):
        raise AssertionError("every canonical gene must receive one outer fold")
    return pd.DataFrame(
        {
            "perturbation_gene": frame["perturbation_gene"],
            "outer_fold": outer_fold,
        }
    )


def build_fixed_split_manifest(
    labels: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Build one deterministic split over the complete gene-level pool."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave an internal test")

    frame = labels[["perturbation_gene", "depmap_gene_effect"]].copy()
    frame["perturbation_gene"] = frame["perturbation_gene"].astype(str).str.upper()
    if frame.empty or frame["perturbation_gene"].nunique() != len(frame):
        raise ValueError("fixed split labels must contain unique genes")
    values = frame["depmap_gene_effect"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("depmap_gene_effect must contain only finite values")
    frame = frame.sort_values("perturbation_gene").reset_index(drop=True)

    strata = quantile_strata(
        frame["depmap_gene_effect"].to_numpy(),
        10,
    )
    train_index, heldout_index = train_test_split(
        np.arange(len(frame), dtype=np.int64),
        train_size=train_fraction,
        random_state=seed,
        stratify=strata,
    )
    heldout = frame.iloc[heldout_index]
    heldout_strata = quantile_strata(
        heldout["depmap_gene_effect"].to_numpy(),
        10,
    )
    validation_count = int(round(len(frame) * validation_fraction))
    validation_local, internal_local = train_test_split(
        np.arange(len(heldout), dtype=np.int64),
        train_size=validation_count,
        random_state=seed + 1,
        stratify=heldout_strata,
    )

    roles = pd.Series(index=frame.index, dtype=object)
    roles.iloc[train_index] = "train"
    roles.iloc[heldout.index.to_numpy(dtype=np.int64)[validation_local]] = "validation"
    roles.iloc[heldout.index.to_numpy(dtype=np.int64)[internal_local]] = "internal_test"
    return pd.DataFrame(
        {
            "perturbation_gene": frame["perturbation_gene"],
            "split_role": roles,
        }
    )


def load_fixed_split_manifest(
    path: Path,
    labels: pd.DataFrame,
    expected_sha256: str,
) -> pd.DataFrame:
    """Load a fixed role manifest only when its hash and universe are exact."""
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"fixed split manifest SHA-256 mismatch: {observed_sha256}")
    manifest = pd.read_csv(path)
    expected_columns = ["perturbation_gene", "split_role"]
    if manifest.columns.tolist() != expected_columns:
        raise ValueError(f"fixed split manifest columns must be {expected_columns}")
    manifest["perturbation_gene"] = (
        manifest["perturbation_gene"].astype(str).str.upper()
    )
    label_genes = labels["perturbation_gene"].astype(str).str.upper()
    exact_universe = (
        len(manifest) == manifest["perturbation_gene"].nunique()
        and len(label_genes) == label_genes.nunique()
        and set(manifest["perturbation_gene"]) == set(label_genes)
    )
    if not exact_universe:
        raise ValueError("fixed split gene universe does not match the label table")
    if set(manifest["split_role"]) != FIXED_SPLIT_ROLES:
        raise ValueError(
            f"fixed split roles must be exactly {sorted(FIXED_SPLIT_ROLES)}"
        )
    return manifest


def fixed_fold_spec(manifest: pd.DataFrame) -> FoldSpec:
    """Convert one fixed role manifest into the existing role-limited contract."""
    required = {"perturbation_gene", "split_role"}
    if not required <= set(manifest):
        raise ValueError("fixed split manifest is missing required columns")
    normalized = manifest.copy()
    normalized["perturbation_gene"] = (
        normalized["perturbation_gene"].astype(str).str.upper()
    )
    if normalized["perturbation_gene"].duplicated().any():
        raise ValueError("fixed split manifest contains duplicate genes")
    if set(normalized["split_role"]) != FIXED_SPLIT_ROLES:
        raise ValueError(
            f"fixed split roles must be exactly {sorted(FIXED_SPLIT_ROLES)}"
        )

    def genes(role: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                normalized.loc[
                    normalized["split_role"] == role,
                    "perturbation_gene",
                ]
            )
        )

    return FoldSpec(
        outer_fold=-1,
        train_genes=genes("train"),
        val_genes=genes("validation"),
        test_genes=genes("internal_test"),
    )


def load_canonical_outer_manifest(
    path: Path,
    labels: pd.DataFrame,
    expected_sha256: str,
) -> pd.DataFrame:
    """Load a manifest only if its provenance and gene universe are exact."""
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"canonical manifest SHA-256 mismatch: {observed_sha256}")
    manifest = pd.read_csv(path)
    expected_columns = ["perturbation_gene", "outer_fold"]
    if manifest.columns.tolist() != expected_columns:
        raise ValueError(f"canonical manifest columns must be {expected_columns}")

    manifest["perturbation_gene"] = (
        manifest["perturbation_gene"].astype(str).str.upper()
    )
    label_genes = labels["perturbation_gene"].astype(str).str.upper()
    exact_universe = (
        len(manifest) == CANONICAL_GENE_COUNT
        and manifest["perturbation_gene"].nunique() == CANONICAL_GENE_COUNT
        and len(label_genes) == CANONICAL_GENE_COUNT
        and label_genes.nunique() == CANONICAL_GENE_COUNT
        and set(manifest["perturbation_gene"]) == set(label_genes)
    )
    if not exact_universe:
        raise ValueError("canonical gene universe does not match the label table")
    if set(manifest["outer_fold"]) != CANONICAL_OUTER_FOLDS:
        raise ValueError("canonical outer folds must be exactly 0..4")
    if not manifest["outer_fold"].map(lambda value: isinstance(value, int)).all():
        raise ValueError("canonical outer folds must be integers")
    return manifest
