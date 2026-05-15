"""NAR cell-viability-axis coefficient scoring utilities."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from dependency_baseline.config import (
    ViabilityAxisArtifactConfig,
    ViabilityAxisConfig,
)


@dataclass(frozen=True)
class CoefficientModel:
    name: str
    coefficients: pd.Series
    intercept: float
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class ViabilityAxisResult:
    scores: pd.DataFrame
    score_columns: tuple[str, ...]
    qa_rows: list[dict[str, object]]


def build_viability_axis_scores(
    *,
    delta: np.ndarray,
    gene_symbols: list[str],
    config: ViabilityAxisConfig,
    default_cache_dir: Path,
) -> ViabilityAxisResult | None:
    """Score delta expression with configured NAR viability-axis models."""
    if not config.enabled:
        return None
    cache_dir = config.cache_dir or default_cache_dir
    models = [
        load_coefficient_model(artifact, cache_dir) for artifact in config.artifacts
    ]
    if not models:
        msg = "viability_axis.enabled=true requires at least one artifact"
        raise ValueError(msg)

    score_data: dict[str, np.ndarray] = {}
    qa_rows: list[dict[str, object]] = []
    for model in models:
        column, qa = score_model(delta, gene_symbols, model)
        score_name = f"nar_{model.name}_score"
        score_data[score_name] = column.astype(np.float32)
        qa_rows.append(qa)

    model_score_columns = tuple(score_data)
    if len(model_score_columns) > 1:
        stacked = np.column_stack(
            [score_data[column] for column in model_score_columns]
        )
        score_data["nar_mean_score"] = stacked.mean(axis=1).astype(np.float32)
    scores = pd.DataFrame(score_data)
    return ViabilityAxisResult(
        scores=scores,
        score_columns=tuple(scores.columns),
        qa_rows=qa_rows,
    )


def load_coefficient_model(
    artifact: ViabilityAxisArtifactConfig,
    cache_dir: Path,
) -> CoefficientModel:
    """Download/cache and parse one NAR coefficient CSV."""
    path = cached_artifact_path(artifact, cache_dir)
    sha256 = file_sha256(path)
    coefficients, intercept = parse_coefficient_csv(path)
    return CoefficientModel(
        name=artifact.name,
        coefficients=coefficients,
        intercept=intercept,
        source_path=path,
        sha256=sha256,
    )


def cached_artifact_path(
    artifact: ViabilityAxisArtifactConfig,
    cache_dir: Path,
) -> Path:
    """Return a checksum-verified local artifact path, downloading if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(artifact.url).path).name or f"{artifact.name}.csv"
    path = cache_dir / filename
    if path.exists():
        _validate_sha256(path, artifact.sha256)
        return path

    tmp_path = path.with_name(f".{path.name}.tmp")
    _download_or_copy(artifact.url, tmp_path)
    _validate_sha256(tmp_path, artifact.sha256)
    tmp_path.replace(path)
    return path


def parse_coefficient_csv(path: Path) -> tuple[pd.Series, float]:
    """Parse a NAR model CSV into gene coefficients and intercept."""
    table = pd.read_csv(path)
    if "coefficient" not in table.columns or "pr_gene_symbol" not in table.columns:
        msg = f"Invalid coefficient CSV columns in {path}"
        raise ValueError(msg)

    first_column = table.columns[0]
    intercept_mask = table[first_column].astype(str).eq("INTERCEPT")
    if not intercept_mask.any():
        intercept_mask = table["pr_gene_symbol"].astype(str).eq("INTERCEPT")
    intercept = (
        float(table.loc[intercept_mask, "coefficient"].iloc[0])
        if intercept_mask.any()
        else 0.0
    )
    coefficient_rows = table.loc[~intercept_mask].copy()
    coefficient_rows["pr_gene_symbol"] = coefficient_rows["pr_gene_symbol"].astype(str)
    coefficients = coefficient_rows.set_index("pr_gene_symbol")["coefficient"].astype(
        float
    )
    coefficients = coefficients.groupby(level=0).sum()
    return coefficients, intercept


def score_model(
    delta: np.ndarray,
    gene_symbols: list[str],
    model: CoefficientModel,
) -> tuple[np.ndarray, dict[str, object]]:
    """Score one delta matrix with one coefficient model."""
    symbol_to_index = {symbol: index for index, symbol in enumerate(gene_symbols)}
    weights = np.zeros(len(gene_symbols), dtype=np.float64)
    matched_symbols = []
    missing_symbols = []
    for symbol, coefficient in model.coefficients.items():
        index = symbol_to_index.get(symbol)
        if index is None:
            missing_symbols.append(symbol)
            continue
        weights[index] = float(coefficient)
        matched_symbols.append(symbol)
    score = delta.astype(np.float64) @ weights + model.intercept
    qa = {
        "model": model.name,
        "source_path": str(model.source_path),
        "sha256": model.sha256,
        "n_coefficients": int(model.coefficients.shape[0]),
        "n_matched_expression_genes": int(len(matched_symbols)),
        "n_missing_expression_genes": int(len(missing_symbols)),
        "matched_fraction": float(len(matched_symbols) / model.coefficients.shape[0]),
        "intercept": model.intercept,
        "missing_gene_examples": ",".join(missing_symbols[:20]),
    }
    return score, qa


def file_sha256(path: Path) -> str:
    """Compute a file SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(path: Path, expected: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        msg = f"SHA256 mismatch for {path}: expected {expected}, observed {observed}"
        raise ValueError(msg)


def _download_or_copy(url: str, target: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        source = Path(parsed.path if parsed.scheme == "file" else url)
        shutil.copyfile(source, target)
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
