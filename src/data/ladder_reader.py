"""Readers of the stage-one inputs (``data/geneeffect_joint/v2/ladder/``, protocol
§15) for the stage-one model, and their GPU-resident form.

numpy, pandas and torch only (never ``pie``), so ``.venv-pie`` and the project
environment both read them. The tables are read as ``ladder_inputs.read_genes`` and
``read_lines`` read them (``data`` never imports ``experiments``); every array is
checked against ``lines.parquet`` and the decode panel before use:

- ``check_pie_index`` fails closed unless ``pie/index.json`` lists the lines of
  ``lines.parquet`` and the panel of ``genes.parquet`` in their order;
- ``open_delta`` memory-maps one gene-major delta (genes x lines x width, centred
  within each gene on the frame lines); ``resident`` copies it to a device by gene
  chunk, in bf16, so the fp32 file is never held in host memory whole;
- ``profile_states``: the before-state ``(x_c - x_frame) C^T`` of every line and each
  gene's shift ``frame_mean_dE[g] + mu_dp C^T``, so the after-state of a pair is
  ``before[c] + dE[g, c] + shift[g]`` (the profile delta stores only coordinates of
  the predicted change on the basis; ``dE`` is centred within gene);
- ``training_targets``: ``(GeneEffect - m_g) / sigma_g`` of training lines only,
  ``m_g`` refitted on the lines kept when a fold is excluded;
- ``scoring_truth``: ``(GeneEffect - mu_g) / sigma_g``, what every prediction is
  scored against (protocol §6).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import torch

from src.data.splits import FixedSplit, assert_fit_eligible

TRAINING_ROLES: Final[tuple[str, ...]] = ("train_cells", "train_bulk")
PIE_ARRAYS: Final[tuple[str, ...]] = ("dH", "dE", "dL", "own_h", "own_dp", "own_x")
#: Panel genes copied to the device at a time (a 768-wide fp32 chunk is ~0.8 GB).
RESIDENT_CHUNK: Final[int] = 256


@dataclass(frozen=True)
class Ladder:
    """The ladder root's line and gene tables."""

    root: Path
    lines: pd.DataFrame  # index model_id: role, fold, lineage, patient_id, paired
    genes: pd.DataFrame  # index gene: gene_text_key, in_panel, m_g, mu_g, sigma_g, ...

    @property
    def line_ids(self) -> list[str]:
        return [str(m) for m in self.lines.index]

    @property
    def panel(self) -> list[str]:
        return [str(g) for g in self.genes.index[self.genes["in_panel"]]]

    def role_lines(self, *roles: str) -> list[str]:
        return [str(m) for m in self.lines.index[self.lines["role"].isin(roles)]]

    @property
    def frame(self) -> list[str]:
        return self.role_lines(*TRAINING_ROLES)

    @property
    def folds(self) -> dict[str, int]:
        """The frame lines' patient-grouped folds."""
        frame = self.lines.loc[self.frame, "fold"]
        return {str(m): int(k) for m, k in frame.items()}


def read_ladder(root: Path) -> Ladder:
    root = Path(root)
    lines = pd.read_parquet(root / "lines.parquet")
    genes = pd.read_parquet(root / "genes.parquet")
    if not lines.index.is_unique or not genes.index.is_unique:
        raise ValueError(f"{root}: duplicate lines or genes")
    return Ladder(root=root, lines=lines, genes=genes)


# --- the frozen PIE's arrays ------------------------------------------------------


def check_pie_index(ladder: Ladder) -> dict[str, Any]:
    """``pie/index.json``, refused unless its lines and genes are the ladder's
    lines and panel in order: a delta row read under another order is another
    line's."""
    index = json.loads((ladder.root / "pie" / "index.json").read_text())
    if list(index["lines"]) != ladder.line_ids:
        raise ValueError(f"{ladder.root}/pie: its lines are not lines.parquet's")
    if list(index["genes"]) != ladder.panel:
        raise ValueError(f"{ladder.root}/pie: its genes are not the decode panel")
    return index


def open_delta(ladder: Ladder, name: str, index: Mapping[str, Any]) -> np.ndarray:
    """One delta or own slot, memory-mapped: panel genes x lines x width."""
    if name not in PIE_ARRAYS:
        raise ValueError(f"unknown PIE array {name!r}; expected one of {PIE_ARRAYS}")
    array = np.load(ladder.root / "pie" / f"{name}.npy", mmap_mode="r")
    expected = tuple(index["arrays"][name])
    if array.shape != expected or array.dtype != np.float32:
        raise ValueError(
            f"pie/{name}.npy is {array.dtype} {array.shape}, the index says {expected}"
        )
    return array


def frame_means(ladder: Ladder, index: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Each array's frame-line mean per gene (panel genes x width)."""
    with np.load(ladder.root / "pie" / "frame_means.npz") as payload:
        means = {name: np.asarray(payload[name], dtype=np.float32) for name in payload}
    for name, shape in index["arrays"].items():
        if means[name].shape != (shape[0], shape[2]):
            raise ValueError(f"frame_means.npz[{name}] does not match pie/{name}.npy")
    return means


def basis(ladder: Ladder) -> dict[str, np.ndarray]:
    """The profile delta's basis: ``mean`` (panel), ``components`` (k x panel)."""
    with np.load(ladder.root / "pie" / "basis.npz") as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def resident(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    chunk: int = RESIDENT_CHUNK,
) -> torch.Tensor:
    """A gene-major array on ``device`` in ``dtype``, copied ``chunk`` genes at a
    time from its memory map."""
    out = torch.empty(array.shape, dtype=dtype, device=device)
    for start in range(0, array.shape[0], chunk):
        block = np.array(array[start : start + chunk], dtype=np.float32)  # a copy
        out[start : start + len(block)] = torch.from_numpy(block).to(device, dtype)
    return out


@dataclass(frozen=True)
class ProfileStates:
    """The profile delta's states on the basis (k components).

    ``before`` (lines x k) is ``(x_c - x_frame) C^T`` from PIE's input; ``shift``
    (panel genes x k) is what the centred ``dE`` lacks of ``Delta_hat C^T``. The
    after-state of (c, g) is ``before[c] + dE[g, c] + shift[g]``.
    ``variance_kept`` is the share of the frame lines' centred input variance the
    basis keeps.
    """

    before: np.ndarray
    shift: np.ndarray
    variance_kept: float

    def after(self, de: np.ndarray, gene: int, lines: Sequence[int]) -> np.ndarray:
        """After-states of one panel gene's ``lines`` from its stored ``dE`` rows."""
        return self.before[list(lines)] + de + self.shift[gene]


def profile_states(
    ladder: Ladder,
    pie_input_frame: pd.DataFrame,
    means: Mapping[str, np.ndarray],
    profile_basis: Mapping[str, np.ndarray],
) -> ProfileStates:
    components = np.asarray(profile_basis["components"], dtype=np.float64)
    if list(pie_input_frame.columns) != ladder.panel:
        raise ValueError("PIE's input is not on the decode panel")
    x = pie_input_frame.loc[ladder.line_ids].to_numpy(dtype=np.float64)
    frame = pie_input_frame.loc[ladder.frame].to_numpy(dtype=np.float64)
    centre = frame.mean(axis=0)
    before = (x - centre) @ components.T
    kept = ((frame - centre) @ components.T) ** 2
    shift = (
        means["dE"].astype(np.float64)
        + np.asarray(profile_basis["mean"], dtype=np.float64) @ components.T
    )
    return ProfileStates(
        before=before.astype(np.float32),
        shift=shift.astype(np.float32),
        variance_kept=float(kept.sum() / ((frame - centre) ** 2).sum()),
    )


@dataclass(frozen=True)
class FullProfile:
    """The profile delta on the full decode panel, for the genes the pass kept it
    (the selective genes in the panel and the pilot genes): ``pie/full/<gene>.npy``,
    lines x panel, PIE's predicted change ``Delta_hat`` as written (uncentred).

    ``before`` (lines x panel) is ``x_c - x_frame`` from PIE's input; the after-state
    of (c, g) is ``before[c] + Delta_hat[g][c]``. ``genes`` are the requested genes
    with a file, in the order asked; ``missing`` those without one.
    """

    genes: tuple[str, ...]
    missing: tuple[str, ...]
    before: np.ndarray
    root: Path

    def change(self, gene: str) -> np.ndarray:
        """Gene g's predicted change, memory-mapped (lines x panel, fp32)."""
        return np.load(self.root / "pie" / "full" / f"{gene}.npy", mmap_mode="r")

    def after(self, gene: str, lines: Sequence[int]) -> np.ndarray:
        return self.before[list(lines)] + np.asarray(self.change(gene)[list(lines)])


def full_panel_genes(ladder: Ladder, genes: Sequence[str]) -> list[str]:
    """The ``genes`` with a full-panel profile delta file, in the order given."""
    folder = ladder.root / "pie" / "full"
    return [g for g in genes if (folder / f"{g}.npy").is_file()]


def full_profile(
    ladder: Ladder, pie_input_frame: pd.DataFrame, genes: Sequence[str]
) -> FullProfile:
    """The full-panel profile delta of ``genes`` (panel genes); every file found is
    checked to be lines x panel fp32."""
    if list(pie_input_frame.columns) != ladder.panel:
        raise ValueError("PIE's input is not on the decode panel")
    folder = ladder.root / "pie" / "full"
    found = full_panel_genes(ladder, genes)
    shape = (len(ladder.line_ids), len(ladder.panel))
    for gene in found:
        array = np.load(folder / f"{gene}.npy", mmap_mode="r")
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError(f"pie/full/{gene}.npy is {array.dtype} {array.shape}")
    x = pie_input_frame.loc[ladder.line_ids].to_numpy(dtype=np.float64)
    centre = pie_input_frame.loc[ladder.frame].to_numpy(dtype=np.float64).mean(axis=0)
    return FullProfile(
        genes=tuple(found),
        missing=tuple(g for g in genes if g not in set(found)),
        before=(x - centre).astype(np.float32),
        root=ladder.root,
    )


class FullStream:
    """The full-panel profile delta streamed to a device by gene chunk: far too
    large to be resident (~300 GB fp32 for the selective genes), it stays in its
    memory maps, and each batch copies its genes' rows for its lines, centred on the
    gene's frame-line mean, to the device in ``dtype``.

    The frame means are computed once at construction, one pass over every gene's
    frame rows (which also brings the files into the host page cache: later epochs
    read from memory when it holds them); ``seconds`` and ``bytes_read`` record it.
    """

    def __init__(
        self,
        profile: FullProfile,
        frame_rows: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        import time

        self.arrays = [profile.change(g) for g in profile.genes]
        self.device, self.dtype = device, dtype
        frame = np.asarray(sorted(frame_rows))
        start = time.monotonic()
        means = np.empty((len(self.arrays), profile.before.shape[1]), np.float32)
        for i, array in enumerate(self.arrays):
            means[i] = np.asarray(array[frame], dtype=np.float64).mean(axis=0)
        self.seconds = time.monotonic() - start
        self.bytes_read = int(len(self.arrays) * len(frame) * means.shape[1] * 4)
        self.means = torch.from_numpy(means).to(device)

    def rows(self, genes: np.ndarray, lines: np.ndarray) -> torch.Tensor:
        """genes x lines x panel on the device: gene i's rows ``lines[i]`` (a 1-d
        ``lines`` is shared by every gene), minus its frame mean."""
        lines = np.broadcast_to(lines, (len(genes), lines.shape[-1]))
        out = np.empty((len(genes), lines.shape[1], self.means.shape[1]), np.float32)
        for i, gene in enumerate(genes):
            out[i] = self.arrays[int(gene)][lines[i]]
        block = torch.from_numpy(out).to(self.device)
        block -= self.means[torch.as_tensor(genes, device=self.device)].unsqueeze(1)
        return block.to(self.dtype)


# --- tables -------------------------------------------------------------------------


def _on_lines(ladder: Ladder, frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if [str(m) for m in frame.index] != ladder.line_ids:
        raise ValueError(f"{name}: its rows are not lines.parquet's lines in order")
    return frame


def pie_input(ladder: Ladder) -> pd.DataFrame:
    """PIE's input (lines x panel): the mean over cells of log1p CP10K."""
    frame = pd.read_parquet(ladder.root / "mean" / "pie_input.parquet")
    if list(frame.columns) != ladder.panel:
        raise ValueError("mean/pie_input.parquet is not on the decode panel")
    return _on_lines(ladder, frame, "mean/pie_input.parquet")


def pie_listed(ladder: Ladder) -> pd.DataFrame:
    """Whether the line's source lists each panel gene (else it took the frame mean)."""
    frame = pd.read_parquet(ladder.root / "mean" / "pie_listed.parquet")
    if list(frame.columns) != ladder.panel:
        raise ValueError("mean/pie_listed.parquet is not on the decode panel")
    return _on_lines(ladder, frame.astype(bool), "mean/pie_listed.parquet")


def mean_channel(ladder: Ladder) -> pd.DataFrame:
    """The mean channel (lines x space genes) in the default prior's space."""
    frame = pd.read_parquet(ladder.root / "mean" / "mean_channel.parquet")
    return _on_lines(ladder, frame, "mean/mean_channel.parquet")


def gene_sources(
    ladder: Ladder, names: Sequence[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each source on the gene axis: vectors (zero where absent) and presence."""
    out = {}
    for name in names:
        with np.load(ladder.root / "gene_sources" / f"{name}.npz") as payload:
            axis = [str(g) for g in payload["axis"]]
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            present = np.asarray(payload["present"], dtype=bool)
        if axis != [str(g) for g in ladder.genes.index]:
            raise ValueError(f"gene_sources/{name}.npz is not on the gene axis")
        out[name] = (np.where(present[:, None], vectors, 0.0), present)
    return out


def context_text(ladder: Ladder) -> tuple[np.ndarray, np.ndarray]:
    """The Cellosaurus text vector of every line (zero without one) and presence."""
    from src.data.gene_sources import read_pie_dense

    table = read_pie_dense(ladder.root / "context_text")
    row = {key: i for i, key in enumerate(table.keys)}
    present = np.asarray([m in row for m in ladder.line_ids])
    vectors = np.zeros((len(ladder.line_ids), table.vectors.shape[1]), np.float32)
    for i, model_id in enumerate(ladder.line_ids):
        if model_id in row:
            vectors[i] = np.asarray(table.vectors[row[model_id]], dtype=np.float32)
    return vectors, present


def text_on_axis(path: Path, axis: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """PIE's ``gene_text`` source dir aligned to any gene axis (the line tokens'),
    case-insensitively: vectors (zero where absent) and presence."""
    from src.data.gene_sources import align, read_pie_dense

    aligned = align(read_pie_dense(Path(path)), axis)
    return np.nan_to_num(aligned.vectors, nan=0.0), aligned.present


@dataclass(frozen=True)
class LineTokens:
    """One token per gene a line's profile lists, carrying both values (§3.3):
    PIE's input (the mean of log1p CP10K, on the panel) and the mean channel (the
    log1p pseudo-bulk in the prior's space). ``values`` is lines x tokens x 4:
    ``[pie value, pie listed, mean-channel value, in the space]``, a value zero
    where its flag is; a token is present when either flag is."""

    axis: tuple[str, ...]
    values: np.ndarray
    present: np.ndarray


def line_tokens(ladder: Ladder) -> LineTokens:
    pie = pie_input(ladder)
    listed = pie_listed(ladder)
    channel = mean_channel(ladder)
    panel = ladder.panel
    inside = set(panel)
    axis = (*panel, *(str(g) for g in channel.columns if g not in inside))
    position = {g: i for i, g in enumerate(axis)}
    n_lines = len(ladder.line_ids)
    values = np.zeros((n_lines, len(axis), 4), dtype=np.float32)
    values[:, : len(panel), 1] = listed.to_numpy(dtype=np.float32)
    values[:, : len(panel), 0] = (
        pie.to_numpy(dtype=np.float32) * values[:, : len(panel), 1]
    )
    columns = [position[str(g)] for g in channel.columns]
    values[:, columns, 2] = channel.to_numpy(dtype=np.float32)
    values[:, columns, 3] = 1.0
    return LineTokens(axis=axis, values=values, present=values[..., [1, 3]].any(axis=2))


# --- labels ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Targets:
    """Training labels: ``(GeneEffect - m_g) / sigma_g`` of the fit lines (NaN where
    unlabelled), with the base ``m_g`` refitted on exactly these lines."""

    lines: tuple[str, ...]
    m_g: pd.Series
    values: pd.DataFrame


def training_targets(
    ladder: Ladder,
    gene_effect: pd.DataFrame,
    split: FixedSplit,
    bulk_labelled: Sequence[str],
    *,
    lines: Sequence[str] | None = None,
    exclude_fold: int | None = None,
) -> Targets:
    """Targets of ``lines`` (default: the frame) without ``exclude_fold``'s lines.

    Every line must be a training line of the ladder: a single-cell training line
    admitted by ``assert_fit_eligible``, a bulk-only line among the split's labelled
    bulk lines (``load_bulk_lines``). ``m_g`` is the mean over the lines kept; over
    the whole frame it must equal ``genes.parquet``'s ``m_g``.
    """
    chosen = list(ladder.frame if lines is None else lines)
    folds = ladder.lines["fold"]
    if exclude_fold is not None:
        if exclude_fold not in set(ladder.folds.values()):
            raise ValueError(f"no frame fold {exclude_fold}")
        chosen = [m for m in chosen if int(folds[m]) != exclude_fold]
    bulk = set(bulk_labelled)
    for model_id in chosen:
        role = (
            ladder.lines.at[model_id, "role"]
            if model_id in ladder.lines.index
            else None
        )
        if role not in TRAINING_ROLES:
            raise ValueError(f"{model_id!r} is not a training line ({role}); refused")
        if role == "train_cells":
            assert_fit_eligible(model_id, split)
        elif model_id not in bulk:
            raise ValueError(f"{model_id!r} is not a labelled bulk-only line; refused")
    absent = [m for m in chosen if m not in gene_effect.index]
    if absent:
        raise ValueError(f"no GeneEffect row for {absent[:10]}")
    genes = [str(g) for g in ladder.genes.index]
    labels = gene_effect.reindex(index=chosen, columns=genes)
    m_g = labels.mean(axis=0, skipna=True)
    if lines is None and exclude_fold is None:
        stored = ladder.genes["m_g"].to_numpy(dtype=np.float64)
        if not np.allclose(m_g.to_numpy(), stored, rtol=0, atol=1e-9, equal_nan=True):
            raise ValueError("the frame's GeneEffect mean is not genes.parquet's m_g")
    sigma = ladder.genes["sigma_g"].to_numpy(dtype=np.float64)
    values = (labels - m_g) / sigma
    return Targets(lines=tuple(chosen), m_g=m_g, values=values)


def scoring_truth(
    ladder: Ladder, gene_effect: pd.DataFrame, lines: Sequence[str]
) -> pd.DataFrame:
    """``(GeneEffect - mu_g) / sigma_g`` of ``lines`` on the gene axis; NaN where
    unlabelled. A line with no GeneEffect row at all is refused."""
    absent = [m for m in lines if m not in gene_effect.index]
    if absent:
        raise ValueError(f"no GeneEffect row for {absent[:10]}")
    genes = [str(g) for g in ladder.genes.index]
    labels = gene_effect.reindex(index=list(lines), columns=genes)
    mu = ladder.genes["mu_g"].to_numpy(dtype=np.float64)
    sigma = ladder.genes["sigma_g"].to_numpy(dtype=np.float64)
    return (labels - mu) / sigma


def base_prediction(
    ladder: Ladder, m_g: pd.Series, lines: Sequence[str]
) -> pd.DataFrame:
    """The base ``m_g`` as a prediction in scoring units, ``(m_g - mu_g) / sigma_g``,
    the same row for every line."""
    genes = ladder.genes
    row = (m_g.loc[genes.index].to_numpy() - genes["mu_g"].to_numpy()) / genes[
        "sigma_g"
    ].to_numpy()
    return pd.DataFrame(
        np.tile(row, (len(lines), 1)), index=list(lines), columns=list(genes.index)
    )
