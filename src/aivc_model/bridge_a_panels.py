"""Independent-N window-budget policy for Bridge A basal panels (Finding A).

Replaces the earlier replacement-expansion approach (bootstrapping a small
observed-cell bag WITH replacement up to a large nominal cell target, e.g.
1024): that equalized STATE window COUNT across arms but not INDEPENDENT
sample count, so a gene with as few as 8 observed cells could look, on
paper, as precise as one with thousands -- the small bag's true Monte Carlo
variance was hidden behind synthetic duplication.

Policy (windowed, capped, sub-window bootstrap-and-flag)
----------------------------------------------------------
For a scored ordered direction ``a -> b`` (gene ``a`` supplies the
a-perturbed basal panel), let ``n_a`` be gene ``a``'s observed-cell count
from the Horlbeck/Replogle K562 coverage table (``load_k562_gene_coverage``,
keyed by canonical symbol -- never hardcoded).

* If ``n_a >= cell_set_len`` (>= 1 independent STATE window): the window
  budget is ``w_a = min(n_a // cell_set_len, max_windows)``. The a-perturbed
  panel draws ``w_a * cell_set_len`` cells WITHOUT replacement from gene
  ``a``'s observed bag; the matched control panel draws the SAME ``w_a``
  windows WITHOUT replacement from the pooled control. Both arms then use
  the same count of INDEPENDENT cells -- the panel-size confound is
  genuinely removed, not just the window count. ``max_windows`` bounds cost
  and stops a handful of outlier-large gene bags (up to ~1963 observed
  cells) from dominating the sweep.
* If ``n_a < cell_set_len`` (fewer observed cells than one STATE window --
  71 of the 408 exp05-covered genes): one independent window cannot be
  formed. The a-perturbed panel is built by bootstrapping gene ``a``'s bag
  WITH replacement to exactly one window; the matched control panel is
  still drawn WITHOUT replacement to that same one window. This direction
  is flagged ``bootstrapped=True`` (low-effective-N) rather than silently
  reported at face value.

Every direction reports ``effective_n = min(n_a, w_a * cell_set_len)`` --
the actual count of independent observed cells behind it (equal to ``n_a``
in the bootstrapped case). A pair is ``high_effective_n`` iff BOTH genes
have ``n >= cell_set_len``; that slice (337 genes in the current exp05
pool / coverage snapshot) is the PRIMARY analysis slice, the rest is
secondary. See ``aivc_model.bridge_a_independent`` for how window budgets
computed here drive the actual STATE panel construction and the matched
per-window-count control-panel cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_MAX_WINDOWS = 8


@dataclass(frozen=True)
class WindowBudget:
    """One gene's independent-cell window budget for its a-perturbed panel.

    Attributes:
        n_cells: Observed-cell count from the coverage table (``n_a``).
        n_windows: STATE windows to draw for the a-perturbed panel --
            ``min(n_cells // cell_set_len, max_windows)``, or ``1`` in the
            sub-window bootstrap fallback.
        bootstrapped: ``True`` iff ``n_cells < cell_set_len``, i.e. the
            a-perturbed panel was built WITH replacement (low-effective-N).
        effective_n: The actual count of independent observed cells behind
            this direction -- ``min(n_cells, n_windows * cell_set_len)`` in
            the without-replacement case, ``n_cells`` in the bootstrapped
            case.
    """

    n_cells: int
    n_windows: int
    bootstrapped: bool
    effective_n: int


def compute_window_budget(
    n_cells: int,
    cell_set_len: int,
    max_windows: int,
) -> WindowBudget:
    """Compute the independent-N window budget for one gene (see module docstring).

    Args:
        n_cells: The gene's observed-cell count (``n_a``).
        cell_set_len: Cells per STATE window (one independent window's size).
        max_windows: The window-budget cap (bounds cost and outlier-bag
            influence).

    Returns:
        The gene's ``WindowBudget``.

    Raises:
        ValueError: If ``n_cells < 0``, ``cell_set_len < 1``, or
            ``max_windows < 1``.
    """
    if n_cells < 0:
        raise ValueError(f"n_cells must be >= 0, got {n_cells}")
    if cell_set_len < 1:
        raise ValueError(f"cell_set_len must be >= 1, got {cell_set_len}")
    if max_windows < 1:
        raise ValueError(f"max_windows must be >= 1, got {max_windows}")

    raw_windows = n_cells // cell_set_len
    if raw_windows >= 1:
        n_windows = min(raw_windows, max_windows)
        return WindowBudget(
            n_cells=n_cells,
            n_windows=n_windows,
            bootstrapped=False,
            effective_n=min(n_cells, n_windows * cell_set_len),
        )
    return WindowBudget(
        n_cells=n_cells,
        n_windows=1,
        bootstrapped=True,
        effective_n=n_cells,
    )


def load_k562_gene_coverage(coverage_csv: Path) -> dict[str, int]:
    """Load per-gene observed Replogle cell counts for the exp05 K562 pool.

    Args:
        coverage_csv: Path to
            ``data/sl_dependency_v0/processed/horlbeck_2018/k562_gene_coverage.csv``.

    Returns:
        ``canonical_symbol -> replogle_n_cells`` for genes that are both
        members of the exp05 fixed training pool AND observed-coverage
        qualified (``in_exp05_fixed_pool`` and ``exp05_observed_covered``
        both true) -- exactly the 408-gene exp05-ready Bridge A universe.
    """
    coverage = pd.read_csv(coverage_csv)
    qualified = coverage.loc[
        coverage["in_exp05_fixed_pool"].astype(bool)
        & coverage["exp05_observed_covered"].astype(bool)
    ]
    return {
        str(row.canonical_symbol).upper(): int(row.replogle_n_cells)
        for row in qualified.itertuples(index=False)
    }
