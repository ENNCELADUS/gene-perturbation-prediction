"""One reader per ready row of configs/benchmarks/sl_label_rules.csv (the ``reader``
column, without its ``read_`` prefix). Each reads a study's files under the data root
and applies that row's label rule, returning one row per (list, gene pair) with the
columns of ``COLUMNS``; gene symbols stay as the file spells them and the builder maps
them. A reader raises when a file's shape is not what it expects. The core screens
wrap the three-screen benchmark's readers; the other modules hold the label-rule
readers written per study group on 2026-10-29."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.data.builders.sl_study_readers import (
    anchor_a_to_l,
    anchor_m_to_z,
    combinatorial,
    combinatorial_slkb,
    core_screens,
)

COLUMNS = core_screens.COLUMNS
Reader = Callable[[Path], pd.DataFrame]

READERS: dict[str, Reader] = {}
for _module in (
    core_screens, combinatorial_slkb, combinatorial, anchor_a_to_l, anchor_m_to_z
):  # fmt: skip
    for _name, _reader in vars(_module).items():
        if _name.startswith("read_") and _module.__name__ == getattr(
            _reader, "__module__", None
        ):
            if _name.removeprefix("read_") in READERS:
                raise ValueError(f"two readers named {_name}")
            READERS[_name.removeprefix("read_")] = _reader
