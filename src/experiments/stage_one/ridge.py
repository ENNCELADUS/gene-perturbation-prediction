"""The per-gene ridge on the decoder-feature delta (a report row): one ridge per gene
and penalty over the frame lines, cached by what it was fitted from."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from src.data import ladder_reader as reader
from src.data.io import read_json, write_json
from src.experiments.stage_one.setup import Setup
from src.experiments.stage_one.run import predictions_frame

log = logging.getLogger(__name__)


#: Panel genes solved at a time (a 64-gene block of dH is ~0.2 GB in fp32).
RIDGE_GENES = 64


def dh_ridge(
    setup: Setup, penalties: Sequence[float], device: torch.device
) -> dict[float, pd.DataFrame]:
    """Per penalty, a ridge per panel gene from the line-centred ``dH`` (768-d) to
    the training target, in its dual (``(K + lambda I) alpha = y``, ``K`` the lines'
    Gram matrix), on the device: the main fit on the frame predicts validation and
    test; fold k's fit, without fold k and its base, predicts fold k's frame lines.
    A pair without a label leaves its fit exactly (its row of dH and its target are
    zeroed). Scoring units on every predicted line; other genes predict the base."""
    ladder = setup.ladder
    index = reader.check_pie_index(ladder)
    features = reader.open_delta(ladder, "dH", index)
    panel, lines = ladder.panel, ladder.line_ids
    position = {m: i for i, m in enumerate(lines)}
    folds = ladder.folds
    queries = [(None, ladder.role_lines("val", "test"))] + [
        (k, [m for m in ladder.frame if folds[m] == k])
        for k in sorted(set(folds.values()))
    ]
    fits = []
    for fold, query in queries:
        targets = setup.targets(fold)
        fits.append(
            (
                targets,
                torch.tensor([position[m] for m in targets.lines], device=device),
                torch.tensor([position[m] for m in query], device=device),
                query,
                targets.values.loc[:, panel].to_numpy(np.float32),
            )
        )
    z = {
        p: [np.empty((len(panel), len(f[3])), np.float32) for f in fits]
        for p in penalties
    }
    for start in range(0, len(panel), RIDGE_GENES):
        end = min(start + RIDGE_GENES, len(panel))
        x = torch.from_numpy(np.array(features[start:end], dtype=np.float32)).to(device)
        for fit, (_, train, query, _, values) in enumerate(fits):
            target = torch.from_numpy(np.ascontiguousarray(values[:, start:end].T)).to(
                device
            )
            have = torch.isfinite(target)
            xt = x[:, train] * have.unsqueeze(-1)
            gram = xt @ xt.transpose(1, 2)
            cross = x[:, query] @ xt.transpose(1, 2)
            target = torch.nan_to_num(target).unsqueeze(-1)
            eye = torch.eye(len(train), device=device)
            for penalty in penalties:
                factor = torch.linalg.cholesky(gram + penalty * eye)
                alpha = torch.cholesky_solve(target, factor)
                z[penalty][fit][start:end] = (cross @ alpha).squeeze(-1).cpu().numpy()
        if start // RIDGE_GENES % 25 == 0:
            log.info("dH ridge: %d of %d panel genes", end, len(panel))
    return {
        penalty: pd.concat(
            [
                predictions_frame(setup, fit[0].m_g, values, fit[3])
                for fit, values in zip(fits, z[penalty])
            ]
        )
        for penalty in penalties
    }


def ridge_fingerprint(setup: Setup, ladder_config: str | Path) -> dict[str, Any]:
    """What the ridge's predictions depend on: the ladder config, the label file,
    the lines with their roles and folds, the panel and the dH file."""

    def stat(path: Path) -> list:
        info = Path(path).stat()
        return [str(Path(path).resolve()), info.st_size, info.st_mtime_ns]

    ladder = setup.ladder
    return {
        "ladder_config": str(Path(ladder_config).resolve()),
        "ladder": dict(setup.config),
        "labels": stat(setup.label_source),
        "lines": {
            m: [str(ladder.lines.at[m, "role"]), int(ladder.lines.at[m, "fold"])]
            for m in ladder.line_ids
        },
        "panel": ladder.panel,
        "dH": stat(ladder.root / "pie" / "dH.npy"),
    }


def cached_ridge(
    setup: Setup,
    ladder_config: str | Path,
    penalties: Sequence[float],
    out: Path,
    device: str,
) -> dict[float, pd.DataFrame]:
    """The ridge's predictions per penalty, from ``out/dh_ridge/`` when its
    ``fingerprint.json`` matches the current inputs (refused otherwise), else
    computed and written with the fingerprint."""
    from src.data.io import jsonable

    folder = out / "dh_ridge"
    fingerprint = jsonable(ridge_fingerprint(setup, ladder_config))
    record = folder / "fingerprint.json"
    if record.is_file() and read_json(record) != fingerprint:
        raise ValueError(f"{folder} was computed from other inputs; move it aside")
    cached = {p: folder / f"penalty_{p:g}.parquet" for p in penalties}
    if any(not path.is_file() for path in cached.values()):
        folder.mkdir(exist_ok=True)
        write_json(record, fingerprint)
        for penalty, frame in dh_ridge(setup, penalties, torch.device(device)).items():
            frame.astype(np.float32).to_parquet(cached[penalty])
    return {penalty: pd.read_parquet(path) for penalty, path in cached.items()}
