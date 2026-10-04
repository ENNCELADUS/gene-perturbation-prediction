# Linear Context Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the linear context prior of the design, from the local groundwork through the prior's build-up: the extra-line membership, pinned reference tables, the shared expression space and bridge, context views and gene features, the stagewise ridge prior with cross-fitting, the learning curve and extra-lines decision, block-by-block selection on validation and one test score.

**Architecture:** A new package `src/context_prior/` holds pure, numpy/pandas/scikit-learn modules (space, folds, targets, bridge, views, gene features, ridge stages, the prior, view weights); it imports `src.data` only. DepMap readers, the extra-line membership and pseudo-bulk live in `src/data/`; reference tables are built on the Mac by `src/data/prepare/build_context_reference.py` and committed, because the H20 host has no internet. One runner, `src/experiments/context_prior.py`, reached through `hpc/run.sh prior`, chains the steps and writes `outputs/context_prior/<run id>/`.

**Tech Stack:** Python 3.11, numpy, pandas, scikit-learn 1.7.2 (LogisticRegression, Ridge), scipy (`rankdata` via pandas), torch (view weights only), PyYAML, pytest, ruff.

**Spec:** `docs/specs/2026-10-04-context-generalization-design.md`

## Global Constraints

- Every Python invocation is `uv run python -m ...` from the repo root; imports are `src.*`.
- Tests: `uv run python -m pytest <file> -q > pytest.txt 2>&1; tail -5 pytest.txt` (the rtk hook breaks foreground pytest). Lint: `.venv/bin/ruff check .`; format only touched files with `.venv/bin/ruff format <files>`.
- `src/context_prior/` and `src/data/` never import `src.training`, `src.eval` or `src.experiments`.
- Configs are strict: unknown or missing keys raise; no `.get(key, default)` config reads.
- Fitting scope: selective genes, `mu_hat` and σ_g come from the 170 labelled training lines (spec §3.5); the bridge, encoders, partner lists and ridges from the training side.
- Excluded from every fit: validation and test lines and every line sharing a `PatientID` with one of them; validation lines' bulk RNA is read only by the oracle diagnostic; test lines' bulk RNA is never read.
- One config is one run at seed 0; validation chooses; the chosen prior is scored once on test.
- Penalties are multiplied by the number of fitted lines; selection keeps a block only if the 95% paired bootstrap interval (27 validation lines, 1,000 resamples, seed 0) of its `val_selective_spearman` gain excludes zero.
- Files over 5 GB are downloaded and processed only on the H20 host; the H20 host has no internet, so small reference tables travel through git.
- H20 access: `ssh -J richard@100.91.229.50 -p <port> root@10.15.171.204`; which container (port) a run uses is the user's call.
- Commits: Conventional Commits, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on branch `feat/context-prior`, created from `feat/geneeffect-revision`.
- Name models, stages and runs by what they are (the linear context prior, the expression-components stage), never by bare labels.

## Review Focus

- A panel gene with no column in the shared expression space (GeneEffect gene without bulk RNA): its own-expression and partner features are undefined, so they standardise to zero and the gene is still predicted from context — pinned in the gene-features task.
- A training label that is NaN (DepMap leaves some line × gene pairs empty): fitting counts it as the residual's expected value, zero; scoring skips it — pinned in the prior task.
- A held-out or patient-sharing line inside a reference table (drivers, MSI): `load_reference` raises instead of silently training an encoder on it — pinned in the reference task.
- A single-cell training line that has both a bulk row (fitted) and a bridged pseudo-bulk row (predicted): predictions are keyed by the query frame, never mixed with the fitted bulk row — pinned in the prior task's cross-fit test.
- A constant feature column (a gene unexpressed in every line, a driver never mutated in a fold): its scale becomes 1 and it contributes nothing; nothing divides by zero — pinned in the ridge and views tasks.

---

### Task 1: DepMap readers

**Files:**
- Create: `src/data/depmap.py`
- Test: `tests/test_depmap.py`

**Interfaces:**
- Produces: `depmap_symbol(column: str) -> str`; `read_depmap_matrix(path: Path) -> pd.DataFrame` (index `model_id`, columns upper-case symbols named `gene_symbol`, float64); `read_model_ids(path: Path) -> frozenset[str]`; `read_models(path: Path) -> pd.DataFrame` (index `model_id`, columns `patient_id`, `lineage`; a missing PatientID becomes the ModelID).

- [ ] **Step 1: Create the branch**

```bash
git switch -c feat/context-prior
```

- [ ] **Step 2: Write the failing tests**

```python
"""DepMap readers: Omics default entries, CRISPR layout, symbols, Model.csv."""

from __future__ import annotations

import math

import pytest

from src.data.depmap import read_depmap_matrix, read_model_ids, read_models

OMICS = (
    ",SequencingID,ModelConditionID,ModelID,IsDefaultEntryForMC,"
    "IsDefaultEntryForModel,TP53 (7157),ZNF781 (Unknown)\n"
    "0,S1,MC1,ACH-1,Yes,Yes,1.5,0\n"
    "1,S2,MC2,ACH-1,No,No,9.0,9\n"
    "2,S3,MC3,ACH-2,Yes,Yes,0.5,2\n"
)
CRISPR = ",A1BG (1),A2M (2)\nACH-1,0.1,\nACH-2,-1.0,0.3\n"


def write(path, text):
    path.write_text(text)
    return path


def test_omics_matrix_keeps_each_models_default_entry(tmp_path):
    frame = read_depmap_matrix(write(tmp_path / "omics.csv", OMICS))
    assert list(frame.index) == ["ACH-1", "ACH-2"]
    assert list(frame.columns) == ["TP53", "ZNF781"]
    assert frame.loc["ACH-1", "TP53"] == 1.5
    assert frame.index.name == "model_id"


def test_crispr_matrix_is_indexed_by_its_first_column(tmp_path):
    frame = read_depmap_matrix(write(tmp_path / "crispr.csv", CRISPR))
    assert list(frame.columns) == ["A1BG", "A2M"]
    assert math.isnan(frame.loc["ACH-1", "A2M"])


def test_duplicate_symbols_raise(tmp_path):
    path = write(tmp_path / "dup.csv", ",TP53 (7157),tp53 (1)\nACH-1,1,2\n")
    with pytest.raises(ValueError, match="duplicate symbols"):
        read_depmap_matrix(path)


def test_model_ids_without_values(tmp_path):
    assert read_model_ids(write(tmp_path / "o.csv", OMICS)) == {"ACH-1", "ACH-2"}
    assert read_model_ids(write(tmp_path / "c.csv", CRISPR)) == {"ACH-1", "ACH-2"}


def test_missing_patient_becomes_its_own_patient(tmp_path):
    path = write(
        tmp_path / "Model.csv",
        "ModelID,PatientID,OncotreeLineage,Other\nACH-1,PT-1,Lung,x\nACH-2,,Bowel,y\n",
    )
    models = read_models(path)
    assert models.loc["ACH-1", "patient_id"] == "PT-1"
    assert models.loc["ACH-2", "patient_id"] == "ACH-2"
    assert models.loc["ACH-2", "lineage"] == "Bowel"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_depmap.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.depmap'`

- [ ] **Step 4: Implement**

```python
"""Pinned DepMap matrices as ModelID x gene-symbol frames, and Model.csv."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

#: DepMap header ``SYMBOL (Entrez)``; some Omics columns carry ``(Unknown)``.
_COLUMN = re.compile(r"^(?P<symbol>\S+) \((?:\d+|Unknown)\)$")
#: Profile metadata of DepMap Omics matrices (one row per sequencing profile).
_OMICS_METADATA = frozenset(
    {
        "SequencingID",
        "ModelConditionID",
        "ModelID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel",
    }
)


def depmap_symbol(column: str) -> str:
    """Upper-case symbol of a DepMap ``SYMBOL (Entrez)`` column."""
    match = _COLUMN.fullmatch(str(column).strip())
    if match is None:
        raise ValueError(f"not a DepMap gene column: {column!r}")
    return match.group("symbol").upper()


def read_depmap_matrix(path: Path) -> pd.DataFrame:
    """A DepMap matrix as float64, indexed by ModelID, columns upper-case symbols.

    Omics matrices (a ``ModelID`` column, one row per profile) keep each model's
    default entry; CRISPR matrices are indexed by ModelID in their first column.
    """
    frame = pd.read_csv(path)
    if "ModelID" in frame.columns:
        default = frame["IsDefaultEntryForModel"].astype(str).str.lower() == "yes"
        frame = frame.loc[default].set_index("ModelID")
        frame = frame.drop(
            columns=[
                column
                for column in frame.columns
                if column in _OMICS_METADATA or str(column).startswith("Unnamed")
            ]
        )
    else:
        frame = frame.set_index(frame.columns[0])
    frame.index = frame.index.astype(str)
    frame.index.name = "model_id"
    if not frame.index.is_unique:
        repeated = sorted(frame.index[frame.index.duplicated()].unique())
        raise ValueError(f"{path}: duplicate ModelIDs {repeated[:10]}")
    symbols = [depmap_symbol(column) for column in frame.columns]
    repeated = sorted(s for s, n in Counter(symbols).items() if n > 1)
    if repeated:
        raise ValueError(f"{path}: columns map to duplicate symbols {repeated[:10]}")
    frame.columns = pd.Index(symbols, name="gene_symbol")
    return frame.astype(np.float64)


def read_model_ids(path: Path) -> frozenset[str]:
    """ModelIDs of a matrix (an Omics matrix's default entries), values unread."""
    header = pd.read_csv(path, nrows=0).columns
    if "ModelID" in header:
        frame = pd.read_csv(
            path, usecols=["ModelID", "IsDefaultEntryForModel"], dtype=str
        )
        default = frame["IsDefaultEntryForModel"].str.lower() == "yes"
        return frozenset(frame.loc[default, "ModelID"])
    return frozenset(pd.read_csv(path, usecols=[0], dtype=str).iloc[:, 0])


def read_models(path: Path) -> pd.DataFrame:
    """Model.csv: ``patient_id`` (a missing one becomes the ModelID) and ``lineage``."""
    frame = pd.read_csv(
        path, usecols=["ModelID", "PatientID", "OncotreeLineage"], dtype=str
    ).set_index("ModelID", verify_integrity=True)
    frame.index.name = "model_id"
    patient = frame["PatientID"].fillna(pd.Series(frame.index, index=frame.index))
    return pd.DataFrame({"patient_id": patient, "lineage": frame["OncotreeLineage"]})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_depmap.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff format src/data/depmap.py tests/test_depmap.py && .venv/bin/ruff check src/data/depmap.py tests/test_depmap.py
git add src/data/depmap.py tests/test_depmap.py
git commit -m "feat(data): DepMap matrix and Model.csv readers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Extra-line membership

**Files:**
- Create: `src/data/extra_lines.py`, `src/data/prepare/build_extra_bulk_lines.py`, `configs/benchmarks/extra_bulk_lines_26Q1.json` (generated)
- Test: `tests/test_extra_lines.py`

**Interfaces:**
- Consumes: `read_models`, `read_model_ids` (DepMap readers); `FixedSplit`, `load_geneeffect_226_split` (`src/data/splits.py`).
- Produces: `SCHEMA_VERSION`; `ExtraLines(labelled: tuple[str, ...], unlabelled: tuple[str, ...], excluded: Mapping[str, str])`; `load_extra_lines(path: Path, split: FixedSplit) -> ExtraLines`; `build_extra_lines(models, *, labelled_ids, bulk_ids, split) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""Extra-line membership: exclusions by patient, labelled vs unlabelled, loader."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.extra_lines import SCHEMA_VERSION, load_extra_lines
from src.data.prepare.build_extra_bulk_lines import build_extra_lines
from src.data.splits import FixedSplit

SPLIT = FixedSplit(train=("T1",), val=("V1",), test=("S1",))
MODELS = pd.DataFrame(
    {
        "patient_id": {
            "T1": "P1", "V1": "P2", "S1": "P3", "E1": "P2",
            "E2": "P1", "E3": "P4", "E4": "P5", "E5": "P3",
        },
        "lineage": "Lung",
    }
).rename_axis("model_id")


def payload():
    return build_extra_lines(
        MODELS,
        labelled_ids={"T1", "V1", "S1", "E1", "E2", "E4"},
        bulk_ids={"T1", "V1", "E1", "E2", "E3", "E5"},
        split=SPLIT,
    )


def test_membership_rules():
    built = payload()
    assert built["schema_version"] == SCHEMA_VERSION
    assert built["labelled"] == ["E2"]  # shares a patient with a train line: kept
    assert built["unlabelled"] == ["E3"]  # bulk RNA without GeneEffect
    assert sorted(built["excluded"]) == ["E1", "E5"]  # share a held-out patient
    assert "V1" in built["excluded"]["E1"]


def test_line_missing_from_model_csv_raises():
    with pytest.raises(ValueError, match="absent from Model.csv"):
        build_extra_lines(
            MODELS, labelled_ids={"E9"}, bulk_ids={"E9"}, split=SPLIT
        )


def test_loader_round_trip_and_overlap_guard(tmp_path):
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(payload()))
    lines = load_extra_lines(path, SPLIT)
    assert lines.labelled == ("E2",) and lines.unlabelled == ("E3",)
    bad = payload()
    bad["unlabelled"] = ["E3", "T1"]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="overlap"):
        load_extra_lines(path, SPLIT)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_extra_lines.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the loader** (`src/data/extra_lines.py`)

```python
"""DepMap lines outside the 226 that join the training side through bulk RNA."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from src.data.splits import FixedSplit

SCHEMA_VERSION = "extra-bulk-lines-26q1-v1"


@dataclass(frozen=True)
class ExtraLines:
    """Training-side lines outside the split, and the lines no fit may read.

    Attributes:
        labelled: Lines with 26Q1 GeneEffect and bulk RNA.
        unlabelled: Lines with bulk RNA and no GeneEffect.
        excluded: Lines outside the split sharing a patient with a validation or
            test line, with the reason; never read by any fit.
    """

    labelled: tuple[str, ...]
    unlabelled: tuple[str, ...]
    excluded: Mapping[str, str]


def load_extra_lines(path: Path, split: FixedSplit) -> ExtraLines:
    """Load the membership file; raise on duplicates or any overlap with the split."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    lines = ExtraLines(
        tuple(payload["labelled"]),
        tuple(payload["unlabelled"]),
        dict(payload["excluded"]),
    )
    groups = {
        "labelled": lines.labelled,
        "unlabelled": lines.unlabelled,
        "excluded": tuple(lines.excluded),
        "split": split.all_model_ids,
    }
    for name, values in groups.items():
        if len(set(values)) != len(values):
            raise ValueError(f"{path}: {name} has duplicate ModelIDs")
    for (left, first), (right, second) in combinations(groups.items(), 2):
        overlap = sorted(set(first) & set(second))
        if overlap:
            raise ValueError(f"{path}: {left} and {right} overlap: {overlap[:10]}")
    return lines


__all__ = ["SCHEMA_VERSION", "ExtraLines", "load_extra_lines"]
```

- [ ] **Step 4: Implement the builder** (`src/data/prepare/build_extra_bulk_lines.py`)

```python
"""Build configs/benchmarks/extra_bulk_lines_26Q1.json from the pinned 26Q1 files.

uv run python -m src.data.prepare.build_extra_bulk_lines \
    --model data/sl_dependency_v0/raw/depmap/Model.csv \
    --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \
    --bulk data/sl_dependency_v0/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --out configs/benchmarks/extra_bulk_lines_26Q1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from src.data.depmap import read_model_ids, read_models
from src.data.extra_lines import SCHEMA_VERSION
from src.data.splits import FixedSplit, load_geneeffect_226_split

POLICY = (
    "Lines outside the 226 with 26Q1 bulk RNA; labelled ones also have 26Q1 "
    "GeneEffect. Every line outside the 226 sharing a PatientID with a validation "
    "or test line is excluded from every fit."
)


def build_extra_lines(
    models: pd.DataFrame,
    *,
    labelled_ids: Collection[str],
    bulk_ids: Collection[str],
    split: FixedSplit,
) -> dict:
    """Membership payload: labelled and unlabelled extra lines, and exclusions."""
    ours = set(split.all_model_ids)
    missing = sorted((set(labelled_ids) | set(bulk_ids) | ours) - set(models.index))
    if missing:
        raise ValueError(f"ModelIDs absent from Model.csv: {missing[:10]}")
    held = {models.at[m, "patient_id"]: m for m in (*split.val, *split.test)}
    excluded = {
        m: (
            f"shares PatientID {models.at[m, 'patient_id']} with held-out line "
            f"{held[models.at[m, 'patient_id']]}"
        )
        for m in sorted(set(models.index) - ours)
        if models.at[m, "patient_id"] in held
    }
    usable = set(bulk_ids) - ours - set(excluded)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "labelled": sorted(usable & set(labelled_ids)),
        "unlabelled": sorted(usable - set(labelled_ids)),
        "excluded": excluded,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "gene-effect", "bulk", "split", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_extra_lines(
        read_models(args.model),
        labelled_ids=read_model_ids(args.gene_effect),
        bulk_ids=read_model_ids(args.bulk),
        split=load_geneeffect_226_split(args.split),
    )
    payload["sources"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in (
            ("model", args.model),
            ("gene_effect", args.gene_effect),
            ("bulk", args.bulk),
        )
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"labelled {len(payload['labelled'])}, unlabelled "
        f"{len(payload['unlabelled'])}, excluded {len(payload['excluded'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_extra_lines.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `3 passed`

- [ ] **Step 6: Build the real membership file**

Run the module with the arguments in its docstring.
Expected stdout: `labelled 919, unlabelled 576, excluded 10` (576 = 1,664 training-side bulk lines − 169 split-train lines with bulk − 919). If the numbers differ, stop and report them; do not edit the file by hand.

- [ ] **Step 7: Commit**

```bash
.venv/bin/ruff format src/data/extra_lines.py src/data/prepare/build_extra_bulk_lines.py tests/test_extra_lines.py
git add src/data/extra_lines.py src/data/prepare/build_extra_bulk_lines.py tests/test_extra_lines.py configs/benchmarks/extra_bulk_lines_26Q1.json
git commit -m "feat(data): extra bulk-line membership for the context prior" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pinned reference tables

**Files:**
- Create: `src/data/prepare/build_context_reference.py`, `src/context_prior/__init__.py`, `src/context_prior/reference.py`, `configs/context_prior/reference/{paralogs.tsv.gz,complexes.tsv,hallmark.tsv,progeny.tsv,drivers.tsv,msi.tsv,provenance.json}` (generated)
- Test: `tests/test_context_reference.py`

**Interfaces:**
- Consumes: `read_depmap_matrix` (DepMap readers), `load_extra_lines` (extra-line membership).
- Produces: `parse_paralogs(text) -> DataFrame[gene, paralog, identity]`; `closest_paralogs(pairs, *, min_identity, per_gene)`; `parse_corum(text) -> DataFrame[complex_id, gene]`; `parse_gmt(text) -> DataFrame[gene_set, gene]`; `parse_progeny(text, *, top) -> DataFrame[pathway, gene, weight]`; `driver_status(hotspot, damaging, lines, *, min_fraction) -> DataFrame` (index `model_id`, 0/1); `Reference(paralogs, complexes, hallmark, progeny, drivers, msi)`; `load_reference(directory: Path, *, blocked: Collection[str]) -> Reference`.

- [ ] **Step 1: Write the failing tests**

```python
"""Reference-table parsers and the leakage guard of the loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.context_prior.reference import load_reference
from src.data.prepare.build_context_reference import (
    closest_paralogs,
    driver_status,
    parse_corum,
    parse_gmt,
    parse_paralogs,
    parse_progeny,
)

BIOMART = (
    "Gene name\tHuman paralogue associated gene name\t"
    "Paralogue %id. target Human gene identical to query gene\t"
    "Paralogue %id. query gene identical to target Human gene\n"
    "ARID1A\tARID1B\t60\t55\n"
    "ARID1A\tARID1A\t100\t100\n"
    "ARID1A\tARID2\t15\t30\n"
    "ARID1A\t\t\t\n"
    "SMARCA4\tSMARCA2\t75\t74\n"
)


def test_paralogs_take_the_lower_identity_and_drop_self_pairs():
    pairs = parse_paralogs(BIOMART)
    row = pairs.set_index(["gene", "paralog"]).loc[("ARID1A", "ARID1B"), "identity"]
    assert row == 55
    assert not ((pairs.gene == pairs.paralog).any())
    assert len(pairs) == 3


def test_biomart_error_text_raises():
    with pytest.raises(ValueError, match="BioMart"):
        parse_paralogs("Query ERROR: caught BioMart::Exception\n")


def test_closest_paralogs_threshold_cap_and_order():
    pairs = pd.DataFrame(
        {
            "gene": ["A", "A", "A", "B"],
            "paralog": ["X", "Y", "Z", "W"],
            "identity": [30.0, 50.0, 10.0, 25.0],
        }
    )
    kept = closest_paralogs(pairs, min_identity=20.0, per_gene=1)
    assert kept.values.tolist() == [["A", "Y", 50.0], ["B", "W", 25.0]]


def test_corum_keeps_human_complexes_as_long_rows():
    text = (
        "complex_id\torganism\tsubunits_gene_name\n"
        "1\tHuman\tBCL6;HDAC4\n2\tMouse\tA;B\n3\tHuman\tsmarca4;ARID1A\n"
    )
    rows = parse_corum(text)
    assert rows.values.tolist() == [
        [1, "BCL6"], [1, "HDAC4"], [3, "ARID1A"], [3, "SMARCA4"],
    ]


def test_gmt_and_progeny():
    sets = parse_gmt("HALLMARK_X\turl\tA\tb\nHALLMARK_Y\turl\tC\n")
    assert sets.values.tolist() == [
        ["HALLMARK_X", "A"], ["HALLMARK_X", "B"], ["HALLMARK_Y", "C"],
    ]
    text = "uniprot\tgenesymbol\tentity_type\tsource\tlabel\tvalue\trecord_id\n" + "".join(
        f"P\t{gene}\tprotein\tPROGENy\t{label}\t{value}\t{record}\n"
        for gene, record, pathway, weight, p in (
            ("A", 1, "EGFR", 0.5, 0.01),
            ("B", 2, "EGFR", -0.2, 0.2),
            ("C", 3, "EGFR", 0.9, 0.001),
        )
        for label, value in (("pathway", pathway), ("weight", weight), ("p_value", p))
    )
    top = parse_progeny(text, top=2)
    assert top.values.tolist() == [["EGFR", "C", 0.9], ["EGFR", "A", 0.5]]


def test_driver_rule_reads_only_given_lines():
    hotspot = pd.DataFrame({"KRAS": [1, 0, 0, 0], "BRAF": [0, 0, 0, 0]},
                           index=["L1", "L2", "L3", "L4"])
    damaging = pd.DataFrame({"KRAS": [0, 0, 0, 0], "BRAF": [0, 1, 0, 0],
                             "TTN": [2, 2, 2, 2]}, index=["L1", "L2", "L3", "L4"])
    status = driver_status(hotspot, damaging, ["L1", "L2", "L3"], min_fraction=0.3)
    assert list(status.columns) == ["KRAS", "BRAF"]  # TTN is not a hotspot gene
    assert list(status.index) == ["L1", "L2", "L3"]
    assert status.loc["L2", "BRAF"] == 1


def test_loader_refuses_blocked_lines(tmp_path):
    pd.DataFrame({"gene": ["A"], "paralog": ["B"], "identity": [50.0]}).to_csv(
        tmp_path / "paralogs.tsv.gz", sep="\t", index=False
    )
    pd.DataFrame({"complex_id": [1], "gene": ["A"]}).to_csv(
        tmp_path / "complexes.tsv", sep="\t", index=False
    )
    pd.DataFrame({"gene_set": ["S"], "gene": ["A"]}).to_csv(
        tmp_path / "hallmark.tsv", sep="\t", index=False
    )
    pd.DataFrame({"pathway": ["P"], "gene": ["A"], "weight": [1.0]}).to_csv(
        tmp_path / "progeny.tsv", sep="\t", index=False
    )
    pd.DataFrame({"model_id": ["L1", "V1"], "KRAS": [1, 0]}).to_csv(
        tmp_path / "drivers.tsv", sep="\t", index=False
    )
    pd.DataFrame({"model_id": ["L1"], "msi_score": [3.0]}).to_csv(
        tmp_path / "msi.tsv", sep="\t", index=False
    )
    assert load_reference(tmp_path, blocked=set()).drivers.shape == (2, 1)
    with pytest.raises(ValueError, match="held-out"):
        load_reference(tmp_path, blocked={"V1"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_reference.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the builder** (`src/data/prepare/build_context_reference.py`)

```python
"""Fetch and pin the reference tables of the linear context prior.

Runs on the Mac: the H20 host has no internet, so the small tables are committed
under configs/context_prior/reference/ and reach it through git. Line tables hold
training-side lines only: validation and test lines and every line sharing their
patients never enter one.

uv run python -m src.data.prepare.build_context_reference \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --extra-lines configs/benchmarks/extra_bulk_lines_26Q1.json \
    --hotspot data/sl_dependency_v0/raw/depmap/OmicsSomaticMutationsMatrixHotspot.csv \
    --damaging data/sl_dependency_v0/raw/depmap/OmicsSomaticMutationsMatrixDamaging.csv \
    --out configs/context_prior/reference
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.parse
import urllib.request
from collections.abc import Collection, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.depmap import read_depmap_matrix
from src.data.extra_lines import load_extra_lines
from src.data.splits import load_geneeffect_226_split

BIOMART_URL = "https://useast.ensembl.org/biomart/martservice"
CORUM_URL = (
    "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/"
    "download_current_file?file_id=human&file_format=txt"
)
HALLMARK_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/"
    "h.all.v2024.1.Hs.symbols.gmt"
)
PROGENY_URL = "https://omnipathdb.org/annotations?resources=PROGENy&format=tsv"
#: DepMap 24Q4 OmicsSignatures.csv (MSIScore), figshare 10.6084/m9.figshare.27993248.
SIGNATURES_URL = "https://ndownloader.figshare.com/files/51065726"
CHROMOSOMES = (*(str(n) for n in range(1, 23)), "X", "Y")
PARALOG_MIN_IDENTITY = 20.0
PARALOGS_PER_GENE = 10
PROGENY_TOP = 100
DRIVER_MIN_FRACTION = 0.05


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=600) as response:
        if response.status != 200:
            raise RuntimeError(f"{url}: HTTP {response.status}")
        return response.read().decode("utf-8")


def paralog_url(chromosome: str) -> str:
    query = (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="1" '
        'uniqueRows="1"><Dataset name="hsapiens_gene_ensembl" interface="default">'
        f'<Filter name="chromosome_name" value="{chromosome}"/>'
        '<Attribute name="external_gene_name"/>'
        '<Attribute name="hsapiens_paralog_associated_gene_name"/>'
        '<Attribute name="hsapiens_paralog_perc_id"/>'
        '<Attribute name="hsapiens_paralog_perc_id_r1"/>'
        "</Dataset></Query>"
    )
    return f"{BIOMART_URL}?{urllib.parse.urlencode({'query': query})}"


def parse_paralogs(text: str) -> pd.DataFrame:
    """BioMart rows -> gene, paralog, identity (the lower %id of both directions)."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    if frame.shape[1] != 4:
        raise ValueError(f"unexpected BioMart response: {text[:200]!r}")
    frame.columns = ["gene", "paralog", "identity", "identity_reverse"]
    frame = frame.dropna(subset=["gene", "paralog"])
    gene, paralog = frame["gene"].str.upper(), frame["paralog"].str.upper()
    keep = gene != paralog
    pairs = pd.DataFrame(
        {
            "gene": gene[keep],
            "paralog": paralog[keep],
            "identity": np.minimum(
                frame.loc[keep, "identity"].astype(float),
                frame.loc[keep, "identity_reverse"].astype(float),
            ),
        }
    )
    return pairs.groupby(["gene", "paralog"], as_index=False)["identity"].max()


def closest_paralogs(
    pairs: pd.DataFrame, *, min_identity: float, per_gene: int
) -> pd.DataFrame:
    """Each gene's ``per_gene`` closest paralogs at or above ``min_identity``."""
    kept = pairs.loc[pairs["identity"] >= min_identity].sort_values(
        ["gene", "identity", "paralog"], ascending=[True, False, True]
    )
    return kept.groupby("gene", sort=False).head(per_gene).reset_index(drop=True)


def parse_corum(text: str) -> pd.DataFrame:
    """CORUM's current human file -> one (complex_id, gene) row per subunit."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    human = frame.loc[frame["organism"] == "Human", ["complex_id", "subunits_gene_name"]]
    rows = [
        (int(complex_id), gene.strip().upper())
        for complex_id, members in human.dropna().itertuples(index=False)
        for gene in members.split(";")
        if gene.strip()
    ]
    return (
        pd.DataFrame(rows, columns=["complex_id", "gene"])
        .drop_duplicates()
        .sort_values(["complex_id", "gene"])
        .reset_index(drop=True)
    )


def parse_gmt(text: str) -> pd.DataFrame:
    """GMT -> one (gene_set, gene) row per member, symbols upper-cased."""
    rows = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            rows += [(fields[0], gene.upper()) for gene in fields[2:] if gene]
    return pd.DataFrame(rows, columns=["gene_set", "gene"]).drop_duplicates()


def parse_progeny(text: str, *, top: int) -> pd.DataFrame:
    """OmniPath PROGENy annotations -> each pathway's ``top`` genes by p-value."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    wide = (
        frame.set_index(["genesymbol", "record_id", "label"])["value"]
        .unstack("label")
        .reset_index()
        .dropna(subset=["pathway", "weight", "p_value"])
    )
    wide["weight"] = wide["weight"].astype(float)
    wide["p_value"] = wide["p_value"].astype(float)
    best = (
        wide.sort_values(["pathway", "p_value", "genesymbol"])
        .groupby("pathway", sort=True)
        .head(top)
    )
    return pd.DataFrame(
        {
            "pathway": best["pathway"].to_numpy(),
            "gene": best["genesymbol"].str.upper().to_numpy(),
            "weight": best["weight"].to_numpy(),
        }
    )


def driver_status(
    hotspot: pd.DataFrame,
    damaging: pd.DataFrame,
    lines: Sequence[str],
    *,
    min_fraction: float,
) -> pd.DataFrame:
    """Lines x driver genes, 1 where a hotspot or damaging mutation is called.

    Drivers are the hotspot matrix's genes called in at least ``min_fraction`` of
    ``lines``; GeneEffect is never read.
    """
    rows = [m for m in lines if m in hotspot.index]
    genes = list(hotspot.columns)
    called = (hotspot.loc[rows, genes].fillna(0) > 0) | (
        damaging.reindex(index=rows, columns=genes).fillna(0) > 0
    )
    drivers = [g for g in genes if called[g].mean() >= min_fraction]
    status = called.loc[:, drivers].astype(int)
    status.index.name = "model_id"
    return status


def training_msi(text: str, blocked: Collection[str]) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text), index_col=0)
    msi = frame["MSIScore"].dropna()
    msi.index = msi.index.astype(str)
    msi = msi.loc[~msi.index.isin(set(blocked))].sort_index()
    msi.index.name, msi.name = "model_id", "msi_score"
    return msi


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split", "extra-lines", "hotspot", "damaging", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    split = load_geneeffect_226_split(args.split)
    extra = load_extra_lines(args.extra_lines, split)
    blocked = {*split.val, *split.test, *extra.excluded}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.concat(parse_paralogs(fetch(paralog_url(c))) for c in CHROMOSOMES)
    pairs = pairs.groupby(["gene", "paralog"], as_index=False)["identity"].max()
    hotspot = read_depmap_matrix(args.hotspot)
    damaging = read_depmap_matrix(args.damaging)
    tables = {
        "paralogs.tsv.gz": (
            closest_paralogs(
                pairs, min_identity=PARALOG_MIN_IDENTITY, per_gene=PARALOGS_PER_GENE
            ),
            BIOMART_URL,
        ),
        "complexes.tsv": (parse_corum(fetch(CORUM_URL)), CORUM_URL),
        "hallmark.tsv": (parse_gmt(fetch(HALLMARK_URL)), HALLMARK_URL),
        "progeny.tsv": (parse_progeny(fetch(PROGENY_URL), top=PROGENY_TOP), PROGENY_URL),
        "drivers.tsv": (
            driver_status(
                hotspot,
                damaging,
                sorted(set(hotspot.index) - blocked),
                min_fraction=DRIVER_MIN_FRACTION,
            ).reset_index(),
            str(args.hotspot),
        ),
        "msi.tsv": (
            training_msi(fetch(SIGNATURES_URL), blocked).reset_index(),
            SIGNATURES_URL,
        ),
    }
    provenance = {
        "retrieved": date.today().isoformat(),
        "settings": {
            "paralog_min_identity": PARALOG_MIN_IDENTITY,
            "paralogs_per_gene": PARALOGS_PER_GENE,
            "progeny_top": PROGENY_TOP,
            "driver_min_fraction": DRIVER_MIN_FRACTION,
        },
        "files": {},
    }
    for name, (frame, source) in tables.items():
        frame.to_csv(out / name, sep="\t", index=False)
        provenance["files"][name] = {
            "source": source,
            "rows": len(frame),
            "sha256": _sha256(out / name),
        }
        print(name, len(frame))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement the package and loader**

`src/context_prior/__init__.py`:

```python
"""The linear context prior: a cross-fitted stagewise ridge from expression to the
GeneEffect residual. Imports ``src.data`` only, never training, eval or experiments."""
```

`src/context_prior/reference.py`:

```python
"""The pinned reference tables of the linear context prior."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Reference:
    """Gene-pair lists, gene sets and training-side line labels.

    Attributes:
        paralogs: ``gene, paralog, identity``, closest first within each gene.
        complexes: ``complex_id, gene``.
        hallmark: ``gene_set, gene``.
        progeny: ``pathway, gene, weight``.
        drivers: training-side lines x driver genes, 0/1.
        msi: training-side line -> MSI score.
    """

    paralogs: pd.DataFrame
    complexes: pd.DataFrame
    hallmark: pd.DataFrame
    progeny: pd.DataFrame
    drivers: pd.DataFrame
    msi: pd.Series


def load_reference(directory: Path, *, blocked: Collection[str]) -> Reference:
    """Load the tables; raise if a line table holds any ``blocked`` line."""
    directory = Path(directory)

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(directory / name, sep="\t")

    drivers = read("drivers.tsv").astype({"model_id": str}).set_index("model_id")
    msi = read("msi.tsv").astype({"model_id": str}).set_index("model_id")["msi_score"]
    leaked = sorted((set(drivers.index) | set(msi.index)) & set(blocked))
    if leaked:
        raise ValueError(
            f"reference tables hold held-out or patient-sharing lines: {leaked[:10]}"
        )
    return Reference(
        paralogs=read("paralogs.tsv.gz"),
        complexes=read("complexes.tsv"),
        hallmark=read("hallmark.tsv"),
        progeny=read("progeny.tsv"),
        drivers=drivers,
        msi=msi,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_reference.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `8 passed`

- [ ] **Step 6: Build the real tables on the Mac**

Run the builder with the arguments in its docstring (network needed; takes a few minutes for 24 BioMart queries).
Expected: printed row counts, roughly paralogs 100k–185k, complexes ~30k–50k, hallmark ~7.3k, progeny 1,400, drivers ~1,900 lines with 10–60 drivers, msi ~1,850. Check `du -sh configs/context_prior/reference` stays under 10 MB. Then:

```bash
uv run python -c "from pathlib import Path; import json; from src.context_prior.reference import load_reference; from src.data.splits import load_geneeffect_226_split as s; from src.data.extra_lines import load_extra_lines as e; sp=s(Path('configs/benchmarks/cell_line_geneeffect_226_split.json')); ex=e(Path('configs/benchmarks/extra_bulk_lines_26Q1.json'), sp); r=load_reference(Path('configs/context_prior/reference'), blocked={*sp.val,*sp.test,*ex.excluded}); print(r.drivers.shape, len(r.msi))"
```

Expected: no error, a shape and a count.

- [ ] **Step 7: Commit**

```bash
.venv/bin/ruff format src/data/prepare/build_context_reference.py src/context_prior tests/test_context_reference.py
git add src/data/prepare/build_context_reference.py src/context_prior tests/test_context_reference.py configs/context_prior/reference
git commit -m "feat(context-prior): pinned paralog, complex, gene-set, driver and MSI tables" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Shared expression space and patient folds

**Files:**
- Create: `src/context_prior/space.py`, `src/context_prior/folds.py`
- Test: `tests/test_context_prior_space.py`

**Interfaces:**
- Produces: `quantile_reference(expression: pd.DataFrame) -> np.ndarray`; `quantile_normalize(expression: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame`; `patient_folds(model_ids, patients, *, n_folds, seed) -> dict[str, int]`; `patient_subset(model_ids, patients, *, size, seed) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Quantile normalisation to a reference and patient-grouped folds and subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.folds import patient_folds, patient_subset
from src.context_prior.space import quantile_normalize, quantile_reference


def test_quantile_normalisation_maps_ranks_to_the_reference():
    frame = pd.DataFrame([[1.0, 5.0, 3.0], [10.0, 0.0, 20.0]], columns=list("abc"))
    reference = quantile_reference(frame)
    assert np.allclose(reference, [0.5, 6.5, 12.5])
    normalized = quantile_normalize(frame, reference)
    assert np.allclose(normalized.iloc[0], [0.5, 12.5, 6.5])
    assert np.allclose(normalized.iloc[1], [6.5, 0.5, 12.5])


def test_ties_share_one_value_and_nan_raises():
    frame = pd.DataFrame([[0.0, 0.0, 2.0]], columns=list("abc"))
    normalized = quantile_normalize(frame, np.array([0.0, 1.0, 2.0]))
    assert normalized.iloc[0, 0] == normalized.iloc[0, 1] == 0.5
    with pytest.raises(ValueError, match="finite"):
        quantile_normalize(frame.assign(a=np.nan), np.array([0.0, 1.0, 2.0]))


def test_patient_folds_keep_patients_together_and_are_seeded():
    lines = [f"L{i}" for i in range(12)]
    patients = {line: f"P{i // 2}" for i, line in enumerate(lines)}
    folds = patient_folds(lines, patients, n_folds=3, seed=0)
    assert folds == patient_folds(lines, patients, n_folds=3, seed=0)
    assert all(folds[f"L{2 * i}"] == folds[f"L{2 * i + 1}"] for i in range(6))
    assert set(folds.values()) == {0, 1, 2}


def test_patient_subset_takes_whole_patients():
    lines = [f"L{i}" for i in range(10)]
    patients = {line: f"P{i // 2}" for i, line in enumerate(lines)}
    subset = patient_subset(lines, patients, size=5, seed=3)
    assert 5 <= len(subset) <= 6
    chosen = {patients[m] for m in subset}
    assert sum(patients[m] in chosen for m in lines) == len(subset)
    with pytest.raises(ValueError):
        patient_subset(lines, patients, size=11, seed=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_space.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `src/context_prior/space.py`

```python
"""One expression space for bulk and pseudo-bulk: quantile normalisation per line."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _finite(expression: pd.DataFrame) -> np.ndarray:
    values = expression.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("expression must be finite")
    return values


def quantile_reference(expression: pd.DataFrame) -> np.ndarray:
    """The mean sorted profile of ``expression`` rows (lines x genes)."""
    return np.sort(_finite(expression), axis=1).mean(axis=0)


def quantile_normalize(expression: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame:
    """Give each line the reference value at each gene's rank; ties share a value."""
    values = _finite(expression)
    if values.shape[1] != reference.size:
        raise ValueError("expression and reference widths differ")
    ranks = rankdata(values, axis=1, method="average") - 1.0
    normalized = np.interp(ranks, np.arange(reference.size), reference)
    return pd.DataFrame(normalized, index=expression.index, columns=expression.columns)
```

`src/context_prior/folds.py`:

```python
"""Patient-grouped, seeded folds and subsets of training-side lines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def patient_folds(
    model_ids: Sequence[str], patients: Mapping[str, str], *, n_folds: int, seed: int
) -> dict[str, int]:
    """Fold of each line; every patient's lines share a fold."""
    groups = sorted({patients[m] for m in model_ids})
    if n_folds < 2 or len(groups) < n_folds:
        raise ValueError(f"{len(groups)} patients cannot fill {n_folds} folds")
    order = np.random.default_rng(seed).permutation(len(groups))
    fold = {groups[g]: rank % n_folds for rank, g in enumerate(order)}
    return {m: fold[patients[m]] for m in model_ids}


def patient_subset(
    model_ids: Sequence[str], patients: Mapping[str, str], *, size: int, seed: int
) -> tuple[str, ...]:
    """Whole patients in seeded random order until at least ``size`` lines."""
    by_patient: dict[str, list[str]] = {}
    for m in model_ids:
        by_patient.setdefault(patients[m], []).append(m)
    groups = sorted(by_patient)
    chosen: list[str] = []
    for g in np.random.default_rng(seed).permutation(len(groups)):
        if len(chosen) >= size:
            break
        chosen.extend(by_patient[groups[g]])
    if len(chosen) < size:
        raise ValueError(f"only {len(chosen)} lines for a subset of {size}")
    return tuple(sorted(chosen))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_prior_space.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/context_prior/space.py src/context_prior/folds.py tests/test_context_prior_space.py
git add src/context_prior/space.py src/context_prior/folds.py tests/test_context_prior_space.py
git commit -m "feat(context-prior): quantile expression space and patient folds" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Metric definitions and the fast selector

**Files:**
- Create: `src/context_prior/targets.py`
- Modify: `src/eval/metrics.py` (append `macro_gene_spearman` after `paired_line_bootstrap`)
- Test: `tests/test_context_prior_targets.py`

**Interfaces:**
- Consumes: `fit_gene_means` (`src/data/residual_target.py`), `fit_variable_gene_membership`, `fit_selective_genes`, `fit_residual_scale` (`src/data/geneeffect.py`), `assert_fit_eligible`, `FixedSplit`.
- Produces: `Definitions(genes, gene_means, selective, variable, residual_scale)`; `fit_definitions(gene_effect: pd.DataFrame, split: FixedSplit, genes: Sequence[str], settings: Mapping) -> Definitions`; `residual_frame(gene_effect, lines, definitions) -> pd.DataFrame`; `macro_gene_spearman(truth: np.ndarray, prediction: np.ndarray) -> float` (genes x lines arrays).

- [ ] **Step 1: Write the failing tests**

```python
"""Train-only metric definitions and the fast selective Spearman."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.targets import fit_definitions, residual_frame
from src.data.splits import FixedSplit
from src.eval.geneeffect import aggregate_geneeffect
from src.eval.metrics import macro_gene_spearman

SETTINGS = {
    "variable_gene_min_observations": 3,
    "variable_gene_percentile": 50,
    "selective_min_lines": 1,
    "selective_max_fraction": 0.9,
    "residual_sd_floor_percentile": 10,
}


def test_definitions_use_labelled_training_lines_only():
    rng = np.random.default_rng(0)
    lines = [f"T{i}" for i in range(6)] + ["V1", "V2"]
    effect = pd.DataFrame(
        rng.normal(-0.3, 0.4, (8, 4)), index=lines, columns=["A", "B", "C", "D"]
    )
    effect.loc["V1"] = 100.0  # a validation value must not move any fit
    effect.loc["T0":"T4", "D"] = np.nan  # D has one training label: dropped
    split = FixedSplit(train=tuple(lines[:6]), val=("V1", "V2"), test=())
    definitions = fit_definitions(effect, split, ["A", "B", "C", "D"], SETTINGS)
    assert definitions.genes == ("A", "B", "C")
    assert np.isclose(definitions.gene_means["A"], effect.loc["T0":"T5", "A"].mean())
    residual = residual_frame(effect, ["V2"], definitions)
    assert np.isclose(
        residual.loc["V2", "B"], effect.loc["V2", "B"] - definitions.gene_means["B"]
    )


def test_fast_selector_matches_aggregate_geneeffect():
    rng = np.random.default_rng(1)
    genes, lines = [f"G{i}" for i in range(5)], [f"L{i}" for i in range(7)]
    truth = rng.normal(size=(5, 7))
    prediction = rng.normal(size=(5, 7))
    truth[0, :3] = np.nan
    prediction[1] = 0.5  # constant: undefined
    prediction[2, :4] = 1.0  # ties
    frame = pd.DataFrame(
        {
            "model_id": np.tile(lines, 5),
            "gene_symbol": np.repeat(genes, 7),
            "residual": truth.ravel(),
            "residual_prediction": prediction.ravel(),
        }
    ).dropna(subset=["residual"])
    frame["gene_effect"] = frame["residual"]
    frame["geneeffect_prediction"] = frame["residual_prediction"]
    metrics, _, _ = aggregate_geneeffect(
        frame, model_ids=lines, genes=genes, variable_genes=genes, selective_genes=genes
    )
    assert np.isclose(macro_gene_spearman(truth, prediction), metrics["selective_spearman"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_targets.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`

- [ ] **Step 3: Implement** `src/context_prior/targets.py`

```python
"""The metric's fixed definitions, fitted on the labelled training lines only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.data.geneeffect import (
    fit_residual_scale,
    fit_selective_genes,
    fit_variable_gene_membership,
)
from src.data.residual_target import fit_gene_means
from src.data.splits import FixedSplit, assert_fit_eligible


@dataclass(frozen=True)
class Definitions:
    """Panel genes with a training mean, their means, gene sets and residual SD."""

    genes: tuple[str, ...]
    gene_means: pd.Series
    selective: tuple[str, ...]
    variable: tuple[str, ...]
    residual_scale: pd.Series


def fit_definitions(
    gene_effect: pd.DataFrame,
    split: FixedSplit,
    genes: Sequence[str],
    settings: Mapping[str, float],
) -> Definitions:
    """Fit as ``load_inputs`` does, on ``split.supervised_train`` only.

    ``settings`` holds the joint config's ``features`` keys. Genes without a
    training mean (fewer than three labels) leave the panel.
    """
    train = split.supervised_train
    for model_id in train:
        assert_fit_eligible(model_id, split)
    candidates = [g for g in genes if g in gene_effect.columns]
    labels = (
        gene_effect.loc[list(train), candidates]
        .rename_axis(index="model_id", columns="gene_symbol")
        .reset_index()
        .melt(id_vars="model_id", var_name="gene_symbol", value_name="gene_effect")
    )
    means = fit_gene_means(labels, train)
    kept = tuple(g for g in candidates if g in means.index)
    labels = labels.loc[labels["gene_symbol"].isin(kept)].copy()
    labels["residual"] = labels["gene_effect"] - labels["gene_symbol"].map(means)
    variable = fit_variable_gene_membership(
        labels,
        train,
        kept,
        min_observations=int(settings["variable_gene_min_observations"]),
        percentile=float(settings["variable_gene_percentile"]),
    )
    selective = fit_selective_genes(
        labels,
        train,
        kept,
        min_lines=int(settings["selective_min_lines"]),
        max_fraction=float(settings["selective_max_fraction"]),
    )
    scale = fit_residual_scale(
        labels,
        train,
        kept,
        floor_percentile=float(settings["residual_sd_floor_percentile"]),
    )
    return Definitions(
        genes=kept,
        gene_means=means.loc[list(kept)],
        selective=tuple(g for g in kept if g in selective),
        variable=tuple(g for g in kept if g in variable),
        residual_scale=scale.loc[list(kept)],
    )


def residual_frame(
    gene_effect: pd.DataFrame, lines: Sequence[str], definitions: Definitions
) -> pd.DataFrame:
    """Lines x genes ``y - mu_hat``; NaN where a line has no label."""
    frame = gene_effect.reindex(index=list(lines), columns=list(definitions.genes))
    return frame.sub(definitions.gene_means, axis=1)
```

Append to `src/eval/metrics.py`:

```python
def macro_gene_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Macro mean over genes (rows) of the Spearman across lines (columns).

    The same quantity as ``aggregate_geneeffect``'s ``selective_spearman``:
    non-finite pairs are dropped, and a gene with fewer than
    :data:`MIN_OBSERVATIONS` pairs or a constant side is undefined and left out.
    """
    if truth.ndim != 2 or truth.shape != prediction.shape:
        raise ValueError("truth and prediction must be equal genes x lines arrays")
    rho = _ResampledSpearman(truth, prediction)(np.ones((1, truth.shape[1])))[0]
    defined = rho[np.isfinite(rho)]
    return float(defined.mean()) if defined.size else math.nan
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_prior_targets.py tests/test_residual_metrics.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/context_prior/targets.py tests/test_context_prior_targets.py
git add src/context_prior/targets.py src/eval/metrics.py tests/test_context_prior_targets.py
git commit -m "feat(context-prior): train-only metric definitions and a fast selective Spearman" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Bridge from pseudo-bulk to bulk

**Files:**
- Create: `src/context_prior/bridge.py`
- Test: `tests/test_context_prior_bridge.py`

**Interfaces:**
- Produces: `Bridge(genes, slope, intercept)` with `apply(pseudobulk: pd.DataFrame) -> pd.DataFrame`; `fit_bridge(pseudobulk, bulk) -> Bridge` (same lines and columns, both quantile-normalised); `bridge_quality(bridged, bulk) -> pd.Series` (per-gene Pearson across lines, NaN when either side is constant).

- [ ] **Step 1: Write the failing tests**

```python
"""Per-gene affine bridge and its quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.bridge import bridge_quality, fit_bridge


def test_bridge_recovers_an_affine_map_and_handles_a_constant_gene():
    rng = np.random.default_rng(0)
    lines = [f"L{i}" for i in range(30)]
    pseudo = pd.DataFrame(rng.normal(size=(30, 2)), index=lines, columns=["A", "B"])
    pseudo["B"] = 1.0
    bulk = pd.DataFrame({"A": 2 * pseudo["A"] + 1, "B": rng.normal(3, 1, 30)}, index=lines)
    bridge = fit_bridge(pseudo, bulk)
    assert np.allclose(bridge.slope, [2.0, 0.0])
    assert np.isclose(bridge.intercept[1], bulk["B"].mean())
    bridged = bridge.apply(pseudo)
    assert np.allclose(bridged["A"], bulk["A"])
    quality = bridge_quality(bridged, bulk)
    assert np.isclose(quality["A"], 1.0) and np.isnan(quality["B"])


def test_bridge_requires_aligned_frames():
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=["L1", "L2"])
    with pytest.raises(ValueError, match="aligned"):
        fit_bridge(frame, frame.rename(index={"L2": "L3"}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_bridge.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `src/context_prior/bridge.py`

```python
"""A per-gene affine map from quantile-normalised pseudo-bulk to bulk.

3' UMI counts carry no gene-length normalisation and TPM does, so the offset and
slope differ by gene; one least-squares line per gene, fitted on lines with both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bridge:
    genes: tuple[str, ...]
    slope: np.ndarray
    intercept: np.ndarray

    def apply(self, pseudobulk: pd.DataFrame) -> pd.DataFrame:
        values = pseudobulk.loc[:, list(self.genes)].to_numpy(dtype=np.float64)
        return pd.DataFrame(
            values * self.slope + self.intercept,
            index=pseudobulk.index,
            columns=list(self.genes),
        )


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if list(left.index) != list(right.index) or list(left.columns) != list(right.columns):
        raise ValueError("pseudo-bulk and bulk must be aligned on lines and genes")
    return left.to_numpy(dtype=np.float64), right.to_numpy(dtype=np.float64)


def fit_bridge(pseudobulk: pd.DataFrame, bulk: pd.DataFrame) -> Bridge:
    """Least squares per gene; a gene constant in pseudo-bulk maps to its bulk mean."""
    x, y = _aligned(pseudobulk, bulk)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    variance = ((x - x_mean) ** 2).mean(axis=0)
    covariance = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    slope = np.where(variance > 0, covariance / np.where(variance > 0, variance, 1.0), 0.0)
    return Bridge(tuple(pseudobulk.columns), slope, y_mean - slope * x_mean)


def bridge_quality(bridged: pd.DataFrame, bulk: pd.DataFrame) -> pd.Series:
    """Per-gene Pearson across lines; NaN where either side is constant."""
    x, y = _aligned(bridged, bulk)
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    denominator = np.sqrt((x**2).sum(axis=0) * (y**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(denominator > 0, (x * y).sum(axis=0) / denominator, np.nan)
    return pd.Series(rho, index=bridged.columns, name="bridge_pearson")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_prior_bridge.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/context_prior/bridge.py tests/test_context_prior_bridge.py
git add src/context_prior/bridge.py tests/test_context_prior_bridge.py
git commit -m "feat(context-prior): per-gene affine bridge from pseudo-bulk to bulk" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Context views

**Files:**
- Create: `src/context_prior/views.py`
- Test: `tests/test_context_prior_views.py`

**Interfaces:**
- Consumes: `fit_context_pca`, `ContextPCA` (`src/data/context_pca.py`); `Reference`.
- Produces: `fit_expression_components(expression, n_components) -> ContextPCA`; `expression_components(pca, expression) -> pd.DataFrame`; `pathway_scores(expression, hallmark, progeny) -> pd.DataFrame`; `GenotypeEncoders.predict(components: pd.DataFrame) -> pd.DataFrame`; `fit_genotype(components, reference, lineage, folds) -> tuple[GenotypeEncoders, pd.DataFrame]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Expression components, pathway scores and out-of-sample predicted genotype."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.reference import Reference
from src.context_prior.views import (
    expression_components,
    fit_expression_components,
    fit_genotype,
    pathway_scores,
)


def reference(lines, drivers, msi):
    return Reference(
        paralogs=pd.DataFrame(columns=["gene", "paralog", "identity"]),
        complexes=pd.DataFrame(columns=["complex_id", "gene"]),
        hallmark=pd.DataFrame({"gene_set": ["S", "S", "T"], "gene": ["A", "B", "ZZ"]}),
        progeny=pd.DataFrame({"pathway": ["P", "P"], "gene": ["A", "C"], "weight": [1.0, -1.0]}),
        drivers=pd.DataFrame(drivers, index=pd.Index(lines, name="model_id")),
        msi=pd.Series(msi, index=pd.Index(lines, name="model_id"), name="msi_score"),
    )


def test_components_are_eigen_scaled_and_named():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(20, 6)), columns=list("ABCDEF"))
    pca = fit_expression_components(frame, 3)
    scores = expression_components(pca, frame)
    assert list(scores.columns) == ["pc1", "pc2", "pc3"]
    assert np.isclose(scores["pc1"].std(ddof=0), 1.0)


def test_pathway_scores_skip_absent_genes():
    frame = pd.DataFrame([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]], columns=["A", "B", "C"])
    ref = reference(["L"], {"KRAS": [0]}, [1.0])
    scores = pathway_scores(frame, ref.hallmark, ref.progeny)
    assert list(scores.columns) == ["hallmark:S", "progeny:P"]  # T has no present gene
    assert np.isclose(scores.iloc[0]["hallmark:S"], (3 / 3 + 2 / 3) / 2 - 0.5)
    assert np.isclose(scores.iloc[0]["progeny:P"], 3.0 - 1.0)


def test_genotype_features_are_out_of_sample_and_handle_a_constant_driver():
    rng = np.random.default_rng(0)
    lines = [f"L{i}" for i in range(60)]
    components = pd.DataFrame(rng.normal(size=(60, 4)), index=lines)
    mutated = (components[0] > 0).astype(int).tolist()
    ref = reference(lines, {"KRAS": mutated, "BRAF": [0] * 60}, components[1].tolist())
    lineage = pd.Series(["Lung"] * 30 + ["Bowel"] * 30, index=lines)
    folds = {m: i % 5 for i, m in enumerate(lines)}
    encoders, own = fit_genotype(components, ref, lineage, folds)
    assert list(own.index) == lines
    assert (own["driver:BRAF"] == 0.0).all()
    assert own.loc[components[0] > 1, "driver:KRAS"].mean() > 0.7
    query = encoders.predict(components.iloc[:3])
    assert list(query.columns) == list(own.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_views.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `src/context_prior/views.py`

```python
"""Context views a query line can supply, computed identically from bulk and
bridged pseudo-bulk: expression components, pathway scores, predicted genotype."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.context_prior.reference import Reference
from src.data.context_pca import ContextPCA, fit_context_pca

#: A lineage with fewer training-side lines is pooled into "other".
MIN_LINEAGE_LINES = 20


def fit_expression_components(expression: pd.DataFrame, n_components: int) -> ContextPCA:
    """Label-free, eigen-scaled principal components of expression rows."""
    return fit_context_pca(expression.to_numpy(dtype=np.float64), n_components)


def expression_components(pca: ContextPCA, expression: pd.DataFrame) -> pd.DataFrame:
    scores = pca.transform(expression.to_numpy(dtype=np.float64))
    columns = [f"pc{i + 1}" for i in range(scores.shape[1])]
    return pd.DataFrame(scores, index=expression.index, columns=columns)


def pathway_scores(
    expression: pd.DataFrame, hallmark: pd.DataFrame, progeny: pd.DataFrame
) -> pd.DataFrame:
    """Hallmark: centred mean within-line rank of the set's genes; PROGENy:
    weighted sum of its footprint genes. Gene sets with no present gene are left out."""
    genes = expression.columns
    ranks = expression.rank(axis=1, method="average").to_numpy() / len(genes) - 0.5
    values = expression.to_numpy(dtype=np.float64)
    columns: dict[str, np.ndarray] = {}
    for name, members in hallmark.groupby("gene_set")["gene"]:
        index = genes.get_indexer(members.unique())
        index = index[index >= 0]
        if index.size:
            columns[f"hallmark:{name}"] = ranks[:, index].mean(axis=1)
    for name, rows in progeny.groupby("pathway"):
        index = genes.get_indexer(rows["gene"])
        present = index >= 0
        if present.any():
            columns[f"progeny:{name}"] = (
                values[:, index[present]] @ rows["weight"].to_numpy()[present]
            )
    return pd.DataFrame(columns, index=expression.index)


@dataclass(frozen=True)
class GenotypeEncoders:
    """Driver status, MSI score and lineage predicted from expression components."""

    drivers: Mapping[str, Any]  # gene -> classifier, or the prevalence if one class
    msi: Ridge
    lineage: Any  # LogisticRegression, or the one class seen
    lineage_classes: tuple[str, ...]

    def predict(self, components: pd.DataFrame) -> pd.DataFrame:
        x = components.to_numpy(dtype=np.float64)
        columns: dict[str, np.ndarray] = {}
        for gene, model in self.drivers.items():
            columns[f"driver:{gene}"] = (
                model.predict_proba(x)[:, 1]
                if isinstance(model, LogisticRegression)
                else np.full(len(x), float(model))
            )
        columns["msi"] = self.msi.predict(x)
        if isinstance(self.lineage, LogisticRegression):
            probability = self.lineage.predict_proba(x)
            classes = list(self.lineage.classes_)
        else:
            probability, classes = np.ones((len(x), 1)), [self.lineage]
        for name in self.lineage_classes:
            columns[f"lineage:{name}"] = (
                probability[:, classes.index(name)] if name in classes else np.zeros(len(x))
            )
        return pd.DataFrame(columns, index=components.index)


def _fit_encoders(
    components: pd.DataFrame,
    reference: Reference,
    lineage: pd.Series,
    drivers: Sequence[str],
    lineage_classes: Sequence[str],
) -> GenotypeEncoders:
    rows = components.index
    driver_rows = rows.intersection(reference.drivers.index)
    models: dict[str, Any] = {}
    for gene in drivers:
        y = reference.drivers.loc[driver_rows, gene].to_numpy()
        if 0 < y.sum() < len(y):
            models[gene] = LogisticRegression(C=1.0, max_iter=2000).fit(
                components.loc[driver_rows].to_numpy(), y
            )
        else:
            models[gene] = float(y.mean()) if len(y) else 0.0
    msi_rows = rows.intersection(reference.msi.index)
    msi = Ridge(alpha=1.0).fit(
        components.loc[msi_rows].to_numpy(), reference.msi.loc[msi_rows].to_numpy()
    )
    labelled = rows.intersection(lineage.dropna().index)
    labels = lineage.loc[labelled].where(lineage.loc[labelled].isin(lineage_classes), "other")
    if labels.nunique() > 1:
        classifier = LogisticRegression(C=1.0, max_iter=2000).fit(
            components.loc[labelled].to_numpy(), labels.to_numpy()
        )
    else:
        classifier = str(labels.iloc[0]) if len(labels) else "other"
    return GenotypeEncoders(models, msi, classifier, tuple(lineage_classes))


def fit_genotype(
    components: pd.DataFrame,
    reference: Reference,
    lineage: pd.Series,
    folds: Mapping[str, int],
) -> tuple[GenotypeEncoders, pd.DataFrame]:
    """Encoders fitted on every row, and each row's own out-of-sample features.

    A row's features come from the encoders of the inner fold that excludes it;
    query rows use the returned encoders, fitted on every row.
    """
    rows = list(components.index)
    drivers = tuple(reference.drivers.columns)
    counts = lineage.reindex(rows).value_counts()
    classes = (*sorted(counts.index[counts >= MIN_LINEAGE_LINES]), "other")
    encoders = _fit_encoders(components, reference, lineage, drivers, classes)
    own = encoders.predict(components)
    for fold in sorted({folds[m] for m in rows}):
        held = [m for m in rows if folds[m] == fold]
        rest = [m for m in rows if folds[m] != fold]
        inner = _fit_encoders(components.loc[rest], reference, lineage, drivers, classes)
        own.loc[held] = inner.predict(components.loc[held])[own.columns].to_numpy()
    return encoders, own
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_prior_views.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/context_prior/views.py tests/test_context_prior_views.py
git add src/context_prior/views.py tests/test_context_prior_views.py
git commit -m "feat(context-prior): expression components, pathway scores and predicted genotype" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Gene features

**Files:**
- Create: `src/context_prior/gene_features.py`
- Test: `tests/test_context_prior_features.py`

**Interfaces:**
- Consumes: `Reference`.
- Produces: `KNOWLEDGE_FEATURES = ("own", "paralog_min", "paralog_sum", "paralog_closest", "complex_mean", "low_partner_fraction")`; `PartnerIndex(own: np.ndarray, paralogs: list[np.ndarray], complex_members: list[np.ndarray], partners: list[np.ndarray])`; `partner_index(genes, space, reference) -> PartnerIndex`; `knowledge_features(expression: np.ndarray, index: PartnerIndex, low_threshold: np.ndarray) -> np.ndarray` (lines x genes x 6, NaN where undefined); `select_genes(expression: np.ndarray, residual: np.ndarray, count: int, own: np.ndarray, chunk: int = 1024) -> np.ndarray` (genes x count positions).

- [ ] **Step 1: Write the failing tests**

```python
"""Own, paralog, complex and low-partner features; data-selected genes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.gene_features import (
    knowledge_features,
    partner_index,
    select_genes,
)
from src.context_prior.reference import Reference

REFERENCE = Reference(
    paralogs=pd.DataFrame(
        {"gene": ["A", "A", "B"], "paralog": ["B", "C", "A"], "identity": [60.0, 40.0, 60.0]}
    ),
    complexes=pd.DataFrame({"complex_id": [1, 1, 1], "gene": ["A", "C", "D"]}),
    hallmark=pd.DataFrame(columns=["gene_set", "gene"]),
    progeny=pd.DataFrame(columns=["pathway", "gene", "weight"]),
    drivers=pd.DataFrame(),
    msi=pd.Series(dtype=float),
)


def test_knowledge_features_and_an_unmeasured_gene():
    space = ["A", "B", "C", "D"]
    genes = ["A", "Q"]  # Q has no expression column and no partners
    index = partner_index(genes, space, REFERENCE)
    expression = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 1.0]])
    features = knowledge_features(expression, index, low_threshold=np.zeros(4))
    a = features[:, 0]
    assert a[0].tolist() == [1.0, 2.0, 5.0, 2.0, 3.5, 0.0]
    assert np.isclose(a[1, 5], 1.0 / 3.0)  # B (paralog) is at its threshold; C, D not
    assert np.isnan(features[:, 1]).all()


def test_selected_genes_exclude_the_gene_itself_and_survive_constant_columns():
    rng = np.random.default_rng(0)
    expression = rng.normal(size=(50, 6))
    expression[:, 5] = 2.0  # constant column
    residual = np.column_stack([expression[:, 0] + 0.1 * rng.normal(size=50), expression[:, 3]])
    selection = select_genes(expression, residual, 2, own=np.array([0, 3]))
    assert 0 not in selection[0] and 3 not in selection[1]
    assert np.isfinite(selection).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_features.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `src/context_prior/gene_features.py`

```python
"""Features of gene g in line c: own, paralog and complex-partner expression, the
low-partner fraction (ISLE's cSL score), and data-selected genes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.context_prior.reference import Reference

KNOWLEDGE_FEATURES = (
    "own",
    "paralog_min",
    "paralog_sum",
    "paralog_closest",
    "complex_mean",
    "low_partner_fraction",
)


@dataclass(frozen=True)
class PartnerIndex:
    """Column positions in the expression space per target gene (-1: absent)."""

    own: np.ndarray
    paralogs: list[np.ndarray]  # closest first
    complex_members: list[np.ndarray]
    partners: list[np.ndarray]  # paralogs and complex members


def partner_index(
    genes: Sequence[str], space: Sequence[str], reference: Reference
) -> PartnerIndex:
    position = {gene: i for i, gene in enumerate(space)}
    paralogs = {
        gene: [position[p] for p in rows["paralog"] if p in position]
        for gene, rows in reference.paralogs.groupby("gene", sort=False)
    }
    members: dict[str, set[str]] = {}
    for _, rows in reference.complexes.groupby("complex_id"):
        group = set(rows["gene"])
        for gene in group:
            members.setdefault(gene, set()).update(group - {gene})
    complex_members = {
        gene: sorted(position[m] for m in others if m in position)
        for gene, others in members.items()
    }
    own = np.array([position.get(g, -1) for g in genes])
    paralog_list = [np.array(paralogs.get(g, []), dtype=int) for g in genes]
    complex_list = [np.array(complex_members.get(g, []), dtype=int) for g in genes]
    partners = [
        np.unique(np.concatenate([p, c])).astype(int)
        for p, c in zip(paralog_list, complex_list, strict=True)
    ]
    return PartnerIndex(own, paralog_list, complex_list, partners)


def knowledge_features(
    expression: np.ndarray, index: PartnerIndex, low_threshold: np.ndarray
) -> np.ndarray:
    """Lines x genes x 6 raw features; NaN where a gene has no such partner.

    A partner counts as low when its expression is at or below its training-side
    10th percentile (``low_threshold``, one value per expression column).
    """
    lines, genes = expression.shape[0], len(index.own)
    out = np.full((lines, genes, len(KNOWLEDGE_FEATURES)), np.nan, dtype=np.float32)
    for g in range(genes):
        if index.own[g] >= 0:
            out[:, g, 0] = expression[:, index.own[g]]
        paralogs = index.paralogs[g]
        if paralogs.size:
            values = expression[:, paralogs]
            out[:, g, 1] = values.min(axis=1)
            out[:, g, 2] = values.sum(axis=1)
            out[:, g, 3] = values[:, 0]
        members = index.complex_members[g]
        if members.size:
            out[:, g, 4] = expression[:, members].mean(axis=1)
        partners = index.partners[g]
        if partners.size:
            out[:, g, 5] = (expression[:, partners] <= low_threshold[partners]).mean(axis=1)
    return out


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    return (values - center) / np.where(scale > 0, scale, 1.0)


def select_genes(
    expression: np.ndarray,
    residual: np.ndarray,
    count: int,
    own: np.ndarray,
    chunk: int = 1024,
) -> np.ndarray:
    """Genes x ``count`` expression columns with the largest absolute Pearson
    correlation with each gene's residual across rows; a gene's own column is
    excluded. Missing residuals count as zero."""
    x = _standardize_columns(expression)
    y = _standardize_columns(np.where(np.isfinite(residual), residual, 0.0))
    genes = y.shape[1]
    out = np.empty((genes, count), dtype=int)
    for start in range(0, genes, chunk):
        stop = min(start + chunk, genes)
        strength = np.abs(x.T @ y[:, start:stop]) / x.shape[0]
        for local, g in enumerate(range(start, stop)):
            if own[g] >= 0:
                strength[own[g], local] = -1.0
        out[start:stop] = np.argpartition(-strength, count - 1, axis=0)[:count].T
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_prior_features.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/context_prior/gene_features.py tests/test_context_prior_features.py
git add src/context_prior/gene_features.py tests/test_context_prior_features.py
git commit -m "feat(context-prior): paralog, complex, low-partner and data-selected gene features" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Ridge stages

**Files:**
- Create: `src/context_prior/ridge.py`
- Test: `tests/test_context_prior_ridge.py`

**Interfaces:**
- Produces: `column_scale(values, axis=0) -> (center, scale)`; `LinearFit.predict(features) -> np.ndarray`; `shared_ridge(features, targets, penalties, *, basis=None) -> list[LinearFit]`; `GeneFit.predict(features) -> np.ndarray`; `gene_ridge(features, targets, shrinkages, *, pooled: bool) -> list[GeneFit]`; `SelectedFit.predict(expression) -> np.ndarray`; `selected_ridge(expression, selection, targets, penalties, *, chunk=128) -> list[SelectedFit]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Closed-form ridge stages against scikit-learn and limiting cases."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.context_prior.ridge import gene_ridge, selected_ridge, shared_ridge


def test_shared_ridge_matches_sklearn_on_standardised_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3)) * [1.0, 5.0, 0.1]
    x = np.column_stack([x, np.full(40, 7.0)])  # constant column: no effect
    y = x[:, :2] @ [[1.0, -1.0], [0.5, 0.2]] + rng.normal(size=(40, 2))
    fit = shared_ridge(x, y, [0.1])[0]
    z = (x[:, :3] - x[:, :3].mean(0)) / x[:, :3].std(0)
    reference = Ridge(alpha=0.1 * 40).fit(z, y)
    assert np.allclose(fit.predict(x), reference.predict(z))


def test_reduced_rank_projects_targets_on_the_basis():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2))
    y = rng.normal(size=(30, 4))
    basis = np.linalg.qr(rng.normal(size=(4, 2)))[0]
    fit = shared_ridge(x, y, [1.0], basis=basis)[0]
    assert np.linalg.matrix_rank(fit.coef) <= 2


def test_pooled_gene_ridge_limits():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(60, 5, 1))
    features = (features - features.mean(axis=0)).astype(np.float32)  # standardised
    targets = 2.0 * features[:, :, 0] + rng.normal(size=(60, 5)) * 0.01
    pooled_only, per_gene = gene_ridge(features, targets, [np.inf, 1e-6], pooled=True)
    assert np.allclose(pooled_only.deviation, 0.0)
    assert np.isclose(pooled_only.pooled[0], 2.0, atol=0.01)
    assert np.allclose(per_gene.predict(features), targets, atol=0.05)


def test_selected_ridge_matches_a_per_gene_fit():
    rng = np.random.default_rng(3)
    expression = rng.normal(size=(50, 6))
    expression -= expression.mean(axis=0)  # the prior passes standardised columns
    selection = np.array([[1, 2], [0, 4]])
    targets = np.column_stack([expression[:, 1], expression[:, 4] - expression[:, 0]])
    fit = selected_ridge(expression, selection, targets, [1e-6])[0]
    assert np.allclose(fit.predict(expression), targets, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_ridge.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `src/context_prior/ridge.py`

```python
"""Closed-form ridge stages of the linear context prior.

Each stage predicts lines x genes and is fitted to what earlier stages left.
Penalties are multiplied by the number of fitted lines, so one grid serves every
training-set size. Features are standardised with the fitted lines' statistics; a
constant column gets scale 1 and contributes nothing. Every stage has a per-gene
intercept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def column_scale(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    center = values.mean(axis=axis)
    scale = values.std(axis=axis)
    return center, np.where(scale > 0, scale, 1.0)


@dataclass(frozen=True)
class LinearFit:
    """One feature matrix shared by every gene."""

    center: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # features x genes
    intercept: np.ndarray  # genes

    def predict(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.center) / self.scale) @ self.coef + self.intercept


def shared_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    penalties: Sequence[float],
    *,
    basis: np.ndarray | None = None,
) -> list[LinearFit]:
    """Ridge of every target column on the same features, one fit per penalty.

    With ``basis`` (genes x k, orthonormal columns) the targets are projected on
    it and the coefficients mapped back: a reduced-rank context term.
    """
    n = features.shape[0]
    center, scale = column_scale(features)
    z = (features - center) / scale
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    if basis is not None:
        centered = centered @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(z.T @ z)
    projected = eigenvectors.T @ (z.T @ centered)
    fits = []
    for penalty in penalties:
        if not penalty > 0:
            raise ValueError("shared ridge penalties must be positive")
        coef = eigenvectors @ (projected / (eigenvalues + penalty * n)[:, None])
        if basis is not None:
            coef = coef @ basis.T
        fits.append(LinearFit(center, scale, coef, intercept))
    return fits


@dataclass(frozen=True)
class GeneFit:
    """Gene-specific features (lines x genes x k): pooled weights plus deviations."""

    pooled: np.ndarray  # k
    deviation: np.ndarray  # genes x k
    intercept: np.ndarray  # genes

    def predict(self, features: np.ndarray) -> np.ndarray:
        weights = self.pooled[None, :] + self.deviation
        return np.einsum("ngk,gk->ng", features, weights) + self.intercept


def gene_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    shrinkages: Sequence[float],
    *,
    pooled: bool,
) -> list[GeneFit]:
    """Per-gene ridge on gene-specific features already standardised per gene.

    With ``pooled``, weights shared by every gene are fitted first by least squares
    over all (line, gene) rows, and each gene's deviation from them is shrunk by
    ``shrinkage * n``; an infinite shrinkage keeps the pooled weights only.
    """
    n, genes, k = features.shape
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    gram = np.einsum("ngk,ngl->gkl", features, features, dtype=np.float64)
    cross = np.einsum("ngk,ng->gk", features, centered, dtype=np.float64)
    shared = np.zeros(k)
    if pooled:
        shared = np.linalg.lstsq(gram.sum(axis=0), cross.sum(axis=0), rcond=None)[0]
        cross = cross - np.einsum("gkl,l->gk", gram, shared)
    fits = []
    for shrinkage in shrinkages:
        if np.isinf(shrinkage):
            deviation = np.zeros((genes, k))
        elif shrinkage > 0:
            system = gram + shrinkage * n * np.eye(k)
            deviation = np.linalg.solve(system, cross[..., None])[..., 0]
        else:
            raise ValueError("gene ridge shrinkages must be positive or infinite")
        fits.append(GeneFit(shared, deviation, intercept))
    return fits


@dataclass(frozen=True)
class SelectedFit:
    """Per-gene ridge on the expression of each gene's selected columns."""

    selection: np.ndarray  # genes x count
    coef: np.ndarray  # genes x count
    intercept: np.ndarray  # genes

    def predict(self, expression: np.ndarray, chunk: int = 128) -> np.ndarray:
        genes = self.selection.shape[0]
        out = np.empty((expression.shape[0], genes))
        for start in range(0, genes, chunk):
            stop = min(start + chunk, genes)
            gathered = expression[:, self.selection[start:stop]]
            out[:, start:stop] = np.einsum("ncm,cm->nc", gathered, self.coef[start:stop])
        return out + self.intercept


def selected_ridge(
    expression: np.ndarray,
    selection: np.ndarray,
    targets: np.ndarray,
    penalties: Sequence[float],
    *,
    chunk: int = 128,
) -> list[SelectedFit]:
    """One ridge per gene on its selected expression columns, which the caller
    standardises on the fitted lines (mean zero), so the intercept is the mean."""
    n = expression.shape[0]
    genes, count = selection.shape
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    coefs = [np.empty((genes, count)) for _ in penalties]
    for start in range(0, genes, chunk):
        stop = min(start + chunk, genes)
        gathered = expression[:, selection[start:stop]]
        gram = np.einsum("ncm,ncl->cml", gathered, gathered)
        cross = np.einsum("ncm,nc->cm", gathered, centered[:, start:stop])
        for coef, penalty in zip(coefs, penalties, strict=True):
            if not penalty > 0:
                raise ValueError("selected ridge penalties must be positive")
            system = gram + penalty * n * np.eye(count)
            coef[start:stop] = np.linalg.solve(system, cross[..., None])[..., 0]
    return [SelectedFit(selection, coef, intercept) for coef in coefs]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_prior_ridge.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/context_prior/ridge.py tests/test_context_prior_ridge.py
git add src/context_prior/ridge.py tests/test_context_prior_ridge.py
git commit -m "feat(context-prior): closed-form shared, pooled and per-gene ridge stages" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: The stagewise prior and cross-fitting

**Files:**
- Create: `src/context_prior/prior.py`
- Test: `tests/test_context_prior_fit.py`

**Interfaces:**
- Consumes: every earlier `src/context_prior` module.
- Produces: `BLOCKS`, `CONTEXT_BLOCKS`; `Stage(block: str, penalty: float, selected: int = 0)`; `PriorSpec(stages: tuple[Stage, ...], rank: int | None = None)`; `PriorInputs(expression, residual, components, reference, lineage, patients)`; `FittedPrior.predict(expression: pd.DataFrame) -> dict[str, pd.DataFrame]`; `fit_prior(spec, inputs, *, fit_lines, encoder_lines) -> FittedPrior`; `total(predictions) -> pd.DataFrame`; `crossfit(spec, inputs, *, folds, queries: Mapping[int, pd.DataFrame], labelled, encoder_lines) -> dict[str, pd.DataFrame]`.

- [ ] **Step 1: Write the failing tests**

```python
"""The stagewise prior on synthetic lines: recovery, missing labels, query keys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.prior import (
    PriorInputs,
    PriorSpec,
    Stage,
    crossfit,
    fit_prior,
    total,
)
from src.context_prior.reference import Reference
from src.context_prior.views import fit_expression_components

SPACE = [f"E{i}" for i in range(12)]
GENES = ["E0", "E1", "Q"]  # Q has no expression column


def synthetic(seed=0, lines=80):
    rng = np.random.default_rng(seed)
    ids = [f"L{i}" for i in range(lines)]
    expression = pd.DataFrame(rng.normal(size=(lines, 12)), index=ids, columns=SPACE)
    residual = pd.DataFrame(
        {
            "E0": expression["E3"] + 0.1 * rng.normal(size=lines),
            "E1": -expression["E1"] + 0.1 * rng.normal(size=lines),
            "Q": expression["E5"] - expression["E6"],
        },
        index=ids,
    )
    residual.iloc[0, 0] = np.nan  # a missing training label
    reference = Reference(
        paralogs=pd.DataFrame({"gene": ["E0"], "paralog": ["E2"], "identity": [50.0]}),
        complexes=pd.DataFrame({"complex_id": [1, 1], "gene": ["E1", "E4"]}),
        hallmark=pd.DataFrame({"gene_set": ["S"], "gene": ["E7"]}),
        progeny=pd.DataFrame({"pathway": ["P"], "gene": ["E8"], "weight": [1.0]}),
        drivers=pd.DataFrame(
            {"KRAS": (expression["E9"] > 0).astype(int)}, index=pd.Index(ids, name="model_id")
        ),
        msi=pd.Series(expression["E10"].to_numpy(), index=pd.Index(ids, name="model_id")),
    )
    inputs = PriorInputs(
        expression=expression,
        residual=residual,
        components=fit_expression_components(expression, 6),
        reference=reference,
        lineage=pd.Series(["Lung"] * lines, index=ids),
        patients={m: m for m in ids},
    )
    return inputs, ids


ALL_BLOCKS = PriorSpec(
    (
        Stage("expression_components", 0.01),
        Stage("pathway_scores", 1.0),
        Stage("predicted_genotype", 1.0),
        Stage("own_expression", 1.0),
        Stage("partners", 1.0),
        Stage("data_selected", 0.01, selected=2),
    )
)


def test_every_block_fits_and_recovers_signal():
    inputs, ids = synthetic()
    fitted = fit_prior(ALL_BLOCKS, inputs, fit_lines=ids[:60], encoder_lines=ids)
    query = inputs.expression.loc[ids[60:]]
    stages = fitted.predict(query)
    assert list(stages) == [s.block for s in ALL_BLOCKS.stages]
    prediction = total(stages)
    truth = inputs.residual.loc[ids[60:]]
    for gene in GENES:
        assert np.corrcoef(prediction[gene], truth[gene])[0, 1] > 0.8
    assert np.isfinite(prediction.to_numpy()).all()


def test_fit_lines_must_be_encoder_lines():
    inputs, ids = synthetic()
    with pytest.raises(ValueError, match="encoder"):
        fit_prior(ALL_BLOCKS, inputs, fit_lines=ids, encoder_lines=ids[:40])


def test_crossfit_predicts_each_fold_from_the_others_keyed_by_query():
    inputs, ids = synthetic()
    folds = {m: i % 4 for i, m in enumerate(ids)}
    queries = {
        k: inputs.expression.loc[[m for m in ids if folds[m] == k]] + 0.5 for k in range(4)
    }
    spec = PriorSpec((Stage("expression_components", 0.1),), rank=2)
    predicted = crossfit(
        spec, inputs, folds=folds, queries=queries, labelled=ids, encoder_lines=ids
    )
    frame = predicted["expression_components"]
    assert sorted(frame.index) == sorted(ids)
    fold_zero = [m for m in ids if folds[m] == 0]
    direct = fit_prior(
        spec,
        inputs,
        fit_lines=[m for m in ids if folds[m] != 0],
        encoder_lines=[m for m in ids if folds[m] != 0],
    ).predict(queries[0])["expression_components"]
    assert np.allclose(frame.loc[fold_zero], direct)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_fit.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `src/context_prior/prior.py`

```python
"""The linear context prior: a stagewise ridge over context views and gene features.

``fit_prior`` fits the stages of a :class:`PriorSpec` in order, each on the residual
the earlier stages leave, from training-side bulk rows. ``FittedPrior.predict`` maps
any expression rows (bulk, or bridged pseudo-bulk) to per-stage predictions in units
of the per-gene residual SD. A missing training label counts as the residual's
expected value, zero, in fitting. Everything that reads labels is fitted on the rows
it is given, so cross-fitting is ``fit_prior`` on the rows outside a fold; the
expression components are label-free and fitted once, outside.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.folds import patient_folds
from src.context_prior.gene_features import (
    PartnerIndex,
    knowledge_features,
    partner_index,
    select_genes,
)
from src.context_prior.reference import Reference
from src.context_prior.ridge import column_scale, gene_ridge, selected_ridge, shared_ridge
from src.context_prior.views import expression_components, fit_genotype, pathway_scores
from src.data.context_pca import ContextPCA

BLOCKS = (
    "expression_components",
    "pathway_scores",
    "predicted_genotype",
    "own_expression",
    "partners",
    "data_selected",
)
CONTEXT_BLOCKS = BLOCKS[:3]
#: Columns of ``knowledge_features`` read by each knowledge block.
_KNOWLEDGE_COLUMNS = {"own_expression": [0], "partners": [1, 2, 3, 4, 5]}
#: Folds of the predicted-genotype encoders, inside whatever lines a fit is given.
INNER_FOLDS, INNER_FOLD_SEED = 5, 1
#: A partner is low at or below this training-side percentile.
LOW_PERCENTILE = 10


@dataclass(frozen=True)
class Stage:
    """One block; ``penalty`` is the ridge strength times n, or for a knowledge
    block the shrinkage toward the pooled weights (infinite: pooled only)."""

    block: str
    penalty: float
    selected: int = 0

    def __post_init__(self) -> None:
        if self.block not in BLOCKS:
            raise ValueError(f"unknown block {self.block!r}")
        if (self.block == "data_selected") != (self.selected > 0):
            raise ValueError("data_selected, and only it, needs a positive count")


@dataclass(frozen=True)
class PriorSpec:
    stages: tuple[Stage, ...]
    rank: int | None = None  # reduced-rank context term; None: per gene

    def __post_init__(self) -> None:
        blocks = [stage.block for stage in self.stages]
        if len(set(blocks)) != len(blocks):
            raise ValueError("a block appears twice")


@dataclass(frozen=True)
class PriorInputs:
    """Everything a fit reads, keyed by ModelID.

    Attributes:
        expression: Quantile-normalised bulk rows of the training side.
        residual: Labelled training-side lines x genes, in residual-SD units.
        components: Expression components, fitted once, label-free.
        reference: Pinned reference tables.
        lineage: Training-side lineage labels.
        patients: PatientID of every line.
    """

    expression: pd.DataFrame
    residual: pd.DataFrame
    components: ContextPCA
    reference: Reference
    lineage: pd.Series
    patients: Mapping[str, str]


@dataclass
class FittedPrior:
    genes: tuple[str, ...]
    space: tuple[str, ...]
    stages: list[tuple[str, Callable[[pd.DataFrame], np.ndarray], Any]]

    def predict(self, expression: pd.DataFrame) -> dict[str, pd.DataFrame]:
        rows = expression.loc[:, list(self.space)]
        return {
            block: pd.DataFrame(
                model.predict(features(rows)), index=rows.index, columns=list(self.genes)
            )
            for block, features, model in self.stages
        }


def total(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = list(predictions.values())
    if not frames:
        raise ValueError("a prior needs at least one stage")
    out = frames[0].copy()
    for frame in frames[1:]:
        out = out + frame
    return out


def _context_view(block, inputs, fit_rows, encoder_rows):
    if block == "expression_components":

        def features(rows):
            return inputs.components.transform(rows.to_numpy(dtype=np.float64))

        return features, features(fit_rows)
    if block == "pathway_scores":

        def features(rows):
            scores = pathway_scores(rows, inputs.reference.hallmark, inputs.reference.progeny)
            return scores.to_numpy()

        return features, features(fit_rows)
    components = expression_components(inputs.components, encoder_rows)
    folds = patient_folds(
        list(encoder_rows.index), inputs.patients, n_folds=INNER_FOLDS, seed=INNER_FOLD_SEED
    )
    encoders, own = fit_genotype(components, inputs.reference, inputs.lineage, folds)

    def features(rows):
        return encoders.predict(expression_components(inputs.components, rows)).to_numpy()

    return features, own.loc[list(fit_rows.index)].to_numpy()


def _knowledge_view(columns, index: PartnerIndex, low, fit_rows):
    raw = knowledge_features(fit_rows.to_numpy(dtype=np.float64), index, low)[:, :, columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN: no such partner
        mean, std = np.nanmean(raw, axis=0), np.nanstd(raw, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)

    def standardize(values):
        z = (values - mean) / std
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)

    def features(rows):
        return standardize(
            knowledge_features(rows.to_numpy(dtype=np.float64), index, low)[:, :, columns]
        )

    return features, standardize(raw)


def fit_prior(
    spec: PriorSpec,
    inputs: PriorInputs,
    *,
    fit_lines: Sequence[str],
    encoder_lines: Sequence[str],
) -> FittedPrior:
    """Fit every stage in order on labelled ``fit_lines``; encoders that read no
    GeneEffect (predicted genotype, low-expression thresholds) use ``encoder_lines``."""
    if not set(fit_lines) <= set(encoder_lines):
        raise ValueError("fit lines must be encoder lines too")
    genes = tuple(inputs.residual.columns)
    space = tuple(inputs.expression.columns)
    fit_rows = inputs.expression.loc[list(fit_lines)]
    encoder_rows = inputs.expression.loc[list(encoder_lines)]
    remaining = inputs.residual.loc[list(fit_lines)].to_numpy(dtype=np.float64)
    remaining = np.where(np.isfinite(remaining), remaining, 0.0)
    basis = None
    if spec.rank is not None:
        basis = np.linalg.svd(remaining, full_matrices=False)[2][: spec.rank].T
    position = {gene: i for i, gene in enumerate(space)}
    index = None
    low = None
    stages = []
    for stage in spec.stages:
        if stage.block in CONTEXT_BLOCKS:
            features, train = _context_view(stage.block, inputs, fit_rows, encoder_rows)
            model = shared_ridge(train, remaining, [stage.penalty], basis=basis)[0]
        elif stage.block in _KNOWLEDGE_COLUMNS:
            if index is None:
                index = partner_index(genes, space, inputs.reference)
                low = np.percentile(encoder_rows.to_numpy(), LOW_PERCENTILE, axis=0)
            features, train = _knowledge_view(
                _KNOWLEDGE_COLUMNS[stage.block], index, low, fit_rows
            )
            model = gene_ridge(train, remaining, [stage.penalty], pooled=True)[0]
        else:
            center, scale = column_scale(fit_rows.to_numpy(dtype=np.float64))

            def features(rows, center=center, scale=scale):
                return (rows.to_numpy(dtype=np.float64) - center) / scale

            train = features(fit_rows)
            own = np.array([position.get(gene, -1) for gene in genes])
            selection = select_genes(train, remaining, stage.selected, own)
            model = selected_ridge(train, selection, remaining, [stage.penalty])[0]
        remaining = remaining - model.predict(train)
        stages.append((stage.block, features, model))
    return FittedPrior(genes, space, stages)


def crossfit(
    spec: PriorSpec,
    inputs: PriorInputs,
    *,
    folds: Mapping[str, int],
    queries: Mapping[int, pd.DataFrame],
    labelled: Sequence[str],
    encoder_lines: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Per-stage predictions of each fold's query rows by the prior fitted on the
    lines outside that fold."""
    parts: dict[str, list[pd.DataFrame]] = {}
    for fold, query in sorted(queries.items()):
        fitted = fit_prior(
            spec,
            inputs,
            fit_lines=[m for m in labelled if folds[m] != fold],
            encoder_lines=[m for m in encoder_lines if folds[m] != fold],
        )
        for block, frame in fitted.predict(query).items():
            parts.setdefault(block, []).append(frame)
    return {block: pd.concat(frames) for block, frames in parts.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_prior_fit.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `3 passed`. If the signal-recovery threshold fails for `Q` (an own-expression-free gene), print each stage's correlation and fix the stage that loses it; do not lower the threshold.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/context_prior/prior.py tests/test_context_prior_fit.py
git add src/context_prior/prior.py tests/test_context_prior_fit.py
git commit -m "feat(context-prior): stagewise prior with cross-fitting" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Gene-conditioned view weights

**Files:**
- Create: `src/context_prior/view_weights.py`
- Test: `tests/test_context_prior_view_weights.py`

**Interfaces:**
- Produces: `ViewWeights(blocks: tuple[str, ...], weights: pd.DataFrame)` with `combine(stage_predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame`; `fit_view_weights(stage_predictions, residual, embeddings: Mapping[str, np.ndarray], blocks, *, steps=300, learning_rate=0.05, seed=0) -> ViewWeights`.

- [ ] **Step 1: Write the failing test**

```python
"""Per-gene view weights from gene embeddings, learned on out-of-fold stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.view_weights import fit_view_weights


def test_view_weights_follow_the_embedding():
    rng = np.random.default_rng(0)
    lines = [f"L{i}" for i in range(200)]
    genes = [f"G{i}" for i in range(40)]
    first = pd.DataFrame(rng.normal(size=(200, 40)), index=lines, columns=genes)
    second = pd.DataFrame(rng.normal(size=(200, 40)), index=lines, columns=genes)
    kind = np.array([i % 2 for i in range(40)])  # even genes follow view one
    truth = np.where(kind == 0, 2 * first, 2 * second)
    residual = pd.DataFrame(truth, index=lines, columns=genes)
    embeddings = {g: np.array([1.0, 0.0]) if k == 0 else np.array([0.0, 1.0])
                  for g, k in zip(genes, kind, strict=True)}
    weights = fit_view_weights(
        {"expression_components": first, "pathway_scores": second},
        residual,
        embeddings,
        ("expression_components", "pathway_scores"),
    )
    assert weights.weights.loc["G0", "expression_components"] > 1.5
    assert weights.weights.loc["G1", "pathway_scores"] > 1.5
    combined = weights.combine({"expression_components": first, "pathway_scores": second})
    assert combined.shape == first.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_context_prior_view_weights.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `src/context_prior/view_weights.py`

```python
"""Gene-conditioned view weights: the prior's only gradient-trained candidate.

The context stages' predictions are mixed per gene with weights
``V * softmax(M e_g + b)`` over the V views, so uniform weights reproduce the
plain stagewise sum. Genes without an embedding keep uniform weights. Trained on
out-of-fold stage predictions; other stages pass through unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ViewWeights:
    blocks: tuple[str, ...]
    weights: pd.DataFrame  # genes x blocks

    def combine(self, stage_predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        out = None
        for block, frame in stage_predictions.items():
            scaled = frame
            if block in self.blocks:
                scaled = frame * self.weights.loc[frame.columns, block].to_numpy()[None, :]
            out = scaled if out is None else out + scaled
        return out


def fit_view_weights(
    stage_predictions: Mapping[str, pd.DataFrame],
    residual: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    blocks: Sequence[str],
    *,
    steps: int = 300,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> ViewWeights:
    torch.manual_seed(seed)
    genes = list(residual.columns)
    lines = list(residual.index)
    views = torch.tensor(
        np.stack([stage_predictions[b].loc[lines, genes].to_numpy() for b in blocks]),
        dtype=torch.float32,
    )  # views x lines x genes
    target = residual.to_numpy(dtype=np.float64)
    observed = torch.tensor(np.isfinite(target))
    target = torch.tensor(np.where(np.isfinite(target), target, 0.0), dtype=torch.float32)
    width = len(next(iter(embeddings.values())))
    has = torch.tensor([g in embeddings for g in genes])
    embedding = torch.tensor(
        np.stack([embeddings.get(g, np.zeros(width)) for g in genes]), dtype=torch.float32
    )
    mixer = torch.nn.Linear(width, len(blocks))
    torch.nn.init.zeros_(mixer.weight)
    torch.nn.init.zeros_(mixer.bias)
    optimizer = torch.optim.Adam(mixer.parameters(), lr=learning_rate)

    def weights() -> torch.Tensor:
        learned = len(blocks) * torch.softmax(mixer(embedding), dim=1)
        return torch.where(has[:, None], learned, torch.ones_like(learned))

    for _ in range(steps):
        optimizer.zero_grad()
        prediction = torch.einsum("vng,gv->ng", views, weights())
        loss = ((prediction - target) ** 2)[observed].mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        values = weights().numpy()
    return ViewWeights(tuple(blocks), pd.DataFrame(values, index=genes, columns=list(blocks)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_context_prior_view_weights.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/context_prior/view_weights.py tests/test_context_prior_view_weights.py
git add src/context_prior/view_weights.py tests/test_context_prior_view_weights.py
git commit -m "feat(context-prior): gene-conditioned view weights candidate" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Pseudo-bulk in preparation

**Files:**
- Create: `src/data/pseudobulk.py`
- Modify: `src/experiments/prepare.py` (add `prepare_pseudobulk` and `_pseudobulk_line` after `prepare_inputs`)
- Test: `tests/test_pseudobulk.py`

**Interfaces:**
- Consumes: `align_columns`, `require_raw_counts`, `symbol_column` (`src/data/basal.py`); `read_registry_source` (`src/data/tx1_cache.py`); `load_source_registry` (`src/data/geneeffect.py`); `_in_processes`, `LINE_PROCESSES` (prepare.py).
- Produces: `PSEUDOBULK_DIR = "pseudobulk"`; `pseudobulk(matrix, symbols, genes) -> np.ndarray` (float32, NaN where a gene is absent); `read_pseudobulk(prepared_root: Path) -> pd.DataFrame`; `prepare_pseudobulk(config: Mapping, genes: Sequence[str]) -> Path` (the manifest path).

- [ ] **Step 1: Write the failing tests**

```python
"""Pseudo-bulk: summed raw UMI, whole-library CPM, log1p; round trip."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.data.pseudobulk import PSEUDOBULK_DIR, pseudobulk, read_pseudobulk


def test_pseudobulk_sums_cells_and_duplicate_symbols():
    counts = sparse.csr_matrix(np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]]))
    values = pseudobulk(counts, ["A", "B", "A"], ["A", "B", "C"])
    assert np.isclose(values[0], np.log1p(5 * 1e6 / 7))
    assert np.isclose(values[1], np.log1p(2 * 1e6 / 7))
    assert np.isnan(values[2])


def test_normalised_input_is_refused():
    with pytest.raises(ValueError, match="raw integer"):
        pseudobulk(sparse.csr_matrix(np.array([[0.5]])), ["A"], ["A"])


def test_read_pseudobulk_round_trip(tmp_path):
    root = tmp_path / PSEUDOBULK_DIR
    root.mkdir()
    frame = pd.DataFrame({"A": [1.0]}, index=pd.Index(["L1"], name="model_id"))
    frame.to_parquet(root / "pseudobulk.parquet")
    (root / "manifest.json").write_text(json.dumps({"genes": ["A"], "lines": ["L1"]}))
    assert read_pseudobulk(tmp_path).equals(frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_pseudobulk.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `src/data/pseudobulk.py`

```python
"""Per-line pseudo-bulk: raw UMI summed over every basal cell, CPM over the whole
library, log1p. The single-cell side of the context prior's shared space."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.basal import align_columns, require_raw_counts

PSEUDOBULK_DIR = "pseudobulk"
TRANSFORM = "sum_umi_cpm_log1p"


def pseudobulk(matrix, symbols: Sequence[str], genes: Sequence[str]) -> np.ndarray:
    """``log1p(1e6 * gene total / library total)`` per gene of ``genes``; NaN where
    the source lacks the gene. Columns sharing a symbol are summed."""
    require_raw_counts(matrix, "pseudo-bulk source")
    library = float(matrix.sum())
    if not library > 0:
        raise ValueError("pseudo-bulk source has no counts")
    counts, available = align_columns(matrix, symbols, genes)
    totals = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    values = np.log1p(totals * 1e6 / library)
    values[~available] = np.nan
    return values.astype(np.float32)


def read_pseudobulk(prepared_root: Path) -> pd.DataFrame:
    root = Path(prepared_root) / PSEUDOBULK_DIR
    manifest = json.loads((root / "manifest.json").read_text())
    frame = pd.read_parquet(root / "pseudobulk.parquet")
    if list(frame.columns) != manifest["genes"] or list(frame.index) != manifest["lines"]:
        raise ValueError(f"{root}: pseudo-bulk does not match its manifest")
    return frame
```

Append to `src/experiments/prepare.py` (before `def main`):

```python
def _pseudobulk_line(
    model_id: str,
    source_path: Path,
    *,
    var_ensembl_col: str,
    hvg_gene_symbol_col: str,
    genes: tuple[str, ...],
):
    from src.data.basal import symbol_column
    from src.data.pseudobulk import pseudobulk
    from src.data.tx1_cache import read_registry_source

    source = read_registry_source(
        source_path, model_id=model_id, var_ensembl_col=var_ensembl_col
    )
    column = symbol_column(source.var, hvg_gene_symbol_col)
    return pseudobulk(source.X, source.var[column].astype(str).tolist(), genes)


def prepare_pseudobulk(config: Mapping[str, Any], genes: Sequence[str]) -> Path:
    """Write ``<prepared_root>/pseudobulk/`` for every registered line, once.

    Reads each line's raw source (preparation stays the only reader of raw data);
    skipped when the manifest exists for the same genes.
    """
    import numpy as np
    import pandas as pd

    from src.data.geneeffect import load_source_registry
    from src.data.pseudobulk import PSEUDOBULK_DIR, TRANSFORM
    from src.data.splits import load_geneeffect_226_split

    root = Path(config["prepared_root"]) / PSEUDOBULK_DIR
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text())["genes"] != list(genes):
            raise ValueError(f"{root} holds pseudo-bulk for other genes")
        return manifest_path
    paths, settings = config["paths"], config["preparation"]
    split = load_geneeffect_226_split(Path(paths["split"]))
    registry = load_source_registry(Path(paths["source_registry"]), split)
    calls = [
        partial(
            _pseudobulk_line,
            str(model_id),
            Path(row["source_path"]),
            var_ensembl_col=settings["var_ensembl_col"],
            hvg_gene_symbol_col=settings["hvg_gene_symbol_col"],
            genes=tuple(genes),
        )
        for model_id, row in registry.iterrows()
    ]
    frame = pd.DataFrame(
        np.stack(list(_in_processes(calls, LINE_PROCESSES))),
        index=pd.Index([str(m) for m in registry.index], name="model_id"),
        columns=list(genes),
    )
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "pseudobulk.parquet")
    manifest = {"transform": TRANSFORM, "genes": list(genes), "lines": list(frame.index)}
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    return manifest_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_pseudobulk.py tests/test_prepare.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff format src/data/pseudobulk.py tests/test_pseudobulk.py
git add src/data/pseudobulk.py src/experiments/prepare.py tests/test_pseudobulk.py
git commit -m "feat(prepare): per-line pseudo-bulk in the prepared root" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Prior config, run data, learning curve and the extra-lines decision

**Files:**
- Modify: `src/experiments/config.py` (add the prior schema at the end)
- Create: `src/experiments/context_prior.py`, `configs/context_prior/prior.yaml`, `configs/context_prior/prior_no_haematopoietic.yaml`
- Test: `tests/test_context_prior_run.py`

**Interfaces:**
- Consumes: everything above; `load_config` (joint), `read_manifest`, `paired_line_bootstrap`, `macro_gene_spearman`.
- Produces: `validate_prior_config(config) -> dict`, `load_prior_config(path) -> dict`; in the runner: `RunData`, `load_run_data(config, *, oracle_only) -> RunData`, `selective_score(prediction, truth, selective) -> float`, `long_frame(prediction, truth, definitions) -> pd.DataFrame`, `curve_points(data) -> list[tuple[str, tuple[str, ...]]]`, `run_curve(data) -> tuple[list[dict], dict]`, `decide(data, curve_predictions) -> dict`.

- [ ] **Step 1: Write the config files**

`configs/context_prior/prior.yaml`:

```yaml
# Linear context prior: learning curve, extra-lines decision, block selection,
# cross-fitting and one test score. Definitions, split, labels and the prepared
# root come from joint_config.
seed: 0
joint_config: configs/revision/frozen_huber.yaml
output_root: outputs/context_prior
paths:
  extra_lines: configs/benchmarks/extra_bulk_lines_26Q1.json
  reference: configs/context_prior/reference
  model: data/sl_dependency_v0/raw/depmap/Model.csv
  bulk_expression: data/sl_dependency_v0/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
training_side:
  exclude_lineages: []
curve:
  sizes: [100, 400, 700]
  subsets: 3
  selected: 50
selection:
  penalties: [0.01, 0.1, 1.0, 10.0, 100.0]
  shrinkages: [0.001, 0.1, 1.0, .inf]
  selected: [10, 50, 200]
  rank: 64
  view_weights: true
prior:
  components: 128
  folds: 5
  bootstrap_repeats: 1000
```

`configs/context_prior/prior_no_haematopoietic.yaml`: the same file with the comment `# The haematopoietic extra lines dropped.` and `exclude_lineages: [Lymphoid, Myeloid]`.

- [ ] **Step 2: Write the failing tests**

```python
"""Prior config schema, block selection, long frames and curve points."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.context_prior.targets import Definitions
from src.experiments import context_prior as run
from src.experiments.config import validate_prior_config

CONFIG = yaml.safe_load(Path("configs/context_prior/prior.yaml").read_text())


def test_prior_config_is_strict():
    validate_prior_config(CONFIG)
    missing = copy.deepcopy(CONFIG)
    del missing["curve"]["subsets"]
    with pytest.raises(ValueError, match="missing"):
        validate_prior_config(missing)
    unknown = copy.deepcopy(CONFIG)
    unknown["prior"]["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_prior_config(unknown)
    assert math.isinf(CONFIG["selection"]["shrinkages"][-1])


def test_long_frame_keeps_finite_labels_in_geneeffect_units():
    definitions = Definitions(
        genes=("A", "B"),
        gene_means=pd.Series({"A": -1.0, "B": 0.0}),
        selective=("A",),
        variable=("A", "B"),
        residual_scale=pd.Series({"A": 2.0, "B": 1.0}),
    )
    truth = pd.DataFrame({"A": [1.0, np.nan], "B": [0.5, 0.0]}, index=["L1", "L2"])
    prediction = pd.DataFrame({"A": [0.5, 0.1], "B": [0.0, 0.2]}, index=["L1", "L2"])
    frame = run.long_frame(prediction, truth, definitions)
    assert len(frame) == 3
    row = frame.set_index(["model_id", "gene_symbol"]).loc[("L1", "A")]
    assert row["residual"] == 2.0 and row["residual_prediction"] == 1.0
    assert row["gene_effect"] == 1.0 and row["geneeffect_prediction"] == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_run.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `ImportError`

- [ ] **Step 4: Add the prior schema** (append to `src/experiments/config.py`)

```python
_PRIOR_GROUPS = {
    "paths": "extra_lines reference model bulk_expression",
    "training_side": "exclude_lineages",
    "curve": "sizes subsets selected",
    "selection": "penalties shrinkages selected rank view_weights",
    "prior": "components folds bootstrap_repeats",
}
_PRIOR_TOP_LEVEL = "seed joint_config output_root"


def validate_prior_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject missing and unknown keys of a linear-context-prior config."""
    _require_keys(config, {*_PRIOR_GROUPS, *_PRIOR_TOP_LEVEL.split()}, "config")
    for name, keys in _PRIOR_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
    return dict(config)


def load_prior_config(path: Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return validate_prior_config(yaml.safe_load(handle))
```

- [ ] **Step 5: Write the runner's data, scoring, curve and decision** (`src/experiments/context_prior.py`)

```python
"""The linear context prior run: learning curve, extra-lines decision, block
selection, cross-fitting and one test score.

``python -m src.experiments.context_prior CONFIG --run-id ID [--oracle-only]`` writes
``<output_root>/<run id>/``. With ``--oracle-only`` (the Mac) only the learning curve
runs, scored from validation lines' bulk RNA: an off-contract upper bound. The full
run (the H20 host) prepares pseudo-bulk, fits the bridge, scores the curve from
bridged pseudo-bulk as well, decides on the extra lines, selects blocks on
validation, cross-fits the chosen prior and scores it once on test against the
existing controls. A step whose output exists is skipped, so rerunning resumes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.bridge import Bridge, bridge_quality, fit_bridge
from src.context_prior.folds import patient_folds, patient_subset
from src.context_prior.prior import (
    BLOCKS,
    CONTEXT_BLOCKS,
    PriorInputs,
    PriorSpec,
    Stage,
    crossfit,
    fit_prior,
    total,
)
from src.context_prior.reference import load_reference
from src.context_prior.space import quantile_normalize, quantile_reference
from src.context_prior.targets import Definitions, fit_definitions, residual_frame
from src.context_prior.views import fit_expression_components
from src.data.depmap import read_depmap_matrix, read_models
from src.data.extra_lines import load_extra_lines
from src.data.splits import FixedSplit, load_geneeffect_226_split
from src.eval.metrics import macro_gene_spearman, paired_line_bootstrap
from src.experiments.config import load_config, load_prior_config

BOOTSTRAP_SEED = 0
CURVE_CONFIGS = ("components", "components_and_selected")


@dataclass
class RunData:
    config: Mapping[str, Any]
    joint: Mapping[str, Any]
    split: FixedSplit
    gene_effect: pd.DataFrame
    definitions: Definitions
    inputs: PriorInputs
    labelled: tuple[str, ...]  # labelled training-side lines with bulk RNA
    single_cell_train: tuple[str, ...]  # labelled single-cell training lines with bulk
    lineage: pd.Series  # every line's lineage, for descriptive tables
    pseudobulk: pd.DataFrame | None  # quantile-normalised, the 226 lines
    bridge: Bridge | None
    queries: dict[str, pd.DataFrame]  # "oracle" (validation bulk), "val" (bridged)

    def truth(self, lines: Sequence[str]) -> pd.DataFrame:
        """Residual in residual-SD units; reads the labels of ``lines``."""
        frame = residual_frame(self.gene_effect, lines, self.definitions)
        return frame / self.definitions.residual_scale


def load_run_data(config: Mapping[str, Any], *, oracle_only: bool) -> RunData:
    from src.data.prepared import read_manifest
    from src.data.pseudobulk import read_pseudobulk
    from src.experiments.prepare import prepare_pseudobulk

    joint = load_config(Path(config["joint_config"]))
    paths = config["paths"]
    split = load_geneeffect_226_split(Path(joint["paths"]["split"]))
    extra = load_extra_lines(Path(paths["extra_lines"]), split)
    models = read_models(Path(paths["model"]))
    blocked = {*split.val, *split.test, *extra.excluded}
    dropped = set(config["training_side"]["exclude_lineages"])

    def kept(ids: Sequence[str]) -> list[str]:
        return [m for m in ids if models.at[m, "lineage"] not in dropped]

    bulk = read_depmap_matrix(Path(paths["bulk_expression"]))
    bulk = bulk.drop(index=[m for m in bulk.index if m in {*split.test, *extra.excluded}])
    gene_effect = read_depmap_matrix(Path(joint["paths"]["gene_effect"]))
    side = [
        m
        for m in (*split.train, *kept(extra.labelled), *kept(extra.unlabelled))
        if m in bulk.index
    ]
    labelled = tuple(
        m for m in (*split.supervised_train, *kept(extra.labelled)) if m in bulk.index
    )
    single_cell_train = tuple(m for m in split.supervised_train if m in bulk.index)
    pseudo = None
    if oracle_only:
        panel = [g for g in gene_effect.columns if g in bulk.columns]
        space = list(bulk.columns)
    else:
        prepared = Path(joint["prepared_root"])
        panel = list(read_manifest(prepared)["common_gene_panel"])
        prepare_pseudobulk(joint, list(bulk.columns))
        pseudo = read_pseudobulk(prepared)
        space = [g for g in bulk.columns if np.isfinite(pseudo[g]).all()]
    definitions = fit_definitions(gene_effect, split, panel, joint["features"])
    reference_profile = quantile_reference(bulk.loc[side, space])
    oracle_lines = [m for m in split.val if m in bulk.index]
    normalized = quantile_normalize(bulk.loc[[*side, *oracle_lines], space], reference_profile)
    data = RunData(
        config=config,
        joint=joint,
        split=split,
        gene_effect=gene_effect,
        definitions=definitions,
        inputs=None,  # set below
        labelled=labelled,
        single_cell_train=single_cell_train,
        lineage=models["lineage"],
        pseudobulk=None,
        bridge=None,
        queries={"oracle": normalized.loc[oracle_lines]},
    )
    data.inputs = PriorInputs(
        expression=normalized.loc[side],
        residual=data.truth(labelled),
        components=fit_expression_components(
            normalized.loc[side], int(config["prior"]["components"])
        ),
        reference=load_reference(Path(paths["reference"]), blocked=blocked),
        lineage=models.loc[side, "lineage"],
        patients=models["patient_id"].to_dict(),
    )
    if pseudo is not None:
        data.pseudobulk = quantile_normalize(pseudo.loc[:, space], reference_profile)
        data.bridge = fit_bridge(
            data.pseudobulk.loc[list(single_cell_train)],
            normalized.loc[list(single_cell_train)],
        )
        data.queries["val"] = data.bridge.apply(data.pseudobulk.loc[list(split.val)])
    return data


def selective_score(
    prediction: pd.DataFrame, truth: pd.DataFrame, selective: Sequence[str]
) -> float:
    genes = list(selective)
    lines = list(prediction.index)
    return macro_gene_spearman(
        truth.loc[lines, genes].to_numpy().T, prediction.loc[:, genes].to_numpy().T
    )


def long_frame(
    prediction: pd.DataFrame, truth: pd.DataFrame, definitions: Definitions
) -> pd.DataFrame:
    """Finite-label rows in GeneEffect units, as ``aggregate_geneeffect`` reads them."""
    lines, genes = list(prediction.index), list(prediction.columns)
    scale = definitions.residual_scale.loc[genes].to_numpy()
    residual = truth.loc[lines, genes].to_numpy() * scale
    predicted = prediction.to_numpy() * scale
    keep = np.isfinite(residual)
    line_index, gene_index = np.nonzero(keep)
    means = definitions.gene_means.loc[genes].to_numpy()[gene_index]
    return pd.DataFrame(
        {
            "model_id": np.asarray(lines)[line_index],
            "gene_symbol": np.asarray(genes)[gene_index],
            "residual": residual[keep],
            "residual_prediction": predicted[keep],
            "gene_effect": residual[keep] + means,
            "geneeffect_prediction": predicted[keep] + means,
        }
    )


def gain_interval(
    data: RunData, better: pd.DataFrame, simpler: pd.DataFrame, truth: pd.DataFrame
) -> list[float]:
    result = paired_line_bootstrap(
        long_frame(better, truth, data.definitions),
        long_frame(simpler, truth, data.definitions),
        data.definitions.selective,
        repeats=int(data.config["prior"]["bootstrap_repeats"]),
        seed=BOOTSTRAP_SEED,
    )
    return [float(value) for value in result["interval"]]


def curve_points(data: RunData) -> list[tuple[str, tuple[str, ...]]]:
    """The single-cell training lines, seeded patient subsets, then every line."""
    curve = data.config["curve"]
    points = [("single_cell_train", data.single_cell_train)]
    for size in curve["sizes"]:
        for subset in range(int(curve["subsets"])):
            lines = patient_subset(
                data.labelled, data.inputs.patients, size=int(size), seed=subset
            )
            points.append((f"random_{size}_{subset}", lines))
    points.append(("all", data.labelled))
    return points


def _best(fits: list[tuple[float, Any, pd.DataFrame]]):
    return max(fits, key=lambda item: item[0] if math.isfinite(item[0]) else -math.inf)


def run_curve(data: RunData) -> tuple[list[dict], dict]:
    """Rows of the learning curve and the chosen predictions of each point, config
    and input; ridge strengths are chosen on validation per input."""
    penalties = data.config["selection"]["penalties"]
    count = int(data.config["curve"]["selected"])
    truth = data.truth(data.split.val)
    encoder = list(data.inputs.expression.index)
    rows, predictions = [], {}
    for point, lines in curve_points(data):
        component_fits = {}
        for penalty in penalties:
            spec = PriorSpec((Stage("expression_components", penalty),))
            fitted = fit_prior(spec, data.inputs, fit_lines=lines, encoder_lines=encoder)
            component_fits[penalty] = {
                name: total(fitted.predict(query)) for name, query in data.queries.items()
            }
        for name in data.queries:
            score, penalty, frame = _best(
                [
                    (selective_score(fits[name], truth, data.definitions.selective), p, fits[name])
                    for p, fits in component_fits.items()
                ]
            )
            rows.append({"point": point, "lines": len(lines), "config": "components",
                         "input": name, "penalty": penalty, "score": score})
            predictions[(point, "components", name)] = frame
            selected_fits = []
            for second in penalties:
                spec = PriorSpec(
                    (
                        Stage("expression_components", penalty),
                        Stage("data_selected", second, selected=count),
                    )
                )
                fitted = fit_prior(spec, data.inputs, fit_lines=lines, encoder_lines=encoder)
                frame = total(fitted.predict(data.queries[name]))
                selected_fits.append(
                    (selective_score(frame, truth, data.definitions.selective), second, frame)
                )
            score, second, frame = _best(selected_fits)
            rows.append({"point": point, "lines": len(lines),
                         "config": "components_and_selected", "input": name,
                         "penalty": [penalty, second], "score": score})
            predictions[(point, "components_and_selected", name)] = frame
    return rows, predictions


def decide(data: RunData, predictions: Mapping[tuple, pd.DataFrame]) -> dict:
    """Extra lines pass when the all-lines prior beats the single-cell-line prior
    (components plus data-selected genes) with an interval excluding zero; binding
    only on bridged input."""
    name = "val" if "val" in data.queries else "oracle"
    truth = data.truth(data.split.val)
    better = predictions[("all", "components_and_selected", name)]
    simpler = predictions[("single_cell_train", "components_and_selected", name)]
    interval = gain_interval(data, better, simpler, truth)
    decision = {
        "input": name,
        "binding": name == "val",
        "difference": selective_score(better, truth, data.definitions.selective)
        - selective_score(simpler, truth, data.definitions.selective),
        "interval": interval,
        "passes": interval[0] > 0,
    }
    if name == "val":
        oracle = gain_interval(
            data,
            predictions[("all", "components_and_selected", "oracle")],
            predictions[("single_cell_train", "components_and_selected", "oracle")],
            truth,
        )
        decision["oracle_interval"] = oracle
        decision["bridge_failing"] = oracle[0] > 0 and interval[0] <= 0
    return decision
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_context_prior_run.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
.venv/bin/ruff format src/experiments/context_prior.py tests/test_context_prior_run.py
git add src/experiments/config.py src/experiments/context_prior.py configs/context_prior/prior.yaml configs/context_prior/prior_no_haematopoietic.yaml tests/test_context_prior_run.py
git commit -m "feat(context-prior): run data, learning curve and extra-lines decision" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Selection, cross-fit, test, summary and the `prior` route

**Files:**
- Modify: `src/experiments/context_prior.py` (append), `hpc/run.sh`
- Test: `tests/test_context_prior_run.py` (append)

**Interfaces:**
- Consumes: Task 13's runner functions; `fit_view_weights`; `run_baselines` (`src/experiments/baselines.py`); `aggregate_geneeffect`; `load_esm2_embeddings` (`src/data/embeddings.py`).
- Produces: `select_blocks(evaluate, interval, selection) -> tuple[PriorSpec, list[dict]]`; `main(argv) -> int`; route `hpc/run.sh prior CONFIG [--run-id ID]`; outputs `curve.json`, `decision.json`, `selection.json`, `predictions.parquet`, `metrics.json`, `summary.md` under `outputs/context_prior/<run id>/`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_context_prior_run.py`)

```python
def test_select_blocks_keeps_only_blocks_whose_gain_interval_excludes_zero():
    from src.context_prior.prior import PriorSpec

    gains = {"pathway_scores": [-0.01, 0.02], "predicted_genotype": [0.01, 0.03],
             "own_expression": [0.001, 0.01], "partners": [-0.02, 0.0],
             "data_selected": [0.02, 0.04], "rank": [-0.01, 0.01]}

    def evaluate(spec: PriorSpec):
        last = spec.stages[-1]
        score = last.penalty if last.block != "data_selected" else last.selected
        return float(score), pd.DataFrame({"A": [float(len(spec.stages))]})

    calls = []

    def interval(prediction, current, block):
        calls.append(block)
        return gains[block]

    selection = {"penalties": [0.1, 1.0], "shrinkages": [0.5, 2.0],
                 "selected": [10, 50], "rank": 64}
    spec, log = run.select_blocks(evaluate, interval, selection)
    assert [s.block for s in spec.stages] == [
        "expression_components", "predicted_genotype", "own_expression", "data_selected",
    ]
    assert spec.stages[2].penalty == 2.0  # best shrinkage by point estimate
    assert [s for s in log if s["block"] == "partners"][0]["penalty"] == 2.0
    assert spec.stages[-1].selected == 50 and spec.rank is None
    assert calls[0] == "pathway_scores"  # the first block is never tested
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_context_prior_run.py -q > pytest.txt 2>&1; tail -5 pytest.txt`
Expected: FAIL with `AttributeError: ... has no attribute 'select_blocks'`

- [ ] **Step 3: Append selection, cross-fit, test, summary and main to the runner**

```python
def select_blocks(
    evaluate: Callable[[PriorSpec], tuple[float, pd.DataFrame]],
    interval: Callable[[pd.DataFrame, pd.DataFrame, str], list[float]],
    selection: Mapping[str, Any],
) -> tuple[PriorSpec, list[dict]]:
    """Add blocks in the fixed order; keep one only if its gain interval over the
    current prior excludes zero. Penalties (and N) are chosen by point estimate;
    the partner group reuses the own-expression shrinkage. The first block is the
    base and is always kept; the reduced rank is tried last."""
    spec, current, log, shrinkage = PriorSpec(()), None, [], None
    for block in BLOCKS:
        if block in CONTEXT_BLOCKS:
            grid = [Stage(block, p) for p in selection["penalties"]]
        elif block == "own_expression":
            grid = [Stage(block, s) for s in selection["shrinkages"]]
        elif block == "partners":
            grid = [Stage(block, shrinkage)]
        else:
            grid = [
                Stage(block, p, n)
                for n in selection["selected"]
                for p in selection["penalties"]
            ]
        results = [(evaluate(PriorSpec((*spec.stages, s), spec.rank)), s) for s in grid]
        (score, prediction), stage = max(
            results,
            key=lambda item: item[0][0] if math.isfinite(item[0][0]) else -math.inf,
        )
        if block == "own_expression":
            shrinkage = stage.penalty
        gain = None if current is None else interval(prediction, current, block)
        kept = gain is None or gain[0] > 0
        log.append({"block": block, "penalty": stage.penalty, "selected": stage.selected,
                    "score": score, "interval": gain, "kept": kept})
        if kept:
            spec, current = PriorSpec((*spec.stages, stage), spec.rank), prediction
    if selection["rank"]:
        candidate = PriorSpec(spec.stages, int(selection["rank"]))
        score, prediction = evaluate(candidate)
        gain = interval(prediction, current, "rank")
        kept = gain[0] > 0
        log.append({"block": "reduced_rank", "rank": candidate.rank, "score": score,
                    "interval": gain, "kept": kept})
        if kept:
            spec, current = candidate, prediction
    return spec, log


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _spec_json(spec: PriorSpec) -> dict:
    return {"stages": [vars(stage) for stage in spec.stages], "rank": spec.rank}


def _spec_from_json(payload: Mapping) -> PriorSpec:
    return PriorSpec(tuple(Stage(**stage) for stage in payload["stages"]), payload["rank"])


def _fit_all(data: RunData, spec: PriorSpec):
    return fit_prior(
        spec,
        data.inputs,
        fit_lines=list(data.labelled),
        encoder_lines=list(data.inputs.expression.index),
    )


def run_selection(data: RunData) -> tuple[PriorSpec, list[dict]]:
    truth = data.truth(data.split.val)
    query = data.queries["val"]

    def evaluate(spec: PriorSpec) -> tuple[float, pd.DataFrame]:
        prediction = total(_fit_all(data, spec).predict(query))
        return selective_score(prediction, truth, data.definitions.selective), prediction

    def interval(prediction, current, block):
        return gain_interval(data, prediction, current, truth)

    return select_blocks(evaluate, interval, data.config["selection"])


def run_crossfit(data: RunData, spec: PriorSpec) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Out-of-fold stage predictions of the 170 single-cell training lines (from
    bridged pseudo-bulk, bridge refitted without their fold) and of the labelled
    extra lines (from bulk), and bridge quality over the out-of-fold lines."""
    lines = sorted({*data.inputs.expression.index, *data.split.supervised_train})
    folds = patient_folds(
        lines, data.inputs.patients, n_folds=int(data.config["prior"]["folds"]),
        seed=int(data.config["seed"]),
    )
    single_cell = list(data.split.supervised_train)
    extras = [m for m in data.labelled if m not in set(single_cell)]
    queries, bridged_parts = {}, []
    for fold in sorted(set(folds.values())):
        bridge_lines = [m for m in data.single_cell_train if folds[m] != fold]
        bridge = fit_bridge(
            data.pseudobulk.loc[bridge_lines], data.inputs.expression.loc[bridge_lines]
        )
        bridged = bridge.apply(data.pseudobulk.loc[[m for m in single_cell if folds[m] == fold]])
        bridged_parts.append(bridged)
        queries[fold] = pd.concat(
            [bridged, data.inputs.expression.loc[[m for m in extras if folds[m] == fold]]]
        )
    stages = crossfit(
        spec, data.inputs, folds=folds, queries=queries,
        labelled=list(data.labelled), encoder_lines=list(data.inputs.expression.index),
    )
    bridged = pd.concat(bridged_parts)
    with_bulk = [m for m in bridged.index if m in data.inputs.expression.index]
    quality = bridge_quality(bridged.loc[with_bulk], data.inputs.expression.loc[with_bulk])
    return stages, quality


def _lineage_rows(data, prediction, truth, split_name) -> list[dict]:
    genes = list(data.definitions.selective)
    rows = []
    for line in prediction.index:
        x, y = prediction.loc[line, genes].to_numpy(), truth.loc[line, genes].to_numpy()
        keep = np.isfinite(y)
        rho = np.corrcoef(x[keep], y[keep])[0, 1] if keep.sum() > 2 else math.nan
        rows.append({"split": split_name, "lineage": data.lineage.get(line), "pearson": rho})
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["split", "lineage"])["pearson"].agg(["count", "mean"])
    return grouped.reset_index().to_dict("records")


def run_test(data: RunData, spec: PriorSpec, weights, run_dir: Path) -> dict:
    """Score the chosen prior on validation and, once, on test, against the controls."""
    from src.eval.geneeffect import aggregate_geneeffect
    from src.experiments.baselines import run_baselines

    fitted = _fit_all(data, spec)
    test_query = data.bridge.apply(data.pseudobulk.loc[list(data.split.test)])
    record: dict[str, Any] = {"splits": {}, "lineages": []}
    frames = []
    for split_name, query in (("val", data.queries["val"]), ("test", test_query),
                              ("oracle_val", data.queries["oracle"])):
        stages = fitted.predict(query)
        prediction = weights.combine(stages) if weights is not None else total(stages)
        truth = data.truth(list(query.index))
        frame = long_frame(prediction, truth, data.definitions)
        metrics, _, _ = aggregate_geneeffect(
            frame, model_ids=list(query.index), genes=list(data.definitions.genes),
            variable_genes=list(data.definitions.variable),
            selective_genes=list(data.definitions.selective),
        )
        entry = {"prior": metrics}
        if split_name != "oracle_val":
            baseline_dir = run_dir / "baselines" / split_name
            if not (baseline_dir / "metrics.json").is_file():
                run_baselines(dict(data.joint), split=split_name, out_dir=baseline_dir)
            entry["controls"] = json.loads((baseline_dir / "metrics.json").read_text())
            tx1 = pd.read_parquet(baseline_dir / "predictions.parquet")
            tx1 = tx1.loc[tx1["method"] == "context_pca_ridge[tx1]"]
            keys = frame.merge(tx1[["model_id", "gene_symbol"]], on=["model_id", "gene_symbol"])
            common = tx1.merge(keys[["model_id", "gene_symbol"]], on=["model_id", "gene_symbol"])
            entry["prior_minus_tx1_ridge"] = paired_line_bootstrap(
                keys, common, data.definitions.selective,
                repeats=int(data.config["prior"]["bootstrap_repeats"]), seed=BOOTSTRAP_SEED,
            )
            record["lineages"] += _lineage_rows(data, prediction, truth, split_name)
        record["splits"][split_name] = entry
        frames.append(frame.assign(split=split_name))
    pd.concat(frames).to_parquet(run_dir / "predictions_eval.parquet")
    return record


def _table(entry: Mapping, split_name: str) -> list[str]:
    columns = (
        ("Selective Spearman", "selective_spearman"),
        ("Selective AUPR lift", "selective_aupr_lift"),
        ("Residual Pearson (per variable gene)", "residual_pearson_macro_per_gene"),
        ("Huber", "geneeffect_loss"),
        ("SD ratio (per gene)", "residual_sd_ratio_macro_per_gene"),
    )
    lines = ["| Method | " + " | ".join(c for c, _ in columns) + " |",
             "| --- |" + " --- |" * len(columns)]

    def cell(value):
        return "undefined" if value is None else f"{value:.4f}"

    name = "Linear context prior" + (" (bulk input, off-contract)" if split_name == "oracle_val" else "")
    lines.append(f"| {name} | " + " | ".join(cell(entry["prior"].get(k)) for _, k in columns) + " |")
    for method, metrics in sorted(entry.get("controls", {}).items()):
        lines.append(f"| {method} | " + " | ".join(
            cell(metrics.get(f"{split_name}_{k}")) for _, k in columns) + " |")
    return lines


def write_summary(run_dir: Path) -> None:
    out = ["# Linear context prior", ""]
    curve = json.loads((run_dir / "curve.json").read_text())
    out += ["## Learning curve (validation selective Spearman)", "",
            "| Point | Lines | Config | Input | Score |", "| --- | --- | --- | --- | --- |"]
    out += [f"| {r['point']} | {r['lines']} | {r['config']} | {r['input']} | {r['score']:.4f} |"
            for r in curve]
    decision = json.loads((run_dir / "decision.json").read_text())
    out += ["", "## Extra-lines decision", "", "```json", json.dumps(decision, indent=2), "```"]
    if (run_dir / "selection.json").is_file():
        selection = json.loads((run_dir / "selection.json").read_text())
        out += ["", "## Block selection", "", "| Block | Setting | Score | Interval | Kept |",
                "| --- | --- | --- | --- | --- |"]
        out += [f"| {r['block']} | {r.get('penalty', r.get('rank'))} {r.get('selected', '')} | "
                f"{r['score']:.4f} | {r['interval']} | {r['kept']} |" for r in selection["log"]]
        out += ["", f"Chosen prior: `{json.dumps(selection['spec'])}`",
                f"View weights kept: {selection.get('view_weights_kept')}",
                f"Bridge quality (median per-gene Pearson, out-of-fold training lines): "
                f"{selection.get('bridge_quality_median')}"]
    if (run_dir / "metrics.json").is_file():
        record = json.loads((run_dir / "metrics.json").read_text())
        for split_name, title in (("val", "Validation"), ("test", "Test"),
                                  ("oracle_val", "Validation, bulk input (off-contract upper bound)")):
            entry = record["splits"][split_name]
            out += ["", f"## {title}", ""] + _table(entry, split_name)
            if "prior_minus_tx1_ridge" in entry:
                out += ["", f"Prior minus Tx1 context-PCA ridge, selective Spearman: "
                        f"{entry['prior_minus_tx1_ridge']}"]
        out += ["", "## Per lineage (mean per-line residual Pearson over selective genes; descriptive)",
                "", "| Split | Lineage | Lines | Mean |", "| --- | --- | --- | --- |"]
        out += [f"| {r['split']} | {r['lineage']} | {r['count']} | {r['mean']:.3f} |"
                for r in record["lineages"]]
    (run_dir / "summary.md").write_text("\n".join(out) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_prior_config(args.config)
    run_dir = Path(config["output_root"]) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config)
    data = load_run_data(config, oracle_only=args.oracle_only)
    if not (run_dir / "decision.json").is_file():
        rows, predictions = run_curve(data)
        _write_json(run_dir / "curve.json", rows)
        _write_json(run_dir / "decision.json", decide(data, predictions))
    decision = json.loads((run_dir / "decision.json").read_text())
    if args.oracle_only:
        write_summary(run_dir)
        return 0
    if not decision["passes"]:
        data.labelled = data.single_cell_train
        data.inputs = PriorInputs(**{**vars(data.inputs),
                                     "residual": data.truth(data.single_cell_train)})
    if not (run_dir / "selection.json").is_file():
        spec, log = run_selection(data)
        stages, quality = run_crossfit(data, spec)
        oof = pd.concat(
            {
                b: f.rename_axis(index="model_id", columns="gene_symbol").stack()
                for b, f in stages.items()
            },
            names=["block"],
        )
        oof.rename("prediction_sigma").reset_index().to_parquet(run_dir / "oof.parquet")
        record = {"spec": _spec_json(spec), "log": log,
                  "bridge_quality_median": float(quality.median())}
        context = [s.block for s in spec.stages if s.block in CONTEXT_BLOCKS]
        if config["selection"]["view_weights"] and len(context) >= 2:
            from src.context_prior.view_weights import fit_view_weights
            from src.data.embeddings import load_esm2_embeddings

            table = load_esm2_embeddings(Path(data.joint["paths"]["esm2_embeddings"]))
            labelled = list(data.labelled)
            weights = fit_view_weights(
                {b: stages[b].loc[labelled] for b in stages},
                data.inputs.residual.loc[labelled], table.vectors_by_symbol, context,
            )
            truth = data.truth(data.split.val)
            val_stages = _fit_all(data, spec).predict(data.queries["val"])
            gain = gain_interval(data, weights.combine(val_stages), total(val_stages), truth)
            record["view_weights_interval"] = gain
            record["view_weights_kept"] = gain[0] > 0
            if record["view_weights_kept"]:
                weights.weights.to_parquet(run_dir / "view_weights.parquet")
        _write_json(run_dir / "selection.json", record)
    selection = json.loads((run_dir / "selection.json").read_text())
    spec = _spec_from_json(selection["spec"])
    weights = None
    if selection.get("view_weights_kept"):
        from src.context_prior.view_weights import ViewWeights

        table = pd.read_parquet(run_dir / "view_weights.parquet")
        weights = ViewWeights(tuple(table.columns), table)
    if not (run_dir / "metrics.json").is_file():
        _write_json(run_dir / "metrics.json", run_test(data, spec, weights, run_dir))
    write_summary(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the route** (`hpc/run.sh`): add the usage line `       hpc/run.sh prior CONFIG --run-id ID   (linear context prior, CPU)`, extend `case "$command" in all|revision|test|prior) ;;`, and add `  prior) exec "$python_bin" -m src.experiments.context_prior "$@" ;;` to the dispatch.

- [ ] **Step 5: Run the tests and the full suite**

Run: `uv run python -m pytest tests -q > pytest.txt 2>&1; tail -5 pytest.txt` and `.venv/bin/ruff check .`
Expected: everything passes that passed before this branch (baseline it with the same command before Task 1 if not already done), ruff `All checks passed!`

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff format src/experiments/context_prior.py tests/test_context_prior_run.py
git add src/experiments/context_prior.py hpc/run.sh tests/test_context_prior_run.py
git commit -m "feat(context-prior): block selection, cross-fit, test score, summary and the prior route" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Documents

**Files:**
- Modify: `docs/01-blueprint.md` (§2, §4), `docs/03-geneeffect-protocol.md` (new §10), `docs/specs/2026-10-04-context-generalization-design.md`, `CLAUDE.md`, `hpc/README.md`
- Create: `docs/data/extra-bulk-lines-26q1.md`

- [ ] **Step 1: Blueprint.** In §2 after the benchmark table add: "DepMap lines outside the 226 with 26Q1 bulk RNA may join the GeneEffect training side of the linear context prior ([membership](../configs/benchmarks/extra_bulk_lines_26Q1.json), [card](data/extra-bulk-lines-26q1.md)): their bulk RNA and other-omics labels fit context encoders, and their GeneEffect labels fit the prior once the learning-curve rule passes. Every line sharing a patient with a validation or test line is excluded from every fit." In §4 add two bullets: "Results that use extra lines are a training-data change, scored on the unchanged validation and test lines." and "Validation lines' bulk RNA is read only by the oracle diagnostic; test lines' bulk RNA is never read."

- [ ] **Step 2: Data card** `docs/data/extra-bulk-lines-26q1.md`: purpose, the builder command, counts (919 labelled, 576 unlabelled, 10 excluded), the exclusion rule, the haematopoietic-lineage note, and that the file is the membership authority for extra lines as the split is for the 226.

- [ ] **Step 3: Protocol §10 "Linear context prior"**: the route `hpc/run.sh prior CONFIG --run-id ID` (and `--oracle-only` locally), the steps (pseudo-bulk, bridge, curve, decision, selection, cross-fit, test), the outputs (`curve.json`, `decision.json`, `selection.json`, `oof.parquet`, `metrics.json`, `summary.md`), and that the prior is a control in its own summary.

- [ ] **Step 4: Spec amendments** (design doc, in place): genes join by symbol from DepMap's `SYMBOL (Entrez)` headers, not Entrez ID; paralogs are each gene's 10 closest at ≥ 20% identity (the lower of both directions); CORUM's current human release; the prior is stagewise (each block fitted to what earlier blocks leave); knowledge features are z-scored per gene with undefined values at the training mean and no mask bit (the per-gene intercept makes it redundant); a missing training label counts as zero residual in fitting; data-selected genes correlate with the residual left by earlier blocks; predicted-genotype encoders read the 128 expression components; the extra-lines decision uses the components-plus-50-data-selected-genes config; reference tables are committed under `configs/context_prior/reference/` because the H20 host has no internet.

- [ ] **Step 5: CLAUDE.md and hpc/README.md.** Add `hpc/run.sh prior CONFIG --run-id <id>` to the commands block; change "Only split, config and provenance files are tracked data" to "Only split, config, provenance files and the context prior's reference tables are tracked data"; add one sentence to "Current state" naming the linear context prior as the current work. In `hpc/README.md` add the route with its output directory.

- [ ] **Step 6: Commit**

```bash
git add docs/01-blueprint.md docs/03-geneeffect-protocol.md docs/specs/2026-10-04-context-generalization-design.md docs/data/extra-bulk-lines-26q1.md CLAUDE.md hpc/README.md
git commit -m "docs(context-prior): blueprint amendment, protocol section, data card and spec details" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Local oracle learning curve (Mac)

- [ ] **Step 1: Run**

```bash
uv run python -m src.experiments.context_prior configs/context_prior/prior.yaml --run-id oracle_20261004 --oracle-only > outputs/oracle_20261004.log 2>&1
```

Expected: `outputs/context_prior/oracle_20261004/{curve.json,decision.json,summary.md}`; about 30 minutes. Read `summary.md`.

- [ ] **Step 2: Report** the curve (score by point and config) and the non-binding oracle decision; the binding decision needs the H20 run.

---

### Task 17: H20 run

- [ ] **Step 1: Ask the user which container (port) to use**, then check it: `ssh -J richard@100.91.229.50 -p <port> root@10.15.171.204 'cd /2023533015/VCC_Project && git branch --show-current && git status --short | head && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader && ps aux | grep "[s]rc\." | head'`.

- [ ] **Step 2: Push the branch and make a worktree**

```bash
GIT_SSH_COMMAND="ssh -J richard@100.91.229.50" git push ssh://root@10.15.171.204:<port>/2023533015/VCC_Project feat/context-prior
ssh -J richard@100.91.229.50 -p <port> root@10.15.171.204 'cd /2023533015/VCC_Project && git worktree add /2023533015/VCC_Project_context_prior feat/context-prior && cd /2023533015/VCC_Project_context_prior && ln -s /2023533015/VCC_Project/data data && ln -s /2023533015/VCC_Project/model model && ln -s /2023533015/VCC_Project/.venv-tx1 .venv-tx1 && ls data/geneeffect_joint/v2/prepared_inputs.json'
```

- [ ] **Step 3: Launch (disconnect-safe)**

```bash
ssh -J richard@100.91.229.50 -p <port> root@10.15.171.204 'cd /2023533015/VCC_Project_context_prior && mkdir -p outputs/launches && OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 nohup hpc/run.sh prior configs/context_prior/prior.yaml --run-id prior_20261004 > outputs/launches/prior_20261004.log 2>&1 &'
```

- [ ] **Step 4: Confirm from a fresh session**: `data/geneeffect_joint/v2/pseudobulk/manifest.json`, then `outputs/context_prior/prior_20261004/decision.json`, `selection.json`, `metrics.json`, `summary.md`. A PID is launch evidence only. On failure read the log and rerun the same command (completed steps are skipped).

- [ ] **Step 5: Run the haematopoietic ablation** with `configs/context_prior/prior_no_haematopoietic.yaml` and run id `prior_no_haematopoietic_20261004` the same way, after the main run finishes.

- [ ] **Step 6: Record results** in `results/context_prior_seed0/README.md` (curve, decision, selection log, validation and test tables, bridge quality, per-lineage table, the oracle row marked off-contract), commit, and report. Merge into `main` only after the H20 run has completed and the user has seen the results.
