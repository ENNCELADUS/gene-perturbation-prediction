"""Single-cell identity map and membership: one source per line, admission, the
proposer's matches and flags."""

import json

import pandas as pd
import pytest

from src.data.extra_lines import load_extra_single_cell_lines
from src.data.prepare.build_extra_single_cell_lines import (
    MAP_COLUMNS,
    admit,
    build_membership,
    choose_sources,
    propose,
    read_map,
)
from src.data.splits import FixedSplit


def _rows(*rows: dict) -> pd.DataFrame:
    base = dict.fromkeys(MAP_COLUMNS, "")
    base.update(condition_class="basal_2d", raw_counts="true", wave="1", review="ok")
    return pd.DataFrame([{**base, **row} for row in rows], columns=list(MAP_COLUMNS))


def test_one_source_per_line_by_condition_then_counts_then_cells():
    candidates = _rows(
        {
            "source": "pilot",
            "model_id": "A",
            "cells": "5000",
            "condition_class": "cutting_control",
        },
        {"source": "geo", "model_id": "A", "cells": "300"},
        {"source": "mixseq", "model_id": "A", "cells": "20"},
        {"source": "cooked", "model_id": "B", "cells": "9000", "raw_counts": "false"},
        {"source": "small", "model_id": "B", "cells": "10"},
        {"source": "x", "model_id": "C", "cells": "50"},
        {"source": "y", "model_id": "C", "cells": "60", "wave": ""},
        {"source": "z", "model_id": "C", "cells": "70", "review": "dropped: swap"},
    )
    assert choose_sources(candidates) == {"A": "geo", "B": "small", "C": "x"}


def test_one_source_sums_a_lines_identifiers():
    candidates = _rows(
        {"source": "a", "model_id": "A", "identifier": "a1", "cells": "60"},
        {"source": "a", "model_id": "A", "identifier": "a2", "cells": "60"},
        {"source": "b", "model_id": "A", "identifier": "b1", "cells": "100"},
    )
    assert choose_sources(candidates) == {"A": "a"}


MODELS = pd.DataFrame(
    {"patient_id": ["P1", "P2", "P3", "P4", "P2", "P5"], "lineage": "Lung"},
    index=["T", "V", "A", "B", "D", "E"],
)
SPLIT = FixedSplit(train=("T",), val=("V",), test=())


def test_admission_skips_the_226_and_excludes_held_out_patients():
    candidates = _rows(
        *({"source": "s", "model_id": m, "identifier": m} for m in "TABDE")
    )
    admission = admit(
        candidates, MODELS, labelled_ids={"T", "A", "D", "E"}, split=SPLIT
    )
    assert admission.labelled == {"A": "s", "E": "s"}
    assert admission.unlabelled == ("B",)
    assert admission.excluded == {"D": "shares PatientID P2 with held-out line V"}


def test_admission_refuses_unknown_model_ids():
    with pytest.raises(ValueError, match="absent from Model.csv"):
        admit(
            _rows({"source": "s", "model_id": "NOPE"}),
            MODELS,
            labelled_ids=(),
            split=SPLIT,
        )


def test_membership_builds_lines_with_cells_and_leaves_the_rest_pending(tmp_path):
    candidates = _rows(
        {"source": "s", "model_id": "A", "identifier": "a", "accession": "GSE1"},
        {"source": "s", "model_id": "E", "identifier": "e", "accession": "GSE1"},
        {"source": "t", "model_id": "A", "identifier": "a", "cells": "1", "wave": ""},
        {"source": "u", "model_id": "B", "identifier": "b"},
        {"source": "v", "model_id": "D", "identifier": "d"},
    )
    qc = {"s": {"source": "s", "lines": {"A": {"cells": 12, "arms": {"dmso": 12}}}}}
    payload = build_membership(
        candidates, MODELS, labelled_ids={"A", "D", "E"}, split=SPLIT, qc=qc
    )
    assert payload["labelled"]["A"] == {
        "source": "s",
        "accession": "GSE1",
        "identifiers": ["a"],
        "condition": "",
        "condition_class": "basal_2d",
        "patient_id": "P3",
        "lineage": "Lung",
        "cells": 12,
        "arms": {"dmso": 12},
        "other_sources": ["t"],
    }
    assert payload["pending"] == {"E": "no cells after QC in s"}
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(payload))
    lines = load_extra_single_cell_lines(path, SPLIT)
    assert set(lines.labelled) == {"A"} and lines.unlabelled == ("B",)
    assert set(lines.excluded) == {"D"}

    payload["pending"]["T"] = "a 226 line"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pending and split overlap"):
        load_extra_single_cell_lines(path, SPLIT)


MODEL_TABLE = pd.DataFrame(
    {
        "ModelID": [
            "ACH-000001",
            "ACH-000002",
            "ACH-000003",
            "ACH-000004",
            "ACH-000005",
        ],
        "CellLineName": ["SK-OV-3", "HEL", "HEL 92.1.7", "MDA-MB-231", "Odd-1"],
        "StrippedCellLineName": ["SKOV3", "HEL", "HEL9217", "MDAMB231", "ODD1"],
        "RRID": ["CVCL_0532", "CVCL_0001", "CVCL_2481", "CVCL_0062", "CVCL_0005"],
    }
)
CELLOSAURUS = {
    "SK-OV-03": [
        {
            "accession": "CVCL_0532",
            "names": ["SK-OV-3", "SK-OV-03"],
            "derived_from": [],
            "problematic": "",
        }
    ],
    "MDA-MB-231-LM2": [
        {
            "accession": "CVCL_ZZ01",
            "names": ["MDA-MB-231-LM2"],
            "derived_from": ["CVCL_0062"],
            "problematic": "",
        }
    ],
    "Nothing": [
        {
            "accession": "CVCL_ZZ02",
            "names": ["Nothingness"],
            "derived_from": [],
            "problematic": "",
        }
    ],
}


def _entry(accession):
    flagged = "Misidentified." if accession == "CVCL_0005" else ""
    return {
        "accession": accession,
        "names": [],
        "derived_from": [],
        "problematic": flagged,
    }


@pytest.mark.parametrize(
    ("identifier", "model_id", "kind", "review"),
    [
        ("sk-ov-3", "ACH-000001", "name", "ok"),
        ("SK-OV-03", "ACH-000001", "synonym", "ok"),
        ("ACH-000004", "ACH-000004", "model_id", "ok"),
        ("CVCL_0062", "ACH-000004", "rrid", "ok"),
        ("MDA-MB-231-LM2", "", "", "derived"),
        ("Nothing", "", "", "no_match"),
        ("ODD 1", "ACH-000005", "name", "problematic"),
    ],
)
def test_proposer_matches_and_flags(identifier, model_id, kind, review):
    candidates = _rows({"source": "s", "identifier": identifier, "review": ""})
    proposed = propose(
        candidates,
        MODEL_TABLE,
        search=lambda name: CELLOSAURUS.get(name, []),
        entry=_entry,
    ).iloc[0]
    assert (proposed["model_id"], proposed["identifier_kind"]) == (model_id, kind)
    assert proposed["review"] == review


def test_proposer_flags_names_two_models_share_and_keeps_reviewed_rows():
    table = MODEL_TABLE.assign(
        StrippedCellLineName=["SKOV3", "HEL", "HEL", "MDAMB231", "ODD1"]
    )
    candidates = _rows(
        {"source": "s", "identifier": "HEL", "review": ""},
        {"source": "s", "identifier": "Ishikawa", "review": "dropped: other stock"},
    )
    proposed = propose(candidates, table, search=lambda name: [], entry=_entry)
    assert proposed.loc[0, "review"] == "ambiguous"
    assert proposed.loc[0, "note"] == "Model.csv: ACH-000002, ACH-000003"
    assert proposed.loc[1, "review"] == "dropped: other stock"


def test_map_refuses_unknown_reviews_and_accepted_rows_without_model(tmp_path):
    path = tmp_path / "map.csv"
    _rows({"source": "s", "identifier": "a", "review": "maybe"}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="unknown review"):
        read_map(path)
    _rows({"source": "s", "identifier": "a"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no model_id"):
        read_map(path)
