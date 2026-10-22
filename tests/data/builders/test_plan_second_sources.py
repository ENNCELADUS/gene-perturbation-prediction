"""One second deposit per labelled single-cell training line."""

from __future__ import annotations

import pandas as pd

from src.data.builders.build_extra_single_cell_lines import MAP_COLUMNS
from src.data.builders.plan_second_sources import plan, primary_sources


def _rows(*rows):
    base = dict.fromkeys(MAP_COLUMNS, "")
    base.update(condition_class="basal_2d", raw_counts="true", wave="1", review="ok")
    return pd.DataFrame([{**base, **row} for row in rows], columns=list(MAP_COLUMNS))


def test_the_second_deposit_is_another_buildable_accession_chosen_by_policy():
    candidates = _rows(
        # A: its own deposit, the same deposit re-counted, and two others.
        {"source": "own", "model_id": "A", "accession": "GSE1", "cells": "100"},
        {"source": "own_recount", "model_id": "A", "accession": "GSE1", "cells": "900"},
        {"source": "big", "model_id": "A", "accession": "GSE2", "cells": "500"},
        {
            "source": "guide",
            "model_id": "A",
            "accession": "GSE3",
            "cells": "5000",
            "condition_class": "control_guide",
        },
        # B: only an unbuildable or a cooked second source.
        {"source": "own", "model_id": "B", "accession": "GSE1", "cells": "10"},
        {"source": "elsewhere", "model_id": "B", "accession": "GSE4", "cells": "10"},
        {
            "source": "big",
            "model_id": "B",
            "accession": "GSE2",
            "cells": "10",
            "raw_counts": "false",
        },
        # C: a line outside the training lines.
        {"source": "big", "model_id": "C", "accession": "GSE2", "cells": "10"},
    )
    buildable = {"own": 1, "own_recount": 2, "big": 1, "guide": 2}
    planned = plan(candidates, {"A": "own", "B": "own"}, buildable)
    assert planned == {
        "A": {"primary": "own", "second": "big", "wave": 1, "accession": "GSE2"}
    }


def test_the_226_names_map_to_identity_map_keys():
    table = pd.DataFrame(
        {
            "model_id": ["A", "B", "C"],
            "expression_source": ["kinker_sccle", "Tahoe-100M DMSO", "mixseq"],
        }
    )
    assert primary_sources(table, ["A", "B", "C"]) == {
        "A": "kinker_scp542",
        "B": "tahoe100m",
        "C": "mixseq",
    }
