"""The pinned cell-state programs: tables from the authors' R objects, and the
count of cell-line studies behind each meta-program."""

from __future__ import annotations

import pytest

from src.data.builders.build_state_programs import (
    cell_cycle_table,
    is_cell_line_study,
    meta_program_table,
    source_table,
)


def test_cell_cycle_phases_come_from_seurats_list_names():
    table = cell_cycle_table({"s.genes": ["MCM5", "PCNA"], "g2m.genes": ["CDK1"]})
    assert table.values.tolist() == [["S", "MCM5"], ["S", "PCNA"], ["G2M", "CDK1"]]
    with pytest.raises(ValueError):
        cell_cycle_table({"s.genes": ["MCM5"]})


def test_sources_join_on_the_mp_number_and_count_cell_line_studies():
    meta = meta_program_table({"MP1  Cell Cycle": ["A", "B"], "MP41 Unassigned": ["C"]})
    assert meta["program"].tolist() == [
        "MP1 Cell Cycle",
        "MP1 Cell Cycle",
        "MP41 Unassigned",
    ]
    names = list(dict.fromkeys(meta["program"]))
    contributions = {
        "MP1  Cell Cycle": [
            "LUNG_Kinker_2020",
            "GBM_cell_lines",
            "Melanoma_Tirosh_2016",
        ],
        "MP41 Unassigned-II": ["Glioma_Neftel_2019"],
    }
    table = source_table(contributions, names)
    assert table.values.tolist() == [
        ["MP1 Cell Cycle", 3, 2],
        ["MP41 Unassigned", 1, 0],
    ]
    with pytest.raises(ValueError, match="different programs"):
        source_table({"MP2 Other": []}, names)


def test_cell_line_studies():
    for study in (
        "BREAST_Kinker_2020",
        "Ewing_CellLines_Aynaud_2020",
        "GBM_cell_lines",
    ):
        assert is_cell_line_study(study)
    for study in (
        "Ewing_PDX_Aynaud_2020",
        "Glioma_Tirosh_2016",
        "SCLC_Mouse_Ireland_2020",
    ):
        assert not is_cell_line_study(study)
