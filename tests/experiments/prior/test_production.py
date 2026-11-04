"""The production fit's patient rule: a labelled line sharing a patient with an SL
held-out line without GeneEffect leaves the fit, is still predicted, and is recorded."""

from __future__ import annotations

from src.data.extra_lines import ExtraLines
from src.data.splits import FixedSplit
from src.experiments.prior.production import left_out, production_split

SPLIT = FixedSplit(
    train=("T0", "T1", "U0", "U1"),
    val=("V0",),
    test=("E0",),
    unlabeled_train=("U0", "U1"),
)
EXTRA = ExtraLines(("B0", "B1"), ("Y0",), {"Z0": "old"})
PATIENTS = {"T0": "P0", "T1": "P1", "U0": "P0", "U1": "P2", "V0": "PV", "E0": "PE",
            "B0": "P0", "B1": "P2"}  # fmt: skip


def test_patient_siblings_of_held_out_lines_without_geneeffect_leave_the_fit():
    sl = {"train": ["T1", "U1"], "val": ["U0", "V0"], "test": ["E0"]}
    labelled = ["T0", "T1", "V0", "E0", "B0", "B1"]
    out = left_out(sl, SPLIT.unlabeled_train, PATIENTS, labelled)
    # U0 (no GeneEffect) is held out: T0 and B0 share its patient P0. U1 is SL
    # training, so B1 stays; V0 and E0 have GeneEffect, so their patients do not count.
    assert set(out) == {"T0", "B0"}
    assert "U0" in out["T0"]
    split, extra = production_split(SPLIT, EXTRA, out)
    assert (split.val, split.test) == ((), ())
    assert set(split.train) == {"T0", "T1", "U0", "U1", "V0", "E0"}  # all predicted
    assert set(split.supervised_train) == {"T1", "V0", "E0"}  # T0 gives no target
    assert extra.labelled == ("B1",) and extra.unlabelled == ("Y0", "B0")
    assert extra.excluded == {"Z0": "old"}  # a left-out line keeps its expression


def test_production_table_predicts_every_line_from_one_full_fit(tmp_path, monkeypatch):
    from src.experiments.prior import production
    from tests.fixtures.prior import TARGETS, synthetic_base, tiny_config

    config = tiny_config(tmp_path)
    base = synthetic_base(config)
    split = FixedSplit(
        train=base.bridge.single_cell_train, val=base.bridge.val, test=base.bridge.test
    )
    extras = tuple(m for m in base.labelled if m not in set(base.bridge.paired))
    monkeypatch.setattr(
        production, "load_config", lambda path: {"paths": {"split": "s"}}
    )
    monkeypatch.setattr(production, "load_geneeffect_split", lambda path: split)
    monkeypatch.setattr(
        production, "load_bulk_lines", lambda path, split: ExtraLines(extras, (), {})
    )
    monkeypatch.setattr(
        production, "read_models", lambda path: base.models.assign(patient_id="P")
    )
    seen = {}

    def on(config, fit_split, extra, space_lines):
        seen.update(split=fit_split, space=list(space_lines))
        return base

    monkeypatch.setattr(production, "base_on", on)
    sl = tmp_path / "sl.json"
    sl.write_text('{"train": [], "val": ["V0"], "test": ["T0"]}')
    table, record = production.production_table(config, sl)
    assert list(table.index) == list(
        seen["split"].train
    )  # every line, val and test too
    assert list(table.columns) == TARGETS and table.notna().all().all()
    assert seen["space"] == ["V0", "T0"] and record["left_out"] == {}
    assert "V0" in record["in_sample"]  # a labelled SL held-out line is fitted on
