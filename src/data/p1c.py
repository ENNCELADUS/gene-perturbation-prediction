"""Leave-one-anchor-out fold membership and control input layouts for P1-C.

Pure data module: no imports from ``src.training``, ``src.eval`` or
``src.experiments``, and no import of ``src.data.p1b`` (``p1b`` imports from
here, not the reverse).
"""

ALL_ANCHORS = ("ACH-000551", "ACH-000739", "ACH-000971", "ACH-000995")

FOLDS = {
    "jurkat": "ACH-000995",
    "k562": "ACH-000551",
    "hepg2": "ACH-000739",
    "hct116": "ACH-000971",
}

INPUT_LAYOUTS = ("tx1", "hvg", "hvg_tx1")
HVG_WIDTH = 2000


def fold_membership(fold):
    """Return (sources, external) for a leave-one-anchor-out fold.

    ``sources`` preserves ``ALL_ANCHORS`` order with the held-out anchor
    removed; ``external`` is the held-out anchor. Raises ``KeyError`` for an
    unknown fold name.
    """
    held = FOLDS[fold]
    return tuple(a for a in ALL_ANCHORS if a != held), held
