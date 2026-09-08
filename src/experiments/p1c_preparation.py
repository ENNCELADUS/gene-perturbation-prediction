"""Fold-aware P1-C bundle preparation, wrapping P1-B's fixed-membership builder."""

import json
from pathlib import Path

from src.data.p1c import fold_membership
from src.experiments.geneeffect import _write_json
from src.experiments.p1b_preparation import P1B_EXPECTATIONS, prepare_bundle


def prepare_fold(checkpoint, fold, directory, *, reference_manifest=None):
    """Prepare a leave-one-anchor-out P1-C bundle for ``fold``.

    Reuses ``prepare_bundle`` with the fold's source/external membership
    (P1-B's exact-count expectations only apply to the ``jurkat`` fold, which
    reproduces P1-B's own membership). When ``reference_manifest`` is given,
    asserts the new bundle's measured coordinates match it exactly. Writes
    ``fold.json`` recording the fold's membership.
    """
    sources, external = fold_membership(fold)
    directory = Path(directory)
    expectations = P1B_EXPECTATIONS if fold == "jurkat" else None
    prepare_bundle(
        checkpoint,
        directory,
        anchors=sources,
        external=external,
        expectations=expectations,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    if reference_manifest is not None:
        reference = json.loads(Path(reference_manifest).read_text())
        if reference["coordinates"] != manifest["coordinates"]:
            raise ValueError("fold coordinates differ from the P1-B reference")
    _write_json(
        directory / "fold.json",
        {"fold": fold, "external": external, "sources": list(sources)},
    )
