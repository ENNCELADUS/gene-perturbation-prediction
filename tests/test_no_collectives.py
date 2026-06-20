"""Guard G1: the sl_dl_model package uses no torch.distributed collective."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN = (
    "gather_object",
    "all_gather",
    "all_reduce",
    ".broadcast(",
    "dist.broadcast",
    ".barrier(",
    "wait_for_everyone",
)


def test_evaluate_has_no_collective_symbols():
    src_dir = Path(__file__).resolve().parents[1] / "src" / "sl_dl_model"
    offenders: list[str] = []
    for py in src_dir.glob("*.py"):
        text = py.read_text()
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{py.name}: {token}")
    assert offenders == [], f"forbidden collective(s) found: {offenders}"
