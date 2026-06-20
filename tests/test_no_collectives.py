"""Guard G1: the sl_dl_model package uses no torch.distributed collective.

The original NCCL/TCPStore timeout came from a single ``gather_object`` collective
on an uneven fold shard. This guard is a regression tripwire: it AST-scans the
whole ``sl_dl_model`` package (recursively) and rejects any ``torch.distributed``
import or collective call. It is deliberately AST-based, not substring-based, so
it does not false-positive on docstrings that mention ``torch.distributed`` or on
the legitimate tensor op ``tensor.gather(dim, idx)``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Collective names that are unambiguous regardless of receiver — finding any of
# these as a called attribute is a violation.
UNAMBIGUOUS_COLLECTIVES = frozenset(
    {
        "all_gather",
        "all_gather_object",
        "gather_object",
        "scatter_object_list",
        "all_gather_into_tensor",
        "all_reduce",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "all_to_all",
        "all_to_all_single",
    }
)

# Collective names that collide with legitimate tensor ops (e.g. tensor.gather).
# Only a violation when called on a distributed receiver (``dist`` / ``torch``
# ``.distributed`` / accelerate ``wait_for_everyone``).
AMBIGUOUS_COLLECTIVES = frozenset(
    {"gather", "scatter", "reduce", "broadcast", "barrier"}
)

# Accelerate's process-group barrier — a distributed collective by another name.
ACCELERATE_COLLECTIVES = frozenset({"wait_for_everyone"})

_DIST_RECEIVERS = frozenset({"dist", "distributed", "torch_dist"})


def _src_files() -> list[Path]:
    pkg = Path(__file__).resolve().parents[1] / "src" / "sl_dl_model"
    return sorted(pkg.rglob("*.py"))


def _receiver_name(node: ast.expr) -> str | None:
    """Return the trailing attribute/name of a call receiver, e.g. ``dist`` in
    ``dist.gather`` or ``distributed`` in ``torch.distributed.gather``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _CollectiveVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.offenders: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "torch.distributed" or alias.name.startswith(
                "torch.distributed."
            ):
                self.offenders.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module == "torch.distributed" or module.startswith("torch.distributed."):
            self.offenders.append(f"from {module} import ...")
        if module in {"torch", ""} or module.startswith("torch"):
            for alias in node.names:
                if alias.name == "distributed":
                    self.offenders.append("from torch import distributed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in UNAMBIGUOUS_COLLECTIVES or attr in ACCELERATE_COLLECTIVES:
                self.offenders.append(f"call .{attr}(")
            elif attr in AMBIGUOUS_COLLECTIVES:
                receiver = _receiver_name(func.value)
                if receiver in _DIST_RECEIVERS:
                    self.offenders.append(f"call {receiver}.{attr}(")
        self.generic_visit(node)


def _scan(source: str) -> list[str]:
    visitor = _CollectiveVisitor()
    visitor.visit(ast.parse(source))
    return visitor.offenders


def test_sl_dl_model_uses_no_distributed_collective():
    offenders: list[str] = []
    for py in _src_files():
        for hit in _scan(py.read_text()):
            offenders.append(f"{py.name}: {hit}")
    assert offenders == [], f"forbidden collective(s) found: {offenders}"


@pytest.mark.parametrize(
    "snippet",
    [
        "import torch.distributed as dist\n",
        "from torch.distributed import all_gather\n",
        "from torch import distributed\n",
        "dist.all_reduce(x)\n",
        "dist.gather(x)\n",
        "dist.barrier()\n",
        "torch.distributed.reduce_scatter(out, ins)\n",
        "accelerator.wait_for_everyone()\n",
        "torch.distributed.gather_object(obj)\n",
    ],
)
def test_guard_rejects_distributed_constructs(snippet: str):
    assert _scan(snippet), f"guard failed to flag: {snippet!r}"


@pytest.mark.parametrize(
    "snippet",
    [
        # Legit tensor op that collides with a collective name — must NOT trip.
        "y = logits.gather(1, idx)\n",
        "y = src.scatter(0, idx, vals)\n",
        "total = losses.reduce(add)\n",
        # A docstring merely mentioning torch.distributed must NOT trip.
        '"""No torch.distributed collective is used here."""\n',
    ],
)
def test_guard_allows_legitimate_lookalikes(snippet: str):
    assert _scan(snippet) == [], f"guard false-positived on: {snippet!r}"
