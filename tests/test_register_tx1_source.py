from __future__ import annotations

from src.data.prepare.register_tx1_source import (
    EXPECTED_WEIGHT_BYTES,
    EXPECTED_WEIGHT_SHA256,
    MODEL_REVISION,
)


def test_tx1_3b_source_constants_are_pinned() -> None:
    assert MODEL_REVISION == "d218a580b9c2500ae9dfc8367a398545e6f017a8"
    assert EXPECTED_WEIGHT_BYTES == 10_868_228_196
    assert len(EXPECTED_WEIGHT_SHA256) == 64
