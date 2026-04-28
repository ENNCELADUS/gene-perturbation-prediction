from __future__ import annotations

import logging
import warnings

from src.utils.distributed import (
    configure_process_output,
    disable_tqdm,
    is_primary_rank,
    suppress_torchtext_deprecation_warning,
)


def test_is_primary_rank_uses_distributed_rank_environment(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert is_primary_rank()

    monkeypatch.setenv("RANK", "1")
    assert not is_primary_rank()

    monkeypatch.setenv("RANK", "0")
    assert is_primary_rank()


def test_worker_rank_suppresses_python_warnings(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "1")

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            configure_process_output(suppress_stdout=False)
            warnings.warn("worker noise", UserWarning, stacklevel=1)
    finally:
        logging.disable(logging.NOTSET)

    assert caught == []


def test_tqdm_is_disabled_on_worker_rank(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "1")
    config = {"run_config": {"disable_tqdm": False}}

    assert disable_tqdm(config)


def test_torchtext_deprecation_warning_is_suppressed() -> None:
    message = (
        "Torchtext is deprecated and the last released version will be 0.18 "
        "(this one)."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        suppress_torchtext_deprecation_warning()
        warnings.warn(message, UserWarning, stacklevel=1)
        warnings.warn("keep this warning", UserWarning, stacklevel=1)

    assert [str(item.message) for item in caught] == ["keep this warning"]
