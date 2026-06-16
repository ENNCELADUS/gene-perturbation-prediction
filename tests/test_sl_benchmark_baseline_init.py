from __future__ import annotations


def test_package_imports_and_has_version() -> None:
    import sl_benchmark_baseline

    assert isinstance(sl_benchmark_baseline.__version__, str)
