import json
import math

import numpy as np
import pytest

from src.data.io import write_json, write_npy, write_npz


def test_json_writes_are_atomic_and_keep_or_null_nan(tmp_path):
    path = tmp_path / "deep" / "record.json"
    write_json(path, {"b": np.float32(1.5), "a": float("nan")}, sort_keys=True)
    assert path.read_text() == '{\n  "a": null,\n  "b": 1.5\n}\n'
    write_json(path, {"a": float("nan")}, keep_nan=True)
    assert math.isnan(json.loads(path.read_text())["a"])
    with pytest.raises(TypeError):
        write_json(path, {"a": object()})
    assert json.loads(path.read_text())["a"] != 0  # a failed write left the old file
    assert sorted(p.name for p in path.parent.iterdir()) == ["record.json"]


def test_array_writes_keep_the_target_name(tmp_path):
    write_npy(tmp_path / "a.npy", np.arange(3))
    write_npz(tmp_path / "b.npz", x=np.ones(2))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.npy", "b.npz"]
    assert np.load(tmp_path / "a.npy").tolist() == [0, 1, 2]
    with np.load(tmp_path / "b.npz") as payload:
        assert payload["x"].tolist() == [1.0, 1.0]


def test_relative_config_paths_resolve_from_the_repository_root():
    from src.experiments.run_dir import REPO_ROOT, absolute

    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert absolute("configs") == REPO_ROOT / "configs"
