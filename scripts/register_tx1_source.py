#!/usr/bin/env python3
"""Validate and register the fixed Tahoe-x1 3B model acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

MODEL_REPO = "tahoebio/Tahoe-x1"
MODEL_REVISION = "d218a580b9c2500ae9dfc8367a398545e6f017a8"
SOURCE_COMMIT = "167cc19fce5888c7738b0f73f61cb4a8d3d1d457"
EXPECTED_WEIGHT_BYTES = 10_868_228_196
EXPECTED_WEIGHT_SHA256 = (
    "424911f1d7425001db3dc6792193ce6470b6b15ab7ec10a35267cc27bd46634c"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register(model_dir: Path, source_repo: Path) -> dict[str, object]:
    required = [
        "model.safetensors",
        "model_config.yml",
        "collator_config.yml",
        "vocab.json",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing model files: {missing}")
    weight = model_dir / "model.safetensors"
    if weight.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise ValueError(f"Unexpected model size: {weight.stat().st_size}")
    weight_sha = sha256_file(weight)
    if weight_sha != EXPECTED_WEIGHT_SHA256:
        raise ValueError(f"Unexpected model SHA256: {weight_sha}")
    config = yaml.safe_load((model_dir / "model_config.yml").read_text())
    if config.get("d_model") != 2560:
        raise ValueError(f"Unexpected 3B d_model: {config.get('d_model')}")
    commit = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Unexpected Tahoe-x1 source commit: {commit}")
    worktree_status = subprocess.run(
        ["git", "-C", str(source_repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if worktree_status:
        raise ValueError("Tahoe-x1 source worktree is not clean")
    files = {
        name: {
            "bytes": (model_dir / name).stat().st_size,
            "sha256": sha256_file(model_dir / name),
        }
        for name in required
    }
    return {
        "status": "verified",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_subdirectory": "3b-model",
        "model_label": "tahoe_x1_3b",
        "rejected_label": "tx-70m-merged",
        "label_resolution": (
            "The official 3b-model artifact is selected explicitly; the top-level "
            "or legacy tx-70m-merged label is not accepted as the 3B encoder."
        ),
        "expected_obsm_width": 2560,
        "source_repository": "https://github.com/tahoebio/tahoe-x1",
        "source_commit": commit,
        "source_worktree_clean": True,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = register(args.model_dir, args.source_repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
