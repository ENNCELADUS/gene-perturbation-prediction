"""Prepare or check local artifacts required by exp08.

This script is intentionally thin: it reuses the ESM2 precompute and gwps bag
builders that the pipeline already consumes. Use ``check`` first on any machine;
run ``esm2`` on a node with Hugging Face/network/GPU access; run ``bags`` on a
node with enough RAM for the gwps h5ad or where a cache can be written.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
from pathlib import Path

import numpy as np

from precompute_esm2_embeddings import (
    check_resolution,
    embed_sequences,
    load_or_fetch_sequences,
    universe_symbols,
)
from sl_dl_model.bags import build_gwps_bags, load_bags_npz, save_bags_npz
from sl_dl_model.config import SLDLConfig, load_config

logger = logging.getLogger("setup_exp08_assets")

DEFAULT_CONFIG = Path(
    "configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml"
)
DEFAULT_ESM2_NPZ = Path("data/esm2/k562_sl_universe_esm2_650M.npz")
DEFAULT_SEQ_CACHE = Path("data/esm2/symbol_to_sequence.json")
DEFAULT_BAGS_NPZ = Path("data/exp08_cache/k562_gwps_bags.npz")
DEFAULT_HF_MODEL = "facebook/esm2_t33_650M_UR50D"


def _version(package: str) -> str:
    """Return installed package version, or ``missing``."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _status(path: Path, label: str, required: bool = True) -> bool:
    """Log whether a setup path exists."""
    exists = path.exists()
    level = logging.INFO if exists or not required else logging.ERROR
    marker = "OK" if exists else "MISSING"
    size = (
        f" ({path.stat().st_size / 1024**2:.1f} MiB)"
        if exists and path.is_file()
        else ""
    )
    logger.log(level, "%s %s: %s%s", marker, label, path, size)
    return exists


def _check_npz(path: Path) -> None:
    """Log shape/resolution for an existing ESM2 cache."""
    if not path.exists():
        return
    with np.load(path, allow_pickle=True) as payload:
        symbols = payload["symbols"]
        vectors = payload["vectors"]
        resolved = payload["resolved"]
    logger.info(
        "ESM2 cache: symbols=%d dim=%d resolved=%d",
        len(symbols),
        int(vectors.shape[1]),
        int(np.asarray(resolved, dtype=bool).sum()),
    )


def check_assets(config: SLDLConfig, esm2_npz: Path, bags_npz: Path) -> bool:
    """Check all artifacts required to run the state_dl producer."""
    logger.info(
        "python deps: transformers=%s torch=%s",
        _version("transformers"),
        _version("torch"),
    )
    hf_token = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    logger.info(
        "HF token env present: %s (not required for public %s)",
        hf_token,
        config.esm2_model,
    )

    ok = True
    ok &= _status(config.input_csv, "SL benchmark CSV")
    ok &= _status(config.gwps_h5ad, "gwps h5ad")
    ok &= _status(config.state_checkpoint, "STATE checkpoint")
    ok &= _status(
        config.state_checkpoint.parent.parent / "pert_onehot_map.pt",
        "STATE pert_onehot_map.pt",
    )
    ok &= _status(esm2_npz, "ESM2 embedding cache")
    ok &= _status(bags_npz, "gwps bags cache")
    _check_npz(esm2_npz)
    if bags_npz.exists():
        bags = load_bags_npz(bags_npz)
        logger.info(
            "gwps bags cache: genes=%d control_cells=%d dim=%d",
            len(bags.bags_by_symbol),
            int(bags.control_template.shape[0]),
            bags.input_dim,
        )
    return ok


def build_esm2_cache(
    config: SLDLConfig,
    out: Path,
    seq_cache: Path,
    cache_dir: Path | None,
    local_files_only: bool,
) -> None:
    """Fetch UniProt sequences and build the ESM2 NPZ cache."""
    symbols = universe_symbols(config.input_csv)
    logger.info("universe size: %d genes", len(symbols))
    seqs = load_or_fetch_sequences(symbols, seq_cache)
    vectors, resolved = embed_sequences(
        symbols,
        seqs,
        config.esm2_model,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    check_resolution(resolved, n_symbols=len(symbols))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        symbols=np.array(symbols, dtype=object),
        vectors=vectors,
        resolved=resolved,
    )
    logger.info("wrote %s (%d resolved / %d)", out, int(resolved.sum()), len(symbols))


def build_bags_cache(config: SLDLConfig, out: Path) -> None:
    """Build and write the gwps bags NPZ cache."""
    bags = build_gwps_bags(config, rng_seed=config.seed)
    save_bags_npz(bags, out)
    logger.info(
        "wrote %s (genes=%d control_cells=%d dim=%d)",
        out,
        len(bags.bags_by_symbol),
        int(bags.control_template.shape[0]),
        bags.input_dim,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up exp08 ESM2/gwps/STATE assets.")
    parser.add_argument("action", choices=("check", "esm2", "bags", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--esm2-npz", type=Path, default=DEFAULT_ESM2_NPZ)
    parser.add_argument("--seq-cache", type=Path, default=DEFAULT_SEQ_CACHE)
    parser.add_argument("--bags-npz", type=Path, default=DEFAULT_BAGS_NPZ)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    config = load_config(args.config)
    if config.esm2_model != DEFAULT_HF_MODEL:
        logger.info("using configured HF model: %s", config.esm2_model)

    if args.action == "check":
        ok = check_assets(config, args.esm2_npz, args.bags_npz)
        if not ok:
            raise SystemExit(1)
        return
    if args.action in {"esm2", "all"}:
        build_esm2_cache(
            config,
            args.esm2_npz,
            args.seq_cache,
            args.hf_cache_dir,
            args.local_files_only,
        )
    if args.action in {"bags", "all"}:
        build_bags_cache(config, args.bags_npz)
    check_assets(config, args.esm2_npz, args.bags_npz)


if __name__ == "__main__":
    main()
