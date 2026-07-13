"""Build the deterministic exp05 GWPS raw-expression cache."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aivc_model.gwps_cache import build_gwps_cache
from aivc_model.prepare import load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    """Build the configured cache."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    cache_dir = args.cache_dir or config.data.prepared_cache_dir
    if cache_dir is None:
        raise ValueError("Configure data.prepared_cache_dir or pass --cache-dir")
    manifest = build_gwps_cache(config, cache_dir)
    LOGGER.info("GWPS cache ready: %s", manifest)


if __name__ == "__main__":
    main()
