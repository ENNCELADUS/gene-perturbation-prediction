# scripts/smoke_accelerate_no_collective.py
"""Guard G2 smoke test: PartialState under accelerate launch with NO collective.

Run on the cluster:

    accelerate launch --num_processes 4 \\
        scripts/smoke_accelerate_no_collective.py --out-dir /tmp/g2_smoke

Confirms that keeping accelerate launch + PartialState (launch-model A) never
triggers a lazy NCCL setup / 600s store timeout when no collective is called.
Expected: exit 0, one file per rank, no timeout traceback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import PartialState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = PartialState()
    # No gather/broadcast/barrier — only local attributes.
    marker = args.out_dir / f"rank_{state.process_index}.ok"
    marker.write_text(
        f"rank={state.process_index} "
        f"num_processes={state.num_processes} "
        f"device={state.device}\n"
    )
    print(  # noqa: T201 — smoke script, stdout is the signal
        f"[rank {state.process_index}/{state.num_processes}] "
        f"device={state.device} wrote {marker}"
    )


if __name__ == "__main__":
    main()
