"""P1-A: extract fixed features once, train selected arms, and compare exports."""

import argparse
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path

import torch


def iter_features(model, dataset, *, batch_size=32, precision="bf16"):
    """Use the real condition-feature boundary, never the checkpoint's head."""
    if batch_size < 1 or precision not in {"bf16", "no"}:
        raise ValueError(
            "positive extraction batch size and bf16/no precision required"
        )
    model.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = dataset.collate(
                range(start, min(start + batch_size, len(dataset)))
            ).to(device)
            with (
                torch.autocast(device.type, dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            ):
                features = model.condition_features(batch.conditions)
            # Feature computation emits FP32; retain these exact cached values.
            yield features.to("cpu"), batch.residual.cpu(), batch.gene_mean.cpu()


def extract_cache(checkpoint, destination, *, device="cpu", batch_size=32):
    from src.data.datasets import DependencyDataset
    from src.data.prepared import load_inputs
    from src.data.readout_cache import write_feature_cache
    from src.experiments.geneeffect import _restore_model, _revision
    from src.training.checkpoint import load_checkpoint

    checkpoint = Path(checkpoint)
    saved = load_checkpoint(checkpoint)
    inputs = load_inputs(
        saved["config"], preprocessing=saved["preprocessing"], include_test=False
    )
    model = _restore_model(saved, inputs).to(device)
    datasets = {
        split: DependencyDataset(inputs, split, device=device)
        for split in ("train", "val")
    }
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    provenance = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_train_state": saved["train_state"],
        "revision": _revision(),
        "precision": saved["config"]["precision"],
        "extraction_batch_size": batch_size,
        "config": saved["config"],
    }
    cache = write_feature_cache(
        destination,
        {
            split: iter_features(
                model,
                data,
                batch_size=batch_size,
                precision=saved["config"]["precision"],
            )
            for split, data in datasets.items()
        },
        row_counts={split: len(data) for split, data in datasets.items()},
        split_lines={
            "train": list(inputs.split.supervised_train),
            "val": list(inputs.split.val),
        },
        genes=list(inputs.genes),
        variable_genes=[g for g in inputs.genes if g in inputs.variable_genes],
        provenance=provenance,
    )
    torch.save(
        {
            "standardizer": saved["normalization_state"],
            "projection": saved["projection_state"],
            "preprocessing": saved["preprocessing"],
        },
        cache.root / "source_state.pt",
    )
    return cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser(
        "extract", help="Extract train/val features once from the P0 checkpoint"
    )
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--out-dir", type=Path, required=True)
    extract.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Extraction batch only; head global batch stays 1024",
    )
    train = sub.add_parser(
        "train", help="Train seed-0 heads using the approved settings"
    )
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument(
        "--arms",
        nargs="+",
        choices=["A0", "A1", "A2", "A3"],
        default=["A0", "A1", "A2", "A3"],
    )
    train.add_argument(
        "--old-scaler", action="store_true", help="Optional A1-only scaler contrast"
    )
    train.add_argument(
        "--resume",
        type=Path,
        help="Resume one arm from its last.pt in the same run directory",
    )
    evaluate = sub.add_parser(
        "evaluate", help="Re-export a trained head without fitting"
    )
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    compare = sub.add_parser(
        "compare",
        help="Compare four completed arms on aligned train/validation exports",
    )
    compare.add_argument("--cache", type=Path, required=True)
    compare.add_argument("--runs", type=Path, required=True)
    compare.add_argument("--out-dir", type=Path, required=True)
    compare.add_argument(
        "--context-map",
        type=Path,
        default=Path("configs/benchmarks/cell_line_geneeffect_226_split.csv"),
    )
    compare.add_argument("--bootstrap-repeats", type=int, default=1000)
    compare.add_argument(
        "--reference-val",
        type=Path,
        action="append",
        default=[],
        help="Existing single-method P0/baseline predictions.parquet for alignment",
    )
    for command in (extract, train, evaluate):
        command.add_argument(
            "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
        )
    args = parser.parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error(
            "P1-A runs one process per arm; do not launch it with distributed workers"
        )
    if args.command == "extract":
        extract_cache(
            args.checkpoint,
            args.out_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        from src.data.readout_cache import ReadoutCache
        from src.eval.readout import export_checkpoint
        from src.training.readout import fit_readout

        cache = ReadoutCache(args.cache)
        if args.command == "compare":
            import pandas as pd
            from src.eval.readout_comparison import compare_runs

            mapping = pd.read_csv(args.context_map).set_index("model_id")
            ids = cache.metadata["split_lines"]["val"]
            if mapping.index.has_duplicates or set(ids) - set(mapping.index):
                parser.error("context map must contain one row per validation ModelID")
            groups = {
                line: str(mapping.loc[line, "patient_id"])
                if pd.notna(mapping.loc[line, "patient_id"])
                else f"line:{line}"
                for line in ids
            }
            compare_runs(
                cache,
                args.runs,
                args.out_dir,
                groups=groups,
                repeats=args.bootstrap_repeats,
                references=args.reference_val,
            )
        elif args.command == "train":
            if len(set(args.arms)) != len(args.arms):
                parser.error("arms must be unique")
            if args.resume is not None and len(args.arms) != 1:
                parser.error("--resume requires exactly one --arms entry")
            scaler = None
            if args.old_scaler:
                if args.arms != ["A1"]:
                    parser.error("--old-scaler requires --arms A1")
                from src.model.normalization import BlockStandardizer

                source = torch.load(cache.root / "source_state.pt", weights_only=True)
                scaler = BlockStandardizer.from_state(source["standardizer"])
            for arm in args.arms:
                fit_readout(
                    cache,
                    arm,
                    args.out_dir / ("A1-old-scaler" if args.old_scaler else arm),
                    device=args.device,
                    standardizer=scaler,
                    resume=args.resume,
                )
        else:
            export_checkpoint(cache, args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
