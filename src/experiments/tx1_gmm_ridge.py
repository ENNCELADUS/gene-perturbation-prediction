"""Fit a basal-only Tx1 GMM64-ridge baseline and evaluate saved models."""

import argparse
from dataclasses import asdict
from pathlib import Path


def evaluate_model(model, inputs, *, split):
    import pandas as pd
    from src.eval.geneeffect import EvalResult, aggregate_geneeffect

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    model_ids = (
        inputs.split.supervised_train
        if split == "train"
        else getattr(inputs.split, split)
    )
    if model.genes != inputs.genes:
        raise ValueError("saved model gene order differs from prepared common panel")
    features = model.context_features(
        {model_id: inputs.lines[model_id].controls_tx1 for model_id in model_ids}
    )
    wide = model.predict(features)
    predictions = wide.reset_index().melt(
        id_vars="model_id",
        var_name="gene_symbol",
        value_name="residual_prediction",
    )
    frame = inputs.labels.loc[
        inputs.labels.model_id.isin(model_ids),
        ["model_id", "gene_symbol", "gene_effect"],
    ].merge(
        predictions, on=["model_id", "gene_symbol"], how="left", validate="one_to_one"
    )
    means = frame.gene_symbol.map(dict(zip(model.genes, model.gene_means, strict=True)))
    frame["residual"] = frame.gene_effect - means
    frame["geneeffect_prediction"] = frame.residual_prediction + means
    metrics, per_line, per_gene = aggregate_geneeffect(
        frame,
        model_ids=model_ids,
        genes=inputs.genes,
        variable_genes=[gene for gene in inputs.genes if gene in inputs.variable_genes],
    )
    prefix = "train_eval" if split == "train" else split
    result = EvalResult(
        {f"{prefix}_{key}": value for key, value in metrics.items()},
        frame,
        per_line,
        per_gene,
        pd.DataFrame(columns=["model_id", "gene_symbol"]),
    )
    return result, features


def _export(model, inputs, run_dir, split):
    from src.experiments.geneeffect import _set_status, export_evaluation

    _set_status(run_dir, "evaluation", "running", split=split)
    try:
        result, features = evaluate_model(model, inputs, split=split)
        destination = run_dir / "evaluation" / split
        export_evaluation(result, destination)
        features.to_csv(destination / "context_features.csv")
        _set_status(
            run_dir, "evaluation", "completed", split=split, output=str(destination)
        )
        return result
    except Exception as exc:
        _set_status(run_dir, "evaluation", "failed", split=split, error=str(exc))
        raise


def fit_baseline(config_path: Path, out_dir: Path) -> Path:
    import joblib
    import sklearn
    from src.baselines.tx1_gmm import Tx1GMMRidge
    from src.data.prepared import load_inputs
    from src.experiments.config import load_config
    from src.experiments.geneeffect import _revision, _set_status, _write_json

    config = load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        out_dir / "run.json",
        {
            "method": "tx1_gmm_ridge",
            "revision": _revision(),
            "config": config,
            "sklearn_version": sklearn.__version__,
            "fitting": {"status": "running"},
            "evaluation": {"status": "not_started"},
        },
    )
    destination = out_dir / "model.joblib"
    temporary = out_dir / "model.joblib.tmp"
    try:
        inputs = load_inputs(config)
        model = Tx1GMMRidge.fit(inputs)
        bundle = {
            "schema_version": 1,
            "model": model,
            "config": config,
            "preprocessing": inputs.preprocessing_state(),
            "split": asdict(inputs.split),
            "sklearn_version": sklearn.__version__,
        }
        joblib.dump(bundle, temporary)
        temporary.replace(destination)
        _write_json(out_dir / "diagnostics.json", model.diagnostics)
        _set_status(
            out_dir,
            "fitting",
            "completed",
            gmm_converged=model.diagnostics["gmm_converged"],
        )
        print(
            f"Saved {destination}; GMM converged={model.gmm.converged_}, "
            f"iterations={model.gmm.n_iter_}",
            flush=True,
        )
    except Exception as exc:
        _set_status(out_dir, "fitting", "failed", error=str(exc))
        raise
    finally:
        temporary.unlink(missing_ok=True)
    for split in ("train", "val"):
        _export(model, inputs, out_dir, split)
    return destination


def evaluate_checkpoint(path: Path, *, split):
    import joblib
    import sklearn
    from src.data.prepared import load_inputs

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    path = Path(path)
    bundle = joblib.load(path)
    if (
        bundle["schema_version"] != 1
        or bundle["sklearn_version"] != sklearn.__version__
    ):
        raise ValueError("unsupported baseline artifact schema or sklearn version")
    inputs = load_inputs(
        bundle["config"],
        preprocessing=bundle["preprocessing"],
        include_test=(split == "test"),
    )
    if asdict(inputs.split) != bundle["split"]:
        raise ValueError("prepared split differs from saved baseline split")
    return _export(bundle["model"], inputs, path.parent, split)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit once and export train/val diagnostics")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate a trusted saved model")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), required=True)
    args = parser.parse_args(argv)
    if args.command == "fit":
        fit_baseline(args.config, args.out_dir)
    else:
        evaluate_checkpoint(args.model, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
