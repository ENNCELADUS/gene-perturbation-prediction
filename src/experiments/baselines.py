"""Evaluate train-fitted residual baselines on the prepared common panel."""

import argparse
import json
from pathlib import Path


def run_baselines(config, *, split, out_dir):
    import numpy as np
    import pandas as pd
    from src.baselines.residual import R1Result, run_r1_ladder
    from src.eval.geneeffect import aggregate_geneeffect
    from src.data.prepared import load_inputs
    from src.data.splits import FixedSplit

    if split not in {"val", "test"}:
        raise ValueError("baseline split must be val or test")
    inputs = load_inputs(config, include_test=(split == "test"))
    train = inputs.split.supervised_train
    requested = getattr(inputs.split, split)
    labels = inputs.labels.loc[inputs.labels.model_id.isin((*train, *requested))].copy()
    donor = (
        labels.loc[labels.model_id == "ACH-000551"].set_index("gene_symbol").gene_effect
    )
    if (
        not set(inputs.genes).issubset(donor.index)
        or not np.isfinite(donor.loc[list(inputs.genes)]).all()
    ):
        raise ValueError(
            "prepared common panel lacks finite training K562 donor coverage"
        )
    line_ids = (*train, *requested)
    views = {
        name: pd.DataFrame(
            np.stack(
                [
                    np.concatenate(
                        (
                            getattr(inputs.lines[line], attribute).mean(0),
                            getattr(inputs.lines[line], attribute).var(0),
                        )
                    )
                    for line in line_ids
                ]
            ),
            index=pd.Index(line_ids, name="model_id"),
        )
        for name, attribute in (("tx1", "controls_tx1"), ("hvg", "basal_hvg"))
    }
    fixed = FixedSplit(
        train=train,
        val=requested if split == "val" else (),
        test=requested if split == "test" else (),
    )
    result = run_r1_ladder(
        labels, views, donor.loc[list(inputs.genes)], seed=0, outer="fixed", split=fixed
    )
    # Re-score the shared predictions on the current absolute-per-line and
    # residual-per-gene axes, using the same fixed training means as the model.
    predictions = result.predictions.copy()
    means = predictions.gene_symbol.map(inputs.train_gene_means)
    predictions["geneeffect_prediction"] = predictions.residual_prediction + means
    metrics, per_line, per_gene = {}, [], []
    for method, frame in predictions.groupby("method", sort=True):
        scalar, lines, genes = aggregate_geneeffect(
            frame,
            model_ids=requested,
            genes=inputs.genes,
            variable_genes=[g for g in inputs.genes if g in inputs.variable_genes],
        )
        metrics[method] = {f"{split}_{key}": value for key, value in scalar.items()}
        per_line.append(lines.assign(method=method))
        per_gene.append(genes.assign(method=method))
    result = R1Result(
        predictions=predictions,
        per_line=pd.concat(per_line, ignore_index=True),
        per_gene=pd.concat(per_gene, ignore_index=True),
        summary=metrics,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    result.per_line.to_csv(out_dir / "per_line.csv", index=False)
    result.per_gene.to_csv(out_dir / "per_gene.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(result.summary, indent=2, allow_nan=False) + "\n"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from src.experiments.config import load_config

    run_baselines(load_config(args.config), split=args.split, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
