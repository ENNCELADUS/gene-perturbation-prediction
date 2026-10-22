"""PIE reproduction: PIE's released checkpoints, run through the vendored code on
PIE's own data, splits and sources, against the metrics PIE released with them.

``hpc/run.sh pie-reproduce --run-id ID [--device cuda] [--steps
fetch,download,evaluate,compare]`` (in ``.venv-pie``)
covers ``PIE_replogle_xdataset`` on Replogle ``test_seen`` and ``test_unseen`` and the
four ``PIE_replogle_wdataset`` folds on their ``unseen_ctx`` test splits. Steps, each
skipped when its outputs exist, so rerunning with the same id resumes:

- ``fetch``: the pinned checkpoints, their configs and data stats, and the released
  ``metrics_best_auprc.csv`` and ``granular_best_auprc.csv`` into
  ``$PIE_DATA_ROOT/released/<repo>@<revision>/``; the released tables copied to
  ``released/<experiment>/<row set>/``; ``runs/<experiment>/`` holds links to the
  checkpoint files only (``pie eval`` refuses to overwrite an existing ``eval/``);
- ``download``: every split, dataset and source the checkpoints pin, through PIE's
  own resolver into ``$PIE_DATA_ROOT/hf/``, each retried until it is complete
  (``downloaded.json``);
- ``evaluate``: ``pie.evaluate.run_eval`` per target, writing
  ``runs/<experiment>/eval/<row set>/``;
- ``compare``: released against reproduced, per context and per (context,
  perturbation) pair, into ``comparison.csv``, ``granular_comparison.csv`` and
  ``results.md``, with the wdataset fold summary beside the model card's.

Everything goes into ``outputs/pie_reproduction/<run id>/``, with ``run.json``
recording the pins, the environment and each step's times. It reports and
decides nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.run_dir import REPO_ROOT, now

log = logging.getLogger(__name__)

OUTPUT_ROOT = REPO_ROOT / "outputs" / "pie_reproduction"

VENDORED_COMMIT = "21a526add9ff3a81b4f962d9b16bbae570052c63"
XDATASET_REPO = "arcinstitute/PIE_replogle_xdataset"
XDATASET_REVISION = "f6da9607226139a1040b6992736118c9b5c7c313"
WDATASET_REPO = "arcinstitute/PIE_replogle_wdataset"
WDATASET_REVISION = "055c7a2cabdaff5659121c64fa2b609c231fc5be"
SPLITS_REVISION = "396ab9563175ee887750c9eed7ccaea6f5fdbf50"
SPLITS_URI = f"hf://datasets/arcinstitute/PIE_splits@{SPLITS_REVISION}"
WDATASET_FOLDS = ("hepg2", "jurkat", "k562", "rpe1")
CKPT = "best_auprc"
CHECKPOINT_FILES = (f"{CKPT}.ckpt", "config.yaml", "data_stats.json")
TABLES = ("metrics", "granular")
ALL_CONTEXT = "all"
STEPS = ("fetch", "download", "evaluate", "compare")
# The H20's connections to the Hugging Face CDN drop for minutes at a time.
DOWNLOAD_ATTEMPTS = 100
DOWNLOAD_BACKOFF_MAX = 600.0
# The wdataset model card's unseen_ctx summary over the four folds (mean, SD).
MODEL_CARD = {"binary_auprc": (0.4155, 0.0286), "spearman_lfc": (0.6568, 0.0411)}
FOLD_METRICS = tuple(MODEL_CARD)


@dataclass(frozen=True)
class Target:
    """One released checkpoint on one test split."""

    experiment: str  # PIE's experiment_name; the run dir is runs/<experiment>
    repo: str
    revision: str
    subdir: str  # the checkpoint's directory inside the model repo ("" = root)
    row_set: str
    split: str

    def repo_file(self, name: str) -> str:
        return f"{self.subdir}/{name}" if self.subdir else name

    def table_name(self, kind: str) -> str:
        return f"{kind}_{CKPT}.csv"


# The wdataset folds first: they need only Replogle and the gene sources.
TARGETS = (
    *(
        Target(
            f"replogle_wdataset/{fold}",
            WDATASET_REPO,
            WDATASET_REVISION,
            fold,
            "unseen_ctx",
            f"{SPLITS_URI}/replogle_wdataset/unseen_ctx/{fold}/test.json",
        )
        for fold in WDATASET_FOLDS
    ),
    *(
        Target(
            "replogle_xdataset",
            XDATASET_REPO,
            XDATASET_REVISION,
            "",
            row_set,
            f"{SPLITS_URI}/replogle_xdataset/{row_set}.json",
        )
        for row_set in ("test_seen", "test_unseen")
    ),
)


# --- comparison (stdlib, numpy and pandas only; pandas 2 and 3) ---------------------


def read_table(path: Path) -> pd.DataFrame:
    """A PIE metrics or granular CSV. Context and perturbation names stay strings
    (a gene called ``NA`` is not missing); only empty fields are NaN."""
    return pd.read_csv(
        path,
        dtype={"context": str, "perturbation": str},
        keep_default_na=False,
        na_values=[""],
    )


def _checked_keys(
    released: pd.DataFrame, reproduced: pd.DataFrame, keys: list[str]
) -> list[str]:
    """The metric columns, once both tables hold the same columns and key rows and
    agree on ``n_rows`` for every key."""
    if set(released.columns) != set(reproduced.columns):
        raise ValueError(
            f"columns differ: released {list(released.columns)}, "
            f"reproduced {list(reproduced.columns)}"
        )
    for name, table in (("released", released), ("reproduced", reproduced)):
        if table.duplicated(keys).any():
            raise ValueError(f"{name} table repeats a {keys} key")
    left = set(released[keys].itertuples(index=False, name=None))
    right = set(reproduced[keys].itertuples(index=False, name=None))
    if left != right:
        raise ValueError(
            f"{keys} keys differ: only released {sorted(left - right)[:5]}, "
            f"only reproduced {sorted(right - left)[:5]}"
        )
    joined = released[[*keys, "n_rows"]].merge(
        reproduced[[*keys, "n_rows"]], on=keys, suffixes=("_released", "_reproduced")
    )
    differ = joined[joined["n_rows_released"] != joined["n_rows_reproduced"]]
    if len(differ):
        raise ValueError(f"n_rows differ for {differ[keys].values.tolist()[:5]}")
    return [c for c in released.columns if c not in (*keys, "n_rows")]


def compare_metrics(released: pd.DataFrame, reproduced: pd.DataFrame) -> pd.DataFrame:
    """Per context and metric: released, reproduced and reproduced minus released."""
    metrics = _checked_keys(released, reproduced, ["context"])
    joined = released.merge(
        reproduced, on="context", suffixes=("_released", "_reproduced"), sort=False
    )
    rows = [
        {
            "context": record["context"],
            "metric": metric,
            "released": float(record[f"{metric}_released"]),
            "reproduced": float(record[f"{metric}_reproduced"]),
        }
        for record in joined.to_dict("records")
        for metric in metrics
    ]
    table = pd.DataFrame(rows, columns=["context", "metric", "released", "reproduced"])
    table["difference"] = table["reproduced"] - table["released"]
    return table


def compare_granular(released: pd.DataFrame, reproduced: pd.DataFrame) -> pd.DataFrame:
    """Per pair metric over the (context, perturbation) pairs: ``n_pairs`` defined on
    both sides, the max and mean absolute difference over them, and
    ``nan_mismatch``, the pairs undefined on exactly one side."""
    keys = ["context", "perturbation"]
    metrics = _checked_keys(released, reproduced, keys)
    joined = released.merge(reproduced, on=keys, suffixes=("_released", "_reproduced"))
    rows = []
    for metric in metrics:
        left = joined[f"{metric}_released"].astype(float).to_numpy()
        right = joined[f"{metric}_reproduced"].astype(float).to_numpy()
        both = ~np.isnan(left) & ~np.isnan(right)
        difference = np.abs(right[both] - left[both])
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(both.sum()),
                "nan_mismatch": int((np.isnan(left) != np.isnan(right)).sum()),
                "max_abs_difference": (
                    float(difference.max()) if difference.size else float("nan")
                ),
                "mean_abs_difference": (
                    float(difference.mean()) if difference.size else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def fold_summary(
    tables: Mapping[str, pd.DataFrame], metrics: Sequence[str] = FOLD_METRICS
) -> pd.DataFrame:
    """Mean and SD of each metric over the folds' ``all`` rows. The SD is the sample
    SD (ddof=1): it regenerates the wdataset model card's 0.0286 and 0.0411 from the
    four released ``all`` rows, where ddof=0 gives 0.0248 and 0.0356."""
    values = {metric: [] for metric in metrics}
    for fold, table in tables.items():
        overall = table[table["context"] == ALL_CONTEXT]
        if len(overall) != 1:
            raise ValueError(f"fold {fold}: expected one {ALL_CONTEXT!r} row")
        for metric in metrics:
            values[metric].append(float(overall[metric].iloc[0]))
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": float(np.mean(found)),
                "sd": float(np.std(found, ddof=1)),
                "n_folds": len(found),
            }
            for metric, found in values.items()
        ]
    )


def _value(value: float) -> str:
    return "undefined" if not np.isfinite(value) else f"{value:.6f}"


def _difference(value: float) -> str:
    return "undefined" if not np.isfinite(value) else f"{value:+.2e}"


def _magnitude(value: float) -> str:
    return "undefined" if not np.isfinite(value) else f"{value:.2e}"


def results_markdown(
    run_id: str,
    comparisons: Mapping[Target, pd.DataFrame],
    granular: Mapping[Target, pd.DataFrame],
    folds: Mapping[str, Mapping[str, pd.DataFrame]],
) -> str:
    """The ``all`` rows, the largest per-context and per-pair differences, and the
    fold summary beside the model card's."""
    lines = [
        f"# PIE reproduction `{run_id}`",
        "",
        f"Released checkpoints (`{XDATASET_REPO}@{XDATASET_REVISION[:7]}`, "
        f"`{WDATASET_REPO}@{WDATASET_REVISION[:7]}`) run through the vendored PIE "
        f"(`{VENDORED_COMMIT[:7]}`) on PIE's splits (`{SPLITS_REVISION[:7]}`), "
        "against the metrics released with them. Differences are reproduced minus "
        "released. Nothing is decided here.",
        "",
    ]
    for target, table in comparisons.items():
        overall = table[table["context"] == ALL_CONTEXT]
        lines += [
            f"## `{target.experiment}` on `{target.row_set}`",
            "",
            "| Metric (`all`) | Released | Reproduced | Difference |",
            "| --- | ---: | ---: | ---: |",
            *(
                f"| {row.metric} | {_value(row.released)} | {_value(row.reproduced)} "
                f"| {_difference(row.difference)} |"
                for row in overall.itertuples(index=False)
            ),
            "",
        ]
        magnitude = table["difference"].abs()
        if magnitude.notna().any():
            worst = table.loc[magnitude.idxmax()]
            lines.append(
                f"Largest |difference| over {table['context'].nunique()} contexts: "
                f"{_difference(worst['difference'])} ({worst['metric']}, "
                f"`{worst['context']}`)."
            )
        else:
            lines.append("Every per-context difference is undefined.")
        lines.append("")
    lines += [
        "## Per-pair metrics",
        "",
        "Pairs defined on both sides; a pair undefined on exactly one side is a NaN "
        "mismatch.",
        "",
        "| Target | Metric | Pairs | NaN mismatch | Max abs difference "
        "| Mean abs difference |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target, table in granular.items():
        for row in table.itertuples(index=False):
            lines.append(
                f"| `{target.experiment}` `{target.row_set}` | {row.metric} "
                f"| {row.n_pairs} | {row.nan_mismatch} "
                f"| {_magnitude(row.max_abs_difference)} "
                f"| {_magnitude(row.mean_abs_difference)} |"
            )
    released = fold_summary(folds["released"]).set_index("metric")
    reproduced = fold_summary(folds["reproduced"]).set_index("metric")
    lines += [
        "",
        f"## `replogle_wdataset` `unseen_ctx`, {len(WDATASET_FOLDS)} folds",
        "",
        "Mean ± sample SD (ddof=1) of the folds' `all` rows.",
        "",
        "| Metric | Released | Reproduced | Model card |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, (card_mean, card_sd) in MODEL_CARD.items():
        cells = [
            f"{frame.loc[metric, 'mean']:.4f} ± {frame.loc[metric, 'sd']:.4f}"
            for frame in (released, reproduced)
        ]
        lines.append(
            f"| {metric} | {cells[0]} | {cells[1]} | {card_mean:.4f} ± {card_sd:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


# --- steps -------------------------------------------------------------------------


def _released_table(run_dir: Path, target: Target, kind: str) -> Path:
    return (
        run_dir
        / "released"
        / target.experiment
        / target.row_set
        / (target.table_name(kind))
    )


def _reproduced_table(run_dir: Path, target: Target, kind: str) -> Path:
    return (
        run_dir
        / "runs"
        / target.experiment
        / "eval"
        / target.row_set
        / (target.table_name(kind))
    )


def _fetch_outputs(run_dir: Path) -> list[Path]:
    paths = [_released_table(run_dir, t, kind) for t in TARGETS for kind in TABLES]
    paths += [
        run_dir / "runs" / t.experiment / name
        for t in TARGETS
        for name in CHECKPOINT_FILES
    ]
    return paths


def _data_root() -> Path:
    value = os.environ.get("PIE_DATA_ROOT")
    if not value:
        raise RuntimeError(
            "PIE_DATA_ROOT is unset; run through hpc/run.sh pie-reproduce"
        )
    return Path(value).resolve()


def fetch(run_dir: Path) -> None:
    # PIE's environment (.venv-pie) has huggingface_hub; the project env may not.
    from huggingface_hub import hf_hub_download

    data_root = _data_root()
    for target in TARGETS:
        local_dir = (
            data_root / "released" / f"{target.repo.split('/')[1]}@{target.revision}"
        )
        names = [target.repo_file(name) for name in CHECKPOINT_FILES]
        names += [
            target.repo_file(f"eval/{target.row_set}/{target.table_name(kind)}")
            for kind in TABLES
        ]
        files = {
            name: Path(
                hf_hub_download(
                    target.repo, name, revision=target.revision, local_dir=local_dir
                )
            )
            for name in names
        }
        for kind in TABLES:
            source = files[
                target.repo_file(f"eval/{target.row_set}/{target.table_name(kind)}")
            ]
            copy = _released_table(run_dir, target, kind)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copy)
        links = run_dir / "runs" / target.experiment
        links.mkdir(parents=True, exist_ok=True)
        for name in CHECKPOINT_FILES:
            link = links / name
            if link.is_symlink():
                link.unlink()
            link.symlink_to(files[target.repo_file(name)])
        log.info("fetched %s %s into %s", target.experiment, target.row_set, local_dir)


def _pie_environment(run_dir: Path) -> None:
    for name in ("PIE_DATA_ROOT", "PIE_CACHE_DIR"):
        if not os.environ.get(name):
            raise RuntimeError(f"{name} is unset; run through hpc/run.sh pie-reproduce")
    # Each checkpoint saves its run_dir as ${PIE_RUNS_ROOT}/<experiment>, which
    # loading resolves; here that is runs/<experiment> of this run.
    os.environ["PIE_RUNS_ROOT"] = str(run_dir / "runs")


def _asset_references(run_dir: Path) -> list[tuple[str, str]]:
    """(kind, reference) of every split, dataset and source the evaluations read,
    in the order the wdataset folds, then the xdataset, need them."""
    from pie.predict import load_checkpoint

    references: dict[tuple[str, str], None] = {}
    for target in TARGETS:
        references[("split_file", target.split)] = None
        experiment_dir = run_dir / "runs" / target.experiment
        # As in evaluate: legacy relative paths must not resolve from this repo.
        with contextlib.chdir(experiment_dir):
            data = load_checkpoint(experiment_dir / f"{CKPT}.ckpt").config.data
        references[("splits", data.split_dir)] = None
        for path in data.preprocessed_dirs:
            references[("preprocessed", path)] = None
        references[("source", data.gene_text_dir)] = None
        for path in data.source_dirs.values():
            references[("source", path)] = None
    return list(references)


def download(run_dir: Path) -> None:
    """Resolve every asset through PIE's own resolver, retrying each one, so that
    evaluate finds them complete. A failed attempt keeps the files it finished."""
    import httpx
    from huggingface_hub.utils import close_session
    from pie.assets import resolve_asset, resolve_split_file

    _pie_environment(run_dir)
    references = _asset_references(run_dir)
    resolved = {}
    for number, (kind, reference) in enumerate(references, start=1):
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                path = (
                    resolve_split_file(reference)
                    if kind == "split_file"
                    else resolve_asset(reference, kind=kind)
                )
                break
            except (OSError, RuntimeError, httpx.HTTPError) as error:
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise
                wait = min(DOWNLOAD_BACKOFF_MAX, 30.0 * attempt)
                log.warning(
                    "download %d/%d %s attempt %d failed (%s: %s); retrying in %.0f s",
                    number,
                    len(references),
                    reference,
                    attempt,
                    type(error).__name__,
                    str(error)[:200],
                    wait,
                )
                # A retry after a failed request can leave the shared client closed.
                close_session()
                time.sleep(wait)
        resolved[reference] = str(path)
        log.info("download %d/%d %s: %s", number, len(references), kind, reference)
    _write_record(run_dir / "downloaded.json", resolved)


def evaluate(run_dir: Path, device: str) -> None:
    # PIE runs only in its own environment (.venv-pie), not the project's.
    from pie.config import EvalConfig
    from pie.evaluate import run_eval

    _pie_environment(run_dir)
    for target in TARGETS:
        metrics = _reproduced_table(run_dir, target, "metrics")
        granular = _reproduced_table(run_dir, target, "granular")
        if granular.exists():
            log.info("%s %s: evaluated, skipping", target.experiment, target.row_set)
            continue
        # run_eval writes metrics, then granular; a crash between leaves metrics alone.
        metrics.unlink(missing_ok=True)
        experiment_dir = run_dir / "runs" / target.experiment
        config = EvalConfig(
            experiment_name=target.experiment,
            run_dir=str(experiment_dir),
            ckpt=CKPT,
            split_path=target.split,
            row_set=target.row_set,
            preprocessed_dirs=None,
            save_predictions=False,
            overwrite=False,
            device=device,
            batch_size=16,
        )
        start = time.monotonic()
        # The checkpoints' relative data/splits/... and data/sources/aliases.yaml
        # map to PIE's HF copies only when they do not resolve from the working
        # directory, so they must not reach this project's data/.
        with contextlib.chdir(experiment_dir):
            run_eval(config)
        log.info(
            "%s %s: evaluated in %.1f min",
            target.experiment,
            target.row_set,
            (time.monotonic() - start) / 60,
        )


def compare(run_dir: Path, run_id: str) -> None:
    comparisons: dict[Target, pd.DataFrame] = {}
    granular: dict[Target, pd.DataFrame] = {}
    folds: dict[str, dict[str, pd.DataFrame]] = {"released": {}, "reproduced": {}}
    for target in TARGETS:
        tables = {
            side: {kind: read_table(path(run_dir, target, kind)) for kind in TABLES}
            for side, path in (
                ("released", _released_table),
                ("reproduced", _reproduced_table),
            )
        }
        comparisons[target] = compare_metrics(
            tables["released"]["metrics"], tables["reproduced"]["metrics"]
        )
        granular[target] = compare_granular(
            tables["released"]["granular"], tables["reproduced"]["granular"]
        )
        if target.repo == WDATASET_REPO:
            for side in folds:
                folds[side][target.subdir] = tables[side]["metrics"]

    def labelled(frames: Mapping[Target, pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(
            [
                frame.assign(experiment=t.experiment, row_set=t.row_set)[
                    ["experiment", "row_set", *frame.columns]
                ]
                for t, frame in frames.items()
            ],
            ignore_index=True,
        )

    labelled(comparisons).to_csv(run_dir / "comparison.csv", index=False)
    labelled(granular).to_csv(run_dir / "granular_comparison.csv", index=False)
    (run_dir / "results.md").write_text(
        results_markdown(run_id, comparisons, granular, folds)
    )
    log.info("PIE reproduction: %s", run_dir / "results.md")


# --- record ------------------------------------------------------------------------


def environment() -> dict[str, object]:
    """Python, torch, PIE and the GPU, each as far as this environment has it."""
    record: dict[str, object] = {
        "python": platform.python_version(),
        "torch": None,
        "torch_cuda": None,
        "pie": None,
        "gpus": None,
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    }
    with contextlib.suppress(ImportError):
        import torch

        record["torch"] = torch.__version__
        record["torch_cuda"] = torch.version.cuda
    with contextlib.suppress(ImportError):
        import pie

        record["pie"] = getattr(pie, "__version__", None)
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        record["gpus"] = [line.strip() for line in query.stdout.splitlines() if line]
    return record


def _write_record(path: Path, record: Mapping[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def _start_record(path: Path, run_id: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    record = {
        "run_id": run_id,
        "pins": {
            "vendored_pie_commit": VENDORED_COMMIT,
            "models": {
                XDATASET_REPO: XDATASET_REVISION,
                WDATASET_REPO: WDATASET_REVISION,
            },
            "splits": SPLITS_URI,
        },
        "environment": environment(),
        "steps": {},
    }
    _write_record(path, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", default=",".join(STEPS))
    args = parser.parse_args(argv)
    requested = [step for step in args.steps.split(",") if step]
    unknown = sorted(set(requested) - set(STEPS))
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {list(STEPS)}")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    run_dir = OUTPUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "run.json"
    record = _start_record(record_path, args.run_id)
    done = {
        "fetch": lambda: all(path.exists() for path in _fetch_outputs(run_dir)),
        "download": lambda: (run_dir / "downloaded.json").exists(),
        "evaluate": lambda: all(
            _reproduced_table(run_dir, t, "granular").exists() for t in TARGETS
        ),
        "compare": lambda: (run_dir / "results.md").exists(),
    }
    run = {
        "fetch": lambda: fetch(run_dir),
        "download": lambda: download(run_dir),
        "evaluate": lambda: evaluate(run_dir, args.device),
        "compare": lambda: compare(run_dir, args.run_id),
    }
    for step in (step for step in STEPS if step in requested):
        if done[step]():
            log.info("%s: outputs exist, skipping", step)
            continue
        started = now()
        run[step]()
        # Re-read: another process may have recorded its steps meanwhile.
        record = json.loads(record_path.read_text())
        record["steps"][step] = {"started": started, "finished": now()}
        _write_record(record_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
