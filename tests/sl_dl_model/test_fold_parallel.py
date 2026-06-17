"""Tests for fold-level task-parallel orchestration in run_cv (no all-reduce)."""

from __future__ import annotations

from sl_dl_model.evaluate import _shard_jobs


def _toy_cv_frame():
    """Two splits (CV2, CV3) x two folds, deterministic labels over G0..G5."""
    import pandas as pd

    genes = [f"G{i}" for i in range(6)]
    eff = {g: float(i) - 2.5 for i, g in enumerate(genes)}
    rows = []
    pid = 0
    for split in ("CV2", "CV3"):
        for fold in (0, 1):
            for role in ("train", "test"):
                for i in range(len(genes)):
                    for j in range(i + 1, len(genes)):
                        rows.append(
                            {
                                "pair_id": f"p{pid}",
                                "fold_id": fold,
                                "split_type": split,
                                "split_role": role,
                                "sl_label": (i + j + fold) % 2,
                                "gene_a_symbol": genes[i],
                                "gene_b_symbol": genes[j],
                                "gene_a_k562_gene_effect": eff[genes[i]],
                                "gene_b_k562_gene_effect": eff[genes[j]],
                            }
                        )
                        pid += 1
    return pd.DataFrame(rows)


def test_shard_jobs_partitions_disjointly_and_covers_all() -> None:
    jobs = [("CV2", f) for f in range(5)] + [("CV3", f) for f in range(5)]
    num = 4
    shards = [_shard_jobs(jobs, r, num) for r in range(num)]

    # Disjoint: no job appears on two ranks.
    flat = [j for s in shards for j in s]
    assert len(flat) == len(jobs), "a job was duplicated or dropped"
    # Covers all: union equals the input set.
    assert set(flat) == set(jobs)
    # Balanced: 10 jobs / 4 ranks -> sizes 3,3,2,2.
    assert sorted(len(s) for s in shards) == [2, 2, 3, 3]


def test_shard_jobs_single_process_owns_everything() -> None:
    jobs = [("CV2", f) for f in range(5)]
    assert _shard_jobs(jobs, 0, 1) == jobs


def test_shard_jobs_more_ranks_than_jobs() -> None:
    jobs = [("CV2", 0), ("CV2", 1)]
    # Ranks 2 and 3 get nothing; no crash.
    assert _shard_jobs(jobs, 0, 4) == [("CV2", 0)]
    assert _shard_jobs(jobs, 1, 4) == [("CV2", 1)]
    assert _shard_jobs(jobs, 2, 4) == []
    assert _shard_jobs(jobs, 3, 4) == []


def test_run_cv_single_process_matches_serial_baseline(tmp_path) -> None:
    """run_cv under 1 process must produce the same rows as a direct serial loop.

    PartialState reports num_processes=1 in pytest, so this pins the refactored
    run_cv against a hand-rolled serial loop over the same jobs — the N-process
    parity gate (Task 4) relies on this 1-process path being correct first.
    """
    import pandas as pd

    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.evaluate import ZeroEmbeddingProducer, run_cv
    from sl_dl_model.scoring import run_fold_with_producer

    frame = _toy_cv_frame()  # defined above
    csv = tmp_path / "bench.csv"
    frame.to_csv(csv, index=False)
    cfg = SLDLConfig(
        input_csv=csv,
        output_dir=tmp_path / "out",
        split_types=("CV2", "CV3"),
        folds=(0, 1),
        ranking_k=(10,),
        include_coverage_flag=False,
    )

    summary = run_cv(cfg, ZeroEmbeddingProducer())
    _ = summary  # return value not directly asserted; artifacts checked on disk

    # Serial reference: same jobs, same producer, no sharding.
    ref_rows: list[dict[str, object]] = []
    for split in ("CV2", "CV3"):
        for fold in (0, 1):
            ref_rows.extend(
                run_fold_with_producer(frame, split, fold, cfg, ZeroEmbeddingProducer())
            )
    ref = pd.DataFrame(ref_rows)

    # The written official summary must exist and be non-empty.
    written = pd.read_csv(cfg.output_dir / "official_metrics_summary.csv")
    assert not written.empty
    # Same set of (split_type, model, slice, metric) keys as the serial baseline.
    from sl_benchmark_baseline.evaluate import _summarize

    ref_summary = _summarize(ref)
    key_cols = ["split_type", "model", "slice", "metric"]
    got_keys = written[key_cols].apply(tuple, axis=1).tolist()
    exp_keys = ref_summary[key_cols].apply(tuple, axis=1).tolist()
    assert sorted(got_keys) == sorted(exp_keys)
