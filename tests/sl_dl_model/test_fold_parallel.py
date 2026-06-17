"""Tests for fold-level task-parallel orchestration in run_cv (no all-reduce)."""

from __future__ import annotations

from sl_dl_model.evaluate import _shard_jobs


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
