---
name: benchmark-harness
description: Use before loading data/SL_benchmark/ splits, computing SL ranking metrics, comparing against SLMGAE/KR4SL, or running the vcc-dep-baseline CLI. The Feng2024 split files and the official cal_metrics have several conventions that silently produce train-on-test or non-comparable numbers.
---

# Feng2024 benchmark harness

## Split files: the suffix is a ratio, not a fold

`data/SL_benchmark/data/data_split/CV{1,2,3}_{1,5,20,50}.npy` — the numeric
suffix is the **negative:positive ratio**. Folds live *inside* the file.

```python
pos, neg = np.load(path, allow_pickle=True)   # whole array is (2, 4, 5)
# axis 1: 0 = graph_train, 1 = graph_test, 2 = train_pair, 3 = test_pair
# axis 2: the 5 folds
```

Verified on `CV2_1.npy`: slots 0/1 hold 9845² adjacency matrices (object-wrapped),
slot 2 fold 0 is `(23770, 2)` train pairs, slot 3 fold 0 is `(10964, 2)` test
pairs. **Swapping 2 and 3 silently trains on the test set.** Correct usage to
copy: `load_feng_fold` in `src/sl_profile_baseline/data.py:50-57`,
`scripts/build_k562_sl_benchmark.py:150`.

The **unsuffixed** file (`CV2_1.npy`) is the `Rand` set — the primary
comparison. The `_Exp` / `_Dep` variants are DepMap-expression/dependency-informed
negatives; pairing them with any DepMap-derived feature is **leakage**
(`docs/data/sl-benchmark-2024.md:125-128`).

`*_indep_test.pkl` / `*_kr4sl.pkl` load through a different branch
(`data/SL_benchmark/src/preprocess.py:573-606`) that handles only CV1 and CV3 —
a CV2 + `--indep_test` invocation falls through with `pos_samples` unbound.

## Metrics

Official `cal_metrics` (`preprocess.py:818`) ranks per-anchor over a full 9845²
score matrix:

- **Omitting `seen_index` leaks train positives into top-k** (`:845-847`).
- Precision@k denominator is `min(k, n_pos)` (`:879-881`).
- The anchor loop covers only test-positive genes (`:854-857`), not all genes.
- The local reimplementation truncates candidates to `max_k = max(ks)`
  (`src/sl_benchmark_baseline/metrics.py:75`, sliced at `:83`); the official one
  takes top-100 then slices. NDCG@50 can disagree. Use the official path for any
  number that goes in a result note.

`src/sl_benchmark_baseline/evaluate.py` builds the candidate gene universe from
the **entire input CSV** (`CandidateUniverse`, `:58`; `n_gene` drives the score
matrix at `:162-164`), so a per-split run and an `all_CV_*.csv` run get different
`n_gene` and therefore different ranking metrics throughout. **Those runs are not
comparable** — never put them in the same table.

Feng `unified_id` *is* the score-matrix row index; rows `0..N-1` of
`fin_entities.csv` must all be `Gene` — `_candidate_symbols` enforces this and
upper-cases the symbols (`sl_profile_baseline/data.py:66-74`). The matrix
dimension comes from the split file itself (`candidate_gene_count`, `:60-63`).
Derived CSVs re-index into a filtered space, so `gene_a_unified_id` no longer
matches score-matrix indices, and only one orientation per pair survives
(`build_k562_sl_benchmark.py:108`).

## The vendored zoo

`data/SL_benchmark/` is third-party Py3.7 / PyG-cu102 code with its own conda
environment. **Never install it into `.venv`.** Its `git status` shows ~119
modified files but `git diff --stat` is 0 — mode-only changes. Do not "restore"
them.

## `vcc-dep-baseline` pipeline order

```
build-features        → run-cv                → fit-final
build-cell-bags       → run-single-cell-cv    → evaluate-single-cell-external
                      → run-distribution-cv   → evaluate-distribution-external
                                              → summarize / organize-artifacts
```

- `build-external-features --reference-features` and
  `build-external-cell-bags --reference-bags` must point at the **reference**
  artifacts, or columns silently misalign.
- `--resume` **without** `--run-id` generates a fresh timestamped id and resumes
  nothing (`artifacts.py:772`). Without `--resume`, an existing dir raises
  (`artifacts.py:326`).
- `--resume` with a *different* `--model`/`--feature-set` appends into the same
  run dir, mixing incompatible results.
- `run-cv --features` silently falls back to a legacy path
  (`artifacts.py:418-423`).

## Config ladder (`configs/experiments/`) — dependency_baseline only

This is the `models:` / `selection:` pattern, and it applies to experiments
01–04 and 11 only. **exp05 / exp12 (`aivc_model`) do not use it.**

- `selection.models` is a set intersection over *generated* spec names
  (`models.py:349-357`). **A typo is silently dropped — no error** — and can
  leave zero models running.
- Generated names ≠ config keys: `ridge` → `ridge_alpha10` when `variants:` is
  set (`models.py:376-383`); `mlp_<variant>` (`:625`); `pca_ridge` expands per
  component.
- `models.elastic_net.selection: random` is sklearn's ElasticNet parameter,
  unrelated to the top-level `selection:` block (`models.py:445`).
- `cv.model_set: quick` silently disables elastic_net — it is the *default* for
  `elastic_net.enabled` (`models.py:432`).
- Unknown `models:` keys are ignored. No inheritance — each YAML is standalone.
- Contrast: `ddgcn/config.py:82` and `sl_profile_baseline/config.py:73` *raise*
  on unknown keys. Know which package you are in.
