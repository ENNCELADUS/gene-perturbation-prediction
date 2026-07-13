# Exp05 STATE+ESM-2 GWPS 5-Fold Minimal Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair exp05 so a frozen K562 STATE checkpoint consumes GWPS in the checkpoint's exact 2,000-gene expression order, uses exp08's ESM-2 gene embedding plus trainable perturbation adapter, and evaluates one pre-frozen 9,338-gene GWPS-DepMap universe under a strict leakage-free outer-five-fold/inner-validation protocol.

**Architecture:** Freeze one canonical outer-fold manifest over all 9,338 overlap genes before ESM resolution, cell sampling, or training, then make every gene-derived label, response cell, transition, prompt, and fine-tuning sample join that manifest. Keep only fold-invariant raw expression alignment/cache work shared; fit the adapter, projector, scVI, GMM, normalizer, layer choice, and scalar C head from the permitted inner-train/inner-validation partitions inside each outer fold. Outer-test observed responses remain sealed until model selection is complete and then flow only to generation-quality evaluation and the observed-B oracle.

**Tech Stack:** Python 3.11, uv, AnnData/h5py, NumPy/Pandas, PyTorch, scikit-learn, Accelerate, scvi-tools, pytest, Ruff.

## Global Constraints

- Preserve the supervised key `(K562, perturbation_gene)` and the continuous K562 DepMap GeneEffect label.
- Use `model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt` with its matching `var_dims.pkl`, `pert_onehot_map.pt`, batch sidecar, and cell-type sidecar.
- STATE expression input/output remains exactly `2000`; ESM-2 expands perturbation identity coverage, not the expression vocabulary.
- Resolve GWPS expression features through `adata.var["gene_name"]` into `var_dims.pkl["gene_names"]` order and require `2000/2000` matches. Never fall back to variance-selected HVG2000 for this config.
- Define the experiment universe as exactly the 9,338 unique genes in the refreshed GWPS-DepMap overlap table. Generate and freeze one canonical five-fold outer split over this full universe before ESM resolution, response-cell sampling, cache fitting, or training; no later filter may add, remove, or reassign a gene.
- Use `facebook/esm2_t33_650M_UR50D` residue-mean embeddings and require all `9338/9338` canonical genes to have `resolved=True`; do not create per-gene learned fallback vectors and do not filter unresolved genes. Any unresolved canonical gene is a preflight failure.
- Every GeneEffect label, GWPS response cell, transition-supervision row, gene-derived prompt cell, and fine-tuning sample must carry `perturbation_gene` and inherit the same canonical `outer_fold`. Non-targeting control prompt cells are fold-neutral; no perturbed outer-test response cell may be used as a prompt for fitting or selection.
- Within each outer fold, derive inner validation only from outer-train genes. Fit learned preprocessing and model parameters on inner-train genes; use inner validation only for early stopping and any explicitly enabled layer/model selection. Outer-test and Adamson labels cannot influence selection.
- Keep STATE weights frozen. Create a fresh ESM-2 adapter, projector, scVI model, GMM, normalizer, scalar C head, optimizer, and checkpoint namespace for every outer fold. If the STATE layer is configurable, lock it in config or select it using inner validation only.
- Outer-test observed responses are sealed during fitting and selection. After the selected checkpoint is frozen, they may be read only by `generation_quality_outer_test` and `observed_b_oracle_outer_test`; the oracle is trained on outer-train genes and never fitted or selected with outer-test labels or responses.
- A shared GWPS cache may contain deterministic raw/aligned cells from all genes only if it computes no learned or dataset-fitted statistics. Every fitted cache/artifact identity must cover the GWPS h5ad, label CSV, canonical split manifest and SHA-256, ESM-2 NPZ, checkpoint, `var_dims.pkl`, `pert_onehot_map.pt`, batch/cell-type sidecars, gene order, sampling settings, outer fold, and exact fit-gene list.
- Label all outputs as GeneEffect/dependency prediction, not single-cell death probability, timing, or death mechanism prediction.
- All Python, pytest, Ruff, mypy, and scripts run through `uv run`; every shell command is prefixed with `rtk`.

## Verified Remote Data Contract (2026-07-13)

Read-only inspection of `wangar2023@10.15.89.192:/public/home/wangar2023/VCC_Project` established:

| Item | Verified value |
| --- | ---: |
| GWPS path | `data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad` |
| GWPS shape | `1,989,578 x 8,248` |
| GWPS file size | `65,830,941,948` bytes |
| `var_names` | Ensembl IDs |
| expression symbol column | `var["gene_name"]` |
| unique perturbation labels | `9,867` including `non-targeting` |
| control cells | `75,328` |
| non-control perturbation genes | `9,866` |
| numeric K562 DepMap genes | `17,787` |
| GWPS x K562 GeneEffect overlap | `9,338` |
| STATE dimensions | `input_dim=2000`, `output_dim=2000`, `pert_dim=2024` |
| STATE expression matches | `2000/2000` through `var["gene_name"]` |
| STATE perturbation vocabulary coverage | `2,023/9,866` |
| duplicated GWPS symbols | `HSPA14`, `TBCE`; neither is in the STATE 2,000-gene list |
| old overlap table | `2,058` rows |
| existing exp08 ESM cache | `9,471` resolved symbols, but only `6,070/9,338` target genes covered |
| Adamson STATE-symbol matches | pilot `1,876/2,000`; UPR epistasis `1,874/2,000`; UPR Perturb-seq `1,874/2,000` |

Important correction: the three Adamson files store gene symbols in `var_names`; `var["ensembl_id"]` contains Ensembl IDs and matches `0/2000` STATE symbols. The exp05 config must therefore use Adamson `var_names`, not `var["ensembl_id"]`. The previous `0/2000` QA arose because the primary Replogle reference features were incorrectly Ensembl-based.

## File Map

- Create `scripts/build_exp05_gwps_labels.py`: generate the single-gene K562 GWPS/GeneEffect label table.
- Create `src/aivc_model/gene_splits.py`: build/load/validate the immutable 9,338-gene outer-fold manifest and construct fold-local inner roles.
- Create `scripts/build_exp05_gene_splits.py`: write the one canonical outer-fold manifest before ESM/cache assets.
- Modify `scripts/precompute_esm2_embeddings.py`: accept either the existing two SL symbol columns or one exp05 `perturbation_gene` column.
- Create `src/aivc_model/gwps_cache.py`: strict STATE-order alignment and memory-mappable GWPS bag-cache build/load.
- Modify `src/aivc_model/prepare.py`: parse primary symbol/cache settings, enforce strict alignment, use Adamson `var_names`, require complete ESM coverage, and expose gene-audited fold views without filtering the universe.
- Modify `src/aivc_model/model.py`: add `Esm2PerturbationAdapter` while preserving the perturbation-provider call interface used by `AivcModel`.
- Modify `src/aivc_model/train.py`: construct the ESM-2 adapter, evaluate both outer-test and Adamson, and expose one-fold training to the CV runner.
- Create `src/aivc_model/cross_validate.py`: consume the frozen outer manifest, create outer-train-only inner validation, enforce fold data-access policy, aggregate results, and run leakage guards.
- Create `configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml`: the only new experiment config.
- Modify `scripts/state.sh`: add an explicit config argument/default without changing unrelated Slurm settings.
- Modify `docs/experiment/05_aivc_a_to_b_to_c.md`: document the repaired contract and invalidate old negative conclusions.
- Modify `tests/sl_dl_model/test_precompute_esm2.py`: single-column universe tests.
- Modify `tests/test_aivc_model.py`: alignment, ESM adapter, cache, external QA, and fold-evaluation tests.
- Create `tests/test_aivc_cross_validate.py`: five-fold, inner split, fingerprint, and aggregation tests.
- Create `tests/test_build_exp05_gwps_labels.py`: deterministic label-table tests.
- Create `tests/test_exp05_gene_splits.py`: canonical-universe, immutable-manifest, provenance, and fold-role tests.

---

### Task 1: Build and Freeze the 9,338-Gene Universe Before Any Other Asset

**Files:**
- Create: `scripts/build_exp05_gwps_labels.py`
- Create: `src/aivc_model/gene_splits.py`
- Create: `scripts/build_exp05_gene_splits.py`
- Modify: `scripts/precompute_esm2_embeddings.py`
- Create: `tests/test_build_exp05_gwps_labels.py`
- Create: `tests/test_exp05_gene_splits.py`
- Modify: `tests/sl_dl_model/test_precompute_esm2.py`

**Interfaces:**
- Produces: `build_gwps_label_table(gwps_h5ad: Path, gene_effect_csv: Path, model_id: str) -> pd.DataFrame`.
- Produces: `build_canonical_outer_manifest(labels: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame`.
- Produces: `load_canonical_outer_manifest(path: Path, labels: pd.DataFrame, expected_sha256: str) -> pd.DataFrame`.
- Produces: `symbols_from_csv(csv_path: Path, symbol_columns: tuple[str, ...]) -> list[str]`.
- Produces remote assets: `data/sl_dependency_v0/interim/k562_gwps_depmap_overlap.csv`, `data/sl_dependency_v0/splits/k562_gwps_depmap_outer5_seed42.csv`, its `.sha256` file, and `data/esm2/k562_gwps_depmap_esm2_650M.npz`.

- [ ] **Step 1: Write failing label-builder tests**

```python
def test_build_gwps_label_table_keeps_numeric_intersection(tmp_path: Path) -> None:
    adata = ad.AnnData(np.ones((5, 2), dtype=np.float32))
    adata.obs["gene"] = ["non-targeting", "TP53", "TP53", "KRAS", "NO_LABEL"]
    h5ad = tmp_path / "gwps.h5ad"
    adata.write_h5ad(h5ad)
    effects = pd.DataFrame(
        [[-0.7, 0.2, np.nan]],
        index=["ACH-000551"],
        columns=["TP53 (7157)", "KRAS (3845)", "NO_LABEL (1)"],
    )
    csv = tmp_path / "CRISPRGeneEffect.csv"
    effects.to_csv(csv)

    result = build_gwps_label_table(h5ad, csv, "ACH-000551")

    assert result.to_dict("records") == [
        {
            "perturbation_gene": "KRAS",
            "depmap_model_id": "ACH-000551",
            "depmap_entrez_id": "3845",
            "depmap_gene_effect": 0.2,
            "has_depmap_label": True,
        },
        {
            "perturbation_gene": "TP53",
            "depmap_model_id": "ACH-000551",
            "depmap_entrez_id": "7157",
            "depmap_gene_effect": -0.7,
            "has_depmap_label": True,
        },
    ]


def test_build_gwps_label_table_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ACH-000551"):
        build_gwps_label_table(
            tmp_path / "gwps.h5ad",
            tmp_path / "CRISPRGeneEffect.csv",
            "ACH-000551",
        )
```

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run: `rtk uv run python -m pytest tests/test_build_exp05_gwps_labels.py -v`

Expected: FAIL because `scripts.build_exp05_gwps_labels` does not exist.

- [ ] **Step 3: Implement the deterministic label builder**

```python
GENE_COLUMN_RE = re.compile(r"^(?P<symbol>.+) \((?P<entrez>\d+)\)$")


def build_gwps_label_table(
    gwps_h5ad: Path,
    gene_effect_csv: Path,
    model_id: str,
) -> pd.DataFrame:
    adata = ad.read_h5ad(gwps_h5ad, backed="r")
    try:
        gwps_genes = {
            str(value).upper()
            for value in adata.obs["gene"].astype(str).unique()
            if str(value) != "non-targeting"
        }
    finally:
        adata.file.close()
    effects = pd.read_csv(gene_effect_csv, index_col=0)
    if model_id not in effects.index:
        raise ValueError(f"{model_id} not found in {gene_effect_csv}")
    rows: list[dict[str, object]] = []
    for column, value in pd.to_numeric(effects.loc[model_id], errors="coerce").dropna().items():
        match = GENE_COLUMN_RE.match(str(column))
        if match is None:
            continue
        symbol = match.group("symbol").upper()
        if symbol not in gwps_genes:
            continue
        rows.append(
            {
                "perturbation_gene": symbol,
                "depmap_model_id": model_id,
                "depmap_entrez_id": match.group("entrez"),
                "depmap_gene_effect": float(value),
                "has_depmap_label": True,
            }
        )
    return pd.DataFrame(rows).sort_values("perturbation_gene").reset_index(drop=True)
```

The CLI defaults must be the verified GWPS, DepMap, model, and output paths and must refuse duplicate `perturbation_gene` rows before writing.

- [ ] **Step 4: Write failing canonical-manifest tests**

```python
def test_canonical_outer_manifest_freezes_all_9338_genes_once() -> None:
    labels = _labels(9338)
    manifest = build_canonical_outer_manifest(labels, n_splits=5, seed=42)
    assert manifest.columns.tolist() == ["perturbation_gene", "outer_fold"]
    assert len(manifest) == 9338
    assert manifest["perturbation_gene"].nunique() == 9338
    assert set(manifest["outer_fold"]) == {0, 1, 2, 3, 4}
    assert manifest.equals(
        build_canonical_outer_manifest(labels, n_splits=5, seed=42)
    )


def test_manifest_loader_rejects_any_universe_or_hash_change(tmp_path: Path) -> None:
    labels = _labels(20)
    path, digest = _write_manifest(tmp_path, labels)
    changed = labels.iloc[:-1].copy()
    with pytest.raises(ValueError, match="canonical gene universe"):
        load_canonical_outer_manifest(path, changed, digest)
    with pytest.raises(ValueError, match="SHA-256"):
        load_canonical_outer_manifest(path, labels, "0" * 64)
```

- [ ] **Step 5: Run the manifest tests and confirm failure**

Run: `rtk uv run python -m pytest tests/test_exp05_gene_splits.py -v`

Expected: FAIL because `aivc_model.gene_splits` does not exist.

- [ ] **Step 6: Implement and write the immutable outer manifest**

```python
def build_canonical_outer_manifest(
    labels: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    frame = labels[["perturbation_gene", "depmap_gene_effect"]].copy()
    frame["perturbation_gene"] = frame["perturbation_gene"].astype(str).str.upper()
    if len(frame) != 9338 or frame["perturbation_gene"].nunique() != 9338:
        raise ValueError("canonical gene universe must contain exactly 9338 unique genes")
    frame = frame.sort_values("perturbation_gene").reset_index(drop=True)
    strata = quantile_strata(frame["depmap_gene_effect"].to_numpy(), n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outer_fold = np.full(len(frame), -1, dtype=np.int64)
    for fold, (_, test_index) in enumerate(splitter.split(frame.index, strata)):
        outer_fold[test_index] = fold
    if np.any(outer_fold < 0):
        raise AssertionError("every canonical gene must receive one outer fold")
    return pd.DataFrame(
        {
            "perturbation_gene": frame["perturbation_gene"],
            "outer_fold": outer_fold,
        }
    )
```

`scripts/build_exp05_gene_splits.py` reads the completed overlap CSV, refuses to overwrite an existing non-identical manifest, writes the sorted CSV atomically, then writes `sha256_file(manifest)` to `<manifest>.sha256`. `load_canonical_outer_manifest` verifies the digest, exact 9,338-gene set equality with the label table, unique symbols, fold values `0..4`, and exactly one assignment per gene. This file is the only authority for outer folds; runtime code must never call `StratifiedKFold` for the outer split.

- [ ] **Step 7: Generalize the ESM-2 CSV symbol reader without breaking exp08**

Write this test:

```python
def test_symbols_from_csv_supports_single_gene_column(tmp_path: Path) -> None:
    csv = tmp_path / "genes.csv"
    pd.DataFrame({"perturbation_gene": ["tp53", "KRAS", "TP53"]}).to_csv(
        csv, index=False
    )
    assert MOD.symbols_from_csv(csv, ("perturbation_gene",)) == ["KRAS", "TP53"]
```

Replace the hard-coded reader with:

```python
def symbols_from_csv(csv_path: Path, symbol_columns: tuple[str, ...]) -> list[str]:
    frame = pd.read_csv(csv_path, usecols=list(symbol_columns))
    symbols: set[str] = set()
    for column in symbol_columns:
        symbols.update(frame[column].dropna().astype(str).str.upper())
    return sorted(symbols)


def universe_symbols(benchmark_csv: Path) -> list[str]:
    return symbols_from_csv(benchmark_csv, ("gene_a_symbol", "gene_b_symbol"))
```

Add repeatable CLI option `--symbol-column`; when absent, retain the existing exp08 two-column default. The exp05 invocation uses `--symbol-column perturbation_gene`.

- [ ] **Step 8: Add strict full-universe ESM coverage validation**

Write this test:

```python
def test_exp05_esm_asset_must_resolve_all_canonical_genes() -> None:
    canonical = ["A", "B", "C"]
    table = Esm2EmbeddingTable(
        symbols=np.asarray(["A", "B"], dtype=object),
        embeddings=np.ones((2, 4), dtype=np.float32),
        resolved=np.asarray([True, True]),
    )
    with pytest.raises(ValueError, match="2/3"):
        require_complete_esm_coverage(canonical, table)
```

Implement `require_complete_esm_coverage` as an exact uppercase set/order check over the canonical manifest. Alias or sequence-resolution improvements occur while creating the NPZ; after creation, `resolved_count` must equal `9338`. Never drop unresolved rows, never regenerate the split, and never substitute a learned fallback. A non-`9338/9338` asset stops preflight and the run.

- [ ] **Step 9: Run asset-builder and manifest tests**

Run: `rtk uv run python -m pytest tests/test_build_exp05_gwps_labels.py tests/test_exp05_gene_splits.py tests/sl_dl_model/test_precompute_esm2.py -v`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
rtk git add scripts/build_exp05_gwps_labels.py src/aivc_model/gene_splits.py scripts/build_exp05_gene_splits.py scripts/precompute_esm2_embeddings.py tests/test_build_exp05_gwps_labels.py tests/test_exp05_gene_splits.py tests/sl_dl_model/test_precompute_esm2.py
rtk git commit -m "feat: freeze exp05 GWPS gene universe"
```

---

### Task 2: Enforce STATE Gene Order and Build a Fingerprinted GWPS Cache

**Files:**
- Create: `src/aivc_model/gwps_cache.py`
- Create: `scripts/build_exp05_gwps_cache.py`
- Modify: `src/aivc_model/prepare.py`
- Modify: `tests/test_aivc_model.py`

**Interfaces:**
- Produces: `resolve_state_gene_order(adata: ad.AnnData, model_dir: Path, symbol_col: str) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `build_gwps_cache(config: AivcConfig, cache_dir: Path) -> Path`.
- Produces: `load_gwps_cache(config: AivcConfig, cache_dir: Path) -> GeneBags`.
- Cache directory contains `cells.npy`, `offsets.npy`, `genes.npy`, `gene_outer_folds.npy`, `batch_labels.npy`, `control_cells.npy`, `control_batch.npy`, `feature_names.npy`, and `manifest.json`.
- Consumes the canonical manifest and its SHA-256; produces only fold-invariant aligned raw arrays, never normalized, projected, scVI-transformed, GMM-fitted, or pooled arrays.

- [ ] **Step 1: Write strict alignment tests**

```python
def test_state_alignment_uses_gene_name_in_checkpoint_order(tmp_path: Path) -> None:
    adata = ad.AnnData(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    adata.var_names = ["ENSG1", "ENSG2", "ENSG3"]
    adata.var["gene_name"] = ["B", "A", "C"]
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": ["A", "B"]}, handle)

    indices, names = resolve_state_gene_order(adata, model_dir, "gene_name")

    np.testing.assert_array_equal(indices, np.asarray([1, 0]))
    np.testing.assert_array_equal(names, np.asarray(["A", "B"], dtype=object))


def test_state_alignment_never_falls_back_when_checkpoint_gene_is_missing(
    tmp_path: Path,
) -> None:
    adata = ad.AnnData(np.ones((1, 1), dtype=np.float32))
    adata.var["gene_name"] = ["A"]
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": ["A", "B"]}, handle)

    with pytest.raises(ValueError, match="1/2"):
        resolve_state_gene_order(adata, model_dir, "gene_name")
```

- [ ] **Step 2: Run alignment tests and confirm failure**

Run: `rtk uv run python -m pytest tests/test_aivc_model.py -k "state_alignment" -v`

Expected: FAIL because `resolve_state_gene_order` does not exist.

- [ ] **Step 3: Implement exact symbol alignment**

```python
def resolve_state_gene_order(
    adata: ad.AnnData,
    model_dir: Path,
    symbol_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    with (model_dir / "var_dims.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    checkpoint_names = np.asarray(payload["gene_names"], dtype=object).astype(str)
    source_names = adata.var[symbol_col].astype(str).to_numpy()
    positions: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, symbol in enumerate(source_names):
        if symbol in positions:
            duplicates.add(symbol)
        else:
            positions[symbol] = index
    duplicate_matches = sorted(set(checkpoint_names).intersection(duplicates))
    missing = [name for name in checkpoint_names if name not in positions]
    if missing or duplicate_matches:
        matched = len(checkpoint_names) - len(missing) - len(duplicate_matches)
        raise ValueError(
            f"STATE expression alignment matched {matched}/{len(checkpoint_names)}; "
            f"missing={missing[:10]}, duplicate_matches={duplicate_matches[:10]}"
        )
    indices = np.asarray([positions[name] for name in checkpoint_names], dtype=np.int64)
    return indices, checkpoint_names.astype(object)
```

Change `_state_input_view()` to call this function when `state.backend == "state_checkpoint"`; return the checkpoint names as `GeneBags.feature_names`. Remove `state_hvg_n_top_genes: 2000` from the new config so this experiment has no variance fallback surface.

- [ ] **Step 4: Write cache fingerprint and round-trip tests**

```python
def test_gwps_cache_manifest_changes_with_state_sidecar(tmp_path: Path) -> None:
    inputs = _toy_cache_inputs(tmp_path)
    first = source_fingerprint(**inputs)
    inputs["var_dims"].write_bytes(b"changed")
    second = source_fingerprint(**inputs)
    assert first != second


def test_gwps_cache_round_trip_preserves_order_and_batches(tmp_path: Path) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    build_gwps_cache(config, cache_dir)
    bags = load_gwps_cache(config, cache_dir)
    assert bags.feature_names.tolist() == ["A", "B"]
    assert bags.genes.tolist() == ["G1", "G2"]
    np.testing.assert_array_equal(bags.gene_outer_folds, np.asarray([0, 1]))
    np.testing.assert_array_equal(bags.control_batch, np.asarray(["25", "31"]))


def test_gwps_cache_rejects_response_gene_outside_canonical_manifest(
    tmp_path: Path,
) -> None:
    config = _toy_gwps_cache_config(tmp_path, response_genes=["G1", "EXTRA"])
    with pytest.raises(ValueError, match="canonical manifest"):
        build_gwps_cache(config, tmp_path / "cache")
```

- [ ] **Step 5: Implement the memory-mappable cache contract**

The cache builder must:

```python
indices, feature_names = resolve_state_gene_order(
    adata, config.state.model_dir, config.data.var_gene_symbol_col
)
expression = _dense_slice(adata.X, indices)
labels = adata.obs[config.data.obs_perturbation_col].astype(str).str.upper().to_numpy()
batches = adata.obs[config.data.obs_batch_col].astype(str).to_numpy()
```

Join every non-control response cell to the canonical manifest by uppercase `perturbation_gene`; retain exactly its 9,338 genes and attach the manifest's `outer_fold` to the gene index. Non-targeting controls have no outer fold and are the only fold-neutral prompt-cell source. The cache builder may perform fixed coordinate reordering and seeded row selection only: no centering, scaling, normalization fitting, scVI fitting, projector fitting, GMM fitting, layer selection, pseudobulk fitting, or statistics pooled across perturbation genes.

It writes all arrays with `np.lib.format.open_memmap`, samples at most `cache_cells_per_gene` rows per perturbation with `np.random.default_rng(cache_seed)`, and writes `manifest.json` last. The manifest is accepted only when its `source_fingerprint` equals a recomputation covering:

```python
{
    "schema_version": 1,
    "h5ad": file_signature(config.data.h5ad_path),
    "checkpoint": file_signature(config.state.checkpoint_path),
    "var_dims": sha256_file(config.state.model_dir / "var_dims.pkl"),
    "pert_onehot_map": sha256_file(config.state.model_dir / "pert_onehot_map.pt"),
    "batch_sidecar": sidecar_signature(config.state.model_dir, "batch_onehot_map"),
    "cell_type_sidecar": sidecar_signature(config.state.model_dir, "cell_type_onehot_map"),
    "feature_names_sha256": sha256_strings(feature_names),
    "canonical_split_sha256": sha256_file(config.cv.outer_split_manifest),
    "canonical_gene_count": 9338,
    "cache_seed": config.data.cache_seed,
    "cache_cells_per_gene": config.data.cache_cells_per_gene,
}
```

For the 65 GB h5ad, `file_signature` is `{resolved_path, size, mtime_ns}`; small checkpoint sidecars use full SHA-256. A mismatched existing cache raises `ValueError("GWPS cache fingerprint mismatch")`; it is never silently reused. Loading also asserts that `genes.npy`, `gene_outer_folds.npy`, and the current canonical manifest are an exact one-to-one match.

- [ ] **Step 6: Run cache and alignment tests**

Run: `rtk uv run python -m pytest tests/test_aivc_model.py -k "state_alignment or gwps_cache" -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add src/aivc_model/gwps_cache.py src/aivc_model/prepare.py scripts/build_exp05_gwps_cache.py tests/test_aivc_model.py
rtk git commit -m "fix: align exp05 GWPS inputs to STATE"
```

---

### Task 3: Replace Per-Gene Fallbacks with the Exp08 ESM-2 Adapter

**Files:**
- Modify: `src/aivc_model/prepare.py`
- Modify: `src/aivc_model/model.py`
- Modify: `src/aivc_model/train.py`
- Modify: `tests/test_aivc_model.py`

**Interfaces:**
- Consumes: `sl_dl_model.gene_embeddings.load_esm2_embeddings(Path) -> Esm2EmbeddingTable`.
- Consumes: `sl_dl_model.encoder.PertAdapter(esm_dim: int, hidden: int, pert_dim: int)`.
- Produces: `Esm2PerturbationAdapter(genes, table, adapter_hidden, pert_dim)` with `forward(gene) -> Tensor`, `has_embedding(gene) -> bool`, and compatibility alias `has_known_vector(gene) -> bool`.

- [ ] **Step 1: Write failing inductive-adapter tests**

```python
def test_esm2_perturbation_adapter_maps_all_genes_through_one_network() -> None:
    table = Esm2EmbeddingTable(
        dim=3,
        vectors_by_symbol={
            "KNOWN": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "HELDOUT": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        },
    )
    adapter = Esm2PerturbationAdapter(
        ["KNOWN", "HELDOUT"], table, adapter_hidden=4, pert_dim=2
    )
    assert adapter("KNOWN").shape == (2,)
    assert adapter("HELDOUT").shape == (2,)
    assert adapter("HELDOUT").requires_grad
    assert not hasattr(adapter, "missing_vectors")


def test_esm2_perturbation_adapter_rejects_unresolved_gene() -> None:
    table = Esm2EmbeddingTable(dim=3, vectors_by_symbol={})
    with pytest.raises(ValueError, match="UNRESOLVED"):
        Esm2PerturbationAdapter(
            ["UNRESOLVED"], table, adapter_hidden=4, pert_dim=2
        )
```

- [ ] **Step 2: Run adapter tests and confirm failure**

Run: `rtk uv run python -m pytest tests/test_aivc_model.py -k "esm2_perturbation_adapter" -v`

Expected: FAIL because `Esm2PerturbationAdapter` does not exist.

- [ ] **Step 3: Implement the adapter by reusing exp08 code**

```python
class Esm2PerturbationAdapter(nn.Module):
    def __init__(
        self,
        genes: list[str],
        table: Esm2EmbeddingTable,
        adapter_hidden: int,
        pert_dim: int,
    ) -> None:
        super().__init__()
        self.genes = [str(gene).upper() for gene in genes]
        missing = [gene for gene in self.genes if gene not in table.vectors_by_symbol]
        if missing:
            raise ValueError(f"Unresolved ESM-2 genes: {missing[:10]}")
        matrix = np.vstack([table.vectors_by_symbol[gene] for gene in self.genes])
        self._gene_to_index = {gene: index for index, gene in enumerate(self.genes)}
        self.register_buffer("esm_matrix", torch.as_tensor(matrix, dtype=torch.float32))
        self.adapter = PertAdapter(table.dim, int(adapter_hidden), int(pert_dim))

    def forward(self, gene: str) -> torch.Tensor:
        index = self._gene_to_index[str(gene).upper()]
        return self.adapter(self.esm_matrix[index].unsqueeze(0)).squeeze(0)

    def has_embedding(self, gene: str) -> bool:
        return str(gene).upper() in self._gene_to_index

    def has_known_vector(self, gene: str) -> bool:
        return self.has_embedding(gene)
```

Change the `AivcModel` constructor annotation to `PerturbationVectorAdapter | Esm2PerturbationAdapter`; its call sites remain unchanged.

- [ ] **Step 4: Parse strict ESM-2 config and reject incomplete coverage without filtering**

Add these `StateConfig` fields and parser entries:

```python
gene_tokenizer: str = "state_onehot"
esm2_npz: Path | None = None
esm2_adapter_hidden: int = 512
require_resolved_esm2: bool = False
```

For `gene_tokenizer: esm2`, first load and validate the canonical outer manifest, then load the NPZ and call `require_complete_esm_coverage(canonical_genes, table)`. Assert the canonical gene list is still exactly 9,338 rows before model construction. Never filter labels, bags, folds, or evaluation rows because of ESM coverage. Do not call `PerturbationVectorAdapter` on this path. If a separately declared Adamson target is missing, fail that external preflight explicitly without changing the internal universe.

Build the model with:

```python
esm = load_esm2_embeddings(config.state.esm2_npz)
canonical_genes = canonical_manifest["perturbation_gene"].tolist()
require_complete_esm_coverage(canonical_genes, esm)
perturbations = Esm2PerturbationAdapter(
    canonical_genes + sorted(set(extra_genes).difference(canonical_genes)),
    esm,
    adapter_hidden=config.state.esm2_adapter_hidden,
    pert_dim=pert_dim,
)
```

Keep the loaded STATE backbone frozen, while `perturbations.adapter`, the projector, and C head remain trainable. ESM vectors are fixed buffers and are not normalized using the full gene universe; any learned ESM normalization must be fold-local and fit on inner-train genes only.

- [ ] **Step 5: Verify gradient and coverage behavior**

Run: `rtk uv run python -m pytest tests/test_aivc_model.py -k "esm2 or freeze_state" -v`

Expected: PASS; a backward test confirms ESM adapter gradients are nonzero and STATE parameter gradients remain absent.

- [ ] **Step 6: Commit**

```bash
rtk git add src/aivc_model/prepare.py src/aivc_model/model.py src/aivc_model/train.py tests/test_aivc_model.py
rtk git commit -m "feat: tokenize exp05 perturbations with ESM2"
```

---

### Task 4: Repair Adamson Expression Alignment and Separate Evaluation Scopes

**Files:**
- Modify: `src/aivc_model/prepare.py`
- Modify: `src/aivc_model/train.py`
- Modify: `tests/test_aivc_model.py`

**Interfaces:**
- Produces: `ExternalSourceConfig.var_gene_symbol_col: str | None`; `None` means `adata.var_names`.
- Produces label-prediction scopes `internal_outer_test` and `external:adamson_k562`; Task 5 adds the two observed-response-only scopes.

- [ ] **Step 1: Write failing Adamson alignment tests**

```python
def test_external_var_names_align_to_state_symbols() -> None:
    adata = ad.AnnData(np.asarray([[1.0, 2.0]], dtype=np.float32))
    adata.var_names = ["A", "B"]
    assert _var_symbols(adata, None) == ["A", "B"]


def test_external_alignment_rejects_zero_matches(toy_reference: GeneBags) -> None:
    adata = ad.AnnData(np.ones((2, 2), dtype=np.float32))
    adata.var_names = ["X", "Y"]
    source = ExternalSourceConfig("adamson", Path("unused"), var_gene_symbol_col=None)
    with pytest.raises(ValueError, match="matched 0"):
        _external_state_input_view(adata, source, _toy_config(), toy_reference)
```

- [ ] **Step 2: Implement explicit `var_names` semantics and QA guard**

```python
def _var_symbols(adata: ad.AnnData, column: str | None) -> list[str]:
    if column is None:
        return adata.var_names.astype(str).tolist()
    if column not in adata.var.columns:
        raise ValueError(f"AnnData var is missing configured symbol column {column!r}")
    return adata.var[column].astype(str).tolist()
```

After alignment, raise when `matched == 0`. Preserve control-mean fill for the verified Adamson missing features and write `matched_input_features`, `missing_input_features`, and `matched_fraction` to QA. The new config sets every Adamson source to `var_gene_symbol_col: null`.

- [ ] **Step 3: Write failing evaluation-scope test**

```python
def test_fold_final_evaluation_keeps_internal_and_external_separate(tmp_path: Path) -> None:
    paths = run_training_fold(**_toy_fold_inputs(tmp_path))
    metrics = pd.read_csv(paths["test_metrics"])
    assert set(metrics["evaluation_scope"]) == {
        "internal_outer_test",
        "external:adamson_k562",
    }
    audit = pd.read_csv(paths["fit_access_audit"])
    assert audit.query(
        "stage == 'internal_outer_test' and reads_observed_response"
    ).empty
```

- [ ] **Step 4: Evaluate outer-test and Adamson independently**

Refactor the current single `eval_data/test_loader` branch into label-prediction requests that do not carry observed GWPS response bags:

```python
evaluation_sets = {
    "internal_outer_test": PredictionRequest(
        genes=fold_spec.test_genes,
        observed_response=None,
    ),
}
if external is not None:
    evaluation_sets[f"external:{config.external_test.name}"] = PredictionRequest(
        genes=tuple(str(gene) for gene in external.data.genes),
        observed_response=external.data,
    )
```

Prepare/evaluate one loader per entry, concatenate metrics and predictions, and keep best-epoch selection tied only to inner `val_loader`. `internal_outer_test` generates B from fold-neutral non-targeting controls plus the test-gene ESM token; it must not open the test-gene observed-response cache. Adamson cannot enter `_evaluate()` during epochs. Task 5 separately gates outer-test observed B behind the selected-checkpoint seal.

- [ ] **Step 5: Run external and final-evaluation tests**

Run: `rtk uv run python -m pytest tests/test_aivc_model.py -k "external or evaluation_scope" -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/aivc_model/prepare.py src/aivc_model/train.py tests/test_aivc_model.py
rtk git commit -m "fix: align Adamson and preserve outer test"
```

---

### Task 5: Enforce the Frozen Outer Split, Inner Validation, and Gene-Level Data Policy

**Files:**
- Modify: `src/aivc_model/gene_splits.py`
- Create: `src/aivc_model/cross_validate.py`
- Modify: `src/aivc_model/prepare.py`
- Modify: `src/aivc_model/train.py`
- Modify: `tests/test_exp05_gene_splits.py`
- Create: `tests/test_aivc_cross_validate.py`

**Interfaces:**
- Produces: `make_inner_fold_spec(manifest, labels, outer_fold, inner_val_fraction, seed) -> FoldSpec` without regenerating outer folds.
- Produces: `attach_gene_provenance(frame, manifest, gene_col, source_kind) -> pd.DataFrame`.
- Produces: `assert_gene_access(stage, genes, fold_spec, checkpoint_frozen) -> None`.
- Produces: `run_cross_validation(config_path: Path, accelerator: Accelerator | None = None) -> Path`.
- Produces per fold: `runs/<run_id>/fold_<k>/...`; aggregate outputs live in `runs/<run_id>/artifacts/`.

- [ ] **Step 1: Write frozen-outer and inner-validation tests**

```python
def test_inner_split_consumes_frozen_outer_assignment() -> None:
    labels, manifest = _toy_labels_and_manifest(50)
    fold = make_inner_fold_spec(manifest, labels, 0, 0.1, 42)
    expected_test = set(manifest.query("outer_fold == 0")["perturbation_gene"])
    assert set(fold.test_genes) == expected_test
    assert set(fold.train_genes).isdisjoint(fold.val_genes)
    assert set(fold.train_genes).isdisjoint(fold.test_genes)
    assert set(fold.val_genes).isdisjoint(fold.test_genes)
    assert set(fold.train_genes) | set(fold.val_genes) | set(fold.test_genes) == set(
        manifest["perturbation_gene"]
    )


def test_outer_test_labels_do_not_change_inner_split() -> None:
    labels, manifest = _toy_labels_and_manifest(50)
    first = make_inner_fold_spec(manifest, labels, 0, 0.1, 42)
    changed = labels.copy()
    changed.loc[changed["perturbation_gene"].isin(first.test_genes), "depmap_gene_effect"] = 999.0
    second = make_inner_fold_spec(manifest, changed, 0, 0.1, 42)
    assert first.train_genes == second.train_genes
    assert first.val_genes == second.val_genes
```

- [ ] **Step 2: Implement inner splitting without an outer splitter**

```python
@dataclass(frozen=True)
class FoldSpec:
    outer_fold: int
    train_genes: tuple[str, ...]
    val_genes: tuple[str, ...]
    test_genes: tuple[str, ...]


def make_inner_fold_spec(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    outer_fold: int,
    inner_val_fraction: float,
    seed: int,
) -> FoldSpec:
    label_by_gene = labels.set_index("perturbation_gene")["depmap_gene_effect"]
    test_genes = sorted(
        manifest.loc[manifest["outer_fold"] == outer_fold, "perturbation_gene"]
    )
    outer_train_genes = sorted(
        manifest.loc[manifest["outer_fold"] != outer_fold, "perturbation_gene"]
    )
    y = label_by_gene.loc[outer_train_genes].to_numpy()
    train_genes, val_genes = train_test_split(
        outer_train_genes,
        test_size=inner_val_fraction,
        random_state=seed + outer_fold + 1,
        stratify=quantile_strata(y, 5),
    )
    return FoldSpec(
        outer_fold=outer_fold,
        train_genes=tuple(sorted(train_genes)),
        val_genes=tuple(sorted(val_genes)),
        test_genes=tuple(test_genes),
    )
```

No runtime path may instantiate an outer `KFold`/`StratifiedKFold`. Only Task 1 creates the outer assignment; Task 5 reads it.

- [ ] **Step 3: Write provenance and access-policy tests**

```python
@pytest.mark.parametrize(
    "source_kind",
    ["gene_effect", "gwps_response", "transition", "prompt", "fine_tuning"],
)
def test_every_gene_derived_row_inherits_canonical_outer_fold(source_kind: str) -> None:
    manifest = pd.DataFrame(
        {"perturbation_gene": ["A", "B"], "outer_fold": [0, 1]}
    )
    rows = pd.DataFrame({"gene": ["A", "A", "B"]})
    result = attach_gene_provenance(rows, manifest, "gene", source_kind)
    assert result[["perturbation_gene", "outer_fold"]].to_dict("records") == [
        {"perturbation_gene": "A", "outer_fold": 0},
        {"perturbation_gene": "A", "outer_fold": 0},
        {"perturbation_gene": "B", "outer_fold": 1},
    ]


@pytest.mark.parametrize(
    "stage",
    [
        "adapter_fit",
        "state_fit",
        "scvi_fit",
        "gmm_fit",
        "normalizer_fit",
        "projector_fit",
        "transition_supervision",
        "gene_prompt_fit",
        "fine_tuning",
    ],
)
def test_outer_test_gene_is_rejected_from_every_fit_stage(stage: str) -> None:
    fold = _toy_fold_spec()
    with pytest.raises(ValueError, match="outer-test"):
        assert_gene_access(stage, [fold.test_genes[0]], fold, checkpoint_frozen=False)


def test_outer_test_response_has_only_two_post_freeze_routes() -> None:
    fold = _toy_fold_spec()
    gene = fold.test_genes[0]
    for stage in ("generation_quality_outer_test", "observed_b_oracle_outer_test"):
        assert_gene_access(stage, [gene], fold, checkpoint_frozen=True)
    with pytest.raises(ValueError, match="selected checkpoint is frozen"):
        assert_gene_access(
            "generation_quality_outer_test", [gene], fold, checkpoint_frozen=False
        )
```

- [ ] **Step 4: Implement the provenance join and stage allowlist**

```python
FIT_STAGES = frozenset(
    {
        "adapter_fit",
        "state_fit",
        "scvi_fit",
        "gmm_fit",
        "normalizer_fit",
        "projector_fit",
        "transition_supervision",
        "gene_prompt_fit",
        "fine_tuning",
    }
)
SELECTION_STAGES = frozenset({"early_stopping", "layer_selection"})
FINAL_RESPONSE_STAGES = frozenset(
    {"generation_quality_outer_test", "observed_b_oracle_outer_test"}
)


def assert_gene_access(
    stage: str,
    genes: Collection[str],
    fold: FoldSpec,
    checkpoint_frozen: bool,
) -> None:
    requested = {str(gene).upper() for gene in genes}
    test = set(fold.test_genes)
    if stage in FIT_STAGES and not requested <= set(fold.train_genes):
        raise ValueError(f"{stage} attempted outer-test or validation gene access")
    if stage in SELECTION_STAGES and not requested <= set(fold.val_genes):
        raise ValueError(f"{stage} must use inner-validation genes only")
    if stage in FINAL_RESPONSE_STAGES:
        if not checkpoint_frozen:
            raise ValueError("selected checkpoint is frozen before outer-test response access")
        if not requested <= test:
            raise ValueError(f"{stage} accepts outer-test genes only")
    if stage not in FIT_STAGES | SELECTION_STAGES | FINAL_RESPONSE_STAGES:
        raise ValueError(f"unknown gene-access stage {stage!r}")
```

`attach_gene_provenance` uppercases the source gene, performs a `many_to_one` merge against the canonical manifest, rejects missing genes and any pre-existing conflicting `outer_fold`, and appends `source_kind`. Transition builders, gene-derived prompt builders, and fine-tuning dataset constructors must call it before sampling. Non-targeting controls use a separate `ControlPromptPool` with `source_kind="non_targeting_control"`, `perturbation_gene=None`, and `outer_fold=None`; no perturbed response cell can enter that pool.

- [ ] **Step 5: Expose one-fold training through audited gene views**

```python
def run_training_fold(
    config: AivcConfig,
    data: GeneBags,
    external: ExternalGeneBags | None,
    fold_spec: FoldSpec,
    run_dir: Path,
    source_fingerprint: str,
    accelerator: Accelerator | None = None,
) -> dict[str, Path]:
    train_data = data.for_genes(fold_spec.train_genes, stage="fine_tuning")
    val_data = data.for_genes(fold_spec.val_genes, stage="early_stopping")
    sealed_test = SealedGeneBags(data, fold_spec.test_genes)
    return run_training(
        config,
        accelerator=accelerator,
        train_data=train_data,
        val_data=val_data,
        sealed_test=sealed_test,
        external_override=external,
        fold_spec=fold_spec,
        run_dir_override=run_dir,
        source_fingerprint=source_fingerprint,
    )
```

Remove the generic `split_override` path for this config: it is too easy for downstream code to receive all bags and index them incorrectly. Every fit function receives `train_data` only. Every early-stopping/layer-selection function receives `val_data` only. `SealedGeneBags.open(stage, checkpoint_frozen=True)` supports exactly the two final-response stage names. STATE parameters remain `requires_grad=False`; `state_fit` exists only as a guard and must never be called.

- [ ] **Step 6: Implement fold-local fit and final evaluation order**

For each outer fold, execute exactly:

```text
load frozen outer manifest and verify SHA-256
derive inner-train / inner-validation from outer-train genes
fit normalizer, scVI, projector, GMM, ESM adapter, and C head on inner-train only
evaluate inner validation for early stopping and optional layer selection
restore selected checkpoint and mark it immutable
open outer-test response for generation_quality_outer_test
apply an observed-B oracle fitted on outer-train data to outer-test observed B
evaluate label predictions and Adamson without further fitting
```

The observed-B oracle may use observed response bags and GeneEffect labels from `train_genes`, and `val_genes` for selection. It must be restored/frozen before `test_genes` responses are opened. Generation-quality metrics compare the frozen model's predicted response with outer-test observed response and cannot update parameters, batch statistics, normalization statistics, thresholds, selected layers, or epochs.

- [ ] **Step 7: Write mutation-based end-to-end leakage regression**

```python
def test_changing_outer_test_responses_cannot_change_fitted_artifacts(
    tmp_path: Path,
) -> None:
    first = run_training_fold(**_toy_fold_inputs(tmp_path / "first"))
    changed = _toy_fold_inputs(tmp_path / "changed")
    changed["data"] = changed["data"].replace_test_responses(value=999.0)
    second = run_training_fold(**changed)
    first_audit = json.loads(first["fit_audit_summary"].read_text())
    second_audit = json.loads(second["fit_audit_summary"].read_text())
    for key in (
        "adapter_sha256",
        "state_sha256",
        "scvi_sha256",
        "gmm_sha256",
        "normalizer_sha256",
        "projector_sha256",
        "selected_layer",
        "best_epoch",
        "checkpoint_sha256",
    ):
        assert first_audit[key] == second_audit[key]
```

The test may observe changed generation-quality/oracle metrics; those are final outputs and are expected to depend on the altered test response. It must also assert the fit audit contains zero outer-test gene IDs for every fit/selection event.

- [ ] **Step 8: Write aggregation tests for all three internal scopes**

```python
def test_cross_validation_writes_each_outer_test_gene_once_per_final_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv, "run_training_fold", _fake_fold_runner)
    run_dir = cv.run_cross_validation(_toy_cv_config(tmp_path))
    predictions = pd.read_csv(run_dir / "artifacts" / "predictions.csv")
    for scope in (
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_oracle_outer_test",
    ):
        rows = predictions.query("evaluation_scope == @scope")
        assert rows["perturbation_gene"].nunique() == 9338
        assert not rows.duplicated(["perturbation_gene", "evaluation_scope"]).any()
```

- [ ] **Step 9: Implement aggregate artifacts and leakage assertions**

Write:

- `artifacts/fold_metrics.csv`: separate rows for internal GeneEffect prediction, generation quality, observed-B oracle, and Adamson.
- `artifacts/predictions.csv`: all fold outputs with `outer_fold`, `inner_role`, and `evaluation_scope`.
- `artifacts/gene_splits.csv`: the canonical `outer_fold` plus each fold's derived inner role; its canonical columns/hash must match the pre-frozen manifest.
- `artifacts/fit_access_audit.csv`: every fit/selection/final-read event with stage, fold, gene-count, gene-set SHA-256, and checkpoint-frozen flag.
- `artifacts/external_alignment_qa.csv`: the three Adamson source QA rows per fold.
- `summary.csv`: mean/std over the five outer-test folds, with the three internal scopes kept separate.
- `run_manifest.json`: canonical split path/SHA-256, `9338/9338` ESM coverage, source fingerprint, fold seeds, checkpoint dimensions, exact feature matches, and artifact paths.

Before aggregation, assert that the canonical universe is exactly 9,338 unique genes, each gene is outer-test exactly once, every derived row's `outer_fold` matches the canonical manifest, no fit audit contains an outer-test gene, no selection audit contains a non-validation gene, and outer-test response reads use only the two allowed post-freeze stages.

- [ ] **Step 10: Run CV tests**

Run: `rtk uv run python -m pytest tests/test_exp05_gene_splits.py tests/test_aivc_cross_validate.py -v`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
rtk git add src/aivc_model/gene_splits.py src/aivc_model/cross_validate.py src/aivc_model/prepare.py src/aivc_model/train.py tests/test_exp05_gene_splits.py tests/test_aivc_cross_validate.py
rtk git commit -m "feat: enforce exp05 gene-level CV protocol"
```

---

### Task 6: Bind Fold Caches and Checkpoints to the Source Fingerprint

**Files:**
- Modify: `src/aivc_model/prepare.py`
- Modify: `src/aivc_model/train.py`
- Modify: `src/aivc_model/cross_validate.py`
- Modify: `tests/test_aivc_model.py`
- Modify: `tests/test_aivc_cross_validate.py`

**Interfaces:**
- Consumes: the `source_fingerprint` generated by `gwps_cache.py` plus label/ESM signatures.
- Produces: every normalizer/scVI/projector/GMM/adapter/C-head/oracle/checkpoint metadata file contains `source_fingerprint`, canonical split SHA-256, outer fold, fit stage, and exact fit-gene SHA-256.

- [ ] **Step 1: Write stale-cache rejection tests**

```python
def test_projector_cache_rejected_when_source_fingerprint_changes(tmp_path: Path) -> None:
    first = _projector_cache_metadata(config, data, split, source_fingerprint="aaa")
    _write_projector_cache(tmp_path, first, np.eye(2), np.zeros(2))
    second = _projector_cache_metadata(config, data, split, source_fingerprint="bbb")
    assert _load_projector_cache(tmp_path, second) is None


def test_fold_manifest_changes_when_esm_cache_changes(tmp_path: Path) -> None:
    config = _toy_config_with_esm(tmp_path)
    first = experiment_source_fingerprint(config)
    config.state.esm2_npz.write_bytes(b"new-cache")
    second = experiment_source_fingerprint(config)
    assert first != second


def test_fitted_cache_rejected_when_canonical_split_or_fit_genes_change(
    tmp_path: Path,
) -> None:
    first = _scvi_cache_metadata(
        config, fold_spec, canonical_split_sha256="aaa", fit_genes=("A", "B")
    )
    second = _scvi_cache_metadata(
        config, fold_spec, canonical_split_sha256="bbb", fit_genes=("A", "B")
    )
    third = _scvi_cache_metadata(
        config, fold_spec, canonical_split_sha256="aaa", fit_genes=("A", "C")
    )
    assert first != second
    assert first != third
```

- [ ] **Step 2: Extend all cache metadata**

Add the same metadata contract to the normalizer, scVI, projector, fixed GMM, ESM adapter, C head, observed-B oracle, and model checkpoint. Bump their schema versions. Each fitted artifact receives `fold_spec.train_genes` directly; it cannot infer fit genes from the full `GeneBags`. Model checkpoint metadata must include:

```python
{
    "source_fingerprint": source_fingerprint,
    "canonical_split_sha256": canonical_split_sha256,
    "outer_fold": fold_spec.outer_fold,
    "fit_stage": "inner_train",
    "fit_genes_sha256": sha256_strings(fold_spec.train_genes),
    "train_genes": list(fold_spec.train_genes),
    "val_genes": list(fold_spec.val_genes),
    "test_genes": list(fold_spec.test_genes),
    "selected_layer": selected_layer,
    "best_epoch": best_epoch,
    "state_checkpoint": str(config.state.checkpoint_path),
    "esm2_npz": str(config.state.esm2_npz),
}
```

`experiment_source_fingerprint` hashes the GWPS cache manifest, full label CSV bytes, canonical outer-manifest bytes and SHA-256 file, ESM NPZ `{path,size,mtime_ns}`, checkpoint `{path,size,mtime_ns}`, and full bytes of every small STATE sidecar. A mismatch causes a rebuild for every fitted fold-local artifact and prevents a best/final checkpoint from being resumable. Cache loading additionally rejects metadata containing any test gene in `train_genes` or any non-validation gene in a selection-gene field.

- [ ] **Step 3: Run cache-fingerprint tests**

Run: `rtk uv run python -m pytest tests/test_aivc_model.py tests/test_aivc_cross_validate.py -k "fingerprint or cache" -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
rtk git add src/aivc_model/prepare.py src/aivc_model/train.py src/aivc_model/cross_validate.py tests/test_aivc_model.py tests/test_aivc_cross_validate.py
rtk git commit -m "fix: fingerprint exp05 fold caches"
```

---

### Task 7: Add the Single Repaired Config, Remote Preflight, and Documentation

**Files:**
- Create: `configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml`
- Modify: `scripts/state.sh`
- Modify: `docs/experiment/05_aivc_a_to_b_to_c.md`
- Modify: `tests/test_aivc_cross_validate.py`

**Interfaces:**
- Produces CLI: `uv run python -m aivc_model.cross_validate --config <yaml>`.
- Produces one authoritative exp05 repaired run configuration.

- [ ] **Step 1: Write config-contract test**

```python
def test_exp05_repaired_config_has_locked_contract() -> None:
    path = Path(
        "configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml"
    )
    config = load_config(path)
    assert config.data.h5ad_path.name == "K562_gwps_normalized_singlecell_01.h5ad"
    assert config.data.var_gene_symbol_col == "gene_name"
    assert config.state.gene_tokenizer == "esm2"
    assert config.state.esm2_npz.name == "k562_gwps_depmap_esm2_650M.npz"
    assert config.state.require_resolved_esm2 is True
    assert config.train.freeze_state is True
    assert config.cv.n_splits == 5
    assert config.cv.expected_gene_count == 9338
    assert config.cv.outer_split_manifest.name == "k562_gwps_depmap_outer5_seed42.csv"
    assert config.cv.outer_split_sha256_file.name == (
        "k562_gwps_depmap_outer5_seed42.csv.sha256"
    )
    assert config.cv.inner_val_fraction == 0.1
    assert all(source.var_gene_symbol_col is None for source in config.external_test.sources)
```

- [ ] **Step 2: Create the locked config**

The YAML must use these result-affecting values:

```yaml
data:
  h5ad_path: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
  overlap_csv: data/sl_dependency_v0/interim/k562_gwps_depmap_overlap.csv
  prepared_cache_dir: data/exp05_cache/k562_gwps_state2000
  output_dir: results/experiments/05_aivc_a_to_b_to_c
  obs_perturbation_col: gene
  control_label: non-targeting
  obs_batch_col: gem_group
  var_gene_symbol_col: gene_name
  state_embed_key: X_hvg
  state_hvg_n_top_genes: null
  depmap_label_col: depmap_gene_effect
  matched_label_col: has_depmap_label
  min_cells_per_gene: 8
  cache_seed: 42
  cache_cells_per_gene: 256

cv:
  n_splits: 5
  expected_gene_count: 9338
  outer_split_manifest: data/sl_dependency_v0/splits/k562_gwps_depmap_outer5_seed42.csv
  outer_split_sha256_file: data/sl_dependency_v0/splits/k562_gwps_depmap_outer5_seed42.csv.sha256
  inner_val_fraction: 0.1
  random_state: 42
  stratify_bins: 10

state:
  backend: state_checkpoint
  model_dir: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562
  checkpoint_path: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt
  gene_tokenizer: esm2
  esm2_npz: data/esm2/k562_gwps_depmap_esm2_650M.npz
  esm2_adapter_hidden: 512
  require_resolved_esm2: true
```

The digest file must contain exactly one lowercase 64-character SHA-256 followed by a newline; the loader verifies it before reading any fold data. Copy the existing ranknet/freeze-state projector, GMM, loss, model, and train values unchanged except set a new `run_id: state_esm2_gwps_5fold`. Lock the STATE representation layer explicitly in config; if the existing pipeline instead searches layers, route that search exclusively through inner validation and record the selected layer in fold metadata. Copy the three Adamson sources unchanged except set `var_gene_symbol_col: null`.

- [ ] **Step 3: Make the Slurm wrapper config-explicit**

At the top of `scripts/state.sh`, set:

```bash
CONFIG_PATH="${CONFIG_PATH:-configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml}"
```

Run the new module with `--config "$CONFIG_PATH"`; do not change resource directives in this task.

- [ ] **Step 4: Document exact claim boundaries and artifacts**

Update the exp05 document to state:

- Old exp05 negative results are invalid as model evidence because primary/external expression coordinates were misaligned.
- The repaired experiment predicts population-level K562 GeneEffect, not cell death probability or mechanism.
- The primary result is the mean/std across five `internal_outer_test` folds.
- Adamson is a secondary assay-transfer evaluation and never participates in epoch selection.
- The universe is exactly the pre-frozen 9,338-gene GWPS-DepMap overlap; its canonical outer-fold manifest and SHA-256 are reported with every run.
- GeneEffect labels, GWPS responses, transitions, gene-derived prompts, and fine-tuning samples inherit the same gene-level outer fold.
- ESM-2 must resolve `9338/9338`; unresolved symbols fail preflight and are never excluded after splitting.
- Outer-test observed response is read only after checkpoint freeze for generation quality and the train-fitted observed-B oracle, never for adapter/STATE/scVI/GMM/normalizer fitting, early stopping, or layer selection.
- STATE expression remains fixed at 2,000 genes.

- [ ] **Step 5: Run all local verification**

```bash
rtk uv run python -m pytest tests/test_build_exp05_gwps_labels.py tests/test_exp05_gene_splits.py tests/test_aivc_model.py tests/test_aivc_cross_validate.py tests/sl_dl_model/test_precompute_esm2.py -v
rtk uv run ruff check src/aivc_model scripts/build_exp05_gwps_labels.py scripts/build_exp05_gene_splits.py scripts/build_exp05_gwps_cache.py scripts/precompute_esm2_embeddings.py tests/test_build_exp05_gwps_labels.py tests/test_exp05_gene_splits.py tests/test_aivc_model.py tests/test_aivc_cross_validate.py tests/sl_dl_model/test_precompute_esm2.py
rtk uv run ruff format --check src/aivc_model scripts/build_exp05_gwps_labels.py scripts/build_exp05_gene_splits.py scripts/build_exp05_gwps_cache.py scripts/precompute_esm2_embeddings.py tests/test_build_exp05_gwps_labels.py tests/test_exp05_gene_splits.py tests/test_aivc_model.py tests/test_aivc_cross_validate.py tests/sl_dl_model/test_precompute_esm2.py
```

Expected: all focused tests pass; Ruff check and format check exit 0.

- [ ] **Step 6: Run read-only remote preflight after assets and before training**

```bash
rtk proxy ssh -o ControlMaster=no wangar2023@10.15.89.192 "cd /public/home/wangar2023/VCC_Project && uv run python -m aivc_model.cross_validate --config configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml --preflight-only"
```

Expected preflight fields:

```text
gwps_shape=1989578x8248
gwps_noncontrol_genes=9866
gwps_depmap_overlap=9338
canonical_split_genes=9338
canonical_split_folds=5
canonical_split_sha256_length=64
esm2_resolved=9338/9338
state_expression_matches=2000/2000
state_input_dim=2000
state_output_dim=2000
state_pert_dim=2024
adamson_pilot_matches=1876/2000
adamson_upr_epistasis_matches=1874/2000
adamson_upr_perturb_seq_matches=1874/2000
```

Preflight must compare the label table, canonical manifest, GWPS cache gene list, and ESM table and assert exact set equality over 9,338 genes. It fails on any missing/extra gene, duplicate, fold reassignment, SHA-256 mismatch, or ESM coverage below `9338/9338`; there is no post-filter universe.

- [ ] **Step 7: Commit**

```bash
rtk git add configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml scripts/state.sh docs/experiment/05_aivc_a_to_b_to_c.md tests/test_aivc_cross_validate.py
rtk git commit -m "docs: lock repaired exp05 STATE ESM2 run"
```

---

## Execution Order on the Remote Host

After Tasks 1-7 pass locally and the code is synchronized, create assets in this order:

```bash
rtk uv run python scripts/build_exp05_gwps_labels.py
rtk uv run python scripts/build_exp05_gene_splits.py
rtk uv run python scripts/precompute_esm2_embeddings.py --benchmark-csv data/sl_dependency_v0/interim/k562_gwps_depmap_overlap.csv --symbol-column perturbation_gene --out data/esm2/k562_gwps_depmap_esm2_650M.npz --seq-cache data/esm2/symbol_to_sequence.json --local-files-only
rtk uv run python scripts/build_exp05_gwps_cache.py --config configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml
rtk uv run python -m aivc_model.cross_validate --config configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml --preflight-only
rtk sbatch scripts/state.sh
```

If missing UniProt sequences are not already present in `symbol_to_sequence.json`, run the ESM command once on a network-enabled node without `--local-files-only`, then rerun it offline. Do not start CV until preflight confirms exact `9338/9338` canonical-set equality, exact split SHA-256, zero unresolved canonical genes, and all five fold leakage guards. Never construct a smaller post-filter universe.

## Self-Review Results

- **Spec coverage:** All ten reported issues and the strict protocol are covered: the canonical 9,338-gene outer split is frozen before downstream assets (Task 1); every gene-derived label/response/transition/prompt/fine-tuning row inherits it (Tasks 2/5); inner validation is outer-train-only (Task 5); outer-test observed response has exactly two post-freeze exits and cannot affect adapter/STATE/scVI/GMM/normalizer/early stopping/layer selection (Task 5); the remaining alignment, ESM, cache, Adamson, and GeneEffect claim requirements are covered by Tasks 1-7.
- **Scope control:** No SL pair head, pair loss, CV1/CV2/CV3 pair split, predicted-transcriptome comparator, STATE checkpoint swap, wider expression checkpoint, or death-mechanism claim is introduced.
- **Placeholder scan:** The plan contains no deferred implementation markers; every task names files, interfaces, tests, commands, expected results, and commit boundaries.
- **Type consistency:** `FoldSpec`, `run_training_fold`, `Esm2PerturbationAdapter`, `resolve_state_gene_order`, and `source_fingerprint` names/signatures are consistent across producer and consumer tasks.
