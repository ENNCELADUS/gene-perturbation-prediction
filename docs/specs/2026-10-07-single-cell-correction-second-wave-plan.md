# Single-Cell Correction, Second Wave: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the single-cell inputs the correction may read, each behind its own config switch so it can be compared on its own against the wave-one winner. The inputs:
- the fraction of cells expressing g's partners;
- STATE's Δ as program scores;
- Tirosh and Gavish state views;
- copy number inferred from single cells;
- the `C` variants: off, linear, over the views, and cross-attention.

**Architecture:** Every input is a deterministic function of a line's basal cells.
- **Per-(g, c) inputs** become new `h` blocks:
  - `partners`: gathers from the existing `q_sc` cache plus the pinned paralog and CORUM tables;
  - `copy_number`: local copy number, smoothed from the `q_sc` log-means along the genome.
- **Per-line inputs** become a state-view vector (cell-cycle and meta-program fractions, arm-level copy number and its spread across cells, aneuploidy).
  - It comes from per-cell scores that a new preparation step writes from raw cells.
  - Thresholds and references are fitted on the labelled training lines in `load_inputs` and stored with the checkpoint.
  - The vector joins `C`'s input, or forms one token per view for cross-attention.
- **Program-score Δ** replaces the random projection of STATE's Δ when `model.delta_projection` is `programs`.

**Tech Stack:** as wave one (PyTorch, Accelerate, NumPy, pandas, pytest).

**Spec:** `docs/specs/2026-10-04-context-generalization-design.md` §3.3, §4 (correction features), §5.2 and §12 (amendments of 2026-10-07). Wave one: `docs/specs/2026-10-07-single-cell-correction-plan.md`.

## When this plan runs

- **Building.** The code is built while wave one's runs train, after wave one's Tasks 1–6 are merged.
- **Refresh first.** Before execution, refresh every "Modify" line reference against the code wave one leaves. The interfaces below assume wave one's `DependencyBatch.prior`, `PreparedInputs.prior` and the zero-initialised `C`.
- **Launching.** Nothing in this plan launches before the user has reviewed wave one. If STATE absent won wave one, the research plan is discussed first. Task 7's program-score Δ runs only if a STATE setting won.

## Global Constraints

All of wave one's Global Constraints apply. Also:
- **New config keys go into every joint YAML:**
  - `model.head_blocks`: `use_partners`, `use_copy_number`, `use_state_views`, all `false` in existing configs;
  - `model.delta_projection`: `random` in existing configs;
  - `model.context_fusion`: `inner_product` in existing configs;
  - `model.context_tower`: `linear_swiglu` in existing configs;
  - `paths.gene_reference`: `configs/context_prior/reference`.
- **Existing configs keep their behaviour exactly.** A checkpoint trained before this plan does not load after it (new architecture keys), as before.
- **Preparation is the only raw-data reader.** The per-cell scores are fixed caches. Everything fitted (thresholds, references, standardisation) is fitted on the labelled training lines in `load_inputs` and restored from the checkpoint.
- **Kinker programs are not used.** Gavish meta-programs must come from tumour samples; the pinning task records the check.

## Review Focus

1. **A training-line threshold leaking validation cells.** If thresholds or the copy-number reference pool validation or test cells, the views are fitted on held-out lines with no error. Expect fitted state to depend on supervised training lines only. Test: `test_state_view_fit_ignores_validation_and_test_cells` (Task 4).
2. **Sources with different gene vocabularies.** If an arm's mean or a program score covers different genes per source, it measures the source, not the line. Expect arm means over genes every source measures, and program scores over the source's measured members with a per-line coverage check. Test: `test_arm_means_use_only_genes_every_source_measures` (Task 4).
3. **A gene without a paralog, complex or genome position.** These must be masked, not zero-filled as data. Expect mask bits. Tests: `test_partner_fractions_mask_genes_without_partners` (Task 2) and `test_local_copy_number_masks_genes_without_position` (Task 5).
4. **Cross-attention not starting at zero.** If it outputs non-zero at init, the stack no longer equals the prior before the first update, and wave one's pre-update validation reports the wrong "prior alone". Expect zero output at init. Test: `test_cross_attention_starts_at_zero` (Task 6).
5. **Program projection rows with too few HVG members.** A program with under 10 member HVGs is noise. Expect it dropped and the width recorded. Test: `test_program_projection_keeps_programs_with_ten_hvg_members` (Task 3).

---

## File structure

| File | Change | Responsibility |
| --- | --- | --- |
| `src/data/prepare/build_correction_reference.py` | create | Pin the Tirosh cell-cycle sets, Gavish meta-programs, gene positions and chromosome arms into `configs/context_prior/reference/` |
| `src/data/gene_reference.py` | create | Read the pinned gene tables on the joint side (paralogs, complexes, hallmark, PROGENy, cell cycle, meta-programs, positions) |
| `src/data/partners.py` | create | Partner fractions per line from `q_sc` |
| `src/data/copy_number.py` | create | Local copy number per (g, c) from `q_sc` log-means; arm means per cell |
| `src/data/state_views.py` | create | Per-cell program scores and arm means (preparation); fitted thresholds, references and the per-line view vector (`load_inputs`) |
| `src/experiments/prepare.py` | modify | `prepare_state_views`: the raw-reading step |
| `src/model/features.py` | modify | `ProgramProjection` beside `FixedSparseProjection` |
| `src/model/head.py` | modify | `partners` and `copy_number` blocks; `C` variants; `ContextCrossAttention` |
| `src/model/geneeffect.py`, `src/model/normalization.py`, `src/model/initialization.py` | modify | Wire the new blocks, projection and views |
| `src/data/prepared.py`, `src/data/batches.py`, `src/data/datasets.py` | modify | Carry the new per-row and per-line inputs |
| `src/experiments/config.py` | modify | New keys |
| `configs/correction/second_wave/*.yaml` | create | One config per input, from the wave-one winner |

---

### Task 1: Pin the correction's gene tables (Mac, internet)

**Files:**
- Create: `src/data/prepare/build_correction_reference.py`, `tests/test_build_correction_reference.py`
- Create (output): `configs/context_prior/reference/{cell_cycle.tsv, meta_programs.tsv, gene_positions.tsv, arms.tsv}`; `provenance.json` gains their entries.

**Interfaces:**
- **Produces:**
  - `cell_cycle.tsv` (`phase` ∈ {S, G2M}, `gene`);
  - `meta_programs.tsv` (`program`, `gene`), 41 Gavish consensus meta-programs;
  - `gene_positions.tsv` (`gene`, `chromosome`, `start`), GRCh38, HGNC symbols, autosomes and X;
  - `arms.tsv` (`arm`, `chromosome`, `start`, `end`): the 39 arms of Cohen-Sharir 2021, i.e. autosomal p and q arms without 13p, 14p, 15p, 21p, 22p, plus Xp and Xq; arm borders from UCSC `cytoBand` `acen` bands.
- **Pure helpers:**
  - `arms_from_cytobands(cytobands: pd.DataFrame) -> pd.DataFrame`;
  - `assign_arms(positions: pd.DataFrame, arms: pd.DataFrame) -> pd.Series`.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from src.data.prepare.build_correction_reference import arms_from_cytobands, assign_arms


def test_arms_split_at_the_centromere_and_drop_acrocentric_p_arms():
    bands = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr13", "chr13", "chr13"],
            "start": [0, 120, 125, 0, 16, 18],
            "end": [120, 125, 248, 16, 18, 114],
            "stain": ["gneg", "acen", "gneg", "gvar", "acen", "gneg"],
        }
    )
    arms = arms_from_cytobands(bands)
    assert list(arms["arm"]) == ["1p", "1q", "13q"]
    assert arms.set_index("arm").loc["1q", "start"] == 125
    positions = pd.DataFrame(
        {"gene": ["A", "B", "C"], "chromosome": ["1", "1", "13"], "start": [5, 200, 10]}
    )
    assert list(assign_arms(positions, arms)) == ["1p", "1q", None]
```

- [ ] **Step 2: Implement the builder**
  - Model the builder on `build_context_reference.py`: fetch, normalise, write TSV, record the source URL, row count and sha256 in `provenance.json`.
  - **Tirosh cell cycle:** from Tirosh et al. 2016 (Science 352:189), Table S5, the S and G2/M gene lists (43 and 54 genes).
  - **Gavish meta-programs:** from Gavish et al. 2023 (Nature 618:598), Supplementary Table 3, the 41 consensus meta-programs (50 genes each).
    - Read the paper's methods for the tumour samples the programs were derived from.
    - Record in `provenance.json["checks"]["gavish_samples"]` the sentence that states the samples are tumours, with page and section.
    - If any CCLE cell line contributed, stop and report: the user ruled out Kinker for exactly this exposure.
  - **Gene positions:** Ensembl BioMart (`hgnc_symbol`, `chromosome_name`, `start_position`; GRCh38), keeping chromosomes 1–22 and X and the first row per symbol.
  - **Cytobands:** `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz`.

```python
ACROCENTRIC_P = {"13", "14", "15", "21", "22"}
CHROMOSOMES = [*map(str, range(1, 23)), "X"]


def arms_from_cytobands(cytobands: pd.DataFrame) -> pd.DataFrame:
    """The 39 arms: each chromosome split at its centromere (``acen`` bands)."""
    rows = []
    for chromosome in CHROMOSOMES:
        bands = cytobands.loc[cytobands["chrom"] == f"chr{chromosome}"]
        if bands.empty:
            continue
        centromere = bands.loc[bands["stain"] == "acen"]
        middle_start, middle_end = centromere["start"].min(), centromere["end"].max()
        if chromosome not in ACROCENTRIC_P:
            rows.append((f"{chromosome}p", chromosome, int(bands["start"].min()), int(middle_start)))
        rows.append((f"{chromosome}q", chromosome, int(middle_end), int(bands["end"].max())))
    return pd.DataFrame(rows, columns=["arm", "chromosome", "start", "end"])


def assign_arms(positions: pd.DataFrame, arms: pd.DataFrame) -> pd.Series:
    """Each gene's arm, or None on an unlisted arm (an acrocentric p arm, Y, MT)."""
    out = []
    for chromosome, start in zip(positions["chromosome"], positions["start"]):
        hit = arms.loc[
            (arms["chromosome"] == str(chromosome))
            & (arms["start"] <= start)
            & (start < arms["end"])
        ]
        out.append(hit["arm"].iloc[0] if len(hit) else None)
    return pd.Series(out, index=positions.index)
```

`gene_positions.tsv` gains an `arm` column from `assign_arms`.

- [ ] **Step 3: Run the test, run the builder, inspect row counts, commit**

```bash
uv run python -m pytest tests/test_build_correction_reference.py -q > /tmp/pytest.txt 2>&1; tail -3 /tmp/pytest.txt
uv run python -m src.data.prepare.build_correction_reference --out configs/context_prior/reference
git add src/data/prepare/build_correction_reference.py tests/test_build_correction_reference.py configs/context_prior/reference
git commit -m "feat(correction): pin cell-cycle sets, meta-programs, gene positions and arms" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected counts: 43 S + 54 G2M; 41 × 50 meta-program rows; about 19k positioned symbols; 39 arms.

---

### Task 2: Partner fractions as an `h` block

**Files:**
- Create: `src/data/gene_reference.py`, `src/data/partners.py`, `tests/test_partners.py`
- Modify:
  - `src/model/head.py`: `GeneEffectFeatureDims.partners = 2`, `GeneEffectBlockConfig.use_partners = False`, `masked_blocks`, `CORRECTION_ENCODER_WIDTHS["partners"] = 16`;
  - `src/model/normalization.py`: add `"partners"` to `STANDARDIZED_BLOCKS`;
  - `src/data/batches.py`: `FeatureBatch` and `OnlineConditionBatch` gain `partners` and `partners_mask`;
  - `src/data/datasets.py`;
  - `src/model/geneeffect.py`: `forward_features`, `condition_features`;
  - `src/experiments/config.py`: `_HEAD_BLOCKS` gains `use_partners`; `paths` gains `gene_reference`;
  - `src/data/prepared.py`: `PreparedInputs.gene_reference`;
  - every joint YAML.

**Definition.** For gene g in line c:
1. the detected fraction (`q_sc` channel 1) of g's closest paralog, the highest-identity row of `paralogs.tsv.gz` whose paralog is in the panel;
2. the mean detected fraction of g's CORUM co-members, over every complex containing g, counting each co-member once.

Each channel is masked when g has no such partner or none is measured in c (`q_sc_available` false).

**Interfaces:**
- `read_gene_reference(path: Path) -> GeneReference`, with fields `paralogs, complexes, hallmark, progeny` read as in `src/context_prior/reference.py`;
  - `cell_cycle, meta_programs, positions, arms` are added by Tasks 1 and 4;
  - no line tables.
- `partner_index(genes: Sequence[str], reference: GeneReference) -> PartnerIndex`:
  - `closest: np.ndarray[int]` [genes], −1 for none;
  - `members: list[np.ndarray[int]]`.
- `partner_fractions(q_sc_values, q_sc_available, index) -> tuple[np.ndarray, np.ndarray]`: values [genes, 2] float32 and available [genes, 2] bool.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from src.data.gene_reference import GeneReference
from src.data.partners import partner_fractions, partner_index

GENES = ("A", "B", "C", "D")


def reference():
    return GeneReference(
        paralogs=pd.DataFrame(
            {"gene": ["A", "A", "B"], "paralog": ["Z", "B", "A"], "identity": [90.0, 50.0, 50.0]}
        ),
        complexes=pd.DataFrame({"complex_id": [1, 1, 1, 2, 2], "gene": ["A", "C", "D", "A", "C"]}),
        hallmark=pd.DataFrame(columns=["gene_set", "gene"]),
        progeny=pd.DataFrame(columns=["pathway", "gene", "weight"]),
    )


def test_partner_fractions_read_the_closest_panel_paralog_and_complex_mates():
    index = partner_index(GENES, reference())
    assert list(index.closest) == [1, 0, -1, -1]  # Z is not in the panel
    values = np.array([[0, 0.1, 0], [0, 0.2, 0], [0, 0.3, 0], [0, 0.5, 0]], dtype=np.float32)
    available = np.array([True, True, True, True])
    fractions, mask = partner_fractions(values, available, index)
    np.testing.assert_allclose(fractions[0], [0.2, (0.3 + 0.5) / 2])
    np.testing.assert_allclose(fractions[2], [0.0, (0.1 + 0.5) / 2])
    assert mask[0].all() and not mask[2, 0] and mask[2, 1]


def test_partner_fractions_mask_genes_without_partners():
    index = partner_index(GENES, reference())
    values = np.zeros((4, 3), dtype=np.float32)
    fractions, mask = partner_fractions(values, np.array([True, False, True, True]), index)
    assert not mask[0, 0]  # the closest paralog B is not measured in this line
    assert not mask[3, 0] and mask[3, 1]
    assert (fractions[~mask] == 0).all()
```

- [ ] **Step 2: Implement `src/data/gene_reference.py` and `src/data/partners.py`**

```python
# src/data/partners.py
"""Fraction of cells expressing a gene's partners, from the q_sc cache."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.data.gene_reference import GeneReference

DETECTED = 1  # q_sc channel: fraction of cells with a nonzero count


@dataclass(frozen=True)
class PartnerIndex:
    closest: np.ndarray  # [genes] panel position of the closest paralog, -1 for none
    members: list[np.ndarray]  # [genes] panel positions of CORUM co-members


def partner_index(genes: Sequence[str], reference: GeneReference) -> PartnerIndex:
    position = {gene: i for i, gene in enumerate(genes)}
    closest = np.full(len(genes), -1, dtype=np.int64)
    ranked = reference.paralogs.sort_values("identity", ascending=False, kind="stable")
    for gene, paralog in zip(ranked["gene"], ranked["paralog"]):
        if gene in position and paralog in position and closest[position[gene]] < 0:
            closest[position[gene]] = position[paralog]
    by_complex = reference.complexes.groupby("complex_id")["gene"].apply(list)
    mates: list[set[int]] = [set() for _ in genes]
    for members in by_complex:
        inside = [position[g] for g in members if g in position]
        for i in inside:
            mates[i].update(j for j in inside if j != i)
    return PartnerIndex(closest, [np.array(sorted(m), dtype=np.int64) for m in mates])


def partner_fractions(
    q_sc_values: np.ndarray, q_sc_available: np.ndarray, index: PartnerIndex
) -> tuple[np.ndarray, np.ndarray]:
    """[genes, 2] closest-paralog and complex-mate detected fractions, with masks."""
    detected = np.nan_to_num(q_sc_values[:, DETECTED].astype(np.float64))
    measured = q_sc_available.astype(bool)
    values = np.zeros((len(index.closest), 2), dtype=np.float32)
    mask = np.zeros((len(index.closest), 2), dtype=bool)
    has = index.closest >= 0
    paralog = np.where(has, index.closest, 0)
    mask[:, 0] = has & measured[paralog]
    values[mask[:, 0], 0] = detected[paralog[mask[:, 0]]]
    for gene, members in enumerate(index.members):
        present = members[measured[members]] if len(members) else members
        if len(present):
            values[gene, 1] = detected[present].mean()
            mask[gene, 1] = True
    return values, mask
```

`GeneReference` is a frozen dataclass with the four tables. `read_gene_reference(path)` reads `paralogs.tsv.gz`, `complexes.tsv`, `hallmark.tsv` and `progeny.tsv` with `pd.read_csv(path / name, sep="\t")`.

- [ ] **Step 3: Plumb the block.** Make each change exactly as for `q_sc`, which has a value tensor and a mask:
  - `masked_blocks` gains `partners: Tensor | None` and `partners_mask: Tensor | None` (shape `[batch, 2]`, bool). When `use_partners`:

    ```python
        parts["partners"] = torch.cat(
            [partners * partners_mask.to(partners.dtype), partners_mask.to(partners.dtype)],
            dim=-1,
        )
    ```

    `_check_block("partners", blocks.use_partners, partners, dims.partners)` validates the value. A 2-D mask check replaces `_check_mask` for this block: shape `(batch, 2)`.
  - The `GeneEffectNestedHead` encoder input width for `partners` is `dims.partners * 2`.
  - `load_inputs`:
    - reads `GeneReference` from `config["paths"]["gene_reference"]` when `use_partners` is set;
    - computes `partner_index(genes, reference)`;
    - stores `PreparedInputs.partners: Mapping[str, tuple[np.ndarray, np.ndarray]] | None`, keyed by line (`partner_fractions` of each exposed line).
  - `DependencyDataset`:
    - stacks the per-line arrays into `_partners` [lines, genes, 2] and `_partners_mask` on the device (zeros and false when `inputs.partners is None`);
    - `collate` gathers `[line, gene]` into `OnlineConditionBatch.partners` and `partners_mask`.
  - `GeneEffectE2EModel.condition_features` copies them into `FeatureBatch`. `forward_features` passes `partners=standardizer.transform("partners", ...)` and `partners_mask` when `blocks.use_partners`.

- [ ] **Step 4: A model test.** In `tests/test_joint.py`, set `use_partners=True` in `make_config` for a new test. Train one CPU epoch on `make_inputs()`: `GeneReference` gets a paralog row G0→G2 and a complex G1/G3. Assert:
  - the head has a `partners` encoder;
  - the loss is finite;
  - a config with `use_partners: false` builds the same parameter set as before this task.

- [ ] **Step 5: Suite, lint, commit** (`feat(correction): partner fractions as a correction block`).

---

### Task 3: STATE's Δ as program scores

**Files:**
- Modify:
  - `src/model/features.py`: `ProgramProjection`;
  - `src/model/initialization.py`: choose the projection from `model.delta_projection`, and record it in `architecture`;
  - `src/model/geneeffect.py`: the projection's state round-trips through `to_state`/`from_state` with a `kind`;
  - `src/experiments/config.py`: `model.delta_projection` ∈ {`random`, `programs`};
  - every joint YAML.
- Test: `tests/test_program_projection.py`

**Definition.**
- Δ is `[mean shift (2000 HVGs), population-variance shift (2000)]`.
- A program's score change is a weighted mean of the mean-shift half over its member HVGs:
  - hallmark sets: weight 1/n over members;
  - PROGENy pathways: footprint weight divided by the sum of absolute weights over members.
- Only programs with at least 10 member HVGs are kept.
- The projection is a fixed `[P, 4000]` matrix with zeros on the variance half, so `delta_proj` has width P and `dims.delta_proj = P`.

**Interfaces:**
- `ProgramProjection.from_tables(hvg_order, hallmark, progeny, *, min_members=10) -> ProgramProjection`;
- attributes `programs: tuple[str, ...]` and `components: np.ndarray [P, 4000]`;
- `transform(delta)`, `metadata`, `to_state`, `from_state`, as `FixedSparseProjection` has them.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import torch

from src.model.features import DELTA_WIDTH, HVG_WIDTH, ProgramProjection


def test_program_projection_keeps_programs_with_ten_hvg_members():
    hvg = [f"G{i}" for i in range(HVG_WIDTH)]
    hallmark = pd.DataFrame(
        {"gene_set": ["BIG"] * 12 + ["SMALL"] * 3, "gene": hvg[:12] + hvg[20:23]}
    )
    progeny = pd.DataFrame(
        {"pathway": ["P"] * 10, "gene": hvg[30:40], "weight": [2.0] * 5 + [-2.0] * 5}
    )
    projection = ProgramProjection.from_tables(hvg, hallmark, progeny)
    assert projection.programs == ("BIG", "P")
    delta = torch.zeros(1, DELTA_WIDTH)
    delta[0, :12] = 1.0
    delta[0, 30:35] = 1.0
    delta[0, HVG_WIDTH:] = 5.0  # the variance half is ignored
    out = projection.transform(delta)
    np.testing.assert_allclose(out.numpy(), [[1.0, 0.5]], rtol=1e-6)
    restored = ProgramProjection.from_state(projection.to_state())
    assert torch.equal(restored.transform(delta), out)
```

- [ ] **Step 2: Implement `ProgramProjection`**

```python
class ProgramProjection:
    """Δ's mean-shift half averaged over each program's member HVGs (hallmark: equal
    weights; PROGENy: footprint weights over their absolute sum), for programs with
    at least ``min_members`` member HVGs; the variance half gets zero weight."""

    def __init__(self, programs: tuple[str, ...], components: np.ndarray) -> None:
        self.programs = tuple(programs)
        self.components = np.asarray(components, dtype=np.float32)
        self._tensors: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @classmethod
    def from_tables(cls, hvg_order, hallmark, progeny, *, min_members: int = 10):
        column = {gene: i for i, gene in enumerate(hvg_order)}
        rows, names = [], []
        sets = [
            (name, group["gene"], np.ones(len(group)))
            for name, group in hallmark.groupby("gene_set", sort=True)
        ] + [
            (name, group["gene"], group["weight"].to_numpy(float))
            for name, group in progeny.groupby("pathway", sort=True)
        ]
        for name, genes, weights in sets:
            keep = [(column[g], w) for g, w in zip(genes, weights) if g in column]
            if len(keep) < min_members:
                continue
            row = np.zeros(DELTA_WIDTH, dtype=np.float64)
            total = sum(abs(w) for _, w in keep)
            for index, weight in keep:
                row[index] = weight / total
            rows.append(row)
            names.append(str(name))
        return cls(tuple(names), np.stack(rows))

    @property
    def metadata(self) -> dict[str, object]:
        return {"algorithm": "program_mean_shift_v1", "input_width": DELTA_WIDTH,
                "output_width": len(self.programs)}

    def transform(self, delta: torch.Tensor) -> torch.Tensor:
        key = (delta.device, delta.dtype)
        if key not in self._tensors:
            self._tensors[key] = torch.as_tensor(
                self.components, device=delta.device, dtype=delta.dtype
            )
        return delta @ self._tensors[key].T

    def to_state(self) -> dict[str, object]:
        return {"kind": "programs", "programs": list(self.programs),
                "components": torch.from_numpy(self.components).clone()}

    @classmethod
    def from_state(cls, state):
        return cls(tuple(state["programs"]), np.asarray(state["components"]))
```

`FixedSparseProjection.to_state` gains `"kind": "random"`. Restoring dispatches on `kind`; a state without `kind` is refused, since checkpoints from before this change do not load anyway. `build_joint_model` builds `ProgramProjection.from_tables(inputs.hvg_order, reference.hallmark, reference.progeny)` when `model.delta_projection == "programs"`, and sets `GeneEffectFeatureDims(delta_proj=len(projection.programs), ...)`.

- [ ] **Step 3: Suite, lint, commit** (`feat(correction): STATE's Δ as hallmark and PROGENy score changes`).

---

### Task 4: Single-cell state views (preparation on the host, fitting in `load_inputs`)

**Files:**
- Create: `src/data/state_views.py`, `tests/test_state_views.py`
- Modify:
  - `src/experiments/prepare.py`: `prepare_state_views(config) -> Path`, called by `prepare_inputs` when any config enables `use_state_views`;
  - `src/data/prepared.py`: fit or restore the view state; `PreparedInputs.state_views: pd.DataFrame | None`, lines × view columns, standardised;
  - `src/data/datasets.py`, `src/model/head.py` (`use_state_views`: `C`'s input becomes `[z̃_c, views]`), `src/model/initialization.py` (`dims.z_c = context_components + view width`).

**Definitions.**
- **Expression.** Per cell, log-normalised expression: whole-library `normalize_total` to the manifest's `target_sum`, then `log1p`, as the expression space defines it.
- **Program score per cell.** For S, G2M and the 41 meta-programs: the mean over the program's genes that the line's source measures, minus the cell's mean over every panel gene the source measures. The line's `coverage[program]` is the fraction of the program's genes measured. A program measured at under 50% in any line raises at preparation, naming the line and the program.
- **Arm mean per cell.** For each of the 39 arms: the mean over the arm genes that **every** registered source measures (the intersection over the 226 sources; recorded in the manifest).
- **Preparation writes** `<prepared_root>/state_views/<ModelID>.npz`, holding `program_scores` [cells, 43] and `arm_means` [cells, 39] over every basal cell, plus `manifest.json` with the programs, arms, arm genes and coverage. It is skipped when the manifest exists for the same programs and arm genes.
- **Fitting** (`load_inputs`, supervised training lines only, each line weighted equally through its `cells_per_context` cells in `select_context_cells` order):
  - per program, the 90th percentile of the pooled scores;
  - per arm, the reference arm mean, which is the mean over training lines of each line's mean arm value.
- **The per-line view** (122 columns):
  - fraction of the line's cells above each program's threshold (43);
  - arm score: the line's mean arm value minus the reference (39);
  - arm spread: the standard deviation of the arm value across the line's cells (39);
  - aneuploidy: the mean absolute arm score (1).
- **Standardisation.** Each column is z-scored with the training lines' mean and SD; a constant column becomes 0.
- **Checkpoint state** (`preprocessing["state_views"]`): the thresholds, the reference arm means, the column means and SDs, the column names, and `None` when unused.

**Interfaces:**
- `cell_program_scores(log_expression: np.ndarray, measured: np.ndarray, members: Sequence[np.ndarray]) -> np.ndarray` [cells, programs];
- `cell_arm_means(log_expression, arm_genes: Sequence[np.ndarray]) -> np.ndarray`;
- `fit_view_state(per_line: Mapping[str, dict[str, np.ndarray]], training: Sequence[str], context_cells: Mapping[str, np.ndarray]) -> dict`;
- `line_views(per_line, state) -> pd.DataFrame`;
- `VIEW_GROUPS = {"cell_cycle": 2, "meta_programs": 41, "copy_number": 79}`, the column order used by Task 6's tokens.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from src.data.state_views import cell_arm_means, cell_program_scores, fit_view_state, line_views


def test_program_scores_subtract_the_cells_mean_over_measured_genes():
    x = np.array([[1.0, 3.0, 2.0, 0.0], [0.0, 0.0, 4.0, 4.0]])
    measured = np.array([True, True, True, False])
    scores = cell_program_scores(x, measured, [np.array([0, 1]), np.array([2, 3])])
    background = np.array([2.0, 4.0 / 3])
    np.testing.assert_allclose(scores[:, 0], [2.0, 0.0] - background)
    np.testing.assert_allclose(scores[:, 1], [2.0, 4.0] - background)  # gene 3 unmeasured


def test_state_view_fit_ignores_validation_and_test_cells():
    rng = np.random.default_rng(0)
    lines = {m: {"program_scores": rng.normal(size=(20, 3)), "arm_means": rng.normal(size=(20, 2))}
             for m in ("T0", "T1", "V0")}
    cells = {m: np.arange(10) for m in lines}
    state = fit_view_state(lines, ["T0", "T1"], cells)
    lines["V0"]["program_scores"] += 100.0
    lines["V0"]["arm_means"] += 100.0
    assert fit_view_state(lines, ["T0", "T1"], cells) == state
    views = line_views(lines, state)
    assert list(views.index) == ["T0", "T1", "V0"]
    assert views.shape[1] == 3 + 2 + 2 + 1


def test_arm_means_use_only_genes_every_source_measures():
    x = np.array([[1.0, 5.0, 3.0]])
    np.testing.assert_allclose(cell_arm_means(x, [np.array([0, 2])]), [[2.0]])
```

The intersection rule is enforced where `arm_genes` is built, in `prepare_state_views`. Its test is in `tests/test_prepare.py`: two sources, one lacking a gene, and that gene does not enter the arm genes. Add it there using the module's `build_world` fixture with a second registry line whose source drops one panel gene.

- [ ] **Step 2: Implement `src/data/state_views.py`** (pure NumPy; equality of `fit_view_state` results uses plain floats and lists so the dict compares exactly)

```python
VIEW_GROUPS = {"cell_cycle": 2, "meta_programs": 41, "copy_number": 79}
THRESHOLD_QUANTILE = 0.9


def cell_program_scores(log_expression, measured, members):
    background = log_expression[:, measured].mean(axis=1)
    out = np.empty((log_expression.shape[0], len(members)))
    for j, genes in enumerate(members):
        present = genes[measured[genes]]
        out[:, j] = log_expression[:, present].mean(axis=1) - background
    return out


def cell_arm_means(log_expression, arm_genes):
    return np.stack([log_expression[:, genes].mean(axis=1) for genes in arm_genes], axis=1)


def fit_view_state(per_line, training, context_cells):
    pooled = np.concatenate(
        [per_line[m]["program_scores"][context_cells[m]] for m in training]
    )
    thresholds = np.quantile(pooled, THRESHOLD_QUANTILE, axis=0)
    reference = np.mean([per_line[m]["arm_means"].mean(axis=0) for m in training], axis=0)
    state = {"thresholds": thresholds.tolist(), "reference": reference.tolist()}
    raw = _raw_views({m: per_line[m] for m in training}, state)
    mean, sd = raw.mean(axis=0), raw.std(axis=0)
    state["mean"], state["sd"] = mean.tolist(), sd.tolist()
    return state


def _raw_views(per_line, state):
    thresholds, reference = np.asarray(state["thresholds"]), np.asarray(state["reference"])
    rows = []
    for m in per_line:
        scores, arms = per_line[m]["program_scores"], per_line[m]["arm_means"]
        arm_score = arms.mean(axis=0) - reference
        rows.append(np.concatenate([
            (scores > thresholds).mean(axis=0), arm_score, arms.std(axis=0),
            [np.abs(arm_score).mean()],
        ]))
    return np.stack(rows)


def line_views(per_line, state):
    raw = _raw_views(per_line, state)
    mean, sd = np.asarray(state["mean"]), np.asarray(state["sd"])
    scaled = np.where(sd > 0, (raw - mean) / np.where(sd > 0, sd, 1.0), 0.0)
    return pd.DataFrame(scaled, index=list(per_line))
```

- [ ] **Step 3: `prepare_state_views`** in `src/experiments/prepare.py`. Follow `prepare_pseudobulk`:
  - read each registered line's raw source with `read_registry_source`;
  - log-normalise every cell with `log_normalize` and `library_sizes`, as `prepare_line` does;
  - align the panel with `align_columns`;
  - compute both arrays;
  - write the per-line `.npz` files and the manifest atomically, in processes (`_in_processes`, `LINE_PROCESSES`).

  The arm genes are the panel genes measured by every registered source; collect the per-source availability first, in a pass over each source's `var` only.

- [ ] **Step 4: Plumb into `load_inputs` and the head.** When `use_state_views` is set:
  - `load_inputs` reads the per-line arrays of the exposed lines;
  - it fits or restores the state and stores `PreparedInputs.state_views`;
  - `DependencyDataset` stacks the rows into `_views` [lines, 122] and concatenates them after the context PCA scores into `z_c`;
  - `dims.z_c` grows by 122;
  - `preprocessing_state()` gains `"state_views"`, the fitted state or `None`, and `_restored_keys` requires it.

- [ ] **Step 5: Run on the host before any config uses it.** On the H20 host, run `uv run python -c "from src.experiments.prepare import prepare_state_views; ..."` through a CLI flag of `src.experiments.prepare`, then report the manifest:
  - arm-gene count per arm;
  - program coverage minimum per line.

  Commit (`feat(correction): single-cell state views from basal cells`).

---

### Task 5: Local copy number as an `h` block

**Files:**
- Create: `src/data/copy_number.py`, `tests/test_copy_number.py`
- Modify: the same plumbing as Task 2, for block `copy_number` (`dims.copy_number = 2`: local copy number of g and of its closest paralog, each with a mask bit).

**Definition.**
1. `fit_copy_number_reference`: on the supervised training lines, the training mean per gene of the `q_sc` log-mean (channel 0), over lines where the gene is measured.
2. Per line, centre the log-mean by that reference.
3. A gene's local copy number is the mean of the centred values of the 100 nearest positioned genes on its arm (50 on each side in position order), excluding the gene itself, over those measured in the line.
4. It is masked when the gene has no arm or fewer than 20 of those neighbours are measured.
5. The paralog channel reads the closest paralog (Task 2's index) the same way.
6. The checkpoint holds the reference.

**Interfaces:**
- `fit_copy_number_reference(q_sc_by_line, training) -> np.ndarray` [genes];
- `local_copy_number(log_mean, measured, reference, order: ArmOrder, *, half_window=50, min_neighbours=20) -> tuple[np.ndarray, np.ndarray]`;
- `ArmOrder`: for each arm, the panel positions in genomic order, from `gene_positions.tsv`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from src.data.copy_number import arm_order, local_copy_number


def test_local_copy_number_is_the_neighbour_mean_without_the_gene():
    positions = {"A": ("1p", 1), "B": ("1p", 2), "C": ("1p", 3), "D": ("1q", 9)}
    order = arm_order(("A", "B", "C", "D"), positions)
    log_mean = np.array([1.0, 2.0, 4.0, 7.0])
    values, mask = local_copy_number(
        log_mean, np.ones(4, bool), np.zeros(4), order, half_window=1, min_neighbours=1
    )
    np.testing.assert_allclose(values[:3], [2.0, (1.0 + 4.0) / 2, 2.0])
    assert mask[:3].all() and not mask[3]  # D has no neighbour on its arm


def test_local_copy_number_masks_genes_without_position():
    order = arm_order(("A", "B"), {"A": ("1p", 1)})
    values, mask = local_copy_number(
        np.ones(2), np.ones(2, bool), np.zeros(2), order, half_window=1, min_neighbours=1
    )
    assert not mask[1] and values[1] == 0.0
```

- [ ] **Step 2: Implement with cumulative sums along each arm**
  - For each arm's ordered positions, take the centred values `v` and the availability `a` along the order.
  - Compute window sums of `v·a` and `a` over `[i − h, i + h]` via `np.cumsum`.
  - Subtract the gene's own term.
  - Divide where the count reaches `min_neighbours`; elsewhere the value is 0 and masked.

- [ ] **Step 3: Plumb the block** as in Task 2 Step 3, with names `copy_number` and `copy_number_mask`; `load_inputs` computes per exposed line when `use_copy_number` is set. Then suite, lint, commit (`feat(correction): local copy number from single-cell log-means`).

---

### Task 6: `C` variants: off, linear, cross-attention

**Files:**
- Modify:
  - `src/model/head.py`;
  - `src/experiments/config.py`: `model.context_tower` ∈ {`linear_swiglu`, `linear`}; `model.context_fusion` ∈ {`inner_product`, `cross_attention`};
  - `src/model/initialization.py`;
  - every joint YAML.
- Test: `tests/test_geneeffect_head.py`

**Definitions.**
- **`C` off.** `use_z_c: false`, and `use_state_views` must then be false. The head is `h` alone: no gene embedding and no `C`, and at least one `h` block is required. This replaces the current refusal of `use_z_c=False`.
- **Linear `C`.** `context_tower: linear` drops `context_residual`.
- **Cross-attention.** `context_fusion: cross_attention`:
  - tokens are one per view group: the Tx1 components (128) and, with `use_state_views`, cell cycle (2), meta-programs (41) and copy number (79);
  - each token is projected by its own `Linear(width → r)` for keys and for values;
  - the value maps start at zero;
  - the query is `G(g)`;
  - the output is `⟨G(g), Σ_t softmax_t(G(g)·K_t/√r) V_t⟩ / √r`;
  - one head; dropout on the attention weights at `model.dropout`.

**Interfaces:** `ContextCrossAttention(widths: Mapping[str, int], rank: int, dropout: float)`, with `forward(gene: Tensor [B, r], context: Tensor [B, Σ widths]) -> Tensor [B]`; the token slices follow `widths`' order.

- [ ] **Step 1: Write the failing tests**

```python
def test_cross_attention_starts_at_zero():
    torch.manual_seed(0)
    module = ContextCrossAttention({"tx1": 4, "cell_cycle": 2}, rank=3, dropout=0.0)
    out = module(torch.randn(5, 3), torch.randn(5, 6))
    assert torch.equal(out, torch.zeros(5))


def test_cross_attention_weights_sum_to_one_over_views():
    torch.manual_seed(0)
    module = ContextCrossAttention({"tx1": 4, "cell_cycle": 2}, rank=3, dropout=0.0)
    for layer in module.values.values():
        torch.nn.init.normal_(layer.weight)
    weights = module.attention(torch.randn(5, 3), torch.randn(5, 6))
    torch.testing.assert_close(weights.sum(-1), torch.ones(5))


def test_head_without_context_is_the_correction_alone():
    dims = _full_dims()
    blocks = GeneEffectBlockConfig(use_z_c=False)
    head = GeneEffectNestedHead(dims, blocks, n_genes=5, factor_rank=RANK, dropout=0.0,
                                context_tower="linear_swiglu", context_fusion="inner_product")
    assert not hasattr(head, "gene_embedding") or head.gene_embedding is None


def test_linear_context_tower_has_no_residual_branch():
    head = GeneEffectNestedHead(_full_dims(), GeneEffectBlockConfig(), n_genes=5,
                                factor_rank=RANK, dropout=0.0, context_tower="linear",
                                context_fusion="inner_product")
    assert head.context_residual is None
```

- [ ] **Step 2: Implement `ContextCrossAttention` and the head arguments**

```python
class ContextCrossAttention(nn.Module):
    """``<G(g), sum_t softmax_t(G(g) K_t / sqrt r) V_t> / sqrt r`` over one token per
    context view; value maps start at zero, so the output starts at zero."""

    def __init__(self, widths, rank: int, dropout: float) -> None:
        super().__init__()
        self.widths = dict(widths)
        self.rank = int(rank)
        self.keys = nn.ModuleDict({n: nn.Linear(w, rank) for n, w in self.widths.items()})
        self.values = nn.ModuleDict({n: nn.Linear(w, rank) for n, w in self.widths.items()})
        for layer in self.values.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.dropout = nn.Dropout(dropout)

    def _tokens(self, context):
        parts = torch.split(context, list(self.widths.values()), dim=-1)
        names = list(self.widths)
        keys = torch.stack([self.keys[n](p) for n, p in zip(names, parts)], dim=1)
        values = torch.stack([self.values[n](p) for n, p in zip(names, parts)], dim=1)
        return keys, values

    def attention(self, gene, context):
        keys, _ = self._tokens(context)
        scores = torch.einsum("br,btr->bt", gene, keys) / math.sqrt(self.rank)
        return torch.softmax(scores, dim=-1)

    def forward(self, gene, context):
        keys, values = self._tokens(context)
        scores = torch.einsum("br,btr->bt", gene, keys) / math.sqrt(self.rank)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = torch.einsum("bt,btr->br", weights, values)
        return (gene * mixed).sum(-1) / math.sqrt(self.rank)
```

In `GeneEffectNestedHead.__init__`, take keyword arguments `context_tower` and `context_fusion`.
- **No context** (`blocks.use_z_c` false): set `gene_embedding`, `gene_projection`, `context_linear`, `context_residual` and `attention` to `None`, and require `self.encoders`.
- **Inner product:** keep `context_linear` (zeroed). `context_residual` is the SwiGLU only for `linear_swiglu`, else `None`.
- **Cross-attention:** build `self.attention = ContextCrossAttention(widths, factor_rank, dropout)`, with `widths = {"tx1": context_components, **(VIEW_GROUPS if use_state_views else {})}`. `dims` gains `context_components` so the split is known, and `context_linear` is `None`.

`forward` computes the context term only when context is used, through whichever fusion is configured. `build_joint_model` and `restore_joint_model` pass both keys from the config and the architecture record.

- [ ] **Step 3: Suite, lint, commit** (`feat(correction): context tower and fusion variants of the correction head`).

---

### Task 7: Second-wave configs, docs and runs (after the user's wave-one review)

- [ ] **Step 1: Write one config per input** under `configs/correction/second_wave/`. Each is a copy of the wave-one winner's config with one change and a header comment naming the input:
  - `partners.yaml`: `use_partners: true`;
  - `program_delta.yaml`: `delta_projection: programs`, and only if a STATE setting won wave one;
  - `state_views.yaml`: `use_state_views: true`;
  - `copy_number.yaml`: `use_copy_number: true`;
  - `context_off.yaml`: `use_z_c: false`;
  - `linear_context.yaml`: `context_tower: linear`;
  - `cross_attention.yaml`: `context_fusion: cross_attention` with `use_state_views: true`.

  A test asserts that each config differs from the winner's in exactly the documented keys.
- [ ] **Step 2: Update the docs.**
  - Protocol §12: the new blocks, views and `C` variants with their definitions.
  - `docs/data/`: a card for the pinned correction tables.
  - `CLAUDE.md` and `hpc/README.md`: the `prepare_state_views` step.
- [ ] **Step 3: Run on port 30838, one at a time, cheapest first.** The order is partners, program Δ, state views, copy number, `C` off, linear `C`, cross-attention.
  1. Read each run against the wave-one winner with `compare_runs` on validation.
  2. Record every row (validation and test) in `results/correction_second_wave_seed0/README.md`.
  3. Then one combined run with the inputs whose validation gain the user keeps.
  4. Stop and report.
