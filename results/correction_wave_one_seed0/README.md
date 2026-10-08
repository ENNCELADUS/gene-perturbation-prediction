# Single-cell correction, wave one, seed 0

Runs of 2026-10-07 and 2026-10-08 on the H20 container on port 30838 (worktree `/2023533015/VCC_Project_correction`,
branch `feat/single-cell-correction`), following the
[wave-one plan](../../docs/specs/2026-10-07-single-cell-correction-plan.md) and
[protocol §12](../../docs/03-geneeffect-protocol.md#12-single-cell-correction). The correction stacks the nested head of
protocol §4 on the default linear context prior: prediction = the prior's offset + the head, and the head starts at zero,
so the stack starts at the prior and validation before the first update (epoch −1) is a `best.pt` candidate. Records:
the [prior export](prior_export.json) and the `summary.md` of every run (`summary_<run>.md`); checkpoints, predictions
and `revision.json` stay in `outputs/geneeffect_correction/<run id>/` on the host. Nothing here is synthetic-lethality
evidence.

**Result: STATE absent wins, and the winner is the prior alone.** In all six runs (four objectives with STATE absent,
then standardised MSE with frozen and with trainable STATE) `best.pt` is the model before its first update, so every
stack scores exactly the prior: selective Spearman 0.2290 validation / 0.2338 test. Every trained epoch of every run
scores lower on validation while the training lines' score rises. The linear version of the correction, the Tx1
context-PCA ridge fitted on what the prior leaves, lowers the prior too (0.1826 / 0.1823). Per the plan, nothing else
launches until the research plan has been discussed.

## The prior export

`hpc/run.sh prior-export configs/context_prior/default_prior.yaml --run-id default_prior_export` (CPU, about two
minutes; code at `fa97aed`) wrote `outputs/context_prior/default_prior_export/export/`: 170 labelled single-cell
training lines out of fold (the bridge and every stage refitted without the line's patient-grouped fold), 27
validation and 27 test lines from the full fit, in residual-SD units.

| Lines | Selective Spearman | Selective AUPR lift | Residual Pearson | SD ratio |
| --- | ---: | ---: | ---: | ---: |
| Training, out of fold | 0.2113 | 0.1036 | 0.2036 | 0.1126 |
| Validation | 0.2290 | 0.1974 | 0.2229 | 0.1179 |
| Test | 0.2338 | 0.1910 | 0.2286 | 0.1139 |

Validation and test reproduce the default prior's row of the
[follow-ups](../default_prior_followups_seed0/README.md) (0.2290 / 0.2338). The out-of-fold training score is close to
validation, so the training offsets carry the error a query line will have. Every run below accepted the export
(gene order, residual SD and lines checked when the inputs open).

## Objective screen, STATE absent

`hpc/run.sh revision configs/correction/<objective>_no_state.yaml --run-id correction_<objective>_no_state_seed0`,
one run after another on 4 GPUs (code at `75b825a`), about 30 s per epoch. In every run `train/best.pt` is the model
before its first update, the prior alone: validation selective Spearman 0.2290 at epoch −1, and each of the five trained
epochs scores lower, so early stopping (patience 5) ends the run after epoch 4.

Validation selective Spearman by epoch (training-diagnostic lines in brackets):

| Epoch | Huber | Standardised MSE | Line ranking | Dependency classification |
| ---: | ---: | ---: | ---: | ---: |
| −1 (the prior alone) | **0.2290** (0.1967) | **0.2290** (0.1967) | **0.2290** (0.1967) | **0.2290** (0.1967) |
| 0 | 0.2026 (0.2371) | 0.2116 (0.2495) | 0.1923 (0.2167) | 0.2096 (0.2555) |
| 1 | 0.2046 (0.2630) | 0.2100 (0.2677) | 0.1825 (0.2450) | 0.2073 (0.2809) |
| 2 | 0.2004 (0.2744) | 0.2071 (0.2908) | 0.1796 (0.2520) | 0.2015 (0.3050) |
| 3 | 0.1867 (0.2858) | 0.1992 (0.3206) | 0.1837 (0.2630) | 0.1948 (0.3308) |
| 4 | 0.1881 (0.3060) | 0.1943 (0.3401) | 0.1739 (0.2633) | 0.1892 (0.3469) |

The head fits the training lines from the first epoch (training-diagnostic Spearman 0.197 → 0.25–0.35) while
validation falls. The validation SD ratio rises from 0.118 at the start to 0.15–0.18 after one epoch and 0.21–0.23
after five: the head's component is larger than the shrunk prior's own prediction, so it sets the per-gene ranking.

Because all four `best.pt` are the prior alone, the four runs have identical rows (from each `summary.md`):

| Model | Val selective Spearman | Test selective Spearman | Val AUPR lift | Test AUPR lift | Val residual Pearson | Test residual Pearson | Val SD ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stack (`best.pt`: the prior alone) | 0.2290 | 0.2338 | 0.1974 | 0.1910 | 0.2229 | 0.2286 | 0.118 |
| Linear context prior | 0.2290 | 0.2338 | 0.1974 | 0.1910 | 0.2229 | 0.2286 | 0.118 |
| Linear context prior + Tx1 context-PCA ridge | 0.1826 | 0.1823 | 0.1683 | 0.1596 | 0.1836 | 0.1824 | 0.294 |
| Context-PCA ridge (Tx1) | 0.1296 | 0.1206 | 0.1335 | 0.1263 | 0.1330 | 0.1216 | 0.273 |
| Context-PCA ridge (HVG) | 0.1174 | 0.1185 | 0.1265 | 0.1211 | 0.1145 | 0.1105 | 0.237 |
| Nearest line (Tx1) | 0.0810 | 0.0599 | 0.1120 | 0.1052 | 0.0851 | 0.0608 | 1.023 |
| Nearest line (HVG) | 0.0806 | 0.0709 | 0.1134 | 0.1019 | 0.0787 | 0.0715 | 0.989 |
| Gene mean, K562 copy prior | undefined | undefined | 0.0000 | 0.0000 | undefined | undefined | 0.000 |

Paired line bootstrap of selective Spearman (1,000 resamples, seed 0):

| Stack minus | Validation | Test |
| --- | --- | --- |
| Context-PCA ridge (Tx1) | +0.0993 [0.0663, 0.1266] | +0.1131 [0.0784, 0.1411] |
| Linear context prior | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| Linear context prior + Tx1 context-PCA ridge | +0.0464 [0.0220, 0.0684] | +0.0514 [0.0264, 0.0730] |

The linear special case of the correction, the Tx1 ridge fitted on what the prior leaves, also lowers the prior:
−0.046 on validation and −0.051 on test, both intervals clear of zero, with an SD ratio of 0.29 against the prior's
0.12. The per-lineage table is [below](#per-lineage).

**Choice.** The reading rule (protocol §9.3) takes the highest validation selective Spearman at `best.pt`; all four
tie exactly (`compare_runs` gives 0.0000 [0.0000, 0.0000] for standardised MSE minus each other objective), and the
plan breaks a tie between the single-term losses by the higher point estimate, which is also tied. The trained epochs
are the only thing that separates them: standardised MSE has the highest trained-epoch validation score (0.2116 at
epoch 0, against 0.2096 for dependency classification, 0.2046 for Huber and 0.1923 for line ranking), so the STATE
runs use standardised MSE.

## STATE, with standardised MSE

`hpc/run.sh revision configs/correction/standardized_mse_<setting>_state.yaml --run-id
correction_standardized_mse_<setting>_state_seed0` for frozen and then trainable STATE (STATE's delta and state
vectors enter the head, no response replay), one after the other on 4 GPUs (code at `d570ee5`). Training took about
36 minutes with frozen STATE and 47 with trainable STATE, against about 3 without it.

Validation selective Spearman by epoch (training-diagnostic lines in brackets):

| Epoch | STATE absent | Frozen STATE | Trainable STATE |
| ---: | ---: | ---: | ---: |
| −1 (the prior alone) | **0.2290** (0.1967) | **0.2290** (0.1967) | **0.2290** (0.1967) |
| 0 | 0.2116 (0.2495) | 0.2102 (0.2490) | 0.2101 (0.2500) |
| 1 | 0.2100 (0.2677) | 0.2085 (0.2697) | 0.2074 (0.2682) |
| 2 | 0.2071 (0.2908) | 0.2004 (0.2923) | 0.2017 (0.2940) |
| 3 | 0.1992 (0.3206) | 0.1964 (0.3229) | 0.1963 (0.3251) |
| 4 | 0.1943 (0.3401) | 0.1907 (0.3403) | 0.1906 (0.3440) |

STATE changes the head's path little: each epoch is within 0.007 of the STATE-absent run, slightly lower on
validation, and the validation SD ratio climbs the same way (0.118 → 0.151 → 0.22). Both `best.pt` are again the
prior alone, so both summaries equal the STATE-absent run's apart from their headers (the table above), and
`compare_runs` gives 0.0000 [0.0000, 0.0000] on validation and on test for frozen minus STATE absent and for
trainable minus STATE absent.

**Choice.** All three tie exactly at `best.pt`, so the order of simplicity decides: STATE absent wins. The trained
epochs point the same way.

## Per lineage

Mean over a lineage's lines of the per-line residual Spearman across the selective genes; the stack equals the prior
in every run. With 1–7 lines per lineage it is descriptive only.

| Lineage | Val lines | Val prior | Val prior + Tx1 ridge | Val Tx1 ridge | Test lines | Test prior | Test prior + Tx1 ridge | Test Tx1 ridge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Biliary Tract | 1 | 0.2950 | 0.1499 | 0.0899 | 1 | 0.2753 | 0.3056 | 0.2248 |
| Bladder/Urinary Tract | 1 | 0.1187 | 0.0876 | 0.0490 | 1 | 0.2795 | 0.1782 | 0.1195 |
| Bowel | 1 | 0.0689 | 0.1138 | 0.0873 | 2 | 0.3516 | 0.2547 | 0.1491 |
| Breast | 4 | 0.3170 | 0.2808 | 0.2389 | 7 | 0.2304 | 0.1720 | 0.1231 |
| CNS/Brain | 3 | 0.2858 | 0.1714 | 0.1238 | 2 | 0.2904 | 0.2347 | 0.1908 |
| Esophagus/Stomach | 2 | 0.1811 | 0.1634 | 0.1336 | 1 | 0.2271 | 0.1690 | 0.1179 |
| Head and Neck | 3 | 0.2455 | 0.2180 | 0.1907 | 3 | 0.2312 | 0.2038 | 0.1744 |
| Kidney | 1 | 0.2871 | 0.3554 | 0.2967 | 1 | 0.4296 | 0.3854 | 0.2201 |
| Liver | 1 | 0.3793 | 0.2929 | 0.1686 | 1 | 0.3593 | 0.1637 | 0.0445 |
| Lung | 4 | 0.2679 | 0.1888 | 0.0957 | 3 | 0.3052 | 0.2622 | 0.1696 |
| Ovary/Fallopian Tube | 2 | 0.3111 | 0.2747 | 0.2455 | 1 | 0.4105 | 0.3678 | 0.3165 |
| Pancreas | 1 | 0.2384 | 0.1175 | 0.0906 | 1 | 0.2582 | 0.3492 | 0.2031 |
| Pleura | 1 | 0.1902 | 0.0741 | -0.0003 | 1 | 0.1804 | 0.2145 | 0.1802 |
| Skin | 1 | 0.3956 | 0.3136 | 0.2238 | 1 | 0.1663 | 0.1767 | 0.1197 |
| Uterus | 1 | 0.2944 | 0.1924 | 0.0767 | 1 | -0.0064 | -0.1062 | -0.1712 |

The prior plus the Tx1 ridge beats the prior alone in 2 of 15 validation lineages (Bowel, Kidney) and 4 of 15 test
lineages (Biliary Tract, Pancreas, Pleura, Skin), each a single line.
