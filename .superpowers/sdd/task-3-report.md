# Task 3 Report: Exp05 ESM-2 Perturbation Adapter

## Result

Implemented the exp08 ESM-2 adapter path for exp05 while preserving the legacy
`state_onehot` path. The ESM-2 path validates the canonical outer manifest and
exact 9,338-gene order before model construction, requires complete ESM-2
coverage without filtering, rejects uncovered external genes, freezes STATE,
and keeps the shared adapter, projector, and C head trainable.

## RED

- `tests/test_aivc_model.py -k "esm2_perturbation_adapter"` failed during
  collection because `Esm2PerturbationAdapter` did not exist.
- `test_state_config_parses_strict_esm2_fields` failed because `StateConfig`
  did not expose the ESM-2 fields.
- The ESM-2 builder tests failed before canonical-universe/ESM integration was
  implemented.

## GREEN

- `rtk uv run python -m pytest tests/test_aivc_model.py -k "esm2 or freeze_state" -v`
  - 5 passed, 78 deselected.
  - Confirms shared-network output, unresolved-gene failure, config parsing,
    nonzero adapter gradients, absent STATE gradients, trainable projector/C
    head, and explicit external-gene coverage failure.
- `rtk uv run ruff format src/aivc_model/prepare.py src/aivc_model/model.py src/aivc_model/train.py tests/test_aivc_model.py`
  - 4 files formatted.
- `rtk uv run ruff check src/aivc_model/prepare.py src/aivc_model/model.py src/aivc_model/train.py tests/test_aivc_model.py`
  - All checks passed.
- `rtk uv run python -m aivc_model.train --help`
  - Passed; the AIVC training module imports and exposes its CLI.
- `rtk git diff --check`
  - Passed.

## Full Suite

`rtk uv run python -m pytest` completed with 425 passed and 20 failed.
The failures are outside the Task 3 diff:

- 10 AIVC tests encounter an existing Accelerate process-global device conflict:
  an earlier test initializes MPS and later tests request `Accelerator(cpu=True)`.
- `test_experiment_configs_follow_grouped_layout` fails on the existing exp07
  augmented config because it has no baseline `data.h5ad_path`.
- 9 report/table tests fail because the worktree lacks required exp06 result
  artifacts such as `official_metrics_summary.csv`.

## Self-review and concerns

- Only the four Task 3 implementation/test files and this report are changed.
- The legacy per-gene `PerturbationVectorAdapter` remains the default for
  `gene_tokenizer: state_onehot`.
- `require_resolved_esm2` is parsed for canonical config compatibility, but the
  `esm2` path is strict unconditionally so the flag cannot enable fallback or
  filtering.
- The exp08 `PertAdapter` import is intentionally local to adapter construction
  because `sl_dl_model.encoder` already imports `aivc_model.model`; a top-level
  import would create a module cycle.
- No Task 3-specific blocker remains.
