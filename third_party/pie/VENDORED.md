# Vendored PIE

Source: https://github.com/ArcInstitute/pie, commit `21a526add9ff3a81b4f962d9b16bbae570052c63` (`main`, 2026-10-05),
copied on 2026-10-16. Every tracked file is copied as released except `.github/`; this note is the only addition.

Licence: CC BY-NC-SA 4.0 (`LICENSE.md`). Our additions under `third_party/pie/` are released under the same licence.

Why it is vendored: PIE's model imports `pie.data` and `pie.utils`, and the stage-one models join PIE's trunk with our
heads in one process, so the whole package is copied rather than installed from a release
(`docs/specs/2026-10-16-pie-geneeffect-ladder-design.md` §3.4). It is installed editable into `.venv-pie` by
`hpc/pie/setup.sh`. Later tasks add `src/pie/model/geneeffect.py` beside PIE's model.

Local changes:
- `src/pie/model/geneeffect.py` (new, 2026-10-17): the stage-one direct readout's trunk (per-channel attention over a
  line's tokens into PIE's latents, gene queries with their own evidence) and a strict, name-mapped warm start from a
  released checkpoint. It imports PIE's layers and changes none of PIE's files.
- `pyproject.toml` and `uv.lock`: a `geneeffect` dependency group (scikit-learn) for the project's scorer; PIE's own
  dependencies are unchanged.
