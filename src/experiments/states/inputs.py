"""The ``inputs`` step: each projected line's state masses and inputs in PIE's input
space, their permuted memberships, and the per-set tables the frozen PIE reads."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data import cell_states as cs
from src.data.io import write_npz, write_parquet
from src.experiments.states.telemetry import IDENTITY_TOLERANCE, Telemetry, npz_keys
from src.experiments.states.vocabularies import read_memberships, vocabulary_labels


def line_cells(t: Telemetry, model_id: str) -> tuple[cs.LineCells, Any, list[str]]:
    """A line's per-cell PIE input, its raw counts and source symbols."""
    from src.data.basal import symbol_column
    from src.data.tx1_cache import read_basal_source

    source = read_basal_source(
        t.source(model_id),
        model_id=model_id,
        var_ensembl_col=t.preparation["var_ensembl_col"],
    )
    symbols = source.var[symbol_column(source.var)].astype(str).tolist()
    cells = cs.per_cell_log_cp10k(
        source.X,
        symbols,
        t.panel,
        t.pie_input.loc[model_id].to_numpy(dtype=np.float64),
        t.pie_listed.loc[model_id].to_numpy(dtype=bool),
    )
    return cells, source.X, symbols


def _check_identity(
    model_id: str, name: str, values: np.ndarray, line: np.ndarray
) -> None:
    """Fail closed when a set's states do not add up to the line's PIE input."""
    difference = float(np.max(np.abs(values - line)))
    if not difference <= IDENTITY_TOLERANCE:
        raise ValueError(
            f"{model_id} {name}: mass-weighted state inputs differ from the line's "
            f"PIE input by {difference:.3g}"
        )


def line_inputs(t: Telemetry, model_id: str) -> dict[str, Any]:
    """Every set's states of one line (``lines/<ModelID>.npz``)."""
    from src.data.noise_inputs import program_scores

    cells, counts, symbols = line_cells(t, model_id)
    line = t.pie_input.loc[model_id].to_numpy(dtype=np.float64)
    _check_identity(model_id, "whole", cells.mean(), line)
    memberships = read_memberships(t, model_id)
    minimum = float(t.config["inputs"]["merge_effective_count"])
    out: dict[str, Any] = {"whole": cells.mean().astype(np.float32)}
    states: dict[str, tuple[cs.StateInputs, dict[int, cs.StateInputs]]] = {}
    permutation_seeds = [int(s) for s in t.config["inputs"]["permutation_seeds"]]
    for set_name in cs.state_sets(t.config):
        if set_name == cs.CYCLING_SET:
            q = memberships[cs.membership_key(cs.CYCLE, t.seed)]
            labels = vocabulary_labels(t, t.seed, cs.CYCLE)
            weights = cs.cycling_weights(
                cs.cycling_probability(q, labels),
                float(t.config["inputs"]["cycling_floor"]),
            )
            inputs = cells.weighted_inputs(weights[:, None])
            state = cs.StateInputs(
                ["0"], np.ones(1), cs.effective_count(weights[:, None]), inputs, [[]]
            )
        else:
            vocabulary = cs.base_vocabulary(set_name)
            if vocabulary not in states:
                # Merged once on the real memberships; the permuted sets permute
                # the merged rows, so they keep the real states and masses.
                states[vocabulary] = cs.permuted_state_inputs(
                    memberships[cs.membership_key(vocabulary, t.seed)],
                    cells,
                    vocabulary_labels(t, t.seed, vocabulary),
                    minimum,
                    model_id=model_id,
                    vocabulary=vocabulary,
                    seeds=permutation_seeds
                    if cs.family(vocabulary) in cs.PERMUTED_FAMILIES
                    else [],
                )
            real, permuted = states[vocabulary]
            state = (
                real
                if set_name == vocabulary
                else permuted[int(set_name.rpartition("__perm")[2])]
            )
            _check_identity(model_id, set_name, state.mass @ state.inputs, line)
        out[f"inputs__{set_name}"] = state.inputs.astype(np.float32)
        out[f"mass__{set_name}"] = state.mass
        out[f"effective__{set_name}"] = state.effective
        out[f"labels__{set_name}"] = np.asarray(state.labels)
        out[f"merged__{set_name}"] = np.asarray([",".join(m) for m in state.merged])
    # Identity and confound telemetry on the unmerged states of every vocabulary.
    references = t.references()
    scores, listed = program_scores(
        counts, symbols, t.panel, references, float(t.recorded["prepared_target_sum"])
    )
    scores = scores.astype(np.float64)
    scores -= scores.mean(axis=0)
    qc = t.cell_arrays(model_id, "qc")[0][:, list(cs.QC_FEATURES)].astype(np.float64)
    qc -= qc.mean(axis=0)
    for vocabulary in cs.vocabularies(t.config):
        q = memberships[cs.membership_key(vocabulary, t.seed)]
        total = q.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"identity__{vocabulary}"] = (q.T @ scores) / total[:, None]
            out[f"qcshift__{vocabulary}"] = (q.T @ qc) / total[:, None]
    out["reference_programs"] = np.asarray(list(references))
    out["reference_listed"] = np.asarray([listed[name] for name in references])
    out["n_cells"] = np.asarray(cells.n_cells)
    return out


def assemble_sets(t: Telemetry) -> dict[str, Any]:
    """``<set>/inputs.parquet`` (contexts x panel, the line's ``|whole`` context
    after its states) and ``<set>/masses.parquet`` for every set."""
    counts = {}
    for set_name in cs.state_sets(t.config):
        names, rows, table = [], [], []
        for model_id in t.projected:
            line = npz_keys(
                t.layout.line(model_id),
                *(f"{part}__{set_name}" for part in ("labels", "inputs", "mass")),
                *(f"{part}__{set_name}" for part in ("effective", "merged")),
                "whole",
                "n_cells",
            )
            labels = [str(s) for s in line[f"labels__{set_name}"]]
            for k, label in enumerate(labels):
                names.append(cs.context_name(model_id, set_name, label))
                rows.append(line[f"inputs__{set_name}"][k])
                table.append(
                    {
                        "model_id": model_id,
                        "state": label,
                        "mass": float(line[f"mass__{set_name}"][k]),
                        "effective_count": float(line[f"effective__{set_name}"][k]),
                        "merged": str(line[f"merged__{set_name}"][k]),
                    }
                )
            names.append(cs.context_name(model_id, set_name, cs.WHOLE))
            rows.append(line["whole"])
            table.append(
                {
                    "model_id": model_id,
                    "state": cs.WHOLE,
                    "mass": float("nan"),
                    "effective_count": float(line["n_cells"]),
                    "merged": "",
                }
            )
        index = pd.Index(names, name="context")
        inputs = pd.DataFrame(np.vstack(rows), index=index, columns=t.panel)
        write_parquet(t.layout.state_set(set_name) / "inputs.parquet", inputs)
        write_parquet(
            t.layout.state_set(set_name) / "masses.parquet",
            pd.DataFrame(table, index=index),
        )
        counts[set_name] = len(names)
    return counts


def step_inputs(t: Telemetry) -> dict[str, Any]:
    done = 0
    for model_id in t.projected:
        if t.layout.line(model_id).is_file():
            continue
        write_npz(t.layout.line(model_id), **line_inputs(t, model_id))
        done += 1
    contexts = assemble_sets(t)
    return {
        "lines": len(t.projected),
        "computed": done,
        "contexts": contexts,
        "contexts_total": int(sum(contexts.values())),
    }
