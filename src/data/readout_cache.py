"""Compact, immutable raw feature cache and train-only P1-A preprocessing."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch

from src.data.batches import FeatureBatch
from src.model.head import GeneEffectFeatureDims
from src.model.normalization import BlockStandardizer

BLOCKS = ("delta_proj", "s", "q_sc", "e_g", "z_c")
MASKS = ("q_sc_mask", "hvg_panel_mask", "own_gene_shift_mask")
PAIR_FIELDS = (
    "delta_proj",
    "s",
    "q_sc",
    *MASKS,
    "gene_index",
    "context_index",
    "residual",
    "gene_mean",
)


def _raw_batch(arrays, shared, genes, lines, indices):
    gi, ci = arrays["gene_index"][indices], arrays["context_index"][indices]
    tensors = {
        name: torch.from_numpy(np.array(arrays[name][indices], copy=True))
        for name in (*BLOCKS[:3], *MASKS)
    }
    tensors.update(
        e_g=torch.from_numpy(np.array(shared["e_g"][gi], copy=True)),
        z_c=torch.from_numpy(np.array(shared["z_c"][ci], copy=True)),
    )
    return FeatureBatch(
        **tensors,
        gene_symbols=tuple(genes[i] for i in gi),
        model_ids=tuple(lines[i] for i in ci),
    )


class ReadoutCache:
    """Memory-map pair features; expand shared gene/context values only per batch."""

    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata["status"] != "complete":
            raise ValueError("feature cache is incomplete")
        self.genes = self.metadata["genes"]
        self.lines = self.metadata["lines"]
        self.variable_genes = self.metadata["variable_genes"]
        self.dims = GeneEffectFeatureDims(**self.metadata["dims"])
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r")
            for name in ("z_c", "e_g")
        }
        self.splits = {
            split: {
                name: np.load(self.root / split / f"{name}.npy", mmap_mode="r")
                for name in PAIR_FIELDS
            }
            for split in ("train", "val")
        }
        state = torch.load(self.root / "preprocessing.pt", weights_only=True)
        self.standardizer = BlockStandardizer.from_state(state["standardizer"])
        self.context_scores = state["context_scores"].numpy()

    def __len__(self):
        return len(self.splits["train"]["residual"])

    def raw_batch(self, split, indices):
        return _raw_batch(
            self.splits[split], self.arrays, self.genes, self.lines, indices
        )

    def batch(self, split, indices, device="cpu", *, standardizer=None):
        raw = self.raw_batch(split, indices).to(device)
        scaler = self.standardizer if standardizer is None else standardizer
        values = {name: scaler.transform(name, getattr(raw, name)) for name in BLOCKS}
        values.update({name: getattr(raw, name) for name in MASKS})
        batch = FeatureBatch(
            **values, gene_symbols=raw.gene_symbols, model_ids=raw.model_ids
        )
        arrays = self.splits[split]
        gi, ci = arrays["gene_index"][indices], arrays["context_index"][indices]
        return (
            batch,
            torch.tensor(gi, device=device),
            torch.tensor(self.context_scores[ci], device=device),
            torch.tensor(arrays["residual"][indices], device=device),
        )

    def labels(self, split):
        arrays = self.splits[split]
        return pd.DataFrame(
            {
                "model_id": np.asarray(self.lines)[arrays["context_index"]],
                "gene_symbol": np.asarray(self.genes)[arrays["gene_index"]],
                "residual": arrays["residual"],
                "gene_effect": arrays["residual"].astype(float) + arrays["gene_mean"],
            }
        )


def write_feature_cache(
    root, batches, *, row_counts, split_lines, genes, variable_genes, provenance
):
    """Stream (raw FeatureBatch, residual, mean) tuples; never fit on validation."""
    root = Path(root)
    if set(batches) != {"train", "val"} or set(split_lines) != {"train", "val"}:
        raise ValueError("P1-A requires train and val only")
    lines = list(split_lines["train"]) + list(split_lines["val"])
    if len(set(lines)) != len(lines) or len(set(genes)) != len(genes):
        raise ValueError("split contexts must be disjoint and identities unique")
    if not set(variable_genes).issubset(genes):
        raise ValueError("variable genes outside gene order")
    root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "writing",
        "genes": list(genes),
        "lines": lines,
        "split_lines": split_lines,
        "variable_genes": list(variable_genes),
        "provenance": provenance,
        "row_counts": row_counts,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    gi_map, ci_map = ({v: i for i, v in enumerate(values)} for values in (genes, lines))
    shared = {}
    seen = {"e_g": set(), "z_c": set()}
    dims = None
    for split in ("train", "val"):
        destination = root / split
        destination.mkdir()
        count = row_counts[split]
        if count < 1:
            raise ValueError("empty feature-cache split")
        arrays, offset = {}, 0
        keys = set()
        for feature, residual, mean in batches[split]:
            feature = feature.to("cpu")
            feature.validate()
            current_dims = {name: getattr(feature, name).shape[1] for name in BLOCKS}
            if dims is None:
                dims = current_dims
                for name, size in (("e_g", len(genes)), ("z_c", len(lines))):
                    shared[name] = np.lib.format.open_memmap(
                        root / f"{name}.npy",
                        mode="w+",
                        dtype="float32",
                        shape=(size, dims[name]),
                    )
            if current_dims != dims:
                raise ValueError("feature dimensions changed during extraction")
            if not set(feature.model_ids).issubset(split_lines[split]):
                raise ValueError("feature contexts outside requested split")
            pair_keys = list(zip(feature.model_ids, feature.gene_symbols, strict=True))
            if len(set(pair_keys)) != len(pair_keys) or keys.intersection(pair_keys):
                raise ValueError("duplicate feature pair keys")
            keys.update(pair_keys)
            gi = np.array([gi_map[g] for g in feature.gene_symbols], dtype="int64")
            ci = np.array([ci_map[c] for c in feature.model_ids], dtype="int64")
            values = {
                name: getattr(feature, name).detach().float().numpy()
                for name in BLOCKS[:3]
            }
            values.update({name: getattr(feature, name).numpy() for name in MASKS})
            values.update(
                gene_index=gi,
                context_index=ci,
                residual=residual.detach().cpu().float().numpy(),
                gene_mean=mean.detach().cpu().float().numpy(),
            )
            size = feature.batch_size
            if any(
                values[n].shape != (size,) or not np.isfinite(values[n]).all()
                for n in ("residual", "gene_mean")
            ):
                raise ValueError("finite labels must align with feature rows")
            if offset + size > count:
                raise ValueError("cache row count exceeded")
            for name, value in values.items():
                if name not in arrays:
                    arrays[name] = np.lib.format.open_memmap(
                        destination / f"{name}.npy",
                        mode="w+",
                        dtype=value.dtype,
                        shape=(count, *value.shape[1:]),
                    )
                arrays[name][offset : offset + size] = value
            for name, indices in (("e_g", gi), ("z_c", ci)):
                value = getattr(feature, name).detach().float().numpy()
                unique, first = np.unique(indices, return_index=True)
                for index, row in zip(unique, first, strict=True):
                    if index in seen[name] and not np.array_equal(
                        shared[name][index], value[row]
                    ):
                        raise ValueError(f"shared {name} changed for a fixed identity")
                    shared[name][index] = value[row]
                    seen[name].add(index)
                if not np.array_equal(shared[name][indices], value):
                    raise ValueError(f"inconsistent repeated {name}")
            offset += size
        if offset != count:
            raise ValueError("cache row count mismatch")
        if split == "train" and seen["e_g"] != set(range(len(genes))):
            raise ValueError("all genes must have training coverage")
        for array in arrays.values():
            array.flush()
    if seen["z_c"] != set(range(len(lines))):
        raise ValueError("missing context features")
    for array in shared.values():
        array.flush()

    # Fit PCA on unique contexts, independently of the number of observed genes.
    train = np.asarray(shared["z_c"][: len(split_lines["train"])], dtype=float)
    keep = train.std(axis=0) > 0
    center, scale = train[:, keep].mean(0), train[:, keep].std(0)
    standardized = (train[:, keep] - center) / scale
    if min(standardized.shape[0] - 1, standardized.shape[1]) < 8:
        raise ValueError(
            "P1-A PCA8 requires at least nine training contexts "
            "and eight nonconstant features"
        )
    pca = PCA(n_components=8, svd_solver="full").fit(standardized)
    pc_scale = pca.transform(standardized).std(0)
    if np.any(pc_scale <= np.finfo(float).eps):
        raise ValueError("P1-A PCA8 has a degenerate component")
    scores = pca.transform((shared["z_c"][:, keep] - center) / scale) / pc_scale

    train_arrays = {
        name: np.load(root / "train" / f"{name}.npy", mmap_mode="r")
        for name in PAIR_FIELDS
    }

    def train_blocks():
        for start in range(0, row_counts["train"], 1024):
            batch = _raw_batch(
                train_arrays, shared, genes, lines, slice(start, start + 1024)
            )
            yield {name: getattr(batch, name) for name in BLOCKS}

    scaler = BlockStandardizer().fit_batches(train_blocks())
    torch.save(
        {
            "standardizer": scaler.to_state(),
            "context_scores": torch.tensor(scores, dtype=torch.float32),
            "pca": {
                key: torch.tensor(value)
                for key, value in {
                    "keep": keep,
                    "center": center,
                    "scale": scale,
                    "mean": pca.mean_,
                    "components": pca.components_,
                    "pc_scale": pc_scale,
                }.items()
            },
        },
        root / "preprocessing.pt",
    )
    metadata.update(status="complete", dims=asdict(GeneEffectFeatureDims(**dims)))
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return ReadoutCache(root)
