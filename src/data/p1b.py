"""Fixed P1-B membership and deterministic three-anchor exposure."""

import math
import numpy as np

from src.data.p1c import INPUT_LAYOUTS, TRANSFORMS, apply_transform, bundle_transform

SOURCE_ANCHORS = ("ACH-000551", "ACH-000739", "ACH-000971")
EXTERNAL_ANCHOR = "ACH-000995"


def split_conditions(keys, holdout, anchors=SOURCE_ANCHORS, external=EXTERNAL_ANCHOR):
    if len(set(keys)) != len(keys) or len(set(anchors)) != 3 or external in anchors:
        raise ValueError(
            "unique conditions and disjoint three-source/external anchors required"
        )
    result = {name: [] for name in ("train", "val", "external")}
    for i, key in enumerate(keys):
        if key[0] == external:
            result["external"].append(i)
        elif key[0] in anchors:
            result["val" if key in holdout else "train"].append(i)
        else:
            raise ValueError(f"unknown anchor {key[0]}")
    for role in ("train", "val"):
        if {keys[i][0] for i in result[role]} != set(anchors):
            raise ValueError(f"{role} must cover all source anchors")
    if not result["external"]:
        raise ValueError("external conditions required")
    return result


def balanced_epoch(keys, indices, anchors, epoch, *, per_anchor=64):
    pools = [[i for i in indices if keys[i][0] == a] for a in anchors]
    if per_anchor < 1 or any(not pool for pool in pools):
        raise ValueError("nonempty anchor pools and positive batch required")
    rngs = [
        np.random.default_rng(np.random.SeedSequence([0, epoch, a]))
        for a in range(len(anchors))
    ]
    orders = [r.permutation(p).tolist() for r, p in zip(rngs, pools)]
    positions = [0] * len(pools)
    for _ in range(math.ceil(max(map(len, pools)) / per_anchor)):
        batch = []
        for a, pool in enumerate(pools):
            for _ in range(per_anchor):
                if positions[a] == len(pool):
                    orders[a] = rngs[a].permutation(pool).tolist()
                    positions[a] = 0
                batch.append(orders[a][positions[a]])
                positions[a] += 1
        yield batch


def common_coordinates(order, vocabularies, *, expected=1957):
    measured = set.intersection(*(set(v) for v in vocabularies))
    indices = [i for i, gene in enumerate(order) if gene in measured]
    if len(indices) != expected:
        raise ValueError(
            f"common measured coordinates: expected {expected}, found {len(indices)}"
        )
    return indices


def derangements(genes, *, repeats=10):
    genes = sorted(set(genes))
    if len(genes) < 2:
        return []
    rng = np.random.default_rng(0)
    result = []
    for _ in range(repeats):
        while True:
            wrong = rng.permutation(genes).tolist()
            if all(a != b for a, b in zip(genes, wrong)):
                break
        result.append(dict(zip(genes, wrong)))
    return result


def source_vocabularies(sources):
    """Read gene metadata only; mirror response alignment without uppercasing."""
    import pandas as pd

    result = {}
    for anchor, source in sources.items():
        column = source.get("target_gene_symbol_col", "gene_name")
        if source["source_type"] == "h5ad":
            import h5py
            from anndata.io import read_elem

            with h5py.File(source["h5ad_path"], "r") as handle:
                result[anchor] = set(read_elem(handle["var"])[column].astype(str))
        elif source["source_type"] == "xatlas_orion_parquet":
            result[anchor] = set(
                pd.read_parquet(source["gene_metadata_path"], columns=[column])[
                    column
                ].astype(str)
            )
        else:
            raise ValueError("unsupported response source")
    return result


def build_snapshot(
    inputs,
    coordinates,
    native_genes,
    *,
    anchors=SOURCE_ANCHORS,
    external=EXTERNAL_ANCHOR,
    transform="raw",
    target_sum=None,
):
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}")
    cache = inputs.response_targets
    keys = cache.keys
    splits = split_conditions(keys, inputs.response_holdout, anchors, external)
    seen = {keys[i][1] for i in splits["train"]}
    # Computed once per source anchor, not once per training condition.
    basal_mean = {
        a: apply_transform(
            np.asarray(inputs.lines[a].basal_hvg), transform, target_sum
        ).mean(0)
        for a in anchors
    }
    baselines = fit_baselines(
        (
            keys[i][0],
            keys[i][1],
            apply_transform(cache.target_bag(i), transform, target_sum).mean(0)
            - basal_mean[keys[i][0]],
        )
        for i in splits["train"]
    )
    panels = {}
    for role, indices in splits.items():
        for anchor in (*anchors, external):
            rows = [i for i in indices if keys[i][0] == anchor]
            if not rows:
                continue
            for name, select in {
                "all": lambda g: True,
                "seen": lambda g: g in seen,
                "unseen": lambda g: g not in seen,
                "native_common": lambda g: g in seen and g in native_genes,
                "native_all": lambda g: g in native_genes,
            }.items():
                selected = [i for i in rows if select(keys[i][1])]
                if selected:
                    panels[f"{role}/{anchor}/{name}"] = {
                        "indices": selected,
                        "derangements": derangements([keys[i][1] for i in selected]),
                    }
    result = {
        "keys": list(keys),
        "splits": splits,
        "anchors": list(anchors),
        "external": external,
        "hvg_order": list(inputs.hvg_order),
        "coordinates": coordinates,
        "joint_holdout": [key in inputs.response_holdout for key in keys],
        "controls": {
            a: {
                "tx1": np.array(inputs.lines[a].controls_tx1),
                "hvg": apply_transform(
                    np.array(inputs.lines[a].basal_hvg), transform, target_sum
                ),
            }
            for a in (*anchors, external)
        },
        "baselines": baselines,
        "panels": panels,
    }
    # Omitted for "raw" so a default-transform bundle is byte-identical to one
    # built before this feature existed; ``bundle_transform`` covers readers.
    if transform != "raw":
        result["transform"] = {
            "name": transform,
            "target_sum": target_sum,
            "row_sum_basis": "hvg_panel",
        }
    return result


class ResponseView:
    """Prepared controls plus memory-mapped targets; no raw-data rebuilding."""

    def __init__(self, bundle, cache, input_layout="tx1"):
        if list(cache.keys) != [tuple(k) for k in bundle["keys"]]:
            raise ValueError("prepared condition identity/order changed")
        if input_layout not in INPUT_LAYOUTS:
            raise ValueError(f"unknown input_layout {input_layout!r}")
        self.bundle, self.cache = bundle, cache
        self.input_layout = input_layout
        self.keys = tuple(cache.keys)
        self._controls = {}
        transform = bundle_transform(bundle)
        self.transform_name = transform["name"]
        self.transform_target_sum = transform["target_sum"]

    def batch(self, indices, device="cpu"):
        import torch
        from src.data.batches import ResponseBatch

        keys = [self.keys[i] for i in indices]

        def tensor(array):
            return torch.tensor(np.asarray(array), dtype=torch.float32, device=device)

        device_key = str(device)
        if device_key not in self._controls:
            per_anchor = {}
            for a, bags in self.bundle["controls"].items():
                hvg, tx1 = tensor(bags["hvg"]), tensor(bags["tx1"])
                if self.input_layout == "tx1":
                    controls_tx1 = tx1
                elif self.input_layout == "hvg":
                    controls_tx1 = hvg
                else:
                    controls_tx1 = torch.cat([hvg, tx1], dim=1)
                per_anchor[a] = {"controls_tx1": controls_tx1, "hvg": hvg}
            self._controls[device_key] = per_anchor
        controls = self._controls[device_key]
        return ResponseBatch(
            tuple(a for a, g in keys),
            tuple(g for a, g in keys),
            tuple(controls[a]["controls_tx1"] for a, g in keys),
            tuple(
                tensor(
                    apply_transform(
                        self.cache.target_bag(i),
                        self.transform_name,
                        self.transform_target_sum,
                    )
                )
                for i in indices
            ),
            tuple(controls[a]["hvg"] for a, g in keys),
        )


def fit_baselines(rows):
    by_anchor, by_gene = {}, {}
    for anchor, gene, effect in rows:
        by_anchor.setdefault(anchor, []).append(effect)
        by_gene.setdefault(gene, []).append(effect)
    return {
        "global": np.mean([np.mean(v, axis=0) for v in by_anchor.values()], axis=0),
        "genes": {g: np.mean(v, axis=0) for g, v in by_gene.items()},
    }
