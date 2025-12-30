"""
Loss utilities for scGPT finetuning.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _unique_perts(perts: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for pert in perts:
        if pert == "ctrl" or pert in seen:
            continue
        ordered.append(pert)
        seen.add(pert)
    return ordered


def _torch_isin(values: torch.Tensor, test_values: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "isin"):
        return torch.isin(values, test_values)
    mask = np.isin(values.detach().cpu().numpy(), test_values.detach().cpu().numpy())
    return torch.from_numpy(mask).to(values.device)


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return x.mean()


def build_de_gene_map(
    adata,
    gene_names: np.ndarray,
    fdr_threshold: float = 0.05,
) -> Dict[str, torch.Tensor]:
    gene_names = np.asarray(gene_names, dtype=str)
    name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    de_map: Dict[str, torch.Tensor] = {}

    uns = getattr(adata, "uns", {}) if adata is not None else {}

    if isinstance(uns, Mapping) and "de_genes" in uns:
        raw = uns.get("de_genes")
        if isinstance(raw, Mapping):
            for pert, genes in raw.items():
                if pert == "ctrl":
                    continue
                genes = [g for g in genes if g in name_to_idx]
                if genes:
                    de_map[str(pert)] = torch.tensor(
                        [name_to_idx[g] for g in genes], dtype=torch.long
                    )

    if de_map:
        return de_map

    if isinstance(uns, Mapping) and "rank_genes_groups" in uns:
        rg = uns.get("rank_genes_groups", {})
        names = rg.get("names")
        pvals_adj = rg.get("pvals_adj")

        if isinstance(names, np.ndarray) and names.dtype.names:
            for group in names.dtype.names:
                genes = np.asarray(names[group]).astype(str)
                if pvals_adj is not None:
                    pvals = np.asarray(pvals_adj[group]).astype(float)
                    genes = genes[pvals < fdr_threshold]
                genes = [g for g in genes if g in name_to_idx]
                if genes and group != "ctrl":
                    de_map[str(group)] = torch.tensor(
                        [name_to_idx[g] for g in genes], dtype=torch.long
                    )
        elif isinstance(names, np.ndarray) and names.ndim == 2:
            groups = rg.get("groups") or rg.get("group_names")
            if groups is None:
                groups = [f"group_{i}" for i in range(names.shape[1])]
            for idx, group in enumerate(groups):
                genes = np.asarray(names[:, idx]).astype(str)
                if pvals_adj is not None:
                    pvals = np.asarray(pvals_adj[:, idx]).astype(float)
                    genes = genes[pvals < fdr_threshold]
                genes = [g for g in genes if g in name_to_idx]
                if genes and group != "ctrl":
                    de_map[str(group)] = torch.tensor(
                        [name_to_idx[g] for g in genes], dtype=torch.long
                    )

    return de_map


def sliced_wasserstein_1(
    pred: torch.Tensor,
    truth: torch.Tensor,
    num_projections: int,
) -> torch.Tensor:
    if pred.shape[0] < 2 or truth.shape[0] < 2 or num_projections <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred = pred.float()
    truth = truth.float()
    n_features = pred.shape[1]

    projections = torch.randn(num_projections, n_features, device=pred.device)
    projections = F.normalize(projections, dim=1)

    proj_pred = pred @ projections.T
    proj_truth = truth @ projections.T

    proj_pred_sorted, _ = torch.sort(proj_pred, dim=0)
    proj_truth_sorted, _ = torch.sort(proj_truth, dim=0)

    return torch.mean(torch.abs(proj_pred_sorted - proj_truth_sorted))


def proto_info_nce_loss(
    pred_deltas: torch.Tensor,
    truth_deltas: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if pred_deltas.shape[0] <= 1:
        return torch.tensor(0.0, device=pred_deltas.device, dtype=pred_deltas.dtype)

    pred_norm = F.normalize(pred_deltas.float(), dim=1)
    truth_norm = F.normalize(truth_deltas.float(), dim=1)
    logits = pred_norm @ truth_norm.T
    logits = logits / max(tau, 1e-6)

    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)


def de_rank_loss(
    pred_delta: torch.Tensor,
    de_mask: torch.Tensor,
    tau: float,
    max_de: int,
    max_non_de: int,
) -> torch.Tensor:
    de_idx = torch.nonzero(de_mask, as_tuple=False).flatten()
    non_idx = torch.nonzero(~de_mask, as_tuple=False).flatten()
    if de_idx.numel() == 0 or non_idx.numel() == 0:
        return torch.tensor(0.0, device=pred_delta.device, dtype=pred_delta.dtype)

    if de_idx.numel() > max_de:
        de_idx = de_idx[
            torch.randperm(de_idx.numel(), device=pred_delta.device)[:max_de]
        ]
    if non_idx.numel() > max_non_de:
        non_idx = non_idx[
            torch.randperm(non_idx.numel(), device=pred_delta.device)[:max_non_de]
        ]

    scores = pred_delta.abs()
    diff = (scores[de_idx][:, None] - scores[non_idx][None, :]) / max(tau, 1e-6)
    return _safe_mean(F.softplus(-diff))


def de_direction_loss(
    pred_delta: torch.Tensor,
    truth_delta: torch.Tensor,
    de_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    sign = torch.sign(truth_delta)
    mask = de_mask & (sign != 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_delta.device, dtype=pred_delta.dtype)

    logits = (sign[mask] * pred_delta[mask]) / max(tau, 1e-6)
    return _safe_mean(F.softplus(-logits))


def compute_composite_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    perts: Iterable[str],
    ctrl_mean: torch.Tensor,
    de_gene_map: Mapping[str, torch.Tensor] | None,
    gene_indices: torch.Tensor,
    config: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = config.get("loss", {})
    sw_weight = loss_cfg.get("sw1_weight", 0.60)
    proto_weight = loss_cfg.get("proto_weight", 0.25)
    de_rank_weight = loss_cfg.get("de_rank_weight", 0.10)
    dir_weight = loss_cfg.get("dir_weight", 0.05)

    num_proj = int(loss_cfg.get("sw1_projections", 32))
    proto_tau = float(loss_cfg.get("proto_tau", 0.1))
    de_rank_tau = float(loss_cfg.get("de_rank_tau", 0.2))
    dir_tau = float(loss_cfg.get("dir_tau", 0.2))
    max_de = int(loss_cfg.get("de_rank_sample_de", 256))
    max_non_de = int(loss_cfg.get("de_rank_sample_non_de", 256))

    unique_perts = _unique_perts(perts)
    if not unique_perts:
        zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return zero, {"dist": 0.0, "proto": 0.0, "de_rank": 0.0, "dir": 0.0}

    pred = pred.float()
    truth = truth.float()
    ctrl_mean = ctrl_mean.float()

    dist_losses = []
    pred_deltas = []
    truth_deltas = []
    de_rank_losses = []
    dir_losses = []

    perts_list = list(perts)
    for pert in unique_perts:
        mask = torch.tensor([p == pert for p in perts_list], device=pred.device)
        if mask.sum() == 0:
            continue
        pred_pert = pred[mask]
        truth_pert = truth[mask]

        dist_losses.append(sliced_wasserstein_1(pred_pert, truth_pert, num_proj))

        pred_mean = pred_pert.mean(dim=0)
        truth_mean = truth_pert.mean(dim=0)
        pred_delta = pred_mean - ctrl_mean
        truth_delta = truth_mean - ctrl_mean
        pred_deltas.append(pred_delta)
        truth_deltas.append(truth_delta)

        if de_gene_map is None or pert not in de_gene_map:
            continue

        de_idx_full = de_gene_map[pert].to(pred.device)
        if de_idx_full.numel() == 0:
            continue

        de_mask = _torch_isin(gene_indices, de_idx_full)
        if de_mask.sum() == 0 or (~de_mask).sum() == 0:
            continue

        de_rank_losses.append(
            de_rank_loss(pred_delta, de_mask, de_rank_tau, max_de, max_non_de)
        )
        dir_losses.append(de_direction_loss(pred_delta, truth_delta, de_mask, dir_tau))

    dist_loss = (
        _safe_mean(torch.stack(dist_losses)) if dist_losses else pred.new_tensor(0.0)
    )
    if pred_deltas and truth_deltas:
        pred_delta_mat = torch.stack(pred_deltas)
        truth_delta_mat = torch.stack(truth_deltas)
        proto_loss = proto_info_nce_loss(pred_delta_mat, truth_delta_mat, proto_tau)
    else:
        proto_loss = pred.new_tensor(0.0)

    de_rank_loss_val = (
        _safe_mean(torch.stack(de_rank_losses))
        if de_rank_losses
        else pred.new_tensor(0.0)
    )
    dir_loss_val = (
        _safe_mean(torch.stack(dir_losses)) if dir_losses else pred.new_tensor(0.0)
    )

    total = (
        sw_weight * dist_loss
        + proto_weight * proto_loss
        + de_rank_weight * de_rank_loss_val
        + dir_weight * dir_loss_val
    )

    return total, {
        "dist": float(dist_loss.detach().cpu().item()),
        "proto": float(proto_loss.detach().cpu().item()),
        "de_rank": float(de_rank_loss_val.detach().cpu().item()),
        "dir": float(dir_loss_val.detach().cpu().item()),
    }
