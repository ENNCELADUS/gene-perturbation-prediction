"""
Compute DES (overlap@N), PDS, and MAE(top-2000) in one run.

Outputs:
- out_dir/perturbation_metrics.json: {pert: {PDS, DES, MAE}, ...} （PDS 为单值；DES/MAE为该扰动的值）
- out_dir/summary.json: {PDS: {...}, DES: {...}, MAE: {...}} 分布统计

Assumptions:
- .X 是 log1p 空间
- 顶基因 JSON 已由真实集生成，形如 {pert: [gene_idx, ...]}
- DES：仅按 FDR 过滤，按 abs log2FC 降序取前 K（默认 |real|）
"""
import argparse, json
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np
import polars as pl
from numpy.linalg import norm
from pdex import parallel_differential_expression


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute DES, PDS, MAE for pred/real h5ad.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--top-genes", required=True, help="JSON: {pert: [gene_idx,...]} from real.")
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--control", default="non-targeting")
    p.add_argument("--des-topk", type=int, default=None, help="Top-K for DES overlap; default |real|.")
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--de-method", default="wilcoxon", choices=["wilcoxon", "anderson", "t-test"])
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--real-de", help="Optional precomputed real DE CSV (Polars-readable).")
    p.add_argument("--pred-de", help="Optional precomputed pred DE CSV (Polars-readable).")
    p.add_argument("--out-dir", required=True, help="Directory to write results.")
    p.add_argument(
        "--baseline-summary",
        default=None,
        help="Optional baseline summary JSON to compute scaled metrics.",
    )
    return p.parse_args()


def compute_means(adata: ad.AnnData, pert_col: str) -> Dict[str, np.ndarray]:
    means = {}
    for pert in adata.obs[pert_col].unique():
        mask = adata.obs[pert_col] == pert
        means[str(pert)] = np.asarray(adata.X[mask.values, :].mean(axis=0)).ravel()
    return means


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def compute_pds_per_pert(pred: ad.AnnData, real: ad.AnnData, pert_col: str, control: str) -> Dict[str, float]:
    means_real = compute_means(real, pert_col)
    means_pred = compute_means(pred, pert_col)
    perts = [p for p in means_real if p != control]
    N = len(perts)
    delta_real = np.stack([means_real[p] - means_real[control] for p in perts], 0)
    delta_pred = np.stack([means_pred[p] - means_real[control] for p in perts], 0)
    S = cosine_matrix(delta_pred, delta_real)
    scores = {}
    for k, pert in enumerate(perts):
        order = np.argsort(-S[k])  # desc
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, N + 1)
        Rk = ranks[k]
        scores[pert] = float((N - Rk + 1) / N)  # high is good
    return scores


def run_de(adata: ad.AnnData, control: str, pert_col: str, method: str, num_workers: int, batch_size: int) -> pl.DataFrame:
    df = parallel_differential_expression(
        adata=adata,
        reference=control,
        groupby_key=pert_col,
        metric=method,
        num_workers=num_workers,
        batch_size=batch_size,
        is_log1p=True,
        as_polars=True,
    )
    if "log2_fold_change" not in df.columns:
        df = df.with_columns(pl.col("fold_change").log(base=2).fill_nan(0.0).alias("log2_fold_change"))
    if "abs_log2_fold_change" not in df.columns:
        df = df.with_columns(pl.col("log2_fold_change").abs().alias("abs_log2_fold_change"))
    return df


def get_sorted_genes(df: pl.DataFrame, fdr: float, topk: int | None) -> Dict[str, np.ndarray]:
    sig = df.filter(pl.col("fdr") < fdr)
    lists: Dict[str, np.ndarray] = {}
    for pert in sig["target"].unique().to_list():
        genes = (
            sig.filter(pl.col("target") == pert)
            .sort("abs_log2_fold_change", descending=True)
            .select("feature")
            .to_numpy()
            .ravel()
        )
        if topk is not None:
            genes = genes[:topk]
        lists[str(pert)] = genes
    return lists


def compute_des_per_pert(pred_df: pl.DataFrame, real_df: pl.DataFrame, fdr: float, topk: int | None) -> Dict[str, float]:
    real_lists = get_sorted_genes(real_df, fdr, topk)
    pred_lists = get_sorted_genes(pred_df, fdr, topk)
    perts = set(real_lists) & set(pred_lists)
    scores: Dict[str, float] = {}
    for pert in perts:
        real_genes = real_lists[pert]
        pred_genes = pred_lists[pert]
        k_eff = len(real_genes) if topk is None else min(len(real_genes), topk)
        if k_eff == 0:
            scores[pert] = 0.0
            continue
        overlap = np.intersect1d(real_genes[:k_eff], pred_genes[:k_eff]).size / k_eff
        scores[pert] = float(overlap)
    return scores


def compute_mae_per_pert(top_json: Path, pred: ad.AnnData, real: ad.AnnData, pert_col: str, control: str) -> Dict[str, float]:
    top = json.loads(top_json.read_text())
    means_real = compute_means(real, pert_col)
    means_pred = compute_means(pred, pert_col)
    scores: Dict[str, float] = {}
    for pert, idxs in top.items():
        if pert == control:
            continue
        if pert in real.obs[pert_col].unique() and pert in pred.obs[pert_col].unique():
            r = means_real[pert][idxs]
            p = means_pred[pert][idxs]
            scores[pert] = float(np.mean(np.abs(p - r)))
    return scores


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "std": float(np.nanstd(arr)),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = ad.read_h5ad(args.real)
    pred = ad.read_h5ad(args.pred)

    pds = compute_pds_per_pert(pred, real, args.pert_col, args.control)
    print("Computed PDS for", len(pds), "perturbations.")
    mae = compute_mae_per_pert(Path(args.top_genes), pred, real, args.pert_col, args.control)
    print("Computed MAE for", len(mae), "perturbations.")
    if args.real_de:
        real_de = pl.read_csv(args.real_de)
        print("Loaded real DE from", args.real_de)
        if "abs_log2_fold_change" not in real_de.columns:
            real_de = real_de.with_columns(
                pl.col("log2_fold_change").abs().alias("abs_log2_fold_change")
                if "log2_fold_change" in real_de.columns
                else pl.col("fold_change").log(base=2).fill_nan(0.0).abs().alias("abs_log2_fold_change")
            )
    else:
        real_de = run_de(real, args.control, args.pert_col, args.de_method, args.num_workers, args.batch_size)

    if args.pred_de:
        pred_de = pl.read_csv(args.pred_de)
        print("Loaded pred DE from", args.pred_de)
        if "abs_log2_fold_change" not in pred_de.columns:
            pred_de = pred_de.with_columns(
                pl.col("log2_fold_change").abs().alias("abs_log2_fold_change")
                if "log2_fold_change" in pred_de.columns
                else pl.col("fold_change").log(base=2).fill_nan(0.0).abs().alias("abs_log2_fold_change")
            )
    else:
        pred_de = run_de(pred, args.control, args.pert_col, args.de_method, args.num_workers, args.batch_size)
    des = compute_des_per_pert(pred_de, real_de, args.fdr, args.des_topk)
    # Assemble per-perturbation metrics (only for perts present in all three).
    perts = set(pds) & set(des) & set(mae)
    per_pert = {pert: {"PDS": pds[pert], "DES": des[pert], "MAE": mae[pert]} for pert in perts}

    # Summary distributions
    summary = {
        "PDS": summarize([pds[p] for p in perts]),
        "DES": summarize([des[p] for p in perts]),
        "MAE": summarize([mae[p] for p in perts]),
    }

    scaled = None
    if args.baseline_summary:
        base = json.loads(Path(args.baseline_summary).read_text())
        try:
            pds_base = float(base.get("PDS", base.get("PDS", np.nan)))
        except Exception:
            pds_base = float(base.get("PDS", np.nan))
        try:
            des_base = float(base.get("DES_overlap", {}).get("mean", np.nan))
        except Exception:
            des_base = np.nan
        try:
            mae_base = float(base.get("MAE_top_genes", {}).get("mean", np.nan))
        except Exception:
            mae_base = np.nan

        def safe_scaled(num, denom):
            if np.isfinite(denom) and denom != 0.0:
                return num / denom
            return np.nan

        pds_scaled = safe_scaled(summary["PDS"]["mean"] - pds_base, 1.0 - pds_base)
        des_scaled = safe_scaled(summary["DES"]["mean"] - des_base, 1.0 - des_base)
        mae_scaled = safe_scaled(mae_base - summary["MAE"]["mean"], mae_base)
        if np.isfinite(pds_scaled) and np.isfinite(des_scaled) and np.isfinite(mae_scaled):
            score = (pds_scaled + des_scaled + mae_scaled) / 3 * 100
        else:
            score = np.nan
        scaled = {
            "PDS_scaled": pds_scaled,
            "DES_scaled": des_scaled,
            "MAE_scaled": mae_scaled,
            "Score": score,
        }

    (out_dir / "perturbation_metrics.json").write_text(json.dumps(per_pert, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_out = {"summary": summary}
    if scaled is not None:
        summary_out["scaled"] = scaled
    (out_dir / "summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
