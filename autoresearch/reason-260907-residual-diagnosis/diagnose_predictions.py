"""Read-only CPU diagnosis of existing, already-observed test predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('outputs/geneeffect_joint/joint_seed0_20260906T174818Z_b1024')
joint = pd.read_parquet(root / 'evaluation/best/test/predictions.parquet')
base = pd.read_parquet(root / 'baselines/test/predictions.parquet')
genes = pd.read_csv(root / 'evaluation/best/test/per_gene.csv').gene_symbol.tolist()
keys = ['model_id', 'gene_symbol']
joint = joint.set_index(keys).sort_index()
out = {'scope': 'Descriptive diagnosis of observed test only; no fitting or selection', 'methods': {}}
frames = {'joint': joint}
for method, frame in base.groupby('method'):
    frame = frame.set_index(keys).sort_index()
    assert frame.index.equals(joint.index)
    assert np.allclose(frame.residual, joint.residual, atol=1e-6)
    frames[method] = frame
out['aligned_rows'] = len(joint)
out['variable_genes'] = len(genes)
correlations = {}
for name, frame in frames.items():
    selected = frame.loc[frame.index.get_level_values('gene_symbol').isin(genes)].reset_index()
    p = selected.pivot(index='model_id', columns='gene_symbol', values='residual_prediction')
    y = selected.pivot(index='model_id', columns='gene_symbol', values='residual')
    ps, ys = p.std(ddof=0), y.std(ddof=0)
    corr = p.corrwith(y).where(ps > 1e-12)
    correlations[name] = corr
    pc, yc = p - p.mean(), y - y.mean()
    mask = pc.notna() & yc.notna()
    pv = pc.to_numpy()[mask.to_numpy()]
    yv = yc.to_numpy()[mask.to_numpy()]
    raw_p, raw_y = selected.residual_prediction.to_numpy(), selected.residual.to_numpy()
    def huber(error):
        a = np.abs(error)
        return float(np.where(a <= 1, .5*a*a, a-.5).mean())
    summary = {
        'macro_pearson': float(corr.mean()) if corr.notna().any() else None,
        'positive_gene_fraction': float((corr > 0).mean()) if corr.notna().any() else None,
        'prediction_std_median': float(ps.median()), 'truth_std_median': float(ys.median()),
        'std_ratio_quantiles': (ps/ys).quantile([.1,.5,.9]).to_dict(),
        'centered_prediction_rms': float(np.sqrt(np.mean(pv*pv))),
        'centered_truth_rms': float(np.sqrt(np.mean(yv*yv))),
        'centered_pooled_correlation': float(np.dot(pv,yv)/np.sqrt(np.dot(pv,pv)*np.dot(yv,yv))) if np.dot(pv,pv)>1e-20 else None,
        'variable_huber': huber(raw_p-raw_y), 'variable_prior_huber': huber(raw_y),
    }
    # Spectrum is descriptive: complete observed genes only, each gene centered
    # across contexts. No missing-label imputation, no model output adjustment.
    complete = pc.loc[:, pc.notna().all()]
    line_offset = complete.mean(axis=1).to_numpy()
    summary['common_line_offset_energy_fraction'] = float(np.sum(line_offset**2)*complete.shape[1]/np.sum(complete.to_numpy()**2)) if np.sum(complete.to_numpy()**2)>1e-20 else None
    s = np.linalg.svd(complete.to_numpy(), compute_uv=False)
    e = s*s
    summary['spectrum_complete_genes'] = complete.shape[1]
    if e.sum() > 1e-20:
        summary['prediction_spectrum_top1'] = float(e[0]/e.sum())
        summary['prediction_spectrum_top3'] = float(e[:3].sum()/e.sum())
        summary['prediction_spectrum_participation_rank'] = float(e.sum()**2/(e*e).sum())
    out['methods'][name] = summary
    if name == 'joint':
        truth_complete = yc.loc[:, complete.columns]
        ts = np.linalg.svd(truth_complete.to_numpy(), compute_uv=False)**2
        out['truth_spectrum'] = {'top1': float(ts[0]/ts.sum()), 'top3': float(ts[:3].sum()/ts.sum()), 'participation_rank': float(ts.sum()**2/np.sum(ts*ts))}
diff = correlations['joint'] - correlations['context_pca_ridge[tx1]']
out['joint_minus_tx1_per_gene'] = {
    'mean': float(diff.mean()), 'median': float(diff.median()),
    'joint_wins': int((diff>0).sum()), 'ridge_wins': int((diff<0).sum()),
    'quantiles': diff.quantile([.1,.25,.5,.75,.9]).to_dict(),
}
print(json.dumps(out, indent=2, allow_nan=False))
