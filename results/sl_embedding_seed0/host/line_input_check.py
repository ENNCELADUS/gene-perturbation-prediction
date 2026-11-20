"""Inference-time check of the finished model without cell tokens.

It shows what the SL score depends on.

Run on the H20 from the repo root, code bee1bcc:
    .venv-tx1/bin/python - < line_input_check.py
No model is refitted; it is no retrained control and it gives no intervals.
"""

from pathlib import Path

import numpy as np
import torch

from src.experiments.sl import embedding as e
from src.experiments.sl.benchmark_baselines import strata

config = e.load_config(Path("configs/sl/embedding_mean_token.yaml"))
data, base, counts = e.load_data(config, torch.device("cuda"))
model = e.build_model(config, data)
run = "outputs/sl/embedding/embedding_mean_token_seed0"
model.load_state_dict(torch.load(f"{run}/best.pt")["weights"])
pairs = e.pair_rows(data)
rows = data.rows
every = np.arange(len(rows))
held = (rows["split"] != "train").to_numpy()


def show(name, score, mask=None):
    chosen = rows if mask is None else rows[mask]
    metrics = e.sl_metrics(chosen, score if mask is None else score[mask], 20)
    cells = [name]
    for split in ("val", "test"):
        m = metrics[split]
        cells += [f"{m['recall_top10']:.4f}", f"{m['auroc']:.4f}", str(m["lines"])]
    print("| " + " | ".join(cells) + " |")


print(
    "| Line input, pairs | Validation recall in the top 10% | Validation AUROC "
    "| Validation lines | Test recall in the top 10% | Test AUROC | Test lines |"
)
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
keep = data.expression.clone()
real = e.pair_scores(model, data, pairs, every, config)
show("The line's own input, all pairs", real)
data.expression = torch.zeros_like(keep)
mean = e.pair_scores(model, data, pairs, every, config)
show("The fit lines' mean input, all pairs", mean)
generator = torch.Generator().manual_seed(0)
lines = torch.tensor(sorted(set(rows["line"])))
swapped = keep.clone()
swapped[lines] = keep[lines[torch.randperm(len(lines), generator=generator)]]
data.expression = swapped
show(
    "The input of another SL line (seeded permutation), all pairs",
    e.pair_scores(model, data, pairs, every, config),
)
data.expression = keep
stratum = strata(rows[rows["split"] == "train"], rows)
names = {
    "seen_agree": "seen pairs whose label agrees with the training lines",
    "seen_flip": "seen pairs whose label differs from the training lines",
    "new_pair": "new pairs, both genes seen",
}
for key, name in names.items():
    mask = stratum == key
    count = int((mask & held).sum())
    show(f"The line's own input, {name} ({count:,} held-out rows)", real, mask)
    show(f"The fit lines' mean input, {name}", mean, mask)
print()
print(
    "Pearson correlation of the two scores (own input, mean input) over the "
    f"held-out rows: {np.corrcoef(real[held], mean[held])[0, 1]:.4f}"
)
