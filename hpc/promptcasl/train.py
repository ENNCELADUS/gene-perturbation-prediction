"""PromptCASL (Lu et al. 2025; official code at third_party/promptcasl, commit afcf7e9)
on
the multi-study SL benchmark: one model on the fit lines, every held-out line scored.

The official leave-one-cell-line-out run (``src/main_loco.py``, ``--loco_mode full
--omics_mode pair_pathway_evidence``, plus ``--omics_encoder 0``, which the README
leaves out and the script needs) trains one model for each target line. Here the fit
lines are its source lines, the validation lines take the place of the target line in
``train_model`` (the official monitor, validation ROC-AUC, picks the checkpoint), and
the held-out lines are scored with the official ``validate_model``. The official
settings are the defaults of ``main_loco.parse_arguments`` (batch 24, 50 epochs, early
stop after 6, LLM learning rate 1e-5, top-16 pathways); the model, the loss and the
training step are imported unchanged.

Rows. ``src/preprocess_datasets.py`` stores 8 arrays for every row: the BiomedBERT
token embeddings (512 x 768 floats), two GenePT vectors, two PrimeKG vectors and the
100 PrimeKG neighbour vectors of each gene, about 2.1 MiB. The 67,000 fit rows would
take
140 GB and the 549,000 held-out rows 1.1 TB, so ``PairDataset`` builds the same 8 arrays
when a batch asks for them (official tokenizer and embedding layer, official neighbour
selection), and nothing is stored. The held-out lines are scored one line at a time;
a finished line is kept in ``parts/`` and skipped on a rerun. Training lines are not
scored: their rows get the median score of the covered held-out rows (``covered``
false).
A row is covered when both genes have a GenePT vector and a PrimeKG entity with at
least 100 neighbours (the official step skips any other row), and its line has omics.

Mean-input control (``scores_mean_input.parquet``): the same weights read the line
``MEAN_INPUT`` (hpc/promptcasl/prepare.py) for each distinct held-out pair, so the score
of a pair is the same in every line.

    .venv-promptcasl/bin/python hpc/promptcasl/train.py \
        --data data/promptcasl/benchmark --out outputs/sl/promptcasl/seed0 --device 3
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OFFICIAL = ROOT / "data/promptcasl/official"
os.environ.setdefault("PROMPTCASL_BIOMEDBERT", str(OFFICIAL / "biomedbert"))
os.environ["WANDB_MODE"] = "offline"

import prepare  # noqa: E402
import sl_common  # noqa: E402

MEAN_INPUT = prepare.MEAN_INPUT
GENEPT = OFFICIAL / "GenePT_gene_protein_embedding_model_3_text.pickle"
_ADJ = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


# --- shared inputs --------------------------------------------------------


class Context:
    """What every row reads: the prompts, the tokenizer and embedding layer, the gene
    vectors and the PrimeKG tables. Built once, shared with the loader workers by
    fork."""

    def __init__(self, data: Path, genes: set[str]):
        from transformers import AutoModel, AutoTokenizer

        path = os.environ["PROMPTCASL_BIOMEDBERT"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        bert = AutoModel.from_pretrained(path)
        # official get_token_embeddings reads ``model.embeddings`` only
        self.embedder = type("Embedder", (), {"embeddings": bert.embeddings.eval()})()
        self.lines_info = json.loads((data / "lines.json").read_text())
        template = data / "text_template" / prepare.PROMPT_VERSION
        self.desc = {
            line: json.loads(p.read_text())
            for line in self.lines_info
            if (
                p := template
                / f"{line}_gene_omics_descriptions_{prepare.PROMPT_VERSION}.json"
            ).exists()
        }
        with open(GENEPT, "rb") as f:
            table = pickle.load(f)
        self.genept = {}
        for symbol, vector in table.items():
            if symbol.upper() in genes and symbol.upper() not in self.genept:
                self.genept[symbol.upper()] = np.asarray(vector, dtype=np.float32)
        conve = np.load(data / "kg" / "conv_primekg_entity_emb768.npy")
        self.kg_emb = conve.astype(np.float32)
        entity = json.loads((data / "kg" / "entity_to_id.json").read_text())
        self.kg_id = {g: entity[g] for g in genes if g in entity}
        triples = np.load(data / "kg" / "train_triples.npy")
        self.adj = prepare.vendored("scripts.build_kg_neighbors").build_adjacency(
            triples
        )


def _neighbors(pair):
    return prepare.neighbor_ids(_ADJ, *pair)


def neighbor_table(ctx: Context, pairs: pd.DataFrame, workers: int):
    """Neighbour ids [n, 2, 100] of each (gene_1, gene_2) and the rows that have
    them."""
    global _ADJ
    _ADJ = ctx.adj
    known = pairs["gene_1"].isin(ctx.kg_id) & pairs["gene_2"].isin(ctx.kg_id)
    keys = [
        (ctx.kg_id[a], ctx.kg_id[b])
        for a, b in zip(pairs.loc[known, "gene_1"], pairs.loc[known, "gene_2"])
    ]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        found = pool.map(_neighbors, keys, chunksize=2000)
    ids = np.zeros((len(pairs), 2, prepare.NEIGHBOR_COUNT), dtype=np.int32)
    ok = np.zeros(len(pairs), dtype=bool)
    for i, got in zip(np.flatnonzero(known.to_numpy()), found):
        if got is not None:
            ids[i], ok[i] = got, True
    return ids, ok


class PairDataset(Dataset):
    """The rows of ``table`` (model_id, gene_1, gene_2, label, plus ``ids`` the
    neighbour
    ids) as the official GenePairDataset item: 8 float tensors, the label and
    (gene_pair, line)."""

    def __init__(self, ctx: Context, table: pd.DataFrame, ids: np.ndarray):
        self.ctx, self.ids = ctx, ids
        self.line = table["model_id"].to_numpy()
        self.a, self.b = table["gene_1"].to_numpy(), table["gene_2"].to_numpy()
        self.label = table["label"].to_numpy()

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        ctx, a, b, line = self.ctx, self.a[i], self.b[i], self.line[i]
        dsp = prepare.vendored("src.preprocess_datasets")
        _, mask, token = dsp.get_biobert_token_emb_for_pair(
            line,
            a,
            b,
            ctx.lines_info,
            ctx.desc[line],
            ctx.embedder,
            ctx.tokenizer,
            "cpu",
        )
        kg = ctx.kg_emb
        item = (
            token.numpy(), mask.numpy(),
            ctx.genept[a], kg[ctx.kg_id[a]], ctx.genept[b], kg[ctx.kg_id[b]],
            kg[self.ids[i, 0]], kg[self.ids[i, 1]],
        )  # fmt: skip
        tensors = tuple(torch.from_numpy(np.ascontiguousarray(x)).float() for x in item)
        return tensors, torch.tensor(int(self.label[i])).long(), (f"{a}|{b}", line)


def worker_init(_):
    torch.set_num_threads(1)


def loader(ds, batch, workers, shuffle=False):
    """A row is 2.1 MiB and the H20 container has 4 GB of /dev/shm (a loader that fills
    it dies with a bus error): at most 600 rows are in flight, one batch per worker."""
    workers = min(workers, max(1, 600 // batch))
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        worker_init_fn=worker_init, prefetch_factor=1,
    )  # fmt: skip


def usable(table: pd.DataFrame, ctx: Context, ids_ok: np.ndarray) -> np.ndarray:
    """Rows whose line has prompts and omics, whose genes have GenePT vectors, and whose
    pair has PrimeKG neighbours."""
    return (
        table["model_id"].isin(set(ctx.desc)).to_numpy()
        & table["gene_1"].isin(set(ctx.genept)).to_numpy()
        & table["gene_2"].isin(set(ctx.genept)).to_numpy()
        & ids_ok
    )


# --- patches of the official pair builder -------------------------------


def patch_pair_builder(data: Path, lineage_of: dict, fit_lines: list[str]) -> None:
    """The official builder reads its tables from the repository's data/multi_omics,
    takes
    the lineage of the six paper lines from a fixed dict, and averages the pathway
    activity over every line of a lineage. Point it at this benchmark's tables, give
    each line its DepMap lineage and average over the fit lines."""
    pp = prepare.vendored("src.pair_pathway")
    pp._DEFAULT_ARTIFACT_DIR = str(data / "multi_omics")
    pp.LINEAGE.update(lineage_of)
    original = pp.PairPathwayEvidenceBuilder.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.lineage_mean = prepare.fit_lineage_means(
            self.activity, lineage_of, fit_lines
        )

    pp.PairPathwayEvidenceBuilder.__init__ = init


# --- scoring ------------------------------------------------------------


def score_pairs(models, ctx, table, ids, batch, workers, device):
    """Official ``validate_model`` over ``table``; labels are dummies (alternating, so
    that
    its AUROC exists) and only its probabilities are read."""
    train_model = prepare.vendored("src.train_model")
    if len(table) == 1:  # validate_model needs both classes
        table, ids = pd.concat([table, table]), np.concatenate([ids, ids])
    table = table.assign(label=np.arange(len(table)) % 2)
    llm, agg, ctx_attn, predictor, encoder, fusion, builder = models
    got = train_model.validate_model(
        loader(PairDataset(ctx, table, ids), batch, workers),
        llm, agg, ctx_attn, predictor, device,
        pathway_encoder=encoder, omics_mode="pair_pathway_evidence",
        text_pathway_fusion=fusion, pair_builder=builder, return_preds=True,
    )  # fmt: skip
    assert got["gene_pairs"] == list(table["gene_1"] + "|" + table["gene_2"])
    return np.asarray(got["probs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument(
        "--epochs", type=int, default=50, help="the official num_epochs"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--score-batch", type=int, default=96)
    cli = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # TF32 matrix products: BiomedBERT runs on 512 padded tokens, a row costs about 25 ms in fp32
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{cli.device}")
    out, data = cli.out, cli.data
    (out / "parts").mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(data / "rows.parquet")
    fit = pd.read_csv(data / "labels" / "fit_full.csv")
    val = pd.read_csv(data / "labels" / "val_full.csv")
    for table in (fit, val):  # the official columns, in this module's names
        table["model_id"] = table["cell_line_origin"]
        table["label"] = (table["SL_or_not"] == "SL").astype(int)
    held = rows[rows["side"].isin(sl_common.HELD_SIDES)].drop_duplicates(
        ["model_id", "gene_1", "gene_2"]
    )[["model_id", "gene_1", "gene_2", "side"]]
    genes = set(pd.concat([fit, val, held])[["gene_1", "gene_2"]].to_numpy().ravel())
    t0 = time.time()
    ctx = Context(data, genes)
    pairs = pd.concat([fit, val, held])[["gene_1", "gene_2"]].drop_duplicates()
    pair_ids, pair_ok = neighbor_table(ctx, pairs.reset_index(drop=True), 32)
    where = pd.MultiIndex.from_frame(pairs.reset_index(drop=True))
    print(
        f"inputs ready in {time.time() - t0:.0f} s; neighbours for "
        f"{pair_ok.sum()} of {len(pairs)} pairs"
    )

    def with_ids(table):
        index = where.get_indexer(pd.MultiIndex.from_frame(table[["gene_1", "gene_2"]]))
        return pair_ids[index], pair_ok[index]

    parts = {}
    for name, table in (("fit", fit), ("val", val), ("held", held)):
        ids, ok = with_ids(table)
        keep = usable(table, ctx, ok)
        parts[name] = (table[keep].reset_index(drop=True), ids[keep])
    lines_info = ctx.lines_info
    lineage_of = {m: (i["lineage"] or "unknown") for m, i in lines_info.items()}
    fit_lines = sorted(
        rows.loc[rows["side"].isin(sl_common.FIT_SIDES), "model_id"].unique()
    )
    patch_pair_builder(data, lineage_of, [m for m in fit_lines if m in ctx.desc])

    # --- official arguments and training (src/main_loco.py: run_loco_target) ----------
    main_loco = prepare.vendored("src.main_loco")
    save = out / "train"
    save.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "main_loco", "--loco_mode", "full", "--omics_mode", "pair_pathway_evidence",
        "--omics_encoder", "0", "--num_epochs", str(cli.epochs),
        "--device", str(cli.device), "--save_folder", str(out),
    ]  # fmt: skip
    args = main_loco.parse_arguments()
    args.pathway_set, args.device = "reactome", device
    args.scenario, args.fold_idx, args.dataset_path = "LOCO", None, str(data)
    logger = main_loco.setup_logger("promptcasl_train", str(save / "training.log"))
    logger.info("Command-line arguments:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    from transformers import AutoModel

    bert = AutoModel.from_pretrained(args.biobert_path).to(device)
    train_ds = PairDataset(ctx, *parts["fit"])
    val_ds = PairDataset(ctx, *parts["val"])
    item_bytes = sum(t.numel() * t.element_size() for t in train_ds[0][0])
    train_loader = loader(train_ds, args.batch_size, cli.workers, shuffle=True)
    val_loader = loader(val_ds, args.batch_size, cli.workers)
    t_train = time.time()
    main_loco.train_model(
        train_loader=train_loader, val_loader=val_loader, bioLLM=bert, args=args,
        save_path=str(save), logger=logger, omics_data=args.omics_data,
        omics_encoder=args.omics_encoder,
    )  # fmt: skip
    t_train = time.time() - t_train
    log = (save / "training.log").read_text()
    epochs_run = len(re.findall(r"Epoch \d+/\d+ - Train Loss", log))

    # --- the checkpoint of the best validation AUROC, then the scores ------------
    llm, agg, ctx_attn, predictor, *_, ckpt_args = main_loco.load_models_from_log(
        str(save), bert, scenario="LOCO", fold_idx=1, device=device, single_model=True
    )
    encoder, _, _, fusion, builder = main_loco._load_pathway_encoder(
        str(save),
        ckpt_args,
        scenario="LOCO",
        fold_idx=1,
        device=device,
        single_model=True,
    )
    models = (llm, agg, ctx_attn, predictor, encoder, fusion, builder)
    held_ok, held_ids = parts["held"]
    t_score, scored = time.time(), 0
    pieces = []
    for line, table in held_ok.groupby("model_id", sort=True):
        part = out / "parts" / f"{line}.parquet"
        if not part.exists():
            ids = held_ids[table.index]
            probs = score_pairs(
                models, ctx, table, ids, cli.score_batch, cli.workers, device
            )
            table.assign(score=probs).to_parquet(part, index=False)
            scored += len(table)
        pieces.append(pd.read_parquet(part))
    own = pd.concat(pieces)
    t_own = time.time() - t_score
    # mean input: each distinct held-out pair once, in chunks
    pair_rows = held_ok.drop_duplicates(["gene_1", "gene_2"]).assign(
        model_id=MEAN_INPUT
    )
    pair_scores = []
    for start in range(0, len(pair_rows), 50_000):
        part = out / "parts" / f"{MEAN_INPUT}_{start}.parquet"
        table = pair_rows.iloc[start : start + 50_000]
        if not part.exists():
            ids = held_ids[table.index]
            probs = score_pairs(
                models, ctx, table, ids, cli.score_batch, cli.workers, device
            )
            table.assign(score=probs).to_parquet(part, index=False)
        pair_scores.append(pd.read_parquet(part))
    pair_scores = pd.concat(pair_scores)[["gene_1", "gene_2", "score"]]
    spread = held_ok.merge(pair_scores, on=["gene_1", "gene_2"], how="left")
    mean = spread.set_index(["model_id", "gene_1", "gene_2"])["score"]
    t_all = time.time() - t_score
    report = sl_common.write_line_scores(
        out, rows, own.set_index(["model_id", "gene_1", "gene_2"])["score"]
    )
    report_mean = sl_common.write_line_scores(out, rows, mean, name="scores_mean_input")
    best = pd.read_csv(save / "final_results.csv").iloc[-1].to_dict()
    inputs = {
        "genept": sha256(GENEPT),
        **{
            p: sha256(data / p)
            for p in (
                "ccle/GE.csv", "ccle/CNGene.csv", "ccle/MU.csv", "kg/train_triples.npy",
                "kg/conv_primekg_entity_emb768.npy", "labels/fit_full.csv",
                "labels/val_full.csv", "rows.parquet",
            )
        },
    }  # fmt: skip
    metrics = {
        "model": "PromptCASL, one model on the fit lines",
        "official_commit": "afcf7e9221f873a6f13e4db8edb570c7810f9bc9",
        "prompt_version": prepare.PROMPT_VERSION,
        "omics_mode": "pair_pathway_evidence",
        "settings": {k: str(v) for k, v in vars(args).items()},
        "rows": {
            "fit": {"table": len(fit), "used": len(parts["fit"][0])},
            "val": {"table": len(val), "used": len(parts["val"][0])},
            "held_pairs": {"table": len(held), "used": len(held_ok)},
        },
        "scores": report,
        "scores_mean_input": report_mean,
        "best_validation": {k: float(v) for k, v in best.items() if k != "Fold"},
        "epochs_run": epochs_run,
        "seconds": {
            "train_including_validation": t_train,
            "score_own_input_rows_scored_this_run": t_own,
            "score_all": t_all,
        },
        "stored_row_bytes": item_bytes,
        "rows_scored_this_run": scored,
        "inputs_sha256": inputs,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {k: metrics[k] for k in ("scores", "scores_mean_input", "seconds")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
