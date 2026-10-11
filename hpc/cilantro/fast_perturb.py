"""The released per-gene Geneformer deletion, computed directly.

The released ``via_perturber.perturb_gene`` runs, per gene: the fork's
InSilicoPerturber (a 16-process filter and map, two gene-mode forward passes over
every line containing the gene, cosine similarities written to pickles nothing reads),
then the EmbExtractor twice (two more model loads, cell-mode forward passes over the
unperturbed and the deleted sequences). Only the extractor's two mean-pooled
penultimate-layer embeddings reach the output CSVs. ``Deleter`` computes those two
embeddings with the same model, the same batches of 50 lines in the same order, the
same padding, attention mask and pooling, and assembles the CSVs with the same pandas
operations, so the files are byte-identical (checked by ``check_equivalence.py``):

- the model is loaded once per process, and only its encoder runs (the masked-language
  head's logits were computed and discarded), cut after the penultimate layer, whose
  hidden state is the one pooled (``emb_layer=-1``);
- the pooling copies the released cell-mode branch: with ``<cls>`` and ``<eos>`` in the
  token dictionary (gc95M), ``get_embs`` averages positions 1 .. length-2, dropping the
  first and the last token of every sequence, even though the tokenizer ran without
  special tokens (``special_token=False``); without them, every position;
- the unperturbed embedding of a line is identical in every batch of the same size
  (measured: equal across batches of 50, 3e-8 apart between batch sizes), so each line
  is embedded once at batch size 50 at start-up and reused for the full batches of
  every gene; a gene's final partial batch is recomputed at its own size;
- the deleted sequences are embedded per gene, one forward pass each.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import torch

from fast_core import delete_token, line_order, lines_with_token


class Deleter:
    BATCH = 50  # the released forward_batch_size

    def __init__(
        self,
        dataset_path: Path,
        model_dir: Path,
        token_dictionary: Path,
        ensembl_mapping: Path,
        gene_effect_csv: Path,
        out_dir: Path,
    ):
        from datasets import load_from_disk
        from geneformer import perturber_utils as pu

        self.pu = pu
        self.out_dir = Path(out_dir)
        for sub in ("subtracted_embs", "perturbed_embs"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)
        with open(token_dictionary, "rb") as f:
            self.token_of = pickle.load(f)
        with open(ensembl_mapping, "rb") as f:
            ensembl_map = pickle.load(f)
        self.pad_token_id = self.token_of.get("<pad>")
        token_gene = {v: k for k, v in self.token_of.items()}
        self.cls_present = any("<cls>" in v for v in token_gene.values())
        self.eos_present = any("<eos>" in v for v in token_gene.values())

        # The released via_perturber's gene-effect table, columns renamed to Ensembl.
        viability = pd.read_csv(gene_effect_csv)
        viability.set_index("Unnamed: 0", inplace=True)
        renames = {}
        for col in viability.columns:
            split = col.split(" ")[0]
            if split in ensembl_map:
                renames[col] = ensembl_map[split]
        self.viability = viability.rename(columns=renames)

        ds = load_from_disk(str(dataset_path))
        self.input_ids = ds["input_ids"]
        self.lengths = ds["length"]
        self.patient_ids = ds["patient_id"]
        self.token_sets = [set(ids) for ids in self.input_ids]
        self.order = line_order(self.lengths)

        model = pu.load_model("Pretrained", 0, str(model_dir), mode="eval")
        self.model_input_size = pu.get_model_input_size(model)
        self.layer = pu.quant_layers(model) + (-1)
        encoder = model.bert.encoder
        encoder.layer = torch.nn.ModuleList(list(encoder.layer)[: self.layer])
        self.encoder = model.bert
        self.cache = self._embed_all()

    def embed(self, seqs: list[list[int]], lengths: list[int]) -> torch.Tensor:
        """The released ``get_embs`` cell-mode minibatch, for one batch."""
        pu = self.pu
        max_len = int(max(lengths))
        original_lens = torch.tensor(lengths, device="cuda")
        batch = pu.pad_tensor_list(
            [torch.tensor(s, dtype=torch.int64) for s in seqs],
            max_len,
            self.pad_token_id,
            self.model_input_size,
        )
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=batch.to("cuda"),
                attention_mask=pu.gen_attention_mask({"length": lengths}),
            )
        embs = outputs.hidden_states[self.layer]
        if self.cls_present:
            non_cls = embs[:, 1:, :]
            if self.eos_present:
                return pu.mean_nonpadding_embs(non_cls, original_lens - 2)
            return pu.mean_nonpadding_embs(non_cls, original_lens - 1)
        return pu.mean_nonpadding_embs(embs, original_lens)

    def _embed_all(self) -> torch.Tensor:
        """Every line's unperturbed embedding, each computed in a batch of exactly
        BATCH lines (the final chunk is filled up with lines already embedded)."""
        n = len(self.order)
        rows = []
        for start in range(0, n, self.BATCH):
            chunk = self.order[start : start + self.BATCH]
            fill = self.order[: self.BATCH - len(chunk)]
            embs = self.embed(
                [self.input_ids[i] for i in chunk + fill],
                [self.lengths[i] for i in chunk + fill],
            )
            rows.append(embs[: len(chunk)].cpu())
        cache = torch.empty((n, rows[0].shape[1]), dtype=rows[0].dtype)
        cache[torch.tensor(self.order)] = torch.cat(rows)
        return cache

    def unperturbed(self, lines: list[int]) -> torch.Tensor:
        n_full = len(lines) // self.BATCH * self.BATCH
        parts = [self.cache[torch.tensor(lines[:n_full], dtype=torch.int64)]]
        rest = lines[n_full:]
        if rest:
            parts.append(
                self.embed(
                    [self.input_ids[i] for i in rest], [self.lengths[i] for i in rest]
                ).cpu()
            )
        return torch.cat(parts)

    def perturbed(self, lines: list[int], token: int) -> torch.Tensor:
        parts = []
        for start in range(0, len(lines), self.BATCH):
            deleted = [
                delete_token(self.input_ids[i], token)
                for i in lines[start : start + self.BATCH]
            ]
            parts.append(self.embed(deleted, [len(s) for s in deleted]).cpu())
        return torch.cat(parts)

    def _label(self, embs: torch.Tensor, lines: list[int]) -> pd.DataFrame:
        """The released ``label_cell_embs`` with the patient_id label."""
        df = pd.DataFrame(embs.numpy())
        df["patient_id"] = [self.patient_ids[i] for i in lines]
        return df

    def perturb_gene(self, gene: str) -> bool:
        """Write subtracted_embs/<gene>.csv and perturbed_embs/<gene>.csv as the
        released ``perturb_gene`` does; False when the released code skips the gene."""
        if gene not in self.viability.columns or gene not in self.token_of:
            print("Error: hvg not in viability mat, continuing")
            return False
        token = self.token_of[gene]
        lines = lines_with_token(self.order, self.token_sets, token)
        if not lines:
            print(f"Gene {gene} not valid, continuing...")
            return False
        embs = self._label(self.unperturbed(lines), lines)
        embs2 = self._label(self.perturbed(lines, token), lines)
        embs.set_index("patient_id", inplace=True)
        embs2.set_index("patient_id", inplace=True)
        sub = embs - embs2
        sub.insert(0, "viability", self.viability.loc[:, gene].to_frame())
        sub.to_csv(self.out_dir / "subtracted_embs" / f"{gene}.csv")
        embs2.to_csv(self.out_dir / "perturbed_embs" / f"{gene}.csv")
        print(f"Finished assembling {gene}")
        return True

    def perturb_all_genes(self, genes: list[str]) -> int:
        """The released ``perturb_all_genes`` loop: skip a gene whose subtracted file
        exists. Returns the number of genes written."""
        written = 0
        for i, gene in enumerate(genes):
            if (self.out_dir / "subtracted_embs" / f"{gene}.csv").is_file():
                print(f"Gene {gene} has already been perturbed, continuing...")
                continue
            print(f"\n\nRunning code on hvg {gene}, iteration {i}/{len(genes)}")
            written += self.perturb_gene(gene)
        print("Finished perturbing all genes (for single knockout)!")
        return written
