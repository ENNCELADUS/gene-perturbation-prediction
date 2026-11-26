"""
This script is modified from ESM's repository (https://github.com/facebookresearch/esm).

Note that long sequences may take lots of memory when inferring. Please set 'batch_size' suitable for your device.

After running this script, you can load the generated ESM-2 representations of proteins by: 
    torch.load('###.pt')
"""

from pathlib import Path
import os
import argparse
from tqdm import tqdm
import pickle
import numpy as np
import torch
import esm

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--root", type=Path, default='/data/qingyuyang/SL/ESM4SL/data')
    return parser.parse_args()


def select_undone_pdbs(input_list: Path, output_dir: Path) -> list[tuple[str, str]]:
    all_pdbs = [i[0] for i in input_list]  # gene_id
    done_pdbs = [int(i[:-4]) for i in os.listdir(output_dir) if i.endswith('.pt')]
    undone_pdbs = list(set(all_pdbs).difference(set(done_pdbs)))  # undone PDB names
    return [pair for pair in input_list if pair[0] in undone_pdbs]


if __name__ == "__main__":
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    batch_size = args.batch_size

    dataset_root = args.root
    save_path = f'{dataset_root}/pretrained_emb/esm2'  # NOTE: can change
    os.makedirs(save_path, exist_ok=True)
    with open(f'{dataset_root}/mapping/all_id_seq.pkl', 'rb') as f:  # NOTE: can change
        all_list = pickle.load(f)
    print('Finish loading protein data')

    all_list = select_undone_pdbs(all_list, save_path)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # can change models of different size; the model is stored in ~/.cache/torch/hub/checkpoints after the 1st time
    model.eval()  # disables dropout for deterministic results
    model.to(device)
    print('Finish loading model')

    batch_converter = alphabet.get_batch_converter()

    def infer_batch(data: list[tuple[str, str]]):
        batch_labels, _, batch_tokens = batch_converter(data)  # all proteins' names, all proteins sequences, torch.Size([#proteins, max_protein_length_in_this_batch])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # all sequences' length
        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations: torch.Tensor = results["representations"][33]  # NOTE: layer number same as model name; less than 1s

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            label = batch_labels[i]
            embedding = token_representations[i, 1 : tokens_len - 1].clone().detach().cpu()
            torch.save(embedding, f"{save_path}/{label}.pt")

    for idx in tqdm(range(0, len(all_list), batch_size)):
        data = all_list[idx : min(idx + batch_size, len(all_list))]
        infer_batch(data)
