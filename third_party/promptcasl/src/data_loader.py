import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pickle
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import random
import os


class GenePairDataset(Dataset):
    def __init__(self, embeddings, labels, gene_names):
        self.embeddings = [
            tuple(torch.from_numpy(e).float() if isinstance(e, np.ndarray) else e.clone().detach().float() for e in emb)
            for emb in embeddings
        ]
        self.labels = [label.clone().detach().long() for label in labels]
        self.gene_names = gene_names 

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.gene_names[idx]


# torch.serialization.add_safe_globals([GenePairDataset])


# def load_data_SCL(args, fold_idx):
def load_data_SCL(dataset_path, cell_line, batch_size, negpos, fold_idx):
    """
    Load training and validation data for Single Cell Line Learning (SCL).
    
    Args:
        args: Argument parser with fields:
              - dataset_path
              - cell_line
              - batch_size
              - negpos
        fold_idx (int): Index of the fold to use as validation set (0-based).
        
    Returns:
        val_loader, train_loader
        Each returns batches of (embeddings, labels, gene_names)
    """
    fold_dir = dataset_path
    fold_prefix = f"{cell_line}_F"

    # Load validation fold
    val_dataset = torch.load(os.path.join(fold_dir, f"{fold_prefix}{fold_idx + 1}.pt"), weights_only=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load and combine remaining folds for training
    train_datasets = []
    for j in range(5):
        if j != fold_idx:
            fold_path = os.path.join(fold_dir, f"{fold_prefix}{j + 1}.pt")
            train_datasets.append(torch.load(fold_path, weights_only=False))
    
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return val_loader, train_loader


def load_data_MCL(dataset_path, cell_line_set, batch_size, negpos, fold_idx): # no matter of which cell line is given
    """
    Load training and validation data for Multi Cell Line Learning (MCL), using fold_idx across all cell lines.
    For fold k, we use fold k from each cell line as validation, and all other folds from all cell lines as training.

    Args:
        args: Argparse namespace with:
              - dataset_path
              - cell_line_set
              - batch_size
              - negpos
        fold_idx (int): Fold to use for validation (0-based)

    Returns:
        val_loader, train_loader
    """
    cell_lines = cell_line_set

    train_datasets = []
    val_datasets = []
    
    for cell_line in cell_lines:
        # fold_folder = os.path.join(dataset_path, cell_line, f"folds_NP{negpos}")
        fold_folder = dataset_path
        # Add validation fold
        val_path = os.path.join(fold_folder, f"{cell_line}_F{fold_idx + 1}.pt")
        if os.path.exists(val_path):
            val_datasets.append(torch.load(val_path, weights_only=False))

        # Add all other folds to training set
        for j in range(5):
            if j != fold_idx:
                train_path = os.path.join(fold_folder, f"{cell_line}_F{j + 1}.pt")
                if os.path.exists(train_path):
                    train_datasets.append(torch.load(train_path, weights_only=False))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return val_loader, train_loader

def load_data_MCTL(dataset_path, target_cell_lines, cell_line_set, negpos, batch_size):
    """
    Load data for Multi Cell Line Transfer Learning (MCTL).
    Uses all folds of target cell line as validation, and all folds of specified source cell lines for training.

    Args:
        args: should include:
              - dataset_path
              - target_cell_lines (target)
              - cell_line_set (sources)
              - negpos
              - batch_size
    Returns:
        val_loader, train_loader
    """
    # === Load validation set (all 5 folds from each target cell line) ===
    val_datasets = []
    for target_cell_line in target_cell_lines:
        for fold_idx in range(5):
            val_fold_path = os.path.join(dataset_path, f"{target_cell_line}_F{fold_idx + 1}.pt")
            if os.path.exists(val_fold_path):
                val_datasets.append(torch.load(val_fold_path, weights_only=False))

    val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Load training set (all folds from source cell lines) ===
    train_datasets = []
    for cell_line in cell_line_set:
        for fold_idx in range(5):
            fold_path = os.path.join(dataset_path, f"{cell_line}_F{fold_idx + 1}.pt")
            if os.path.exists(fold_path):
                train_datasets.append(torch.load(fold_path, weights_only=False))

    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return val_loader, train_loader


def load_data_LOCO(dataset_path, target_cell_line, all_cell_lines, batch_size):
    """
    Leave-one-cell-line-out data loading.
    Train: all available folds from every cell line except target.
    Val: all available folds from the target cell line.
    Auto-detects folds — works for NP1 (F1-F5) and full-eval (F1 only).
    """
    source_cell_lines = [cl for cl in all_cell_lines if cl != target_cell_line]

    train_datasets = []
    for cl in source_cell_lines:
        for fold_idx in range(1, 10):
            fold_path = os.path.join(dataset_path, f"{cl}_F{fold_idx}.pt")
            if not os.path.exists(fold_path):
                break
            train_datasets.append(torch.load(fold_path, weights_only=False))

    val_datasets = []
    for fold_idx in range(1, 10):
        fold_path = os.path.join(dataset_path, f"{target_cell_line}_F{fold_idx}.pt")
        if not os.path.exists(fold_path):
            break
        val_datasets.append(torch.load(fold_path, weights_only=False))

    assert train_datasets, f"No training data found in {dataset_path} for LOCO (sources: {source_cell_lines})"
    assert val_datasets, f"No val data found in {dataset_path} for target {target_cell_line}"

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    return (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    )


# for case study
def load_train_test_datasets_infer(cell_line_condition, folder=None, shuffle=True, logger=None):
    train_path, train_emb, train_labels = None, None, None
    test_path = f"{folder}/{cell_line_condition}_test.pt"
    # Load test dataset
    test_dataset = torch.load(test_path, weights_only=True)
    test_emb, test_labels = zip(*[(sample[0], sample[1]) for sample in test_dataset])

    # Shuffle the test dataset
    test_data = list(zip(test_emb, test_labels))
    print("len of test data", len(test_data)) 
    if shuffle:
        random.seed(42)  # Set seed for reproducibility
        random.shuffle(test_data)
    test_emb, test_labels = zip(*test_data)

    if logger:
        logger.info(f"Loaded {train_path} and {test_path}")

    return train_emb, train_labels, test_emb, test_labels

