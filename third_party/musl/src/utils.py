import torch
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm, trange
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import pickle as pkl
import scanpy as sc
import os
# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer


class customAdamW(Optimizer):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.init_lr = lr

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # apply weight decay
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                mask = (exp_avg * grad > 0).to(grad.dtype)
                # mask = mask * (mask.numel() / (mask.sum() + 1)) ## original implementation, leaving it here for record
                mask.div_(mask.mean().clamp_(min=1e-3)) # https://huggingface.co/rwightman/timm-optim-caution found this implementation is more favoarable in many cases
                norm_grad = (exp_avg * mask) / denom
                p.add_(norm_grad, alpha=-step_size)
        return loss

def vector_state(exp_data, log_exp_data, norm_exp_data, train_pairs, test_pairs=None):
    """
    Calculate statistical features for gene pairs and perform standardization
    Standardize the training set data, then use the training set standardization parameters to standardize the test set
    
    Parameters:
    exp_data: Raw expression data
    log_exp_data: Log-transformed expression data
    norm_exp_data: Normalized expression data
    train_pairs: Training set gene pairs
    test_pairs: Test set gene pairs, if None then only process training set
    
    Returns:
    If test_pairs is None, returns standardized training set features
    Otherwise returns (standardized training set features, standardized test set features)
    """
    if log_exp_data is None:
        log_exp_data = torch.log1p(exp_data)
    if norm_exp_data is None:
        mean = torch.mean(log_exp_data, dim=1, keepdim=True)
        std = torch.std(log_exp_data, dim=1, keepdim=True)
        std = torch.where(std < 1e-10, torch.ones_like(std), std)
        norm_exp_data = (log_exp_data - mean) / std
    train_state = calculate_features(exp_data, log_exp_data, norm_exp_data, train_pairs)
    
    train_state = torch.nan_to_num(train_state, nan=0.0).float()
    
    if test_pairs is None:
        return normalize_features(train_state), None
    
    test_state = calculate_features(exp_data, log_exp_data, norm_exp_data, test_pairs)
    
    test_state = torch.nan_to_num(test_state, nan=0.0).float()
    
    norm_params = get_normalize_params(train_state)
    
    norm_train_state = normalize_with_params(train_state, norm_params)
    norm_test_state = normalize_with_params(test_state, norm_params)
    
    return norm_train_state, norm_test_state

def calculate_features(exp_data, log_exp_data, norm_exp_data, gene_pairs):
    """
    Calculate statistical features for gene pairs
    """
    log_exp_a = log_exp_data[gene_pairs[0]]
    log_exp_b = log_exp_data[gene_pairs[1]]
    log_exp_a_q75 = torch.quantile(log_exp_a, 0.75, dim=1)
    log_exp_a_q25 = torch.quantile(log_exp_a, 0.25, dim=1)
    log_exp_a_q50 = torch.quantile(log_exp_a, 0.5, dim=1)
    log_exp_b_q75 = torch.quantile(log_exp_b, 0.75, dim=1)
    log_exp_b_q25 = torch.quantile(log_exp_b, 0.25, dim=1)
    log_exp_b_q50 = torch.quantile(log_exp_b, 0.5, dim=1)
    
    state = torch.zeros(gene_pairs.shape[1], 35)
    
    print('Calculating basic features...')
    state[:,0] = exp_data[gene_pairs[0]].mean(dim=1)
    state[:,1] = exp_data[gene_pairs[0]].std(dim=1)
    state[:,2] = exp_data[gene_pairs[0]].max(dim=1)[0] - exp_data[gene_pairs[0]].min(dim=1)[0]
    state[:,3] = exp_data[gene_pairs[0]].median(dim=1).values
    state[:,4] = exp_data[gene_pairs[0]].quantile(0.25, dim=1)
    state[:,5] = exp_data[gene_pairs[0]].quantile(0.75, dim=1)
    
    state[:,6] = exp_data[gene_pairs[1]].mean(dim=1)
    state[:,7] = exp_data[gene_pairs[1]].std(dim=1)
    state[:,8] = exp_data[gene_pairs[1]].max(dim=1)[0] - exp_data[gene_pairs[1]].min(dim=1)[0]
    state[:,9] = exp_data[gene_pairs[1]].median(dim=1).values
    state[:,10] = exp_data[gene_pairs[1]].quantile(0.25, dim=1)
    state[:,11] = exp_data[gene_pairs[1]].quantile(0.75, dim=1)
    
    state[:,12] = abs(state[:,0] - state[:,6])
    state[:,13] = state[:,[1,7]].max(dim=1)[0] / state[:,[1,7]].min(dim=1)[0]
    state[:,14] = state[:,1] / (state[:,0] + 1e-10)
    state[:,15] = state[:,7] / (state[:,6] + 1e-10)
    state[:,16] = 1 - abs(state[:,14] - state[:,15]) / (state[:,14] + state[:,15] + 1e-10)
    
    print('Calculating gene pair features...')
    state[:,17] = get_pearson(log_exp_a,log_exp_b)
    state[:,18] = get_pearson(norm_exp_data[gene_pairs[0]],norm_exp_data[gene_pairs[1]])
    state[:,19] = get_spearman(log_exp_a,log_exp_b)
    state[:,20] = get_spearman(norm_exp_data[gene_pairs[0]],norm_exp_data[gene_pairs[1]])
    state[:,21] = get_mutual_info(log_exp_a,log_exp_b)
    state[:,22:26] = get_divergence(log_exp_a,log_exp_b)
    state[:,26:28] = get_conditional_entropy(log_exp_a,log_exp_b)
    state[:,28:33] = get_distance(log_exp_a,log_exp_b)
    
    print('Calculating IQR features...')
    iqr_a = log_exp_a_q75 - log_exp_a_q25
    iqr_b = log_exp_b_q75 - log_exp_b_q25
    max_iqr = torch.maximum(iqr_a, iqr_b)
    min_iqr = torch.minimum(iqr_a, iqr_b)
    state[:,33] = torch.where(min_iqr > 0, max_iqr / min_iqr, torch.zeros_like(min_iqr))
    
    print('Calculating range features...')
    max_val = torch.maximum(log_exp_a.max(dim=1)[0], log_exp_b.max(dim=1)[0])
    min_val = torch.minimum(log_exp_a.min(dim=1)[0], log_exp_b.min(dim=1)[0])
    range_val = max_val - min_val
    state[:,34] = torch.abs(log_exp_a_q50 - log_exp_b_q50) / torch.where(range_val > 0, range_val, torch.ones_like(range_val))
    
    return state

def get_pearson(exp_a, exp_b):
    n = exp_a.size(1)  # m
    sum_a = exp_a.sum(dim=1)  # N,
    sum_b = exp_b.sum(dim=1)  # N,
    sum_ab = (exp_a * exp_b).sum(dim=1)  # N,
    sum_a2 = (exp_a ** 2).sum(dim=1)  # N,
    sum_b2 = (exp_b ** 2).sum(dim=1)  # N,
    
    numerator = n * sum_ab - sum_a * sum_b  # N,
    denominator = torch.sqrt((n * sum_a2 - sum_a ** 2) * (n * sum_b2 - sum_b ** 2) + 1e-10)  # N,
    return numerator / denominator  # N,

def compute_mi_row(args):
    i, a_row, b_row, n_neighbors = args
    return i, mutual_info_regression(a_row.reshape(-1, 1), b_row, n_neighbors=n_neighbors)[0]

def get_mutual_info(exp_a, exp_b, n_neighbors=3, max_workers=None):
    """
    Multi-process parallel computation of mutual information for two N*M matrices
    
    Parameters:
    exp_a: N*M matrix
    exp_b: N*M matrix
    n_neighbors: number of neighbors for mutual information calculation
    max_workers: maximum number of processes (None means automatic selection)
    
    Returns: N*1 tensor of mutual information values
    """
    N = exp_a.shape[0]
    
    if max_workers is None:
        max_workers = 48 
    
    print(f'Calculating mutual information using {max_workers} processes...')
    
    a_np = exp_a.numpy()
    b_np = exp_b.numpy()
    
    tasks = [(i, a_np[i], b_np[i], n_neighbors) for i in range(N)]
    
    results = torch.full((N,), float('nan'))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_mi_row, task) for task in tasks]
        
        for future in tqdm(futures, total=N, desc="Processing rows"):
            i, mi = future.result()
            results[i] = mi
            
    return results

def get_divergence(exp_a, exp_b):
    
    N = exp_a.shape[0]
    M = exp_a.shape[1]
    bins = min(32, M // 10 + 1)
    results = torch.zeros(N, 4)
    hist_a = np.zeros((N,bins))
    hist_b = np.zeros((N,bins))
    
    print('Calculating divergence...')
    for i in trange(N):
        hist_a[i], bin_edges = np.histogram(exp_a[i].numpy(), bins=bins)
        hist_b[i] = np.histogram(exp_b[i].numpy(), bins=bin_edges)[0]
    
    eps = 1e-10
    prob_a = torch.from_numpy((hist_a + eps) / (np.sum(hist_a,axis=1,keepdims=True) + bins * eps))
    prob_b = torch.from_numpy((hist_b + eps) / (np.sum(hist_b,axis=1,keepdims=True) + bins * eps))
    
    log_ratio = torch.log(prob_a) - torch.log(prob_b)
    kl_div_ab = torch.sum(prob_a * log_ratio,dim=1)
    kl_div_ba = torch.sum(prob_b * (-log_ratio),dim=1)
    
    m = 0.5 * (prob_a + prob_b)
    
    kl_pm = torch.sum(prob_a * (torch.log(prob_a) - torch.log(m)),dim=1)
    kl_qm = torch.sum(prob_b * (torch.log(prob_b) - torch.log(m)),dim=1)
    
    results[:,0] = (kl_div_ab+kl_div_ba)/2
    results[:,1] = kl_div_ab
    results[:,2] = kl_div_ba
    results[:,3] = (kl_pm + kl_qm)/2
    
    return results

def get_spearman(exp_a, exp_b):
    """
    Calculate Spearman correlation coefficient between two N*M matrices, return N*1 result
    exp_a: N*M matrix
    exp_b: N*M matrix
    Returns: N*1 Spearman correlation coefficient
    """

    def compute_ranks_vectorized(x):
        N, M = x.shape
        
        sorted_x, sorted_indices = torch.sort(x, dim=1)
        
        initial_ranks = torch.arange(M, device=x.device, dtype=torch.float).expand(N, -1).clone()
        
        diffs = torch.diff(sorted_x, dim=1)
        duplicate_mask = torch.cat([
            torch.zeros(N, 1, device=x.device, dtype=torch.bool), 
            diffs == 0
        ], dim=1)
        
        group_starts = ~duplicate_mask
        group_ids = torch.cumsum(group_starts.int(), dim=1) - 1
        
        group_ids = torch.clamp(group_ids, 0, M-1)
        
        group_sizes = torch.zeros(N, M, device=x.device, dtype=torch.float)
        group_sizes.scatter_add_(1, group_ids, torch.ones_like(group_sizes))
        
        group_starts = torch.cat([
            torch.zeros(N, 1, device=x.device, dtype=torch.float),
            torch.cumsum(group_sizes[:, :-1], dim=1)
        ], dim=1)
        
        avg_ranks = group_starts + (group_sizes - 1) / 2.0
        
        final_ranks = torch.where(duplicate_mask, avg_ranks.gather(1, group_ids), initial_ranks)
        
        reverse_indices = torch.argsort(sorted_indices, dim=1)
        ranks = final_ranks.gather(1, reverse_indices)
        
        return ranks
        
    rank_a = compute_ranks_vectorized(exp_a)
    rank_b = compute_ranks_vectorized(exp_b)
    
    spearman = get_pearson(rank_a,rank_b)
    return spearman

def get_conditional_entropy(exp_a,exp_b):
    N = exp_a.shape[0]
    M = exp_a.shape[1]
    bins = min(32, M // 10 + 1)
    results = torch.zeros(N, 2)
    image = torch.zeros(N,bins,bins)
    print('Calculating conditional entropy...')
    for i in trange(N):
        np.histogram2d
        image[i] = torch.histogramdd(torch.stack([exp_a[i],exp_b[i]],dim=1),bins=[bins,bins])[0]
    eps = 1e-10
    joint_prob = (image + eps) / (image.sum(dim=(1,2)).unsqueeze(1).unsqueeze(1) + eps)
    p_a = joint_prob.sum(dim=2).unsqueeze(1)
    p_b = joint_prob.sum(dim=1).unsqueeze(2)
    cond_prob_ab = joint_prob / (p_b + eps)
    cond_prob_ba = joint_prob / (p_a + eps)
    log_cond_prob_ab = torch.log(cond_prob_ab+eps)
    log_cond_prob_ba = torch.log(cond_prob_ba+eps)
    entropy_terms_ab = torch.einsum('nij,nij->nij', cond_prob_ab, log_cond_prob_ab)
    entropy_terms_ba = torch.einsum('nij,nij->nij', cond_prob_ba, log_cond_prob_ba)
    sum_per_ab = torch.sum(entropy_terms_ab,dim=(1,2)).unsqueeze(1).unsqueeze(1)
    sum_per_ba = torch.sum(entropy_terms_ba,dim=(1,2)).unsqueeze(1).unsqueeze(1)
    cond_entropy_ab = -torch.sum(p_b*sum_per_ab,dim=(1,2))
    cond_entropy_ba = -torch.sum(p_a*sum_per_ba,dim=(1,2))
    results[:,0] = cond_entropy_ab
    results[:,1] = cond_entropy_ba
    return results
    

def get_distance(exp_a, exp_b):

    euclidean = torch.norm(exp_a - exp_b, p=2, dim=1)
    cosine = torch.nn.functional.cosine_similarity(exp_a, exp_b, dim=1)
    manhattan = torch.norm(exp_a - exp_b, p=1, dim=1)
    chebyshev = torch.norm(exp_a - exp_b, p=np.inf, dim=1)
    
    a_sorted = torch.sort(exp_a, dim=1)[0]
    b_sorted = torch.sort(exp_b, dim=1)[0]
    wasserstein = torch.mean(torch.abs(a_sorted - b_sorted), dim=1)
    
    distances = torch.stack([euclidean, cosine, manhattan, chebyshev, wasserstein], dim=1)
    return distances


def get_gene_list():
    gene_table = pd.read_csv('./processed_data/meta_table_7684.csv')
    gene_id = dict(zip(gene_table['uni_gene'],gene_table.index.values))
    id_gene = dict(zip(gene_table.index.values,gene_table['uni_gene']))
    return gene_id,id_gene

def get_cell_gene_list(cl_name):
    gene_table = pd.read_csv(f'./processed_data/{cl_name}_filtered_NP1_F_mapped.csv')
    genes = set(gene_table['gene_1']).union(set(gene_table['gene_2']))
    genes = sorted(list(genes))
    gene_id = dict(zip(genes,range(len(genes))))
    id_gene = dict(zip(range(len(genes)),genes))
    return gene_id,id_gene

def get_gene_embedding(emb_type='protein'):
    id_gene = get_gene_list()[1]
    if emb_type == 'protein':
        gene_embeddings = torch.load('./processed_data/protein_embeddings.pt')
        emb_dim = gene_embeddings[list(gene_embeddings.keys())[0]].shape[0]
        id_embeddings = torch.zeros(len(id_gene),emb_dim)
        for id,gene in id_gene.items():
            id_embeddings[id] = gene_embeddings[gene].to(torch.float32)
    elif emb_type == 'scgpt':
        gene_embeddings = pkl.load(open('./processed_data/all_emb_scgpt.pkl','rb'))
        emb_dim = gene_embeddings[list(gene_embeddings.keys())[0]].shape[0]
        id_embeddings = torch.zeros(len(id_gene),emb_dim)
        for id,gene in id_gene.items():
            id_embeddings[id] = torch.tensor(gene_embeddings[gene]).to(torch.float32)
    elif emb_type == 'genept':
        gene_embeddings = pkl.load(open('./processed_data/all_emb_genept.pkl','rb'))
        emb_dim = np.array(gene_embeddings[list(gene_embeddings.keys())[0]]).shape[0]
        id_embeddings = torch.zeros(len(id_gene),emb_dim)
        for id,gene in id_gene.items():
            id_embeddings[id] = torch.tensor(gene_embeddings[gene]).to(torch.float32)
    else:
        raise ValueError(f'Invalid embedding type: {emb_type}')
    
    return id_embeddings

def get_expression():
    id_gene = get_gene_list()[1]
    expre_data = sc.read_h5ad('./processed_data/all_tcga.h5ad')
    expre_data.var.index = expre_data.var['gene_name']
    expre_data = expre_data.to_df().T
    id_expre = torch.zeros(len(id_gene),expre_data.shape[1])
    for id,gene in id_gene.items():
        id_expre[id] = torch.from_numpy(expre_data.loc[gene].values)
    return id_expre

def get_cell_expression(cl_name):
    id_gene = get_cell_gene_list(cl_name)[1]
    expre_data = sc.read_h5ad(f'./processed_data/{cl_name}_cell_line_imputed.h5ad')
    expre_data.var.index = expre_data.var['gene_name']
    expre_data = expre_data.to_df().T
    id_expre = torch.zeros(len(id_gene),expre_data.shape[1])
    for id,gene in id_gene.items():
        if gene in expre_data.index:
            id_expre[id] = torch.from_numpy(expre_data.loc[gene].values)
        else:
            id_expre[id] = torch.zeros(expre_data.shape[1])
    return id_expre

def get_edge_index():
    gene_id = get_gene_list()[0]
    all_ppi_data = pd.read_csv('./processed_data/merged_ppi_gene_names.csv')
    all_ppi_data = all_ppi_data[['gene1','gene2']]
    all_ppi_data.columns = ['source', 'target']
    all_ppi_data['source'] = all_ppi_data['source'].map(gene_id)
    all_ppi_data['target'] = all_ppi_data['target'].map(gene_id)
    all_ppi_data = all_ppi_data[all_ppi_data['source'].notna()&all_ppi_data['target'].notna()]
    edge_index = all_ppi_data[['source', 'target']].values.T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def get_cell_edge_index(cl_name):
    gene_id = get_cell_gene_list(cl_name)[0]
    all_ppi_data = pd.read_csv('./processed_data/merged_ppi_gene_names.csv')
    all_ppi_data = all_ppi_data[['gene1','gene2']]
    all_ppi_data.columns = ['source', 'target']
    all_ppi_data['source'] = all_ppi_data['source'].map(gene_id)
    all_ppi_data['target'] = all_ppi_data['target'].map(gene_id)
    all_ppi_data = all_ppi_data[all_ppi_data['source'].notna()&all_ppi_data['target'].notna()]
    edge_index = all_ppi_data[['source', 'target']].values.T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def get_cell_pos_lable(train_or_test,fold,cl_name):
    gene_pairs = np.load(f'./processed_data/data_cell_line/CV1_{cl_name}_bins_32/fold_data/{train_or_test}_pos_gene_pair_fold{fold}.npy')
    gene_id = get_cell_gene_list(cl_name)[0]
    gene_pair_ids = torch.zeros(gene_pairs.shape[0],2)
    for i in range(gene_pairs.shape[0]):
        gene_pair_ids[i,0] = gene_id[gene_pairs[i,0]]
        gene_pair_ids[i,1] = gene_id[gene_pairs[i,1]]
    gene_pair_ids = gene_pair_ids.long()
    return gene_pair_ids.T

def get_cell_neg_lable(train_or_test,fold,cl_name):
    gene_pairs = np.load(f'./processed_data/data_cell_line/CV1_{cl_name}_bins_32/fold_data/{train_or_test}_neg_gene_pair_fold{fold}.npy')
    gene_id = get_cell_gene_list(cl_name)[0]
    gene_pair_ids = torch.zeros(gene_pairs.shape[0],2)
    for i in range(gene_pairs.shape[0]):
        gene_pair_ids[i,0] = gene_id[gene_pairs[i,0]]
        gene_pair_ids[i,1] = gene_id[gene_pairs[i,1]]
    gene_pair_ids = gene_pair_ids.long()
    return gene_pair_ids.T

def get_normalize_params(features):
    """
    Get parameters required for feature normalization
    
    Parameters:
    features: Feature tensor to be normalized
    
    Returns:
    Dictionary containing normalization parameters:
    - high_var_cols: Column indices with standard deviation greater than 10
    - has_neg_vals: Whether each column contains negative values
    - min_vals: Minimum values for each column
    - max_vals: Maximum values for each column
    """
    if isinstance(features, torch.Tensor):
        features_np = features.numpy()
    else:
        features_np = features
        
    # 找出值范围较大的列（标准差大于10的列）
    high_var_cols = np.std(features_np, axis=0) > 10
    
    # 检查每列是否包含负值
    has_neg_vals = np.zeros(features_np.shape[1], dtype=bool)
    min_vals = np.zeros(features_np.shape[1])
    max_vals = np.zeros(features_np.shape[1])
    
    for col in range(features_np.shape[1]):
        has_neg_vals[col] = np.any(features_np[:, col] < 0)
        min_vals[col] = np.min(features_np[:, col])
        max_vals[col] = np.max(features_np[:, col])
        
    return {
        'high_var_cols': high_var_cols,
        'has_neg_vals': has_neg_vals,
        'min_vals': min_vals,
        'max_vals': max_vals
    }

def normalize_with_params(features, norm_params):
    """
    Normalize features using the given normalization parameters
    
    Parameters:
    features: Feature tensor to be normalized
    norm_params: Normalization parameters generated by get_normalize_params
    
    Returns:
    Normalized feature tensor
    """
    if isinstance(features, torch.Tensor):
        norm_features = features.numpy()
    else:
        norm_features = features
        
    high_var_cols = norm_params['high_var_cols']
    has_neg_vals = norm_params['has_neg_vals']
    min_vals = norm_params['min_vals']
    max_vals = norm_params['max_vals']
    
    if np.any(high_var_cols):
        for col in np.where(high_var_cols)[0]:
            if has_neg_vals[col]:
                if min_vals[col] != max_vals[col]:
                    norm_features[:, col] = (norm_features[:, col] - min_vals[col]) / (max_vals[col] - min_vals[col])
            else:
                epsilon = 1e-10
                norm_features[:, col] = np.log1p(norm_features[:, col] + epsilon)
    
    for col in range(norm_features.shape[1]):
        if min_vals[col] != max_vals[col]:
            norm_features[:, col] = (norm_features[:, col] - min_vals[col]) / (max_vals[col] - min_vals[col])
    
    return torch.tensor(norm_features)

def normalize_features(features):
    """
    Normalize features
    
    Parameters:
    features: Feature tensor to be normalized
    
    Returns:
    Normalized feature tensor
    """
    norm_params = get_normalize_params(features)
    
    return normalize_with_params(features, norm_params)

def generate_neg_pairs(genes, pos_pairs_set, num_samples, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    neg_pairs = set()
    while len(neg_pairs) < num_samples and len(genes) >= 2:
        i, j = np.random.choice(len(genes), 2, replace=False)
        pair = tuple(sorted((genes[i], genes[j])))
        if pair not in pos_pairs_set:
            neg_pairs.add(pair)
    neg_pairs = np.array(list(neg_pairs))

    assert (neg_pairs[:,0] < neg_pairs[:,1]).all()
    assert len(neg_pairs) == num_samples
    return neg_pairs

def create_pairs_and_labels(pos_set, neg_set):
    pos_set = torch.tensor(list(pos_set))
    neg_set = torch.tensor(list(neg_set))
    pairs = torch.cat([pos_set, neg_set])
    pairs = pairs[pairs[:,0] < pairs[:,1]]
    labels = torch.cat([
        torch.ones(pos_set.size(0)//2, dtype=torch.long),
        torch.zeros(neg_set.size(0)//2, dtype=torch.long)
    ], dim=0)
    return pairs, labels

def get_all_pos_pairs():
    all_pos_pairs = pd.read_csv('./processed_data/human_sl_7684.csv')[['unified_id_A','unified_id_B']]
    gene_table = pd.read_csv('./processed_data/meta_table_7684.csv')
    uni_id_uni_gene = dict(zip(gene_table['unified_id'],gene_table['uni_gene']))
    gene_id = dict(zip(gene_table['uni_gene'],gene_table.index.values))
    all_pos_pairs['gene1'] = all_pos_pairs['unified_id_A'].map(uni_id_uni_gene)
    all_pos_pairs['gene2'] = all_pos_pairs['unified_id_B'].map(uni_id_uni_gene)
    all_pos_pairs['gene1'] = all_pos_pairs['gene1'].map(gene_id)
    all_pos_pairs['gene2'] = all_pos_pairs['gene2'].map(gene_id)
    all_pos_pairs = all_pos_pairs[['gene1','gene2']].values
    all_pos_pairs = torch.tensor(all_pos_pairs,dtype=torch.long)
    return all_pos_pairs

def cv1_sampling(all_gene_pairs, genes, pos_pairs_set, num_neg_samples, random_seed):
    neg_pairs = generate_neg_pairs(genes, pos_pairs_set, num_neg_samples, random_seed)
    
    pair_ind = np.arange(len(all_gene_pairs))
    np.random.shuffle(pair_ind)
    pair_ind = pair_ind.tolist()
    train_size = int(len(all_gene_pairs)*0.8)
    
    train_pairs_fold, test_pairs_fold = [], []
    train_labels_fold, test_labels_fold = [], []
    
    for i in range(5):
        np.random.seed(random_seed+i)
        train_pos_pairs = all_gene_pairs[pair_ind[:train_size]]
        train_pos_set = set((a, b) for a, b in train_pos_pairs) | set((b, a) for a, b in train_pos_pairs)
        test_pos_pairs = all_gene_pairs[pair_ind[train_size:]]
        test_pos_set = set((a, b) for a, b in test_pos_pairs) | set((b, a) for a, b in test_pos_pairs)
        
        train_neg_pairs = neg_pairs[pair_ind[:train_size]]
        train_neg_set = set((a, b) for a, b in train_neg_pairs) | set((b, a) for a, b in train_neg_pairs)
        test_neg_pairs = neg_pairs[pair_ind[train_size:]]
        test_neg_set = set((a, b) for a, b in test_neg_pairs) | set((b, a) for a, b in test_neg_pairs)
        
        train_pairs, train_labels = create_pairs_and_labels(train_pos_set, train_neg_set)
        test_pairs, test_labels = create_pairs_and_labels(test_pos_set, test_neg_set)
        
        assert train_pairs.shape[0] == train_labels.shape[0]
        assert test_pairs.shape[0] == test_labels.shape[0]
        
        pair_ind = pair_ind[train_size:] + pair_ind[:train_size]
        train_pairs_fold.append(train_pairs)
        test_pairs_fold.append(test_pairs)
        train_labels_fold.append(train_labels)
        test_labels_fold.append(test_labels)
    
    return train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold

def cv2_cv3_sampling(genes, pos_pairs_set, num_neg_samples, random_seed, cv_type='CV2'):
    gene_index = np.arange(len(genes))
    genes = np.array(genes)
    np.random.shuffle(gene_index)
    gene_index = gene_index.tolist()
    train_size = int(len(gene_index)*0.8)
    pos_pairs = np.array(list(pos_pairs_set))
    
    train_pairs_fold, test_pairs_fold = [], []
    train_labels_fold, test_labels_fold = [], []
    
    for i in range(5):
        np.random.seed(random_seed+i)
        train_genes = list(genes[gene_index[:train_size]])
        test_genes = list(genes[gene_index[train_size:]])
        if cv_type == 'CV2':
            train_mask = np.isin(pos_pairs[:, 0], train_genes) & np.isin(pos_pairs[:, 1], train_genes)
            train_pos_set = {(a,b) for a,b in pos_pairs[train_mask]} | {(b,a) for a,b in pos_pairs[train_mask]}
            test_mask = (np.isin(pos_pairs[:, 0], train_genes) & np.isin(pos_pairs[:, 1], test_genes)) | (np.isin(pos_pairs[:, 0], test_genes) & np.isin(pos_pairs[:, 1], train_genes))
            test_pos_set = {(a,b) for a,b in pos_pairs[test_mask]} | {(b,a) for a,b in pos_pairs[test_mask]}
        else:  # CV3
            train_mask = np.isin(pos_pairs[:, 0], train_genes) & np.isin(pos_pairs[:, 1], train_genes)
            train_pos_set = {(a,b) for a,b in pos_pairs[train_mask]} | {(b,a) for a,b in pos_pairs[train_mask]}
            test_mask = np.isin(pos_pairs[:, 0], test_genes) & np.isin(pos_pairs[:, 1], test_genes)
            test_pos_set = {(a,b) for a,b in pos_pairs[test_mask]} | {(b,a) for a,b in pos_pairs[test_mask]}
        
        train_neg_pairs = generate_neg_pairs(train_genes, pos_pairs_set, len(train_pos_set)//2, random_seed+i)
        train_neg_set = set((a, b) for a, b in train_neg_pairs) | set((b, a) for a, b in train_neg_pairs)
        
        if cv_type == 'CV2':
            test_neg_pairs = set()
            while len(test_neg_pairs) < len(test_pos_set)//2 and len(test_genes) >= 1:
                i = np.random.choice(len(train_genes), 1, replace=False)[0]
                j = np.random.choice(len(test_genes), 1, replace=False)[0]
                pair = tuple(sorted((train_genes[i], test_genes[j])))
                if pair not in pos_pairs_set:
                    test_neg_pairs.add(pair)
        else:  # CV3
            test_neg_pairs = generate_neg_pairs(test_genes, pos_pairs_set, len(test_pos_set)//2, random_seed+i)
        
        test_neg_set = set((a, b) for a, b in test_neg_pairs) | set((b, a) for a, b in test_neg_pairs)
        
        train_pairs, train_labels = create_pairs_and_labels(train_pos_set, train_neg_set)
        test_pairs, test_labels = create_pairs_and_labels(test_pos_set, test_neg_set)
        
        assert train_pairs.shape[0] == train_labels.shape[0]
        assert test_pairs.shape[0] == test_labels.shape[0]
        
        gene_index = gene_index[train_size:] + gene_index[:train_size]
        train_pairs_fold.append(train_pairs)
        test_pairs_fold.append(test_pairs)
        train_labels_fold.append(train_labels)
        test_labels_fold.append(test_labels)
    
    return train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold

def sampling_neg_label(all_gene_pairs, gene_id, id_gene, random_seed=42, sampling_type='CV1'):
    if os.path.exists(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_pairs_seed{random_seed}.pkl'):
        train_pairs_fold = pkl.load(open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_pairs_seed{random_seed}.pkl','rb'))
        test_pairs_fold = pkl.load(open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/test_pairs_seed{random_seed}.pkl','rb'))
        train_labels_fold = pkl.load(open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_labels_seed{random_seed}.pkl','rb'))
        test_labels_fold = pkl.load(open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/test_labels_seed{random_seed}.pkl','rb'))
        return train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold
    
    num_neg_samples = all_gene_pairs.shape[0]
    pos_pairs_set = set((a, b) for a, b in all_gene_pairs) | set((b, a) for a, b in all_gene_pairs)
    pos_pairs_set = torch.tensor(list(pos_pairs_set))
    pos_pairs_set = pos_pairs_set[pos_pairs_set[:,0] < pos_pairs_set[:,1]]
    pos_pairs_set = set((a, b) for a, b in pos_pairs_set)
    genes = list(id_gene.keys())
    
    if sampling_type == 'CV1':
        train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold = cv1_sampling(all_gene_pairs, genes, pos_pairs_set, num_neg_samples, random_seed)
        pkl.dump(train_pairs_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_pairs_seed{random_seed}.pkl','wb'))
        pkl.dump(test_pairs_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/test_pairs_seed{random_seed}.pkl','wb'))
        pkl.dump(train_labels_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_labels_seed{random_seed}.pkl','wb'))
        pkl.dump(test_labels_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/test_labels_seed{random_seed}.pkl','wb'))
        return train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold
    elif sampling_type == 'CV2':
        train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold = cv2_cv3_sampling(genes, pos_pairs_set, num_neg_samples, random_seed, 'CV2')
        pkl.dump(train_pairs_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_pairs_seed{random_seed}.pkl','wb'))
        pkl.dump(test_pairs_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/test_pairs_seed{random_seed}.pkl','wb'))
        pkl.dump(train_labels_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_labels_seed{random_seed}.pkl','wb'))
        pkl.dump(test_labels_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/test_labels_seed{random_seed}.pkl','wb'))
        return train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold
    elif sampling_type == 'CV3':
        train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold = cv2_cv3_sampling(genes, pos_pairs_set, num_neg_samples, random_seed, 'CV3')
        pkl.dump(train_pairs_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_pairs_seed{random_seed}.pkl','wb'))
        pkl.dump(test_pairs_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/test_pairs_seed{random_seed}.pkl','wb'))
        pkl.dump(train_labels_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/train_labels_seed{random_seed}.pkl','wb'))
        pkl.dump(test_labels_fold, open(f'./processed_data/data/{sampling_type}_bins_32/fold_data/test_labels_seed{random_seed}.pkl','wb'))
        return train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold
    else:
        raise ValueError(f"Unknown sampling type: {sampling_type}")