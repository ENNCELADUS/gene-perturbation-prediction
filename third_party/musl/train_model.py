import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import wandb
import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm

from src.module import (
    MuSL, EdgeDataset, train_step, 
    compute_f1, min_max_normalize, get_default_config
)
from src.utils import (
    get_gene_list, get_gene_embedding, get_expression, get_edge_index,
    get_all_pos_pairs, sampling_neg_label, get_cell_gene_list, get_cell_edge_index,
    get_cell_expression, get_cell_pos_lable, get_cell_neg_lable, vector_state,
    customAdamW
)

def set_seed(seed):
    """Random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_musl(config):
    """Train model"""
    print(f"Config: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备数据
    print("Prepare data...")
    
    if config['cl']:
        edge_index = get_cell_edge_index(config['cl_name'])
        gene_id, id_gene = get_cell_gene_list(config['cl_name'])
        exp_data = get_cell_expression(config['cl_name'])
    else:
        edge_index = get_edge_index()
        gene_id, id_gene = get_gene_list()
        exp_data = get_expression()
        
    if config['use_pre_embeddings']:
        gene_embedding = get_gene_embedding(emb_type=config['emb_type'])
    else:
        gene_embedding = None
    
    # 模型参数
    NUM_NODES = len(gene_id)
    NODE_FEATURE_DIM = gene_embedding.shape[1] if gene_embedding is not None else None
    
    best_metrics = []
    
    all_pairs = get_all_pos_pairs()
    
    train_pairs_fold, test_pairs_fold, train_labels_fold, test_labels_fold = sampling_neg_label(all_pairs, gene_id, id_gene, random_seed=config['seed'], sampling_type=config['cv'])
    
    # 五折交叉验证
    for fold in range(5):
        # 设置随机种子
        set_seed(config['seed'])
        # print(f"随机种子: {config['seed']+fold}")
        
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*60}")
        
        # 设置wandb
        if config.get('use_wandb', False):
            experiment_name = f"{config['experiment_name']}_fold{fold}" if config.get('experiment_name') else f"fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=config['project_name'],
                name=experiment_name,
                config=config,
                mode=config.get('wandb_mode', 'offline')
            )
        
        
        # 准备数据
        if config['cl']:
            train_pos_pairs = get_cell_pos_lable(train_or_test='train', fold=fold, cl_name=config['cl_name'])
            train_neg_pairs = get_cell_neg_lable(train_or_test='train', fold=fold, cl_name=config['cl_name'])
            test_pos_pairs = get_cell_pos_lable(train_or_test='test', fold=fold, cl_name=config['cl_name'])
            test_neg_pairs = get_cell_neg_lable(train_or_test='test', fold=fold, cl_name=config['cl_name'])
            train_pairs = torch.cat((train_pos_pairs, train_neg_pairs), dim=1)
            test_pairs = torch.cat((test_pos_pairs, test_neg_pairs), dim=1)
            train_labels = torch.cat((torch.ones(train_pos_pairs.shape[1]), torch.zeros(train_neg_pairs.shape[1])), dim=0).long()
            test_labels = torch.cat((torch.ones(test_pos_pairs.shape[1]), torch.zeros(test_neg_pairs.shape[1])), dim=0).long()
        else:
            train_pairs = train_pairs_fold[fold].T
            test_pairs = test_pairs_fold[fold].T
            train_labels = train_labels_fold[fold]
            test_labels = test_labels_fold[fold]
            
        
        # if config['use_pre_cal_state'] and not config['cal_state']:
        #     if config['cl']:
        #         train_stat, test_stat = get_cell_pre_stat(fold, config['cl_name'])
        #     else:
        #         train_stat, test_stat = get_pre_stat(fold, config['cv'])
        # el
        if config['cal_state']:
            if config['cl']:
                if os.path.exists(f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt"):
                    train_stat = torch.load(f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt")
                    test_stat = torch.load(f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt")
                else:
                    train_stat, test_stat = vector_state(exp_data, None, None, train_pairs, test_pairs)
                    torch.save(train_stat, f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt")
                    torch.save(test_stat, f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt")
                # if os.path.exists(f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt"):
                # else:
                #     test_stat = None
            else:
                if os.path.exists(f"./processed_data/da ta/{config['cv']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt"):
                    train_stat = torch.load(f"./processed_data/data/{config['cv']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt")
                    test_stat = torch.load(f"./processed_data/data/{config['cv']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt")
                else:
                    train_stat, test_stat = vector_state(exp_data, None, None, train_pairs, test_pairs)
                    torch.save(train_stat, f"./processed_data/data/{config['cv']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt")
                    torch.save(test_stat, f"./processed_data/data/{config['cv']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt")
                # if os.path.exists(f"./processed_data/data/{config['cv']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt"):
                # else:
                #     test_stat = None
        else:
            train_stat = None
            test_stat = None
        
        train_dataset = EdgeDataset(train_pairs, train_labels, exp_data, state=train_stat, 
                                    bins=config['bins'], cal_state=config['cal_state'])
        test_dataset = EdgeDataset(test_pairs, test_labels, exp_data, state=test_stat, 
                                    bins=config['bins'], cal_state=config['cal_state'])
        # # save cal_state
        # if config['cal_state']:
        #     if config['cl']:
        #         if not os.path.exists(f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt"):
        #             torch.save(train_dataset.state, f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt")
        #         if not os.path.exists(f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt"):
        #             torch.save(test_dataset.state, f"./processed_data/data_cell_line/{config['cv']}_{config['cl_name']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt")
        #     else:
        #         if not os.path.exists(f"./processed_data/data/{config['cv']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt"):
        #             torch.save(train_dataset.state, f"./processed_data/data/{config['cv']}_bins_32/fold_data/train_stat_fold{fold}_seed{config['seed']}.pt")
        #         if not os.path.exists(f"./processed_data/data/{config['cv']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt"):
        #             torch.save(test_dataset.state, f"./processed_data/data/{config['cv']}_bins_32/fold_data/test_stat_fold{fold}_seed{config['seed']}.pt")
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=config['num_workers'])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                shuffle=False, num_workers=config['num_workers'])
        # continue

        model = MuSL(
            num_nodes=NUM_NODES,
            node_embedding_dim=NODE_FEATURE_DIM,
            hidden_dim=config['hidden_dim'],
            target_node_dim=config['target_node_dim'],
            node_embeddings=gene_embedding.to(device) if gene_embedding is not None else None,
            unified_dim=config['unified_dim'],
            enable_cross_attention=config['enable_cross_attention'],
            enable_adaptive_fusion=config['enable_adaptive_fusion'],
            enable_contrastive_learning=config['enable_contrastive_learning'],
            use_cnn=config['use_cnn'],
            use_gnn=config['use_gnn'],
            use_stat=config['use_stat']
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n Model info:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Unified dimention: {config['unified_dim']}")
        print(f"  - Enable cross attention: {'Y' if config['enable_cross_attention'] else 'N'}")
        print(f"  - Enable adaptive fusion: {'Y' if config['enable_adaptive_fusion'] else 'N'}")
        print(f"  - Enable contrastive learning: {'Y' if config['enable_contrastive_learning'] else 'N'}")
        print(f"  - Use pretrained embeddings: {'Y' if config['use_pre_embeddings'] else 'N'}")
        
        # Optimizer and loss
        if not config['use_custom_adamw']:
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=config['learning_rate'], 
                                        weight_decay=config['weight_decay'])
        else:
            optimizer = customAdamW(model.parameters(), 
                                    lr=config['learning_rate'], 
                                    weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        scheduler = None
        scheduler_type = config.get('lr_scheduler', 'none')
        if scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=config.get('lr_factor', 0.5), patience=config.get('lr_patience', 5), verbose=True, min_lr=config.get('min_lr', 1e-6)
            )
            print(f"[LR Scheduler] Using ReduceLROnPlateau: factor={config.get('lr_factor', 0.5)}, patience={config.get('lr_patience', 5)}, min_lr={config.get('min_lr', 1e-6)}")
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.get('lr_T_max', 10), eta_min=config.get('min_lr', 1e-6)
            )
            print(f"[LR Scheduler] Using CosineAnnealingLR: T_max={config.get('lr_T_max', 10)}, min_lr={config.get('min_lr', 1e-6)}")
        elif scheduler_type == 'none':
            print("[LR Scheduler] Disabled, using fixed learning rate.")
        else:
            print(f"[LR Scheduler] Unknown type: {scheduler_type}, using fixed learning rate.")

        # Train loop
        best_score = -float('inf')
        best_epoch = 0
        early_stopping_counter = 0
        best_auc = None
        best_aupr = None
        best_f1 = None
        best_gate_weights = None
        
        for epoch in range(config['epochs']):
            model.train()
            train_losses = {'total': 0, 'cnn': 0, 'gnn': 0, 'stat': 0, 'final': 0, 'contrastive': 0}
            train_predictions = {'cnn': [], 'gnn': [], 'stat': [], 'final': []}
            train_labels_all = []
            gate_weights_epoch = []
            
            train_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]', leave=False)
            
            for batch_idx, batch in enumerate(train_iterator):
                optimizer.zero_grad()
                
                train_results = train_step(
                    model, batch, edge_index, criterion, device,
                    alpha=config['contrastive_weight']
                )
                
                if not config['use_cnn'] and not config['use_gnn'] and not config['use_stat']:
                    train_results['total_loss'].backward()
                else:
                    if config['use_cnn'] and config['use_gnn'] and config['use_stat']:
                        train_results['total_loss'].backward()
                    else:
                        loss = 0
                        if config['use_cnn']:
                            loss += train_results['loss_cnn']
                        if config['use_gnn']:
                            loss += train_results['loss_gnn']
                        if config['use_stat']:
                            loss += train_results['loss_stat']
                        loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                for key in train_losses:
                    train_losses[key] += train_results[f'loss_{key}' if key != 'total' else 'total_loss'].item()
                
                with torch.no_grad():
                    for key in train_predictions:
                        pred = train_results['predictions'][f'{key}_pred']
                        probs = F.softmax(pred, dim=1)[:, 1]
                        train_predictions[key].append(probs.cpu())
                    
                    train_labels_all.append(batch['label'].cpu())
                    
                    if train_results['gate_weights'] is not None:
                        gate_weights_epoch.append(train_results['gate_weights'].mean(0).cpu())
                
                train_iterator.set_postfix({
                    'Total': f"{train_results['total_loss'].item():.4f}",
                    'CNN': f"{train_results['loss_cnn'].item():.4f}",
                    'GNN': f"{train_results['loss_gnn'].item():.4f}",
                    'Final': f"{train_results['loss_final'].item():.4f}"
                })
            
            for key in train_losses:
                train_losses[key] /= len(train_loader)
            
            train_labels_all = torch.cat(train_labels_all)
            for key in train_predictions:
                train_predictions[key] = min_max_normalize(torch.cat(train_predictions[key]))
            
            # Eval stage
            model.eval()
            test_losses = {'total': 0, 'cnn': 0, 'gnn': 0, 'stat': 0, 'final': 0, 'contrastive': 0}
            test_predictions = {'cnn': [], 'gnn': [], 'stat': [], 'final': []}
            test_labels_all = []
            
            with torch.no_grad():
                test_iterator = tqdm(test_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Test]', leave=False)
                
                for batch in test_iterator:
                    test_results = train_step(
                        model, batch, edge_index, criterion, device, 
                        alpha=config['contrastive_weight']
                    )
                    
                    for key in test_losses:
                        test_losses[key] += test_results[f'loss_{key}' if key != 'total' else 'total_loss'].item()
                    
                    for key in test_predictions:
                        pred = test_results['predictions'][f'{key}_pred']
                        probs = F.softmax(pred, dim=1)[:, 1]
                        test_predictions[key].append(probs.cpu())
                    
                    test_labels_all.append(batch['label'].cpu())
            
            for key in test_losses:
                test_losses[key] /= len(test_loader)
            
            test_labels_all = torch.cat(test_labels_all)
            for key in test_predictions:
                test_predictions[key] = min_max_normalize(torch.cat(test_predictions[key]))
            
            # AUC, AUPR, F1
            metrics = {}
            for key in ['cnn', 'gnn', 'stat', 'final']:
                train_auc = roc_auc_score(train_labels_all.numpy(), train_predictions[key].numpy())
                train_precision, train_recall, _ = precision_recall_curve(train_labels_all.numpy(), train_predictions[key].numpy())
                train_aupr = auc(train_recall, train_precision)
                train_f1, _ = compute_f1(train_labels_all.numpy(), train_predictions[key].numpy())
                
                test_auc = roc_auc_score(test_labels_all.numpy(), test_predictions[key].numpy())
                test_precision, test_recall, _ = precision_recall_curve(test_labels_all.numpy(), test_predictions[key].numpy())
                test_aupr = auc(test_recall, test_precision)
                test_f1, _ = compute_f1(test_labels_all.numpy(), test_predictions[key].numpy())
                
                metrics[key] = {
                    'train_auc': train_auc, 'train_aupr': train_aupr, 'train_f1': train_f1,
                    'test_auc': test_auc, 'test_aupr': test_aupr, 'test_f1': test_f1
                }
                
            # current_score = metrics['final']['test_f1']
            if config['use_stat'] and not config['use_cnn'] and not config['use_gnn']:
                current_score = metrics['stat']['test_aupr']
            elif config['use_cnn'] and not config['use_gnn'] and not config['use_stat']:
                current_score = metrics['cnn']['test_aupr']
            elif config['use_gnn'] and not config['use_cnn'] and not config['use_stat']:
                current_score = metrics['gnn']['test_aupr']
            else:
                current_score = metrics['final']['test_aupr']
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                early_stopping_counter = 0
                # Save best
                if config['use_stat'] and not config['use_cnn'] and not config['use_gnn']:
                    best_auc = metrics['stat']['test_auc']
                    best_aupr = metrics['stat']['test_aupr']
                    best_f1 = metrics['stat']['test_f1']
                elif config['use_cnn'] and not config['use_gnn'] and not config['use_stat']:
                    best_auc = metrics['cnn']['test_auc']
                    best_aupr = metrics['cnn']['test_aupr']
                    best_f1 = metrics['cnn']['test_f1']
                elif config['use_gnn'] and not config['use_cnn'] and not config['use_stat']:
                    best_auc = metrics['gnn']['test_auc']
                    best_aupr = metrics['gnn']['test_aupr']
                    best_f1 = metrics['gnn']['test_f1']
                else:
                    best_auc = metrics['final']['test_auc']
                    best_aupr = metrics['final']['test_aupr']
                    best_f1 = metrics['final']['test_f1']

                if config.get('save_checkpoints', True):
                    checkpoint_dir = f"{config['checkpoint_dir']}/fold_{fold}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

                if gate_weights_epoch:
                    best_gate_weights = torch.stack(gate_weights_epoch).mean(0).cpu().numpy().tolist()
            else:
                early_stopping_counter += 1
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(current_score)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[LR Scheduler] Current learning rate: {current_lr:.6e}")

            if epoch % config['log_frequency'] == 0:
                print(f"\nEpoch {epoch+1} Results:")
                print(f"  Train Loss - Total: {train_losses['total']:.4f}, CNN: {train_losses['cnn']:.4f}, GNN: {train_losses['gnn']:.4f}, Stat: {train_losses['stat']:.4f}, Final: {train_losses['final']:.4f}")
                print(f"  Train AUPR - CNN: {metrics['cnn']['train_aupr']:.4f}, GNN: {metrics['gnn']['train_aupr']:.4f}, Stat: {metrics['stat']['train_aupr']:.4f}, Final: {metrics['final']['train_aupr']:.4f}")
                print(f"  Test Loss - Total: {test_losses['total']:.4f}, CNN: {test_losses['cnn']:.4f}, GNN: {test_losses['gnn']:.4f}, Stat: {test_losses['stat']:.4f}, Final: {test_losses['final']:.4f}")
                print(f"  Test AUPR - CNN: {metrics['cnn']['test_aupr']:.4f}, GNN: {metrics['gnn']['test_aupr']:.4f}, Stat: {metrics['stat']['test_aupr']:.4f}, Final: {metrics['final']['test_aupr']:.4f}")
                
                if gate_weights_epoch and all([config['use_cnn'], config['use_gnn'], config['use_stat']]):
                    avg_gate_weights = torch.stack(gate_weights_epoch).mean(0)
                    print(f"  Gate weight - CNN: {avg_gate_weights[0]:.3f}, GNN: {avg_gate_weights[1]:.3f}, Stat: {avg_gate_weights[2]:.3f}")
                
                print(f"  Best AUPR: {best_score:.4f} (Epoch {best_epoch+1}), early stopping count: {early_stopping_counter}/{config['patience']}")
            
            # wandb
            if config.get('use_wandb', False):
                log_dict = {
                    'epoch': epoch,
                    'fold': fold,
                    **{f'train_loss_{k}': v for k, v in train_losses.items()},
                    **{f'test_loss_{k}': v for k, v in test_losses.items()},
                }
                
                for key, vals in metrics.items():
                    for metric, val in vals.items():
                        log_dict[f'{metric}_{key}'] = val
                
                if gate_weights_epoch:
                    avg_gate_weights = torch.stack(gate_weights_epoch).mean(0)
                    for i, name in enumerate(['cnn', 'gnn', 'stat']):
                        log_dict[f'gate_weight_{name}'] = avg_gate_weights[i].item()
                
                wandb.log(log_dict)
            
            # 早停检查
            if config['early_stopping'] and early_stopping_counter >= config['patience']:
                print(f"\nEarly stop!")
                break

        best_metrics.append({
            'fold': fold,
            'best_epoch': best_epoch,
            'test_auc': best_auc,
            'test_aupr': best_aupr,
            'test_f1': best_f1,
            'gate_weight_cnn': best_gate_weights[0] if best_gate_weights else None,
            'gate_weight_gnn': best_gate_weights[1] if best_gate_weights else None,
            'gate_weight_stat': best_gate_weights[2] if best_gate_weights and len(best_gate_weights)>2 else None
        })
        print(f"\nFold {fold+1} finish! Best AUPR: {best_score:.4f} (Epoch {best_epoch+1})")
        if config.get('use_wandb', False):
            wandb.finish()

    aucs = [m['test_auc'] for m in best_metrics]
    auprs = [m['test_aupr'] for m in best_metrics]
    f1s = [m['test_f1'] for m in best_metrics]
    print(f"Final AUPR: {np.mean(auprs):.4f} ± {np.std(auprs, ddof=1):.4f}")
    for i, metric in enumerate(best_metrics):
        print(f"  Fold {i+1}: {metric['test_aupr']:.4f} (Epoch {metric['best_epoch']+1})")

    fold_results = []
    for m in best_metrics:
        fold_results.append({
            'fold': m['fold']+1,
            'best_epoch': m['best_epoch']+1,
            'test_auc': m['test_auc'],
            'test_aupr': m['test_aupr'],
            'test_f1': m['test_f1'],
            'gate_weight_cnn': m.get('gate_weight_cnn', None),
            'gate_weight_gnn': m.get('gate_weight_gnn', None),
            'gate_weight_stat': m.get('gate_weight_stat', None)
        })
    mean_std_row = {
        'fold': 'mean±std',
        'best_epoch': '-',
        'test_auc': f"{np.mean(aucs):.4f}±{np.std(aucs, ddof=1):.4f}",
        'test_aupr': f"{np.mean(auprs):.4f}±{np.std(auprs, ddof=1):.4f}",
        'test_f1': f"{np.mean(f1s):.4f}±{np.std(f1s, ddof=1):.4f}",
        'gate_weight_cnn': '-',
        'gate_weight_gnn': '-',
        'gate_weight_stat': '-'
    }
    config_row = {k: v for k, v in config.items()}
    df = pd.DataFrame(fold_results + [mean_std_row])
    config_df = pd.DataFrame([config_row])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_name = f"new_musl_results_{timestamp}_{config['cv']}.csv"
    with open(csv_name, 'w', encoding='utf-8') as f:
        f.write('# config\n')
        config_df.to_csv(f, index=False)
        f.write('\n# results\n')
        df.to_csv(f, index=False)
    print(f"\nResults are saving to: {csv_name}")
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description='MuSL Training Script')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--use_cnn', action='store_true', help='Whether to use CNN')
    parser.add_argument('--use_gnn', action='store_true', help='Whether to use GNN')
    parser.add_argument('--use_stat', action='store_true', help='Whether to use Stat')
    parser.add_argument('--use_custom_adamw', action='store_true', help='Whether to use custom AdamW')
    parser.add_argument('--use_pre_embeddings', action='store_true', help='Whether to use pre-trained node embeddings')
    parser.add_argument('--use_pre_cal_state', action='store_true', help='Whether to use pre-calculated statistical metrics')
    parser.add_argument('--cal_state', action='store_true', help='Whether to calculate statistical metrics')
    parser.add_argument('--enable_cross_attention', action='store_true', help='Whether to enable cross-modal attention')
    parser.add_argument('--enable_adaptive_fusion', action='store_true', help='Whether to enable adaptive fusion')
    parser.add_argument('--enable_contrastive_learning', action='store_true', help='Whether to enable contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, help='Contrastive learning weight')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--target_node_dim', type=int, default=1024)
    parser.add_argument('--cv', type=str, help='Cross-validation method')
    parser.add_argument('--cl', action='store_true', help='Whether to use cell line data')
    parser.add_argument('--cl_name', type=str, help='Cell line name')
    parser.add_argument('--emb_type', type=str, default='protein', help='Embedding type')
    args = parser.parse_args()
    
    # Get config
    config = get_default_config()
    config['learning_rate'] = args.learning_rate if args.learning_rate else config['learning_rate']
    config['weight_decay'] = args.weight_decay if args.weight_decay else config['weight_decay']
    config['batch_size'] = args.batch_size if args.batch_size else config['batch_size']
    config['patience'] = args.patience if args.patience else config['patience']
    config['cv'] = args.cv if args.cv else config['cv']
    config['seed'] = args.seed if args.seed else config['seed']
    config['contrastive_weight'] = args.contrastive_weight if args.contrastive_weight else config['contrastive_weight']
    config['use_wandb'] = args.use_wandb
    config['use_cnn'] = args.use_cnn
    config['use_gnn'] = args.use_gnn
    config['use_stat'] = args.use_stat
    config['cal_state'] = args.cal_state
    config['enable_cross_attention'] = args.enable_cross_attention if args.enable_cross_attention else config['enable_cross_attention']
    config['enable_adaptive_fusion'] = args.enable_adaptive_fusion if args.enable_adaptive_fusion else config['enable_adaptive_fusion']
    config['enable_contrastive_learning'] = args.enable_contrastive_learning if args.enable_contrastive_learning else config['enable_contrastive_learning']
    config['use_custom_adamw'] = args.use_custom_adamw
    config['use_pre_embeddings'] = args.use_pre_embeddings
    config['use_pre_cal_state'] = args.use_pre_cal_state
    config['cl'] = args.cl
    config['cl_name'] = args.cl_name
    config['emb_type'] = args.emb_type
    config['target_node_dim'] = args.target_node_dim
    
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        config.update(custom_config)
        
    best_metrics = train_musl(config)
    
    return best_metrics

if __name__ == "__main__":
    main() 