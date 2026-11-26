import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import SAGEConv

import numpy as np
from sklearn.metrics import f1_score
from .utils import vector_state

class EdgeDataset(Dataset):
    def __init__(self, edges, labels, exp_data, state = None, bins=32, cal_state=True):
        self.edges = edges
        self.labels = labels
        self.exp_data = exp_data
        self.bins = bins
        self.log_exp_data = torch.log1p(exp_data)
        mean = torch.mean(self.log_exp_data, dim=1, keepdim=True)
        std = torch.std(self.log_exp_data, dim=1, keepdim=True)
        std = torch.where(std < 1e-10, torch.ones_like(std), std)
        self.norm_exp_data = (self.log_exp_data - mean) / std
        self.norm_exp_data = torch.where(std == 1, torch.zeros_like(self.norm_exp_data), self.norm_exp_data)
        
        self.state = state
        # if self.state is None and cal_state:
        #     self.state = vector_state(exp_data,self.log_exp_data,self.norm_exp_data,self.edges)
        if self.state is None and not cal_state:
            self.state = torch.zeros(self.edges.shape[1],35,device=self.edges.device)
        
    def __len__(self):
        return self.edges.size(1)
    
    def __getitem__(self, idx):
        exp_a_raw = self.norm_exp_data[self.edges[0,idx]].numpy()
        exp_b_raw = self.norm_exp_data[self.edges[1,idx]].numpy()
        exp_a_log = self.log_exp_data[self.edges[0,idx]].numpy()
        exp_b_log = self.log_exp_data[self.edges[1,idx]].numpy()
        exp_a_norm = self.norm_exp_data[self.edges[0,idx]].numpy()
        exp_b_norm = self.norm_exp_data[self.edges[1,idx]].numpy()
        image_raw = torch.from_numpy(np.histogram2d(
                                    exp_a_raw, 
                                    exp_b_raw, 
                                    bins=[self.bins,self.bins],
                                )[0]).float().unsqueeze(0)
        image_log = torch.from_numpy(np.histogram2d(
                                    exp_a_log, 
                                    exp_b_log, 
                                    bins=[self.bins,self.bins],
                                )[0]).float().unsqueeze(0)
        image_norm = torch.from_numpy(np.histogram2d(
                                    exp_a_norm, 
                                    exp_b_norm, 
                                    bins=[self.bins,self.bins],
                                )[0]).float().unsqueeze(0)
        image = torch.cat([image_raw, image_log, image_norm], dim=0)

        state = self.state[idx]
        
        return {
            'image': image,
            'state': state,
            'edge': self.edges[:, idx],
            'label': self.labels[idx]
        }

class CrossModalAttention(nn.Module):
    
    def __init__(self, feature_dims, hidden_dim=256):
        super(CrossModalAttention, self).__init__()
        self.feature_dims = feature_dims  # [cnn_dim, gnn_dim, stat_dim]
        self.hidden_dim = hidden_dim
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features):
        """
        Args:
            features: list of [batch_size, feature_dim]
        Returns:
            enhanced_features: list of [batch_size, hidden_dim]
        """
        projected_features = []
        for feat, proj in zip(features, self.projections):
            projected_features.append(proj(feat))
        
        stacked_features = torch.stack(projected_features, dim=1)
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        enhanced_features = self.norm(attended_features + stacked_features)
        enhanced_list = [enhanced_features[:, i, :] for i in range(len(features))]
        
        return enhanced_list, attention_weights

class TeLU(nn.Module):
    def __init__(self):
        super(TeLU, self).__init__()
    def forward(self, input):
        return input * torch.tanh( torch.exp(input) )

class AdaptiveFusion(nn.Module):
    
    def __init__(self, feature_dims, output_dim=256):
        super(AdaptiveFusion, self).__init__()
        self.feature_dims = feature_dims
        self.total_dim = sum(feature_dims)
        
        self.gate_network = nn.Sequential(
            nn.Linear(self.total_dim, len(feature_dims)),
            nn.Softmax(dim=1)
        )
        
        self.fusion_network = nn.Sequential(
            nn.Linear(self.total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, features):
        """
        Args:
            features: list of tensors with different dimensions
        Returns:
            fused_features: [batch_size, output_dim]
            gate_weights: [batch_size, num_modalities]
        """
        concatenated = torch.cat(features, dim=1)
        gate_weights = self.gate_network(concatenated)
        
        min_weight = 0.1
        max_weight = 0.8
        gate_weights = min_weight + (max_weight - min_weight) * gate_weights
        
        gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True)
        
        weighted_features = []
        for i, feat in enumerate(features):
            weight = gate_weights[:, i:i+1]  # [batch_size, 1]
            weighted_feat = feat * weight
            weighted_features.append(weighted_feat)
        
        weighted_concat = torch.cat(weighted_features, dim=1)
        fused_features = self.fusion_network(weighted_concat)
        
        return fused_features, gate_weights

class ContrastiveLoss(nn.Module):
    
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features_list, labels):
        """
        Args:
            features_list: list of [batch_size, feature_dim]
            labels: [batch_size]
        """
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                feat_i = F.normalize(features_list[i], dim=1)
                feat_j = F.normalize(features_list[j], dim=1)
                
                similarity_matrix = torch.matmul(feat_i, feat_j.T) / self.temperature
                
                labels_expand = labels.unsqueeze(1)
                target_matrix = (labels_expand == labels_expand.T).float()
                loss = F.cross_entropy(similarity_matrix, target_matrix)
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs

class CNNModule(nn.Module):
    
    def __init__(self, input_channels=1, output_dim=256):
        super(CNNModule, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        return features

class GNNModule(nn.Module):
    
    def __init__(self, node_feature_dim, num_graphs=1, hidden_dim=128, output_dim=256):
        super(GNNModule, self).__init__()
        self.num_graphs = num_graphs
        self.output_dim = output_dim
        
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_graphs):
            gcn_seq = nn.Sequential(
                SAGEConv(node_feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                SAGEConv(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
            self.gcn_layers.append(gcn_seq)
        
        self.pair_processor = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
    def forward(self, node_features, edge_indices, gene_pairs):
        graph_outputs = []
        for i in range(self.num_graphs):
            x = node_features
            for layer in self.gcn_layers[i]:
                if isinstance(layer, SAGEConv):
                    x = layer(x, edge_indices[i])
                else:
                    x = layer(x)
            graph_outputs.append(x)
        
        if self.num_graphs == 1:
            node_features_out = graph_outputs[0]
        else:
            node_features_out = torch.mean(torch.stack(graph_outputs), dim=0)
        
        gene_emb0 = node_features_out[gene_pairs[:, 0]]
        gene_emb1 = node_features_out[gene_pairs[:, 1]]
        pair_features_raw = torch.cat([gene_emb0, gene_emb1], dim=1)
        
        pair_features = self.pair_processor(pair_features_raw)
        
        return pair_features

class StatModule(nn.Module):
    
    def __init__(self, input_dim=35, output_dim=256):
        super(StatModule, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        self.use_residual = input_dim <= output_dim
        if self.use_residual:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        if x is None:
            batch_size = x.size(0) if hasattr(x, 'size') else 1
            return torch.zeros(batch_size, self.feature_extractor[-2].out_features, device=x.device if hasattr(x, 'device') else 'cpu')
        
        x_norm = self.input_norm(x)
        features = self.feature_extractor(x_norm)
        if self.use_residual:
            residual = self.residual_projection(x_norm)
            features = features + residual
        
        return features

class MuSL(nn.Module):
    
    def __init__(self, num_nodes, node_embedding_dim, hidden_dim, target_node_dim=None, node_embeddings=None, 
                 unified_dim=256, enable_cross_attention=True, enable_adaptive_fusion=True,
                 enable_contrastive_learning=True, use_cnn=True, use_gnn=True, use_stat=True):
        super(MuSL, self).__init__()
        
        self.unified_dim = unified_dim
        self.enable_cross_attention = enable_cross_attention
        self.enable_adaptive_fusion = enable_adaptive_fusion
        self.enable_contrastive_learning = enable_contrastive_learning
        
        self.use_cnn = use_cnn
        self.use_gnn = use_gnn
        self.use_stat = use_stat
        if target_node_dim is not None:
            self.target_node_dim = target_node_dim
        else:
            self.target_node_dim = node_embedding_dim
        
        if node_embeddings is not None:
            self.node_embeddings = nn.Embedding.from_pretrained(node_embeddings, freeze=False)
        else:
            self.node_embeddings = nn.Embedding(num_nodes, target_node_dim)
            nn.init.xavier_uniform_(self.node_embeddings.weight)
        
        if self.target_node_dim != node_embedding_dim and node_embeddings is not None:
            self.node_projector = nn.Sequential(
                    nn.Linear(node_embedding_dim,target_node_dim),
                    nn.LayerNorm(target_node_dim),
                    nn.ReLU(),
                    nn.Dropout(0.25)
                    )
        
        if self.use_cnn:
            self.cnn_module = CNNModule(input_channels=3, output_dim=unified_dim)
        if self.use_gnn:
            self.gnn_module = GNNModule(
                node_feature_dim=self.target_node_dim, 
                hidden_dim=hidden_dim, 
                output_dim=unified_dim
            )
        if self.use_stat:
            self.stat_module = StatModule(input_dim=35, output_dim=unified_dim)
        
        if self.enable_cross_attention:
            feature_dims = []
            if self.use_cnn:
                feature_dims.append(unified_dim)
            if self.use_gnn:
                feature_dims.append(unified_dim)
            if self.use_stat:
                feature_dims.append(unified_dim)
            
            if len(feature_dims) > 2:
                self.cross_attention = CrossModalAttention(
                    feature_dims=feature_dims,
                    hidden_dim=unified_dim
                )
            else:
                self.enable_cross_attention = False
        
        if self.enable_adaptive_fusion:
            feature_dims = []
            if self.use_cnn:
                feature_dims.append(unified_dim)
            if self.use_gnn:
                feature_dims.append(unified_dim)
            if self.use_stat:
                feature_dims.append(unified_dim)
            
            if len(feature_dims) > 2:
                self.adaptive_fusion = AdaptiveFusion(
                    feature_dims=feature_dims,
                    output_dim=unified_dim
                )
            else:
                self.enable_adaptive_fusion = False
        
        if not self.enable_adaptive_fusion:
            total_dim = sum([unified_dim if self.use_cnn else 0,
                           unified_dim if self.use_gnn else 0,
                           unified_dim if self.use_stat else 0])
            if total_dim > 0:
                self.traditional_fusion = nn.Sequential(
                    nn.Linear(total_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, unified_dim)
                )
        
        if self.use_cnn:
            self.cnn_classifier = nn.Sequential(
                nn.Linear(unified_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )
        if self.use_gnn:
            self.gnn_classifier = nn.Sequential(
                nn.Linear(unified_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )
        if self.use_stat:
            self.stat_classifier = nn.Sequential(
                nn.Linear(unified_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(unified_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        if self.enable_contrastive_learning:
            self.contrastive_loss = ContrastiveLoss(temperature=0.1)
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images, state, edge_index, gene_pairs, return_attention=False):
        batch_size = images.size(0)
        
        if self.use_cnn:
            x_cnn = self.cnn_module(images)  # [batch, unified_dim]
        else:
            x_cnn = torch.zeros(batch_size, self.unified_dim, device=images.device)
        
        if self.use_gnn:
            node_features = self.node_embeddings.weight
            if hasattr(self, 'node_projector') and self.node_projector is not None:
                node_features = self.node_projector(node_features)
            x_gnn = self.gnn_module(node_features, edge_index, gene_pairs)  # [batch, unified_dim]
        else:
            x_gnn = torch.zeros(batch_size, self.unified_dim, device=images.device)
        
        if self.use_stat and state is not None:
            x_stat = self.stat_module(state)  # [batch, unified_dim]
        else:
            x_stat = torch.zeros(batch_size, self.unified_dim, device=images.device)
        
        attention_weights = None
        if self.enable_cross_attention:
            features_list = []
            if self.use_cnn:
                features_list.append(x_cnn)
            if self.use_gnn:
                features_list.append(x_gnn)
            if self.use_stat:
                features_list.append(x_stat)
            
            enhanced_features, attention_weights = self.cross_attention(features_list)
            
            idx = 0
            x_cnn_enhanced = enhanced_features[idx] if self.use_cnn else x_cnn
            idx += 1 if self.use_cnn else 0
            x_gnn_enhanced = enhanced_features[idx] if self.use_gnn else x_gnn
            idx += 1 if self.use_gnn else 0
            x_stat_enhanced = enhanced_features[idx] if self.use_stat else x_stat
        else:
            x_cnn_enhanced, x_gnn_enhanced, x_stat_enhanced = x_cnn, x_gnn, x_stat
        
        if self.use_cnn:
            cnn_pred = self.cnn_classifier(x_cnn_enhanced)
        else:
            cnn_pred = torch.zeros(batch_size, 2, device=images.device)
        
        if self.use_gnn:
            gnn_pred = self.gnn_classifier(x_gnn_enhanced)
        else:
            gnn_pred = torch.zeros(batch_size, 2, device=images.device)
        
        if self.use_stat:
            stat_pred = self.stat_classifier(x_stat_enhanced)
        else:
            stat_pred = torch.zeros(batch_size, 2, device=images.device)
        
        gate_weights = None
        if self.enable_adaptive_fusion:
            features_list = []
            if self.use_cnn:
                features_list.append(x_cnn_enhanced)
            if self.use_gnn:
                features_list.append(x_gnn_enhanced)
            if self.use_stat:
                features_list.append(x_stat_enhanced)
            
            fused_features, gate_weights = self.adaptive_fusion(features_list)
        else:
            features_list = []
            if self.use_cnn:
                features_list.append(x_cnn_enhanced)
            if self.use_gnn:
                features_list.append(x_gnn_enhanced)
            if self.use_stat:
                features_list.append(x_stat_enhanced)
            
            if len(features_list) > 0:
                concatenated = torch.cat(features_list, dim=1)
                fused_features = self.traditional_fusion(concatenated)
            else:
                fused_features = torch.zeros(batch_size, self.unified_dim, device=images.device)
        
        final_pred = self.final_classifier(fused_features)
        
        results = {
            'cnn_pred': cnn_pred,
            'gnn_pred': gnn_pred,
            'stat_pred': stat_pred,
            'final_pred': final_pred,
            'features': {
                'cnn': x_cnn_enhanced,
                'gnn': x_gnn_enhanced,
                'stat': x_stat_enhanced,
                'fused': fused_features
            }
        }
        
        if return_attention and attention_weights is not None:
            results['attention_weights'] = attention_weights
        if gate_weights is not None:
            results['gate_weights'] = gate_weights
            
        return results
    
    def compute_contrastive_loss(self, features, labels):
        if not self.enable_contrastive_learning:
            return torch.tensor(0.0, device=labels.device)
        
        feature_list = []
        if self.use_cnn:
            feature_list.append(features['cnn'])
        if self.use_gnn:
            feature_list.append(features['gnn'])
        if self.use_stat:
            feature_list.append(features['stat'])
        
        if len(feature_list) < 2:
            return torch.tensor(0.0, device=labels.device)
        
        return self.contrastive_loss(feature_list, labels)

def train_step(model, batch, edge_index, criterion, device, alpha=0.1):

    images = batch['image'].to(device)
    state = batch['state'].to(device) if 'state' in batch and batch['state'] is not None else None
    gene_pairs = batch['edge'].to(device)
    labels = batch['label'].to(device)
    results = model(images, state, [edge_index.to(device)], gene_pairs)
    
    total_loss = 0.0
    loss_cnn = torch.tensor(0.0, device=device)
    loss_gnn = torch.tensor(0.0, device=device)
    loss_stat = torch.tensor(0.0, device=device)
    
    if model.use_cnn:
        loss_cnn = criterion(results['cnn_pred'], labels)
        total_loss += loss_cnn
    
    if model.use_gnn:
        loss_gnn = criterion(results['gnn_pred'], labels)
        total_loss += loss_gnn
    
    if model.use_stat:
        loss_stat = criterion(results['stat_pred'], labels)
        total_loss += loss_stat
    
    loss_final = criterion(results['final_pred'], labels)
    total_loss += loss_final
    
    contrastive_loss = model.compute_contrastive_loss(results['features'], labels)
    total_loss += alpha * contrastive_loss
    
    return {
        'total_loss': total_loss,
        'loss_cnn': loss_cnn,
        'loss_gnn': loss_gnn,
        'loss_stat': loss_stat,
        'loss_final': loss_final,
        'loss_contrastive': contrastive_loss,
        'predictions': {
            'cnn_pred': results['cnn_pred'],
            'gnn_pred': results['gnn_pred'],
            'stat_pred': results['stat_pred'],
            'final_pred': results['final_pred']
        },
        'gate_weights': results.get('gate_weights', None),
        'attention_weights': results.get('attention_weights', None)
    }

def compute_f1(y_true, y_pred_float):
    min_val, max_val = np.min(y_pred_float), np.max(y_pred_float)
    if min_val == max_val:
        return 0, 0
        
    best_threshold = 0
    best_f1 = 0
    thresholds = np.linspace(min_val, max_val, 50)

    for _ in range(3):
        f1_scores = [f1_score(y_true, (y_pred_float >= t).astype(int)) for t in thresholds]
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_threshold = thresholds[best_idx]

        if best_idx > 0 and best_idx < len(thresholds) - 1:
            min_val = thresholds[best_idx - 1]
            max_val = thresholds[best_idx + 1]
        thresholds = np.linspace(min_val, max_val, 50)

    return best_f1, best_threshold

def min_max_normalize(probs):
    min_val = probs.min()
    max_val = probs.max()
    return (probs - min_val) / (max_val - min_val + 1e-10)

def get_default_config():

    return {
        'hidden_dim': 128,
        'unified_dim': 256, 
        'enable_cross_attention': False,
        'enable_adaptive_fusion': False,
        'enable_contrastive_learning': False,
        'contrastive_weight': 0.1,
        
        'use_cnn': True,
        'use_gnn': True,
        'use_stat': True,
        
        'epochs': 100,
        'batch_size': 128,
        'num_workers': 12,
        'learning_rate': 0.00005,
        'weight_decay': 0.001,
        'target_node_dim': 1024,
        
        'fold': 0,
        'cv': 'CV3',
        'use_state': True,
        'bins': 32,
        
        'project_name': 'MuSL',
        'experiment_name': None,
        'save_checkpoints': True,
        'checkpoint_dir': 'checkpoints',
        
        'early_stopping': True,
        'patience': 7,
        'min_delta': 1e-6,
        
        'seed': 432,
        'log_frequency': 1,
        'lr_scheduler': 'plateau',
        'lr_patience': 3,
        'use_telu': False,
        'correct_norm': True
    }