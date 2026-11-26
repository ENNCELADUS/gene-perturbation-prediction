from collections import defaultdict
from typing import Any
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from omegaconf import DictConfig
# import deepspeed

from coach_pl.configuration import configurable
from coach_pl.model import build_model  #, build_criterion
from coach_pl.module import MODULE_REGISTRY
from esm4sl.utils.metric import mean_loss, compute_metrics


@MODULE_REGISTRY.register()
class ClsModule(pl.LightningModule):
    @configurable
    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model

        self.batch_size = cfg.DATALOADER.TRAIN.BATCH_SIZE
        self.base_lr = cfg.MODULE.OPTIMIZER.BASE_LR
        self.monitor = cfg.TRAINER.CHECKPOINT.MONITOR

        self.save_hyperparameters(cfg)

        self.loss_func = nn.BCEWithLogitsLoss()

        self.losses = defaultdict(list)
        self.logits = defaultdict(list)
        self.labels = defaultdict(list)
        self.g1_indices = defaultdict(list)
        self.g2_indices = defaultdict(list)
        self.attns = defaultdict(list)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        model = build_model(cfg)
        return {
            "model": model,
            "cfg": cfg,
        }

    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        if gradient_clip_algorithm == "value":
            nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), gradient_clip_val)
        elif gradient_clip_algorithm == "norm":
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_val)

    def configure_optimizers(self) -> dict[str, Optimizer | dict[str, str | Optimizer]]:
        lr = self.base_lr * self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size / 16

        # if 'deepspeed' in self.cfg.TRAINER.STRATEGY:
        #     optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        # else:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70], gamma=0.2)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            } 
        }

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        gene1, gene2, label, g1_idx, g2_idx = batch[0].cuda(), batch[1].cuda(), \
                batch[2].cuda(), batch[3].cuda(), batch[4].cuda()

        logit = self.model(torch.cat([gene1, gene2], dim=1))

        loss = self.loss_func(logit, label)
        return loss, logit, label, g1_idx, g2_idx, None

    def step_common(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: RunningStage, stage_str: str) -> dict[str, Any]:
        loss, logit, label, g1_idx, g2_idx, attn = self.forward(batch)

        if stage != RunningStage.TESTING:
            self.log(f"{stage_str}/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=len(label), sync_dist=True, rank_zero_only=True)

        self.losses[stage].append((loss.detach().item(), len(label)))
        self.logits[stage].append(logit.detach().cpu())
        self.labels[stage].append(label.detach().cpu())
        self.g1_indices[stage].append(g1_idx.detach().cpu())
        self.g2_indices[stage].append(g2_idx.detach().cpu())
        return {"loss": loss}

    def end_common(self, stage: RunningStage, stage_str: str) -> None:
        ml = mean_loss(self.losses[stage])
        labels = torch.cat(self.labels[stage])  # [N]
        logits = torch.cat(self.logits[stage])  # [N]
        
        probs = torch.sigmoid(logits)
        metrics = compute_metrics(labels, probs)

        self.log(f"{stage_str}/mean_loss", ml, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log(f"{stage_str}/auroc", metrics['auroc'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log(f"{stage_str}/auprc", metrics['auprc'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        if stage == RunningStage.VALIDATING:
            if self.monitor == 'avgmtr':
                self.log(self.monitor, (metrics['auroc'] + metrics['auprc']) / 2, logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
            elif self.monitor != 'loss':
                self.log(self.monitor, metrics[self.monitor.lower()], logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
            else:
                self.log(self.monitor, ml, logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        if stage == RunningStage.TESTING:
            for metric in ['acc', 'precision', 'recall', 'f1', 'bacc']:
                self.log(f"{stage_str}/{metric}", metrics[metric], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)

            g1_indices = torch.cat(self.g1_indices[stage])
            g2_indices = torch.cat(self.g2_indices[stage])
            df = pd.DataFrame({'gene1': g1_indices.detach().cpu().numpy(), 'gene2': g2_indices.detach().cpu().numpy(), 'probs': probs.detach().cpu().numpy()})
            output_dir = f'{self.cfg.OUTPUT_DIR}/csv_log'
            version_num = len(os.listdir(output_dir)) - 1
            df.to_csv(f'{output_dir}/version_{version_num}/test_logits.csv', index=False)
            # np.save(f'{output_dir}/version_{version_num}/test_metrics.npy', metrics, allow_pickle=True)

        self.losses[stage].clear()
        self.logits[stage].clear()
        self.labels[stage].clear()
        self.g1_indices[stage].clear()
        self.g2_indices[stage].clear()
        self.attns[stage].clear()

    def training_step(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        return self.step_common(batch, RunningStage.TRAINING, 'train')

    def on_train_epoch_end(self) -> None:
        self.end_common(RunningStage.TRAINING, 'train')

    def validation_step(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        return self.step_common(batch, RunningStage.VALIDATING, 'val')

    def on_validation_epoch_end(self) -> None:
        self.end_common(RunningStage.VALIDATING, 'val')

    def test_step(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Any]:        
        return self.step_common(batch, RunningStage.TESTING, 'test')

    def on_test_epoch_end(self) -> None:
        self.end_common(RunningStage.TESTING, 'test')


@MODULE_REGISTRY.register()
class AttnModule(ClsModule):
    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        g1_idx, g1, g1_mask, g2_idx, g2, g2_mask, label = batch[0].cuda(), batch[1].cuda(), \
            batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda()
        cell_embedding = batch[7].cuda() if self.cfg.DATASET.CELL_LINE else None

        logit, attns = self.model(g1, g1_mask, g2, g2_mask, cell_embedding)

        attn: torch.Tensor = attns[-1]
        _, L1, L2 = attn.shape
        attn = attn.reshape(len(logit), -1, L1, L2)
        attn = attn.mean(dim=1).squeeze(1)

        loss = self.loss_func(logit, label)
        return loss, logit, label, g1_idx, g2_idx, attn


@MODULE_REGISTRY.register()
class LoraModule(ClsModule):
    def forward(self, batch: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
        g1_list, g2_list, label, g1_idx, g2_idx = batch[0], batch[1], \
                batch[2].cuda(), batch[3].cuda(), batch[4].cuda()

        logit = self.model(g1_list, g2_list)

        loss = self.loss_func(logit, label)
        return loss, logit, label, g1_idx, g2_idx, None
