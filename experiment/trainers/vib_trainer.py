"""
VIB训练器类 - 专门用于Variational Information Bottleneck训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import numpy as np

from .base_trainer import BaseExperimentTrainer
from .core_trainer import CoreTrainer
from models.vib_model import build_vib_model, BetaWarmup


class VIBCoreTrainer(CoreTrainer):
    """VIB专用的核心训练器"""

    def __init__(self, config, model, optimizer, criterion, device, experiment_mode, train_loader_len, train_loader, beta_scheduler):
        super().__init__(config, model, optimizer, criterion, device, experiment_mode, train_loader_len, train_loader)
        self.beta_scheduler = beta_scheduler
        self.global_step = 0

    def train_epoch(self, train_loader):
        """VIB训练的epoch"""
        self.model.train()
        total_loss = 0.0
        total_util_loss = 0.0
        total_kl_loss = 0.0

        # 创建进度条
        from tqdm import tqdm
        import sys
        iterable_loader = tqdm(train_loader, desc="VIB Training", leave=False,
                              file=sys.stderr, dynamic_ncols=True,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch in iterable_loader:
            x, y, mask = batch[0], batch[1], batch[2]
            x = torch.nan_to_num(x, nan=0.0)
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

            if mask.sum().item() == 0:
                mask = None

            self.optimizer.zero_grad()

            # VIB前向传播
            logits, pooled, vib_out = self.model(x, padding_mask=mask, sample=True)

            # 获取当前的beta值
            current_beta = self.beta_scheduler(self.global_step)

            # 计算VIB损失
            loss, metrics = self.model.vib.compute_loss(pooled, y, current_beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_util_loss += metrics['util'].item()
            total_kl_loss += metrics['kl'].item()
            self.global_step += 1

            # 更新进度条
            current_batch = iterable_loader.n
            current_avg_loss = total_loss / current_batch if current_batch > 0 else 0.0
            current_avg_util_loss = total_util_loss / current_batch if current_batch > 0 else 0.0
            current_avg_kl_loss = total_kl_loss / current_batch if current_batch > 0 else 0.0
            iterable_loader.set_postfix(
                loss=f"{current_avg_loss:.4f}",
                util=f"{current_avg_util_loss:.4f}",
                kl=f"{current_avg_kl_loss:.4f}",
                beta=f"{current_beta:.6f}"
            )

        iterable_loader.close()
        avg_loss = total_loss / len(train_loader)
        avg_util_loss = total_util_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)

        return avg_loss, {
            'util_loss': avg_util_loss,
            'kl_loss': avg_kl_loss,
            'beta': current_beta
        }

    def evaluate(self, val_loader):
        """VIB评估"""
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
        from collections import defaultdict, Counter
        from tqdm import tqdm
        import sys

        self.model.eval()
        speaker_preds = defaultdict(list)
        speaker_labels = {}
        speaker_logits = defaultdict(list)

        with torch.no_grad():
            eval_loader = tqdm(val_loader, desc="VIB Evaluating", leave=False,
                              file=sys.stderr, dynamic_ncols=True,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

            for batch in eval_loader:
                x, y, mask, speaker_ids = batch[0], batch[1], batch[2], batch[3]
                x = torch.nan_to_num(x, nan=0.0)
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

                if mask.sum().item() == 0:
                    mask = None

                # VIB MC预测
                h = self.model.encoder(x, padding_mask=mask)
                logits = self.model.vib.mc_predict(h)

                preds = torch.argmax(logits, dim=1).cpu().tolist()
                
                labels = y.cpu().tolist()

                if logits.shape[1] == 2:
                    pos_probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                else:
                    pos_probs = torch.sigmoid(logits).cpu().tolist() if logits.ndim == 1 else [0.5] * len(labels)

                for i, (spk, pred, true, prob) in enumerate(zip(speaker_ids, preds, labels, pos_probs)):
                    speaker_preds[spk].append(pred)
                    speaker_labels[spk] = true
                    speaker_logits[spk].append(prob)

        eval_loader.close()

        # 计算speaker级别的指标
        final_preds, final_labels, final_probs = [], [], []
        for sid, pred_list in speaker_preds.items():
            if pred_list:
                majority_pred = Counter(pred_list).most_common(1)[0][0]
                if sid in speaker_labels:
                    final_preds.append(majority_pred)
                    final_labels.append(speaker_labels[sid])
                    final_probs.append(np.mean(speaker_logits[sid]))

        if not final_preds:
            return {'acc': 0.0, 'p': 0.0, 'r': 0.0, 'f1': 0.0, 'auc': 0.0, 'wer': 0.0}

        acc = accuracy_score(final_labels, final_preds)
        p, r, f1, _ = precision_recall_fscore_support(final_labels, final_preds, average='binary', zero_division=0)

        try:
            auc = roc_auc_score(final_labels, final_probs)
        except ValueError:
            auc = 0.5

        return {
            'acc': float(acc), 'p': float(p), 'r': float(r),
            'f1': float(f1), 'auc': float(auc), 'wer': 0.0
        }


class VIBTrainer(BaseExperimentTrainer):
    """VIB专用训练器"""

    def __init__(self, config):
        super().__init__(config)
        self.beta_scheduler = None
        self.vib_model = None

    def create_model(self, train_set_size: int = None) -> nn.Module:
        """创建VIB模型"""
        print("Creating VIB model...")

        # 计算steps_per_epoch
        if train_set_size is not None:
            steps_per_epoch = max(1, train_set_size // self.config.data['batch_size'])
        else:
            # 使用估算值
            estimated_dataset_size = 10000
            steps_per_epoch = estimated_dataset_size // self.config.data['batch_size']

        print(f"Estimated steps per epoch: {steps_per_epoch}")

        # 构建VIB模型
        model, beta_sched, _ = build_vib_model(
            self.config,
            self.config.vib_cfg.__dict__ if self.config.vib_cfg else {},
            steps_per_epoch
        )

        self.beta_scheduler = beta_sched
        self.vib_model = model

        return model.to(self.config.general['device'])

    def create_trainer(self, model: nn.Module, train_set, train_loader: DataLoader) -> VIBCoreTrainer:
        """创建VIB训练器"""
        print("Creating VIB trainer with AdamW optimizer...")

        # VIB使用标准的AdamW优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.training['lr'],
            weight_decay=self.config.training['weight_decay']
        )

        criterion = nn.CrossEntropyLoss()  # VIB内部会处理损失计算，这里主要是为了兼容性

        trainer = VIBCoreTrainer(
            self.config, model, optimizer, criterion,
            self.config.general['device'], 'VIB',
            len(train_set), train_loader, self.beta_scheduler
        )

        return trainer

    def train_single_fold(self, fold_idx: int, val_fold_num: int,
                         train_df, val_df) -> Dict[str, float]:
        """训练单个fold - VIB版本"""
        print(f"\n--- Starting VIB Fold {fold_idx + 1}/{self.config.general['num_folds']} (Validation Fold: {val_fold_num}) ---")

        # 创建数据集和数据加载器
        train_set, val_set = self.create_datasets(train_df, val_df)
        train_loader, val_loader = self.create_data_loaders(train_set, val_set)

        # 创建VIB模型和训练器
        model = self.create_model(len(train_set))
        trainer = self.create_trainer(model, train_set, train_loader)
    
        # 固定最后一个epoch记录策略
        target_epoch = self.config.general['epochs']

        # 初始验证
        val_metrics = trainer.evaluate(val_loader)
        print(f"  [VIB Fold {fold_idx+1}, Epoch {0}] | Val Acc: {val_metrics['acc']:.4f}, P: {val_metrics['p']:.4f}, R: {val_metrics['r']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        for epoch in range(1, self.config.general['epochs'] + 1):
            train_result = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)

            # 处理VIB训练结果
            if isinstance(train_result, tuple):
                avg_train_loss, train_details = train_result
                loss_str = f"Train Loss: {avg_train_loss:.4f}, Util: {train_details['util_loss']:.4f}, KL: {train_details['kl_loss']:.4f}, Beta: {train_details['beta']:.6f}"
            else:
                avg_train_loss = train_result
                loss_str = f"Train Loss: {avg_train_loss:.4f}"

            print(f"  [VIB Fold {fold_idx+1}, Epoch {epoch}] {loss_str} | Val Acc: {val_metrics['acc']:.4f}, P: {val_metrics['p']:.4f}, R: {val_metrics['r']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

            if epoch == target_epoch:
                target_epoch_metrics = val_metrics.copy()
                target_epoch_metrics['loss'] = avg_train_loss
                # 保存最后一个epoch的模型
                model_name = f'checkpoints/last_epoch_model_{self.config.feature_type}_{self.config.experiment_mode}_fold_{val_fold_num}.pth'
                torch.save(model.state_dict(), model_name)

        #     if val_metrics['acc'] > best_fold_val_acc:
        #         best_fold_val_acc = val_metrics['acc']
        #         best_fold_metrics = val_metrics
        #         best_fold_metrics['loss'] = avg_train_loss
        #         # 保存模型到checkpoints文件夹
        #         model_name = f'checkpoints/best_model_{self.config.feature_type}_vib_fold_{val_fold_num}.pth'
        #         torch.save(model.state_dict(), model_name)

        # print(f"\n--- VIB Fold {fold_idx + 1}/{self.config.general['num_folds']} (Validation Fold: {val_fold_num}) BEST Results ---")
        # print(f"  Best Training Loss (at best ACC epoch): {best_fold_metrics['loss']:.4f}")
        # print(f"  Best Val Speaker-Level - Acc: {best_fold_metrics['acc']:.4f}, P: {best_fold_metrics['p']:.4f}, R: {best_fold_metrics['r']:.4f}, F1: {best_fold_metrics['f1']:.4f}, AUC: {best_fold_metrics['auc']:.4f}")

        print(f"\n--- Fold {fold_idx + 1}/{self.config.general['num_folds']} (Validation Fold: {val_fold_num}) Epoch {target_epoch} Results ---")
        if target_epoch_metrics:
            print(f"  Last Training Loss: {target_epoch_metrics['loss']:.4f}")
            print(f"  Last Val Speaker-Level - Acc: {target_epoch_metrics['acc']:.4f}, P: {target_epoch_metrics['p']:.4f}, R: {target_epoch_metrics['r']:.4f}, F1: {target_epoch_metrics['f1']:.4f}, AUC: {target_epoch_metrics['auc']:.4f}")

        return target_epoch_metrics