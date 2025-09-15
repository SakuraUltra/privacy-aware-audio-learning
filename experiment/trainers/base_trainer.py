"""
基础训练器类
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional
import warnings

from utils.utils import set_seed
from models.transformer import TransformerClassifier
from .core_trainer import CoreTrainer
from utils.data_utils import load_and_validate_data, split_data_by_folds


class BaseExperimentTrainer:
    """基础实验训练器"""
    
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        # 抑制警告
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='opacus')
        warnings.filterwarnings('ignore', category=UserWarning, module='opacus')
        warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载数据"""
        return load_and_validate_data(self.config)
    
    def create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """创建数据集"""
        dataset_class = self.config.get_dataset_class()
        
        if self.config.feature_type == 'mel':
            train_set = dataset_class(train_df, from_df=True, normalize=True, config=self.config)
            val_set = dataset_class(val_df, from_df=True, normalize=True, config=self.config)
        else:
            train_set = dataset_class(train_df, from_df=True, normalize=True)
            val_set = dataset_class(val_df, from_df=True, normalize=True)
        
        return train_set, val_set
    
    def create_data_loaders(self, train_set, val_set) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        collate_fn = self.config.get_collate_fn()
        shuffle_train = False if self.config.experiment_mode == 'dp' else True
        
        train_loader = DataLoader(
            train_set, 
            batch_size=self.config.data['batch_size'], 
            shuffle=shuffle_train, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=self.config.data['batch_size'], 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
    
    def create_model(self) -> nn.Module:
        """创建模型"""
        if self.model is not None:
            print("Using externally provided model (VIB mode)")
            return self.model
        dp_mode = (self.config.experiment_mode == 'dp')
        print(f"Creating model with dp_mode={dp_mode}")
        model = TransformerClassifier(
            d_model=self.config.model['d_model'],
            nhead=self.config.model['nhead'],
            num_layers=self.config.model['num_layers'],
            num_classes=self.config.model['num_classes'],
            dim_feedforward=self.config.model['dim_feedforward'],
            dropout=self.config.model['dropout'],
            dp_mode=dp_mode
        ).to(self.config.general['device'])
        return model
    
    def create_trainer(self, model: nn.Module, train_set, train_loader: DataLoader) -> CoreTrainer:
        """创建训练器"""
        criterion = nn.CrossEntropyLoss()
        
        if self.config.experiment_mode == 'dp':
            # DP模式：使用更高的学习率来克服噪声干扰
            dp_lr = self.config.training['lr'] * 10.0  # 提高10倍学习率，并且使用Adam而非AdamW
            print(f"DP Mode: Creating separate optimizers with enhanced learning rate {dp_lr}")
            optimizer_encoder = optim.Adam(
                model.encoder.parameters(),
                lr=dp_lr,
                weight_decay=self.config.training['weight_decay']
            )
            optimizer_classifier = optim.Adam(
                model.classifier.parameters(),
                lr=dp_lr,
                weight_decay=self.config.training['weight_decay']
            )
            optimizers = (optimizer_encoder, optimizer_classifier)
            trainer = CoreTrainer(
                self.config, model, optimizers, criterion,
                self.config.general['device'], self.config.experiment_mode.upper(),
                len(train_set), train_loader
            )
        else:
            print("Normal Mode: Creating standard optimizer.")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.training['lr'],
                weight_decay=self.config.training['weight_decay']
            )
            trainer = CoreTrainer(
                self.config, model, optimizer, criterion,
                self.config.general['device'], self.config.experiment_mode.upper(),
                len(train_set), train_loader
            )
        
        return trainer

    def train_single_fold(self, fold_idx: int, val_fold_num: int,
                         train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
        """训练单个fold"""
        # print(f"\n[DEBUG] Starting Fold {fold_idx + 1}. Configured epochs: {self.config.general['epochs']}")

        print(f"\n--- Starting Fold {fold_idx + 1}/{self.config.general['num_folds']} (Validation Fold: {val_fold_num}) ---")

        # 创建数据集和数据加载器
        train_set, val_set = self.create_datasets(train_df, val_df)
        train_loader, val_loader = self.create_data_loaders(train_set, val_set)

        # 创建模型和训练器
        model = self.create_model()
        trainer = self.create_trainer(model, train_set, train_loader)

        # 训练循环
        best_fold_val_acc = -np.inf
        best_fold_metrics = {}

        # 只取最后一个epoch记录策略
        target_epoch = self.config.general['epochs']
        target_epoch_metrics = {}

        # 初始验证
        val_metrics = trainer.evaluate(val_loader)
        print(f"  [Fold {fold_idx+1}, Epoch {0}] | Val Acc: {val_metrics['acc']:.4f}, P: {val_metrics['p']:.4f}, R: {val_metrics['r']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        for epoch in range(1, self.config.general['epochs'] + 1):
            train_result = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)

            # 处理训练结果
            if isinstance(train_result, tuple):
                avg_train_loss = train_result[0]
                loss_str = f"Train Loss: {avg_train_loss:.4f}"
            else:
                avg_train_loss = train_result
                loss_str = f"Train Loss: {avg_train_loss:.4f}"

            print(f"  [Fold {fold_idx+1}, Epoch {epoch}] {loss_str} | Val Acc: {val_metrics['acc']:.4f}, P: {val_metrics['p']:.4f}, R: {val_metrics['r']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

            # if val_metrics['acc'] > best_fold_val_acc:
            #     best_fold_val_acc = val_metrics['acc']
            #     best_fold_metrics = val_metrics
            #     best_fold_metrics['loss'] = avg_train_loss
            #     # 保存模型到checkpoints文件夹
            #     model_name = f'checkpoints/best_model_{self.config.feature_type}_{self.config.experiment_mode}_fold_{val_fold_num}.pth'
            #     torch.save(model.state_dict(), model_name)

            if epoch == target_epoch:
                target_epoch_metrics = val_metrics.copy()
                target_epoch_metrics['loss'] = avg_train_loss
                model_name = f'checkpoints/last_epoch_model_{self.config.feature_type}_{self.config.experiment_mode}_fold_{val_fold_num}.pth'
                torch.save(model.state_dict(), model_name)

        print(f"\n--- Fold {fold_idx + 1}/{self.config.general['num_folds']} (Validation Fold: {val_fold_num}) BEST Results ---")
        if target_epoch_metrics:
            print(f"  Last Training Loss: {target_epoch_metrics['loss']:.4f}")
            print(f"  Last Val Speaker-Level - Acc: {target_epoch_metrics['acc']:.4f}, P: {target_epoch_metrics['p']:.4f}, R: {target_epoch_metrics['r']:.4f}, F1: {target_epoch_metrics['f1']:.4f}, AUC: {target_epoch_metrics['auc']:.4f}")
        
        # print(f"[DEBUG] Returning from Fold {fold_idx + 1}. Metrics dictionary being returned: {target_epoch_metrics}")

        return target_epoch_metrics

    def run_cross_validation(self) -> Dict[str, float]:
        """运行交叉验证"""
        print(f"Experiment Mode: {self.config.experiment_mode.upper()}")
        print(f"Feature Type: {self.config.feature_type.upper()}")
        print(f"General Config: {self.config.general}")
        print(f"Data Config: {self.config.data}")
        print(f"Model Config: {self.config.model}")
        print(f"Training Config: {self.config.training}")

        if self.config.experiment_mode == 'dp':
            print("Running in Differential Privacy mode.")
            print(f"DP Params: {self.config.dp_params}")
        else:
            print("Running in Normal mode.")

        # 设置随机种子
        set_seed(self.config.general['seed'])
        print(f"Running in {self.config.experiment_mode.upper()} mode with seed {self.config.general['seed']}")

        # 加载数据
        speaker_fold_df, full_df = self.load_data()

        # 交叉验证
        all_folds = range(1, self.config.general['num_folds'] + 1)
        all_fold_best_metrics = defaultdict(list)

        for fold_idx, val_fold_num in enumerate(all_folds):
            # 划分训练和验证集
            train_df, val_df = split_data_by_folds(speaker_fold_df, full_df, val_fold_num)

            # 训练单个fold
            target_epoch_metrics = self.train_single_fold(fold_idx, val_fold_num, train_df, val_df)

            # 收集结果
            all_fold_best_metrics['loss'].append(target_epoch_metrics['loss'])
            all_fold_best_metrics['accuracy'].append(target_epoch_metrics['acc'])
            all_fold_best_metrics['precision'].append(target_epoch_metrics['p'])
            all_fold_best_metrics['recall'].append(target_epoch_metrics['r'])
            all_fold_best_metrics['f1'].append(target_epoch_metrics['f1'])
            all_fold_best_metrics['auc'].append(target_epoch_metrics['auc'])

        # 计算平均结果和标准差
        print(f"\n--- Average BEST Metrics Across All {self.config.general['num_folds']} Folds ---")
        avg_metrics = {}
        for metric_name, values in all_fold_best_metrics.items():
            avg_value = sum(values) / len(values)
            std_value = (sum((x - avg_value) ** 2 for x in values) / len(values)) ** 0.5
            avg_metrics[metric_name] = avg_value
            print(f"  Average BEST {metric_name.capitalize()}: {avg_value:.4f} ± {std_value:.4f}")

        return avg_metrics