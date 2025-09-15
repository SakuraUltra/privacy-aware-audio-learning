"""
MINE隐私保护训练器模块
实现两阶段交替训练：MINE模型更新阶段和主模型更新阶段
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import pandas as pd
from torch.utils.data import DataLoader
from models.mine_model import TimeAwareMINE, create_timeaware_mine
from models.transformer import TransformerClassifier
from trainers.base_trainer import BaseExperimentTrainer
from trainers.core_trainer import CoreTrainer
from tqdm import tqdm
from opacus import PrivacyEngine
import logging
import os


class MINEPrivacyTrainer(BaseExperimentTrainer):
    """MINE隐私保护训练器 - 实现两阶段交替训练"""
    
    def __init__(self, config):
        super().__init__(config)
        self.main_model = None  # 主模型（TransformerClassifier）
        self.mine_model = None  # MINE模型
        self.main_optimizer = None  # 主模型优化器
        self.privacy_engine = None  # DP隐私引擎
        self.criterion = nn.CrossEntropyLoss()  # 效用损失函数
        self.best_mi = float('-inf')  # 最佳互信息估计
        self.best_f1 = float('-inf')  # 最佳F1分数
        self.current_epoch = -1  # 当前训练的轮数
        self.mine_model_type = None  # MINE模型类型
    
    def create_model(self) -> Tuple[nn.Module, nn.Module]:
        """创建主模型和MINE模型"""
        # 1. 创建主模型（Transformer分类器）
        dp_mode = (self.config.experiment_mode == 'dp')
        
        # 根据特征类型获取正确的d_model
        d_model = self.config.get_model_input_dim()
        
        main_model = TransformerClassifier(
            d_model=d_model,
            nhead=8 if d_model % 8 == 0 else 4,  # 根据d_model调整注意力头数
            num_layers=4,
            num_classes=self.config.model['num_classes'],
            dim_feedforward=256,
            dropout=0.3,
            dp_mode=dp_mode
        ).to(self.config.general['device'])
        
        # 2. 创建MINE模型
        mine_config = self.config.get_mine_model_config()
        
        if mine_config['mine_model_type'] == 'timeaware':
            self.mine_model_type = 'timeaware'
            from models.mine_model import create_timeaware_mine
            mine_model = create_timeaware_mine(
                input_dim=mine_config['input_dim'],
                mlp_hidden=mine_config['hidden_dim']
            ).to(self.config.general['device'])
        elif mine_config['mine_model_type'] == 'standard':
            self.mine_model_type = 'standard'
            from models.standard_mine_model import create_standard_mine
            mine_model = create_standard_mine(
                input_dim=mine_config['standard_input_dim'],
                mlp_hidden=mine_config['hidden_dim']
            ).to(self.config.general['device'])
        else:
            raise ValueError(f"Unknown MINE model type: {mine_config['mine_model_type']}")
        
        return main_model, mine_model
    
    def create_optimizers(self, main_model: nn.Module, train_loader: DataLoader) -> Tuple[optim.Optimizer, Optional[PrivacyEngine]]:
        """创建优化器和隐私引擎"""
        # 1. 创建主模型优化器
        main_optimizer = optim.Adam(
            main_model.parameters(),
            lr=self.config.training['lr'],
            weight_decay=self.config.training['weight_decay']
        )
        
        # 2. 如果是DP模式，添加差分隐私保护
        privacy_engine = None
        if self.config.experiment_mode == 'dp':
            privacy_engine = PrivacyEngine()
            main_model, main_optimizer, _ = privacy_engine.make_private_with_epsilon(
                module=main_model,
                optimizer=main_optimizer,
                data_loader=train_loader,
                epochs=self.config.general['epochs'],
                target_epsilon=self.config.DP_PARAMS['target_epsilon'],
                target_delta=self.config.DP_PARAMS['target_delta'],
                max_grad_norm=self.config.DP_PARAMS['max_grad_norm']
            )
        
        return main_optimizer, privacy_engine
    
    def mine_update_step(self, mel_spec: torch.Tensor, texts: list) -> Dict[str, float]:
        """MINE更新阶段：仅更新MINE模型参数
        
        Args:
            mel_spec: MEL频谱图输入
            texts: 文本输入列表（批次）
            
        Returns:
            包含互信息估计等训练统计信息的字典
        """
        # 1. 冻结主模型参数
        for param in self.main_model.parameters():
            param.requires_grad = False
        self.main_model.eval()
        
        # 2. 解冻MINE模型参数
        for param in self.mine_model.parameters():
            param.requires_grad = True
        self.mine_model.train()
        
        # 3. 使用主模型提取特征（无需梯度）
        with torch.no_grad():
            mel_spec = mel_spec.to(self.config.general['device'])
            if len(mel_spec.shape) == 2:
                mel_spec = mel_spec.unsqueeze(0)  # 只添加批次维度
            
            # 直接使用encoder获取时序特征
            transformer_output = self.main_model.encoder(mel_spec)
            # 转置为 MINE 期望的格式 [batch, seq_len, features] -> [batch, features, seq_len]
            transformer_output = transformer_output.transpose(1, 2)
        
        # 4. MINE前向传播（使用encoded模式）
        # 为了匹配批次大小，我们将代表性文本重复以匹配音频批次大小
        batch_size = transformer_output.shape[0]
        
        if isinstance(texts, (list, tuple)) and len(texts) > 0:
            representative_text = texts[0]
        elif isinstance(texts, str):
            representative_text = texts
        else:
            # 如果texts不是预期的格式，使用一个默认文本
            representative_text = "This is audio content for privacy analysis."
        
        # 将单个文本重复为批次大小，这样MINE可以为每个音频样本使用相同的文本参考
        batch_texts = [representative_text] * batch_size
            
        outputs = self.mine_model(transformer_output, batch_texts, mode='raw')
        h = outputs['h']  # 获取联合表示
        
        # 5. 计算和更新互信息估计
        stats = self.mine_model.update_estimator(h, batch_size=self.config.data['batch_size'])
        
        return {
            'mi_estimate': stats['mi_estimate']
        }
    
    def main_model_update_step(self, mel_spec: torch.Tensor, texts: list, labels: torch.Tensor) -> Dict[str, float]:
        """主模型更新阶段：更新主模型（编码器φ、预测器θ）参数，固定MINE网络参数ψ
        
        通过DP-SGD实现"隐私-效用"的双目标优化：
        1. 效用损失L_Utility：最大化I(Z;Y)，保证心理健康评估能力
        2. 隐私损失L_Privacy：最小化I(Z;S)，消减语音中的语义信息
        
        Args:
            mel_spec (torch.Tensor): MEL频谱图输入 [B, seq_len=1000, features=80]
            texts (list): 文本输入列表（用于MINE计算互信息）
            labels (torch.Tensor): 心理健康标签 [B] 或 [B, 1]
            
        Returns:
            Dict[str, float]: 包含各损失值的字典
        """
        # 1. 数据准备
        mel_spec = mel_spec.to(self.config.general['device'])
        labels = labels.to(self.config.general['device'])
        batch_size = mel_spec.shape[0]
        
        # 2. 模型状态设置
        self.main_model.train()  # 启用训练模式
        # 确保主模型的参数可以更新
        for param in self.main_model.parameters():
            param.requires_grad = True
            
        self.mine_model.eval()   # 固定MINE参数
        for param in self.mine_model.parameters():
            param.requires_grad = False
        
        # 3. 使用encoder获取时序特征Z
        self.main_optimizer.zero_grad()
        transformer_output = self.main_model.encoder(mel_spec)  # 直接使用encoder
        # 转置为 MINE 期望的格式 [batch, seq_len, features] -> [batch, features, seq_len]
        mine_input = transformer_output.transpose(1, 2)
        # 为了获得分类结果，需要池化后通过分类器（对序列维度取平均）
        pooled_output = torch.mean(transformer_output, dim=1)  # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        logits = self.main_model.classifier(pooled_output)
        
        # 4. 计算效用损失L_Utility（最大化I(Z;Y)）
        if self.config.mine_params['task_type'] == 'classification':
            # 分类任务：使用交叉熵
            utility_loss = self.criterion(logits, labels)  # -E[log P_theta(Y|Z)]
        else:
            # 回归任务：使用MSE
            utility_loss = F.mse_loss(logits.squeeze(), labels.float())
            
        # 5. 计算隐私损失L_Privacy（最小化I(Z;S)）
        # 5.1 MINE前向传播
        # 为了匹配批次大小，将代表性文本重复以匹配音频批次大小
        if isinstance(texts, (list, tuple)) and len(texts) > 0:
            representative_text = texts[0]
        elif isinstance(texts, str):
            representative_text = texts
        else:
            representative_text = "This is audio content for privacy analysis."
        
        batch_texts = [representative_text] * batch_size
        mine_outputs = self.mine_model(mine_input, batch_texts, mode='raw')  # 使用转置后的特征
        h = mine_outputs['h']
        
        # 5.2 计算互信息估计I(Z;S)
        h_marginal = self.mine_model.create_marginal_samples(
            h, batch_size=self.config.data['batch_size']
        )
        privacy_loss = self.mine_model.compute_mutual_information(
            h_joint=h, h_marginal=h_marginal
        )
        
        # 5.3 批量平均
        privacy_loss = privacy_loss.mean()  # E[I(Z;S)]
        
        # 6. 总损失计算
        gamma = self.config.mine_params['privacy_weight']  # 隐私权重γ
        total_loss = utility_loss + gamma * privacy_loss
        
        # 7. DP-SGD优化
        total_loss.backward()  # 反向传播
        
        if self.config.experiment_mode == 'dp' and self.privacy_engine is not None:
            # PrivacyEngine自动处理:
            # 1. Per-sample梯度计算
            # 2. 梯度裁剪
            # 3. 噪声添加
            # 4. 隐私预算追踪
            epsilon, delta = self.privacy_engine.get_privacy_spent()
            logging.info(f"Privacy budget spent: (ε={epsilon:.2f}, δ={delta:.2e})")

        # 参数更新（PrivacyEngine会自动应用DP-SGD）
        self.main_optimizer.step()
        
        return {
            'utility_loss': utility_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'total_loss': total_loss.item(),
            'logits': logits.detach(),  # 用于评估
            'batch_size': batch_size
        }
    
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch - 实现两阶段交替训练
        
        对每个batch:
        1. 先执行MINE更新，估计互信息
        2. 再执行主模型更新，使用估计的互信息作为正则项
        
        Args:
            epoch: 当前epoch索引
            train_loader: 训练数据加载器
            
        Returns:
            包含训练统计信息的字典
        """
        total_utility_loss = 0.0
        total_privacy_loss = 0.0
        total_loss = 0.0
        total_mi = 0.0
        n_samples = 0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.general["epochs"]}')
        
        for batch_idx, (mel_specs, labels, _, speaker_ids, texts) in enumerate(progress_bar):
            batch_size = len(mel_specs)
            n_samples += batch_size
            
            batch_stats = {}
            
            # 阶段1：MINE更新 - 整批处理
            mine_stats = self.mine_update_step(mel_specs, texts)  # 使用整个批次的MEL特征和文本
            batch_stats['mi'] = mine_stats['mi_estimate']
            
            # 阶段2：主模型更新 - 整批处理
            main_stats = self.main_model_update_step(mel_specs, texts, labels)  # 使用整个批次的文本
            batch_stats.update(main_stats)
            
            # 累积统计信息
            total_utility_loss += batch_stats['utility_loss'] * batch_size
            total_privacy_loss += batch_stats['privacy_loss'] * batch_size
            total_loss += batch_stats['total_loss'] * batch_size
            total_mi += batch_stats['mi'] * batch_size
            
            # 更新进度条
            progress_bar.set_postfix({
                'utility_loss': f"{batch_stats['utility_loss']:.4f}",
                'privacy_loss': f"{batch_stats['privacy_loss']:.4f}",
                'total_loss': f"{batch_stats['total_loss']:.4f}",
                'mi': f"{batch_stats['mi']:.4f}"
            })
            
            # 定期记录日志（仅记录主要指标）
            if batch_idx % self.config.training['logging_steps'] == 0:
                logging.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}: Loss={batch_stats['total_loss']:.4f}"
                )
        
        # 计算平均值
        return {
            'avg_utility_loss': total_utility_loss / n_samples,
            'avg_privacy_loss': total_privacy_loss / n_samples,
            'avg_total_loss': total_loss / n_samples,
            'avg_mi': total_mi / n_samples
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """评估模型性能
        
        评估内容：
        1. 主模型的效用指标（准确率/MSE）
        2. MINE估计的互信息
        3. 总体隐私-效用平衡
        """
        self.main_model.eval()
        self.mine_model.eval()
        
        total_utility_loss = 0.0
        total_privacy_loss = 0.0
        total_loss = 0.0
        total_mi = 0.0
        n_samples = 0
        
        # 分类任务的指标
        total_correct = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for mel_specs, labels, _, speaker_ids, texts in val_loader:
                batch_size = len(mel_specs)
                n_samples += batch_size
                
                # 1. 主模型预测
                mel_specs = mel_specs.to(self.config.general['device'])
                transformer_output = self.main_model.encoder(mel_specs)
                # 为 MINE 转置特征
                mine_input = transformer_output.transpose(1, 2)
                # 池化原始输出用于分类（对序列维度取平均）
                pooled_output = torch.mean(transformer_output, dim=1)  # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
                logits = self.main_model.classifier(pooled_output)
                labels = labels.to(self.config.general['device'])
                
                # 2. 计算效用损失
                if self.config.mine_params['task_type'] == 'classification':
                    utility_loss = self.criterion(logits, labels)
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    utility_loss = F.mse_loss(logits.squeeze(), labels.float())
                
                # 3. 计算隐私损失（使用MINE）
                # 为了匹配批次大小，将代表性文本重复以匹配音频批次大小
                if isinstance(texts, (list, tuple)) and len(texts) > 0:
                    representative_text = texts[0]
                elif isinstance(texts, str):
                    representative_text = texts
                else:
                    representative_text = "This is audio content for privacy analysis."
                
                batch_texts = [representative_text] * batch_size
                mine_outputs = self.mine_model(mine_input, batch_texts, mode='raw')
                h = mine_outputs['h']
                h_marginal = self.mine_model.create_marginal_samples(
                    h, batch_size=self.config.data['batch_size']
                )
                privacy_loss = self.mine_model.compute_mutual_information(
                    h_joint=h, h_marginal=h_marginal
                ).mean()
                
                # 4. 计算总损失
                gamma = self.config.mine_params['privacy_weight']
                total_loss += (utility_loss + gamma * privacy_loss).item() * batch_size
                total_utility_loss += utility_loss.item() * batch_size
                total_privacy_loss += privacy_loss.item() * batch_size
                total_mi += privacy_loss.item() * batch_size
        
        # 计算平均值
        metrics = {
            'utility_loss': total_utility_loss / n_samples,
            'privacy_loss': total_privacy_loss / n_samples,
            'loss': total_loss / n_samples,  # 这个是 base_trainer 期望的键名
            'mi': total_mi / n_samples,
            'acc': 0.0,  # 默认值
            'p': 0.0,    # 默认值
            'r': 0.0,    # 默认值
            'f1': 0.0,   # 默认值
            'auc': 0.0   # 默认值
        }
        
        # 添加分类指标
        if self.config.mine_params['task_type'] == 'classification':
            from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
            metrics['acc'] = total_correct / n_samples
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
            metrics['p'] = precision
            metrics['r'] = recall
            metrics['f1'] = f1
            try:
                metrics['auc'] = roc_auc_score(all_labels, all_preds)
            except:
                metrics['auc'] = 0.5  # 如果计算AUC失败，使用0.5作为默认值
        
        # 无需额外的checkpoint保存逻辑，统一在训练结束时保存
            
        return metrics
    
    def train_single_fold(self, fold_idx: int, val_fold_num: int,
                         train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
        """训练单个fold
        
        完整的训练流程:
        1. 创建数据集和加载器
        2. 创建/重置模型和优化器
        3. 执行训练循环
        4. 保存最终模型和指标
        
        Args:
            fold_idx: 当前fold的索引
            val_fold_num: 验证fold的编号
            train_df: 训练数据DataFrame
            val_df: 验证数据DataFrame
            
        Returns:
            Dict[str, float]: 最终的验证指标
        """
        print(f"\n--- Starting Fold {fold_idx + 1}/{self.config.general['num_folds']} (Validation Fold: {val_fold_num}) ---")
        
        # 1. 创建数据集和加载器
        train_set, val_set = self.create_datasets(train_df, val_df)
        train_loader, val_loader = self.create_data_loaders(train_set, val_set)
        
        # 2. 为新的fold创建新的模型和优化器
        # 每个fold都需要从头开始训练
        self.main_model, self.mine_model = self.create_model()  # 总是创建新模型
        self.main_optimizer, self.privacy_engine = self.create_optimizers(
            self.main_model, train_loader
        )
        self.current_epoch = -1  # 重置epoch计数
        
        # 4. 训练循环
        # 记录所有epoch的指标
        all_metrics = {
            'train_utility': [], 'train_privacy': [], 'train_mi': [],
            'val_utility': [], 'val_privacy': [], 'val_mi': [],
            'val_acc': [], 'val_auc': [], 'val_f1': []
        }
        
        for epoch in range(self.current_epoch + 1, self.config.general['epochs']):
            self.current_epoch = epoch
            
            # 4.1 训练一个epoch
            train_stats = self.train_epoch(epoch, train_loader)
            
            # 4.2 验证
            val_metrics = self.evaluate(val_loader)
            
            # 4.3 记录指标
            all_metrics['train_utility'].append(train_stats['avg_utility_loss'])
            all_metrics['train_privacy'].append(train_stats['avg_privacy_loss'])
            all_metrics['train_mi'].append(train_stats['avg_mi'])
            all_metrics['val_utility'].append(val_metrics['utility_loss'])
            all_metrics['val_privacy'].append(val_metrics['privacy_loss'])
            all_metrics['val_mi'].append(val_metrics['mi'])
            all_metrics['val_acc'].append(val_metrics['acc'])
            all_metrics['val_auc'].append(val_metrics['auc'])
            all_metrics['val_f1'].append(val_metrics['f1'])
            
            # 4.4 打印当前epoch指标
            print(f"\nEpoch {epoch + 1}/{self.config.general['epochs']}:")
            print(f"  Train - Utility: {train_stats['avg_utility_loss']:.4f}, "
                  f"Privacy: {train_stats['avg_privacy_loss']:.4f}, "
                  f"MI: {train_stats['avg_mi']:.4f}")
            print(f"  Val   - Utility: {val_metrics['utility_loss']:.4f}, "
                  f"Privacy: {val_metrics['privacy_loss']:.4f}, "
                  f"MI: {val_metrics['mi']:.4f}")
            
            if self.config.mine_params['task_type'] == 'classification':
                print(f"  Val Metrics - Acc: {val_metrics['acc']:.4f}, "
                      f"AUC: {val_metrics['auc']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}")
            
            # 4.5 保存最后一个epoch的模型（与其他trainer保持一致）
            if epoch == self.config.general['epochs'] - 1:
                model_name = f'checkpoints/last_epoch_model_{self.config.feature_type}_{self.mine_model_type}_fold_{val_fold_num}.pth'
                os.makedirs('checkpoints', exist_ok=True)
                
                # 保存主模型的encoder部分（用于AIA攻击的表征提取）
                # 只保存encoder，避免复杂对象导致的加载问题
                if hasattr(self.main_model, 'encoder'):
                    # 如果模型有encoder属性，只保存encoder
                    torch.save(self.main_model.encoder.state_dict(), model_name)
                    print(f"Saved encoder to: {model_name}")
                else:
                    # 如果没有明确的encoder属性，保存整个模型（但只保存state_dict）
                    torch.save(self.main_model.state_dict(), model_name)
                    print(f"Saved full model state_dict to: {model_name}")
        
        # 返回最后一个epoch的验证指标，保持与原接口兼容
        final_metrics = {
            'loss': val_metrics['loss'],
            'utility_loss': val_metrics['utility_loss'],
            'privacy_loss': val_metrics['privacy_loss'],
            'mi': val_metrics['mi'],
            'acc': val_metrics['acc'],
            'p': val_metrics['p'],    # 添加precision
            'r': val_metrics['r'],    # 添加recall
            'auc': val_metrics['auc'],
            'f1': val_metrics['f1']
        }
        
        return final_metrics
