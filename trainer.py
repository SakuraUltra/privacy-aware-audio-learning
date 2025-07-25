import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import warnings


class Trainer:
    def __init__(self, config, model, optimizer, criterion, device, experiment_mode, train_loader_len):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.experiment_mode = experiment_mode
        self.privacy_engine = None
        
        # if DP mode, initialize PrivacyEngine properly
        if self.experiment_mode == 'DP':
            # 抑制 Opacus 的运行时警告
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='opacus')
            
            try:
                self.privacy_engine = PrivacyEngine()
                print("Successfully enabled DP model with PrivacyEngine"+(str(self.privacy_engine is not None)))
                # 正确地初始化PrivacyEngine，需要传入训练数据加载器
                # 注意：这里我们不能在__init__中直接绑定dataloader，需要在train_epoch中处理
            except Exception:
                # 如果PrivacyEngine初始化失败，关闭DP模式
                self.experiment_mode = 'Normal'
                self.privacy_engine = None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        # 如果是DP模式，需要正确设置PrivacyEngine
        if self.experiment_mode == 'DP' and self.privacy_engine is not None:
            try:
                # 确保模型、优化器和数据加载器都绑定到PrivacyEngine
                if not hasattr(self.privacy_engine, '_module_registry') or len(self.privacy_engine._module_registry) == 0:
                    self.model, self.optimizer, train_loader = self.privacy_engine.make_private_with_epsilon(
                        module=self.model,
                        optimizer=self.optimizer,
                        data_loader=train_loader,
                        criterion=self.criterion,
                        target_epsilon=self.config.DP_PARAMS['target_epsilon'],
                        target_delta=self.config.DP_PARAMS['target_delta'],
                        max_grad_norm=self.config.DP_PARAMS['max_grad_norm'],
                        epochs=self.config.GENERAL['epochs']
                    )
            except Exception as e:
                print(e)
                # 如果DP设置失败，回退到正常模式
                self.experiment_mode = 'Normal'
                self.privacy_engine = None
                raise RuntimeError
        
        # 使用Opacus的BatchMemoryManager来处理可能的OOM问题
        if self.experiment_mode == 'DP' and self.config.DATA['batch_size'] > 256: # 示例阈值
             with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=256,
                optimizer=self.optimizer
            ) as memory_safe_loader:
                iterable_loader = memory_safe_loader
        else:
            iterable_loader = train_loader

        for x, y, mask, _ in iterable_loader:
            # 自动处理NaN值
            x = torch.nan_to_num(x, nan=0.0)
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(x, padding_mask=mask)
            loss = self.criterion(logits, y)
            loss.backward()
            # clip gradients with config.DP_PARAMS['max_grad_norm'] for DP and Normal modes
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.DP_PARAMS['max_grad_norm'])
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # 如果是DP模式，获取隐私预算消耗（安全地）
        if self.experiment_mode == 'DP' and self.privacy_engine is not None:
            try:
                epsilon = self.privacy_engine.get_epsilon(self.config.DP_PARAMS['target_delta'])
                print("Get current epsilon", epsilon)
            except Exception as e:
                # 忽略隐私预算计算错误
                input(e)
                pass
            
        return avg_loss

    def evaluate(self, val_loader):
        self.model.eval()
        speaker_preds = defaultdict(list)
        speaker_labels = {}
        with torch.no_grad():
            for x, y, mask, speaker_ids in val_loader:
                x = torch.nan_to_num(x, nan=0.0)
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                logits, _ = self.model(x, padding_mask=mask)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                labels = y.cpu().tolist()

                for spk, pred, true in zip(speaker_ids, preds, labels):
                    speaker_preds[spk].append(pred)
                    speaker_labels[spk] = true

        final_preds, final_labels = [], []
        for sid, pred_list in speaker_preds.items():
            if pred_list:
                majority_pred = Counter(pred_list).most_common(1)[0][0]
                if sid in speaker_labels:
                    final_preds.append(majority_pred)
                    final_labels.append(speaker_labels[sid])

        if not final_preds:
            return {'acc': 0.0, 'p': 0.0, 'r': 0.0, 'f1': 0.0}

        acc = accuracy_score(final_labels, final_preds)
        p, r, f1, _ = precision_recall_fscore_support(final_labels, final_preds, average='binary', zero_division=0)
        
        return {
            'acc': float(acc),
            'p': float(p),
            'r': float(r),
            'f1': float(f1)
        }
