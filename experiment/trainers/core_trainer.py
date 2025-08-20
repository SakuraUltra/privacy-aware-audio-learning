import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import warnings
from models.transformer import pool
from tqdm import tqdm
import sys

class CoreTrainer:
    """
    Core training logic for individual epochs and evaluation.
    This class handles the low-level training operations including:
    - Forward/backward passes
    - Differential privacy integration
    - Evaluation metrics computation
    """
    
    def __init__(self, config, model, optimizer, criterion, device, experiment_mode, train_loader_len, train_loader):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.device = device
        self.experiment_mode = experiment_mode
        self.privacy_engine = None

        # --- MODIFIED: Improved partial DP implementation ---
        if self.experiment_mode == 'DP':
            # In DP mode, we expect a tuple of two optimizers (back to partial DP)
            assert isinstance(optimizer, tuple) and len(optimizer) == 2, "DP mode requires a tuple of two optimizers."
            self.optimizer_encoder, self.optimizer_classifier = optimizer
            self.privacy_engine = PrivacyEngine()
            print("DP mode enabled: PrivacyEngine will be attached to the ENCODER only (partial DP).")
            
            print("Attaching PrivacyEngine to Encoder...")
            self.model.encoder, self.optimizer_encoder, _ = self.privacy_engine.make_private_with_epsilon(
                module=self.model.encoder,
                optimizer=self.optimizer_encoder,
                data_loader=train_loader,
                epochs=self.config.general['epochs'],
                target_epsilon=self.config.DP_PARAMS['target_epsilon'],
                target_delta=self.config.DP_PARAMS['target_delta'],
                max_grad_norm=self.config.DP_PARAMS['max_grad_norm']
            )
        else:
            # In Normal mode, we expect a single optimizer
            self.optimizer = optimizer


    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        # 创建进度条，使用stderr避免写入日志文件
        iterable_loader = tqdm(train_loader, desc="Training", leave=False, 
                              file=sys.stderr, dynamic_ncols=True,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch in iterable_loader:
            # Simplified batch unpacking
            x, y, mask = batch[0], batch[1], batch[2]
            x = torch.nan_to_num(x, nan=0.0)
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            if mask.sum().item()==0: 
                mask = None # 如果mask全为0，则设置为None，因为不需要mask



            if self.experiment_mode == 'DP':
                # 在DP模式下，使用分离的优化器
                self.optimizer_encoder.zero_grad()
                self.optimizer_classifier.zero_grad()

                # 使用完整的模型forward方法，但不使用eval_encoder
                logits, _ = self.model(x, padding_mask=mask, use_eval_encoder=False)
                loss = self.criterion(logits, y)
                loss.backward()

                self.optimizer_classifier.step()
                self.optimizer_encoder.step()

            else: # Normal Mode
                self.optimizer.zero_grad()
                logits, _ = self.model(x, padding_mask=mask)
                loss = self.criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            
            # 更新进度条显示当前loss
            current_batch = iterable_loader.n
            current_avg_loss = total_loss / current_batch if current_batch > 0 else 0.0
            iterable_loader.set_postfix(loss=f"{current_avg_loss:.4f}")

        iterable_loader.close()
        avg_loss = total_loss / len(train_loader)

        # Get privacy budget if DP mode
        if self.experiment_mode == 'DP' and self.privacy_engine is not None:
            try:
                epsilon = self.privacy_engine.get_epsilon(self.config.DP_PARAMS['target_delta'])
                print(f"Epsilon: {epsilon:.4f}")
            except Exception as e:
                print(f"Could not compute epsilon: {e}")

        return avg_loss

    def evaluate(self, val_loader):
        # This evaluation function is well-structured and can remain as is.
        # I have just tidied up the commented-out code for clarity.
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        speaker_preds = defaultdict(list)
        speaker_labels = {}
        speaker_logits = defaultdict(list)

        with torch.no_grad():
            eval_loader = tqdm(val_loader, desc="Evaluating", leave=False, 
                              file=sys.stderr, dynamic_ncols=True,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            for batch in eval_loader:
                x, y, mask, speaker_ids = batch[0], batch[1], batch[2], batch[3]
                x = torch.nan_to_num(x, nan=0.0)
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                if self.experiment_mode == 'DP':
                    model_output = self.model(x, padding_mask=mask, use_eval_encoder=True)
                    logits = model_output[0] if isinstance(model_output, tuple) else model_output
                else:
                    model_output = self.model(x, padding_mask=mask)
                    logits = model_output[0] if isinstance(model_output, tuple) else model_output

                
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
        except ValueError: # Happens if only one class is present in a batch
            auc = 0.5
            
        return {
            'acc': float(acc), 'p': float(p), 'r': float(r),
            'f1': float(f1), 'auc': float(auc), 'wer': 0.0
        }
