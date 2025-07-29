import torch
import numpy as np
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
        self.criterion = criterion
        self.device = device
        self.experiment_mode = experiment_mode
        self.privacy_engine = None

        # --- MODIFIED: Simplified and corrected optimizer handling ---
        if self.experiment_mode == 'DP':
            # In DP mode, we expect a tuple of two optimizers
            assert isinstance(optimizer, tuple) and len(optimizer) == 2, "DP mode requires a tuple of two optimizers."
            self.optimizer_encoder, self.optimizer_classifier = optimizer
            self.privacy_engine = PrivacyEngine()
            print("DP mode enabled: PrivacyEngine will be attached to the ENCODER optimizer.")
            
            if not hasattr(self.privacy_engine, '_module_registry') or len(self.privacy_engine._module_registry) == 0:
                print("Attaching PrivacyEngine to Encoder...")
                self.model.encoder, self.optimizer_encoder, train_loader = self.privacy_engine.make_private_with_epsilon(
                    module=self.model.encoder,
                    optimizer=self.optimizer_encoder,
                    data_loader=train_loader,
                    target_epsilon=self.config.DP_PARAMS['target_epsilon'],
                    target_delta=self.config.DP_PARAMS['target_delta'],
                    max_grad_norm=self.config.DP_PARAMS['max_grad_norm'],
                    epochs=self.config.GENERAL['epochs']
                )
        else:
            # In Normal mode, we expect a single optimizer
            self.optimizer = optimizer


    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        iterable_loader = train_loader
        
        for batch in iterable_loader:
            # Simplified batch unpacking
            x, y, mask = batch[0], batch[1], batch[2]
            x = torch.nan_to_num(x, nan=0.0)
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

            if self.experiment_mode == 'DP':
                # 在DP模式下，只使用被PrivacyEngine包装的encoder optimizer
                self.optimizer_encoder.zero_grad()
                self.optimizer_classifier.zero_grad()
                
                transformer_output = self.model.encoder(x, padding_mask=mask)
                pooled_output = self.model.pool(transformer_output, padding_mask=mask)
                logits = self.model.classifier(pooled_output)
                logits = torch.nan_to_num(logits, nan=0.0)
                loss = self.criterion(logits, y)
                loss.backward()
                
                self.optimizer_classifier.step()
                self.optimizer_encoder.step()  # 只调用encoder optimizer的step

            else: # Normal Mode
                self.optimizer.zero_grad()
                logits, _ = self.model(x, padding_mask=mask)
                loss = self.criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Get privacy budget if DP mode
        if self.experiment_mode == 'DP' and self.privacy_engine is not None:
            try:
                epsilon = self.privacy_engine.get_epsilon(self.config.DP_PARAMS['target_delta'])
                print(f"Epsilon: {epsilon:.4f}")
            except Exception:
                pass

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
            for batch in val_loader:
                x, y, mask, speaker_ids = batch[0], batch[1], batch[2], batch[3]
                x = torch.nan_to_num(x, nan=0.0)
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                
                # The model's forward pass might return one or two items.
                # We only need the first one (utility logits) for evaluation.
                model_output = self.model(x, padding_mask=mask)
                logits = model_output[0] if isinstance(model_output, tuple) else model_output

                preds = torch.argmax(logits, dim=1).cpu().tolist()
                labels = y.cpu().tolist()
                
                if logits.shape[1] == 2:
                    pos_probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                else: # Handles single-class output case
                    pos_probs = torch.sigmoid(logits).cpu().tolist() if logits.ndim == 1 else [0.5] * len(labels)


                for i, (spk, pred, true, prob) in enumerate(zip(speaker_ids, preds, labels, pos_probs)):
                    speaker_preds[spk].append(pred)
                    speaker_labels[spk] = true
                    speaker_logits[spk].append(prob)

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
