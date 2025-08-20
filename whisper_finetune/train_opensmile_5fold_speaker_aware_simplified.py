#!/usr/bin/env python3
"""
Simplified speaker-aware 5-fold cross-validation training script.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whisper_finetune.configs import config
from data_loaders.data_loader_opensmile import OpenSMILEWhisperDataset
from models_def.model_opensmile import OpenSMILEWhisperModel # Import the corrected model
from transformers import WhisperTokenizer

class SimplifiedCrossValidationDataModule(pl.LightningDataModule):
    """Simplified DataModule for cross-validation."""
    
    def __init__(self, train_df, val_df, feature_dir, scaler, batch_size=4, num_workers=4):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.feature_dir = feature_dir
        self.scaler = scaler
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.tokenizer = WhisperTokenizer.from_pretrained(
            config.MODEL_NAME,
            language=config.LANGUAGE,
            task=config.TASK
        )
    
    def setup(self, stage=None):
        """Setup datasets."""
        self.train_dataset = OpenSMILEWhisperDataset(
            df=self.train_df,
            tokenizer=self.tokenizer,
            feature_dir=self.feature_dir,
            shared_scaler=self.scaler
        )
        
        self.val_dataset = OpenSMILEWhisperDataset(
            df=self.val_df,
            tokenizer=self.tokenizer,
            feature_dir=self.feature_dir,
            shared_scaler=self.scaler
        )
    
    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def _collate_fn(self, batch):
        """Custom collate function to pad sequences."""
        features = [item["input_features"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad features
        max_feature_len = max(f.shape[0] for f in features)
        padded_features = [
            torch.nn.functional.pad(f, (0, 0, 0, max_feature_len - f.shape[0])) for f in features
        ]
        features_tensor = torch.stack(padded_features)
        
        # Pad labels
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return {
            'input_features': features_tensor,
            'labels': padded_labels
        }

class SpeakerAwareCrossValidationTrainer:
    """Trainer for speaker-aware cross-validation."""
    
    def __init__(self):
        # Configuration
        self.model_name = config.MODEL_NAME
        self.feature_dim = config.OPENSMILE_FEATURE_DIM
        self.learning_rate = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        self.max_epochs = config.NUM_EPOCHS
        self.num_workers = 4
        
        self.global_scaler = StandardScaler()
        self.cv_results = []
        
        # Paths
        self.speaker_folds_path = "/home/siyuan/Opensmile_project/data/speaker_folds.csv"
        self.dataset_path = "/home/siyuan/Opensmile_project/whisper_finetune/train_dataset_dim32.csv"
        self.feature_dir = "/home/siyuan/Opensmile_project/data/extracted_features_train"
        
        # Output directories
        self.output_dir = "/home/siyuan/Opensmile_project/whisper_finetune/outputs"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints_speaker_aware")
        self.log_dir = os.path.join(self.output_dir, "logs_speaker_aware")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def load_speaker_folds(self):
        """Loads pre-defined speaker folds."""
        if not os.path.exists(self.speaker_folds_path):
            raise FileNotFoundError(f"Speaker folds file not found: {self.speaker_folds_path}")
        
        speaker_folds_df = pd.read_csv(self.speaker_folds_path)
        speaker_folds_df['speaker_id'] = speaker_folds_df['speaker_id'].str.strip("'\"")
        
        print(f"✓ Speaker folds loaded: {len(speaker_folds_df)} speakers.")
        fold_counts = speaker_folds_df['fold'].value_counts().sort_index()
        print("Speaker distribution per fold:")
        for fold, count in fold_counts.items():
            print(f"  Fold {fold}: {count} speakers")
        
        return speaker_folds_df
    
    def extract_speaker_id(self, audio_path):
        """Extracts speaker ID from audio path."""
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        parts = base_name.split('_')
        return '_'.join(parts[:3]) if len(parts) >= 3 else None
    
    def load_and_filter_data(self):
        """Loads and filters the main dataset."""
        print("\n" + "="*60 + "\nLoading and Filtering Data\n" + "="*60)
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        print(f"Original dataset size: {len(df)} samples")
        
        speaker_folds_df = self.load_speaker_folds()
        valid_speakers = set(speaker_folds_df['speaker_id'])
        
        df['speaker_id'] = df['audio_path'].apply(self.extract_speaker_id)
        df = df[df['speaker_id'].isin(valid_speakers)].copy()
        
        valid_indices = []
        for idx, row in df.iterrows():
            feature_path = os.path.join(self.feature_dir, f"{os.path.splitext(os.path.basename(row['audio_path']))[0]}.csv")
            if os.path.exists(feature_path):
                valid_indices.append(idx)
        
        valid_df = df.loc[valid_indices].reset_index(drop=True)
        print(f"✓ Valid data after filtering: {len(valid_df)} samples")
        
        return valid_df, speaker_folds_df
    
    def fit_global_scaler(self, valid_df):
        """Fits a global StandardScaler on a subset of the training data."""
        print("\nFitting global StandardScaler...")
        
        # Use a fraction of the data to fit the scaler to save time/memory
        sample_df = valid_df.sample(n=min(len(valid_df), 5000), random_state=42)
        
        feature_paths = [os.path.join(self.feature_dir, f"{os.path.splitext(os.path.basename(p))[0]}.csv") for p in sample_df['audio_path']]
        
        # Read features in chunks to manage memory
        all_features = np.vstack([pd.read_csv(p).values for p in feature_paths])
        
        self.global_scaler.fit(all_features)
        print(f"✓ Global scaler fitted on {len(all_features)} feature vectors.")
    
    def create_fold_data(self, valid_df, speaker_folds_df, fold):
        """Creates train/validation splits for a specific fold."""
        val_speakers = set(speaker_folds_df[speaker_folds_df['fold'] == fold]['speaker_id'])
        train_speakers = set(speaker_folds_df[speaker_folds_df['fold'] != fold]['speaker_id'])
        
        train_data = valid_df[valid_df['speaker_id'].isin(train_speakers)].reset_index(drop=True)
        val_data = valid_df[valid_df['speaker_id'].isin(val_speakers)].reset_index(drop=True)
        
        print(f"\nFold {fold} Data Split:")
        print(f"  Training samples: {len(train_data)} from {len(train_speakers)} speakers")
        print(f"  Validation samples: {len(val_data)} from {len(val_speakers)} speakers")
        
        assert train_speakers.isdisjoint(val_speakers), "Speaker overlap detected between train and val sets!"
        print("✓ Speaker separation confirmed.")
        
        return train_data, val_data
    
    def train_fold(self, fold, train_data, val_data):
        """Trains a single cross-validation fold."""
        print(f"\n{'='*60}\nTraining Fold {fold}\n{'='*60}")
        
        # Save the scaler for this fold
        scaler_path = os.path.join(self.checkpoint_dir, f"scaler_fold_{fold}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.global_scaler, f)
        
        data_module = SimplifiedCrossValidationDataModule(
            train_df=train_data, val_df=val_data, feature_dir=self.feature_dir,
            scaler=self.global_scaler, batch_size=self.batch_size, num_workers=self.num_workers
        )
        
        model = OpenSMILEWhisperModel(
            model_name=self.model_name, opensmile_dim=self.feature_dim, learning_rate=self.learning_rate
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.checkpoint_dir, f"fold_{fold}"),
            filename='best-{epoch:02d}-{val/wer:.3f}',
            monitor='val/wer', mode='min', save_top_k=1, save_last=True
        )
        
        early_stop_callback = EarlyStopping(monitor='val/wer', mode='min', patience=5, verbose=True)
        
        logger = TensorBoardLogger(save_dir=self.log_dir, name=f"fold_{fold}", version=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision="16-mixed",
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2
        )
        
        print(f"Starting training for fold {fold}...")
        trainer.fit(model, data_module)
        
        # Check if a checkpoint was saved before accessing best_model_score
        if checkpoint_callback.best_model_score:
            best_wer = checkpoint_callback.best_model_score.item()
        else:
            print("Warning: No checkpoint was saved. Using NaN for best WER.")
            best_wer = float('nan')

        
        fold_result = {
            'fold': fold, 'best_val_wer': best_wer,
            'train_samples': len(train_data), 'val_samples': len(val_data),
            'best_model_path': checkpoint_callback.best_model_path, 'scaler_path': scaler_path
        }
        
        print(f"✓ Fold {fold} training complete. Best Validation WER: {best_wer:.4f}")
        return fold_result
    
    def run_cross_validation(self):
        """Runs the full cross-validation pipeline."""
        print("\n" + "="*80 + "\nStarting Speaker-Aware 5-Fold Cross-Validation\n" + "="*80)
        
        valid_df, speaker_folds_df = self.load_and_filter_data()
        self.fit_global_scaler(valid_df)
        
        for fold in range(1, 6):
            try:
                train_data, val_data = self.create_fold_data(valid_df, speaker_folds_df, fold)
                fold_result = self.train_fold(fold, train_data, val_data)
                self.cv_results.append(fold_result)
            except Exception as e:
                print(f"❌ Fold {fold} training failed: {e}")
                import traceback
                traceback.print_exc()
        
        self.save_and_display_final_results()
    
    def save_and_display_final_results(self):
        """Saves and displays the final CV results."""
        if not self.cv_results:
            print("❌ No cross-validation results to display.")
            return
            
        results_path = os.path.join(self.output_dir, "cv_results_speaker_aware.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.cv_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Cross-validation results saved to: {results_path}")
        
        print("\n" + "="*80 + "\nFinal Cross-Validation Results\n" + "="*80)
        
        val_wers = [r['best_val_wer'] for r in self.cv_results if pd.notna(r['best_val_wer'])]
        
        print("Per-fold Validation WER:")
        for r in self.cv_results:
            print(f"  Fold {r['fold']}: {r['best_val_wer']:.4f} (Train: {r['train_samples']}, Val: {r['val_samples']})")
        
        if val_wers:
            print("\nCV Statistics:")
            print(f"  Average Validation WER: {np.mean(val_wers):.4f} ± {np.std(val_wers):.4f}")
            print(f"  Best Validation WER:  {np.min(val_wers):.4f}")
            
            best_fold = self.cv_results[np.argmin(val_wers)]
            print("\nBest Overall Model:")
            print(f"  Fold: {best_fold['fold']}")
            print(f"  Validation WER: {best_fold['best_val_wer']:.4f}")
            print(f"  Model Path: {best_fold['best_model_path']}")

def main():
    """Main execution function."""
    print("Initializing 5-Fold Cross-Validation Training for OpenSMILE-Whisper")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ CUDA not available, training will use CPU.")
    
    trainer = SpeakerAwareCrossValidationTrainer()
    trainer.run_cross_validation()

if __name__ == "__main__":
    main()
