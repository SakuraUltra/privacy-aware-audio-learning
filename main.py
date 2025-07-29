import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from trainer import Trainer

# 抑制特定的运行时警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='opacus')
warnings.filterwarnings('ignore', category=UserWarning, module='opacus')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# 从新结构中导入模块
from utils.utils import set_seed
from data.dataset import OpenSMILEAudioDataset, collate_fn_with_padding
from models.transformer import TransformerClassifier

import argparse


parser = argparse.ArgumentParser(description="OpenSMILE Audio Classification Training")
parser.add_argument('--em', type=str, default='Normal',
                    help="Experiment mode: 'DP' for Differential Privacy, 'Normal' for standard training")
parser.add_argument('--epsilon', type=float, default=8.0,
                    help="Epsilon value for Differential Privacy")
parser.add_argument('--lambda_adv', type=float, default=0.5,
                    help="Lambda value for adversarial training")
args = parser.parse_args()


# logger
import sys
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

logfile = open(f"results/training_log_{args.em.lower()}{f'_eps{args.epsilon}' if args.em.lower()=='dp' else ''}.txt", "w")
sys.stdout = Tee(sys.stdout, logfile)

# Set exp mode
if args.em.lower() == "dp":
    import config_DP as config
    if args.epsilon:
        config.DP_PARAMS['target_epsilon'] = args.epsilon
elif args.em.lower() == "advasr":
    import config_AdvASR as config
    if args.lambda_adv is not None:
        config.ADVERSARIAL_PARAMS['lambda_adv'] = args.lambda_adv
else:
    import config

def main():
    # print config settings
    print(f"Experiment Mode: {config.EXPERIMENT_MODE}")
    print(f"General Config: {config.GENERAL}")
    print(f"Data Config: {config.DATA}")
    print(f"Model Config: {config.MODEL}")
    print(f"Training Config: {config.TRAINING}")
    if config.EXPERIMENT_MODE == 'DP':
        print("Running in Differential Privacy mode.")
        print(f"DP Params: {config.DP_PARAMS}")
    else:
        print("Running in Normal mode.")
    set_seed(config.GENERAL['seed'])
    print(f"Running in {config.EXPERIMENT_MODE} mode with seed {config.GENERAL['seed']}")

    try:
        speaker_fold_df = pd.read_csv(config.DATA['speaker_folds_csv'])
        speaker_fold_df['speaker_id'] = speaker_fold_df['speaker_id'].str.strip("'")
        full_df = pd.read_csv(config.DATA['all_expanded_features_csv'])
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the correct path.")
        return

    # 3. K-Fold
    all_folds = range(1, config.GENERAL['num_folds'] + 1)
    all_fold_best_metrics = defaultdict(list)

    for fold_idx, val_fold_num in enumerate(all_folds):
        print(f"\n--- Starting Fold {fold_idx + 1}/{config.GENERAL['num_folds']} (Validation Fold: {val_fold_num}) ---")

        # split traning and val
        val_speakers = speaker_fold_df[speaker_fold_df['fold'] == val_fold_num]['speaker_id'].tolist()
        train_folds = [f for f in all_folds if f != val_fold_num]
        train_speakers = speaker_fold_df[speaker_fold_df['fold'].isin(train_folds)]['speaker_id'].tolist()
        
        train_df = full_df[full_df['speaker_id'].isin(train_speakers)].reset_index(drop=True)
        val_df = full_df[full_df['speaker_id'].isin(val_speakers)].reset_index(drop=True)

        # Dataset 和 DataLoader
        train_set = OpenSMILEAudioDataset(train_df, from_df=True)
        val_set = OpenSMILEAudioDataset(val_df, from_df=True)
        
        shuffle_train = False if config.EXPERIMENT_MODE == 'DP' else True
        train_loader = DataLoader(train_set, batch_size=config.DATA['batch_size'], shuffle=shuffle_train, collate_fn=collate_fn_with_padding)
        val_loader = DataLoader(val_set, batch_size=config.DATA['batch_size'], shuffle=False, collate_fn=collate_fn_with_padding)

        # initialize model, optimizer, and criterion
        model = TransformerClassifier(
            d_model=config.MODEL['d_model'],
            nhead=config.MODEL['nhead'],
            num_layers=config.MODEL['num_layers'],
            num_classes=config.MODEL['num_classes'],
            dim_feedforward=config.MODEL['dim_feedforward'],
            dropout=config.MODEL['dropout'],
            dp_mode=(config.EXPERIMENT_MODE == 'DP')
        ).to(config.GENERAL['device'])
        
        # Initialize trainer
        if config.EXPERIMENT_MODE == 'DP':
            print("DP Mode: Creating separate optimizers for Encoder and Classifier.")
            optimizer_encoder = optim.AdamW(model.encoder.parameters(), lr=config.TRAINING['lr'], weight_decay=config.TRAINING['weight_decay'])
            optimizer_classifier = optim.AdamW(model.classifier.parameters(), lr=config.TRAINING['lr'], weight_decay=config.TRAINING['weight_decay'])
            optimizers = (optimizer_encoder, optimizer_classifier)
            criterion = nn.CrossEntropyLoss()
            trainer = Trainer(config, model, optimizers, criterion, config.GENERAL['device'], config.EXPERIMENT_MODE, len(train_set))
        elif config.EXPERIMENT_MODE == 'ADVASR':
            # 暂时停用ADVASR相关逻辑
            print("ADVASR mode is currently disabled.")
            return
        else:  # NORMAL
            optimizer = optim.AdamW(model.parameters(), lr=config.TRAINING['lr'], weight_decay=config.TRAINING['weight_decay'])
            criterion = nn.CrossEntropyLoss()
            trainer = Trainer(config, model, optimizer, criterion, config.GENERAL['device'], config.EXPERIMENT_MODE, len(train_set))

        # traning and evaluation loop
        # reset best metrics for each fold
        best_fold_val_acc = -np.inf
        best_fold_metrics = {}
        val_metrics = trainer.evaluate(val_loader)
        print(f"  [Fold {fold_idx+1}, Epoch {0}] | Val Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")

        for epoch in range(1, config.GENERAL['epochs'] + 1):
            train_result = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)
            
            # Handle different return formats from train_epoch
            if isinstance(train_result, tuple):
                # AdversarialTrainer returns (avg_loss, avg_utility_loss, avg_adv_loss)
                avg_train_loss = train_result[0]
                loss_str = f"Train Loss: {avg_train_loss:.4f}"
            else:
                # Normal Trainer returns just avg_loss
                avg_train_loss = train_result
                loss_str = f"Train Loss: {avg_train_loss:.4f}"
                
            print(f"  [Fold {fold_idx+1}, Epoch {epoch}] {loss_str} | Val Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
                
            if val_metrics['acc'] > best_fold_val_acc:
                best_fold_val_acc = val_metrics['acc']
                best_fold_metrics = val_metrics
                best_fold_metrics['loss'] = avg_train_loss
                torch.save(model.state_dict(), f'best_model_fold_{val_fold_num}.pth')

        # 打印当前折的最佳结果
        print(f"\n--- Fold {fold_idx + 1}/{config.GENERAL['num_folds']} (Validation Fold: {val_fold_num}) BEST Results ---")
        print(f"  Best Training Loss (at best ACC epoch): {best_fold_metrics['loss']:.4f}")
        print(f"  Best Val Speaker-Level - Acc: {best_fold_metrics['acc']:.4f}, P: {best_fold_metrics['p']:.4f}, R: {best_fold_metrics['r']:.4f}, F1: {best_fold_metrics['f1']:.4f}")
        
        # 存储结果用于最终平均
        all_fold_best_metrics['loss'].append(best_fold_metrics['loss'])
        all_fold_best_metrics['accuracy'].append(best_fold_metrics['acc'])
        all_fold_best_metrics['precision'].append(best_fold_metrics['p'])
        all_fold_best_metrics['recall'].append(best_fold_metrics['r'])
        all_fold_best_metrics['f1'].append(best_fold_metrics['f1'])

    # 打印所有折的平均结果
    print(f"\n--- Average BEST Metrics Across All {config.GENERAL['num_folds']} Folds ---")
    for metric_name, values in all_fold_best_metrics.items():
        avg_value = sum(values) / len(values)
        print(f"  Average BEST {metric_name.capitalize()}: {avg_value:.4f}")


if __name__ == '__main__':
    main()
