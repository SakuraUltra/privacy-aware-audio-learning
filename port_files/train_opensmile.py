import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from mydataset_opensmile import OpenSMILEAudioDataset, collate_fn_with_padding
from model_transformer import TransformerClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np # 导入 numpy for inf

speaker_fold_df = pd.read_csv("speaker_folds.csv")
speaker_fold_df['speaker_id'] = speaker_fold_df['speaker_id'].str.strip("'")

try:
    full_df = pd.read_csv("all_expanded_features.csv")
except FileNotFoundError:
    print("error: cannot find the file 'all_expanded_features.csv'.")
    # 退出程序或添加其他错误处理逻辑
    # sys.exit(1) # 如果希望在文件未找到时退出，可以启用此行

device = torch.device("cpu") # Or "cuda" if you have a GPU and want to use it

# K-Fold Cross-Validation Loop
num_folds = 5
all_folds = [1, 2, 3, 4, 5]

# 用于存储每个折叠的最佳结果
all_fold_best_metrics = defaultdict(list)

for fold_idx, val_fold_num in enumerate(all_folds):
    print(f"\n--- Starting Fold {fold_idx + 1}/{num_folds} (Validation Fold: {val_fold_num}) ---")

    # Determine train and validation speakers for the current fold
    val_speakers = speaker_fold_df[speaker_fold_df['fold'] == val_fold_num]['speaker_id'].tolist()
    train_folds = [f for f in all_folds if f != val_fold_num]
    train_speakers = speaker_fold_df[speaker_fold_df['fold'].isin(train_folds)]['speaker_id'].tolist()

    # Filter dataframes based on current fold's speaker lists
    train_df = full_df[full_df['speaker_id'].isin(train_speakers)].reset_index(drop=True)
    val_df = full_df[full_df['speaker_id'].isin(val_speakers)].reset_index(drop=True)

    # Create datasets and dataloaders for the current fold
    train_set = OpenSMILEAudioDataset(train_df, from_df=True)
    val_set = OpenSMILEAudioDataset(val_df, from_df=True)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn_with_padding)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn_with_padding)

    # Initialize model, criterion, and optimizer for each fold
    # This ensures a fresh start for each fold
    model = TransformerClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Variables to track best metrics for the current fold
    best_fold_val_acc = -np.inf # 初始化为负无穷大，确保第一次都会更新
    best_fold_metrics = {
        'loss': 0.0,
        'acc': 0.0,
        'p': 0.0,
        'r': 0.0,
        'f1': 0.0
    }
    
    num_epochs_per_fold = 5
    for epoch in range(1, num_epochs_per_fold + 1):
        model.train()
        total_loss = 0.0
        for x, y, mask, _ in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, padding_mask=mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"  [Fold {fold_idx + 1}/{num_folds}, Epoch {epoch}/{num_epochs_per_fold}] Training Loss: {avg_train_loss:.4f}")

        # Validation loop for the current fold (speaker-level)
        model.eval()
        speaker_preds = defaultdict(list)
        speaker_labels = {}
        with torch.no_grad():
            for x, y, mask, speaker_ids in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                logits = model(x, padding_mask=mask)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                labels = y.cpu().tolist()

                for spk, pred, true in zip(speaker_ids, preds, labels):
                    speaker_preds[spk].append(pred)
                    speaker_labels[spk] = true

        final_preds, final_labels = [], []
        for sid, pred_list in speaker_preds.items():
            majority_pred = Counter(pred_list).most_common(1)[0][0]
            final_preds.append(majority_pred)
            final_labels.append(speaker_labels[sid])

        acc = accuracy_score(final_labels, final_preds)
        p, r, f1, _ = precision_recall_fscore_support(final_labels, final_preds, average='binary', zero_division=0)
        acc = float(acc)
        p = float(p)
        r = float(r)
        f1 = float(f1)
        
        print(f"  [Fold {fold_idx + 1}/{num_folds}, Epoch {epoch}/{num_epochs_per_fold}] Val Speaker-Level - Acc: {acc:.4f}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

        # Update best metrics for the current fold if current accuracy is better
        if acc > best_fold_val_acc:
            best_fold_val_acc = acc
            best_fold_metrics['loss'] = avg_train_loss # 记录对应最佳准确率的训练损失
            best_fold_metrics['acc'] = acc
            best_fold_metrics['p'] = p
            best_fold_metrics['r'] = r
            best_fold_metrics['f1'] = f1
            # 可以在这里保存模型权重：
            torch.save(model.state_dict(), f'best_model_fold_{val_fold_num}.pth')

    # After all epochs for the current fold, print its BEST metrics
    print(f"\n--- Fold {fold_idx + 1}/{num_folds} (Validation Fold: {val_fold_num}) BEST Results ---")
    print(f"  Best Training Loss (at best ACC epoch): {best_fold_metrics['loss']:.4f}")
    print(f"  Best Val Speaker-Level - Acc: {best_fold_metrics['acc']:.4f}, P: {best_fold_metrics['p']:.4f}, R: {best_fold_metrics['r']:.4f}, F1: {best_fold_metrics['f1']:.4f}")

    # Store best metrics for overall averaging at the end
    all_fold_best_metrics['loss'].append(best_fold_metrics['loss'])
    all_fold_best_metrics['accuracy'].append(best_fold_metrics['acc'])
    all_fold_best_metrics['precision'].append(best_fold_metrics['p'])
    all_fold_best_metrics['recall'].append(best_fold_metrics['r'])
    all_fold_best_metrics['f1'].append(best_fold_metrics['f1'])

# Print average BEST metrics across all folds
print(f"\n--- Average BEST Metrics Across All {num_folds} Folds ---")
for metric_name, values in all_fold_best_metrics.items():
    avg_value = sum(values) / len(values)
    print(f"  Average BEST {metric_name.capitalize()}: {avg_value:.4f}")