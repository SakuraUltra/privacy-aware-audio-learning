from cv2 import norm, normalize
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import sys
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

__all__ = ['OpenSMILEAudioDataset', 'collate_fn_with_padding']

class OpenSMILEAudioDataset(Dataset):
    def __init__(self, csv_input, from_df=False):
        self.data = csv_input.reset_index(drop=True) if from_df else pd.read_csv(csv_input)
        self.d_model = 32  # feature dimension
        self.normalize = normalize

        if self.normalize:
            self.scaler = StandardScaler()
            self._fit_scaler()

    def _fit_scaler(self):
        """计算特征标准化参数"""
        all_features = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            path = row.get('feature_path') or row.get('path')
            if path is None:
                continue
            try:
                feats_df = pd.read_csv(path)
                # 检查并填充 NaN 值
                if feats_df.isnull().any().any():
                    feats_df = feats_df.fillna(0.0)
                feats = feats_df.values
                all_features.append(feats)
            except Exception:
                continue
        if all_features:
            all_features = np.vstack(all_features)
            self.scaler.fit(all_features)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            path = row.get('feature_path', row.get('path'))
            
            # 安全地处理 label，检查是否为 NaN
            label_val = row['label']
            if pd.isna(label_val):
                label = 0
            else:
                label = int(label_val)
                
            speaker_id = row.get('speaker_id', "unknown") if not pd.isna(row.get('speaker_id')) else "unknown"
            
            try:
                # 读取特征文件并检查 NaN 值
                feats_df = pd.read_csv(path)
                if feats_df.isnull().any().any():
                    feats_df = feats_df.fillna(0.0)
                
                feats = torch.tensor(feats_df.values, dtype=torch.float32)
                if feats.shape[1] != self.d_model:
                    raise ValueError(f"Feature dim mismatch {feats.shape[1]} != {self.d_model}")
                if self.normalize:
                    feats = torch.tensor(self.scaler.transform(feats), dtype=torch.float32)
            except Exception:
                feats = torch.zeros((1, self.d_model), dtype=torch.float32)

            # 安全地处理 speaker_id
            try:
                # 确保speaker_id是字符串
                if pd.isna(speaker_id) or speaker_id == "unknown" or not isinstance(speaker_id, str):
                    speaker_idx = 0
                else:
                    speaker_parts = speaker_id.split('_')
                    if len(speaker_parts) > 0 and speaker_parts[0].isdigit():
                        speaker_idx = int(speaker_parts[0])
                    else:
                        speaker_idx = 0
            except Exception:
                speaker_idx = 0

            return feats, torch.tensor(label, dtype=torch.long), torch.tensor(speaker_idx), speaker_id
        except Exception:
            # 返回一个默认的样本
            feats = torch.zeros((1, self.d_model), dtype=torch.float32)
            label = 0
            speaker_idx = 0
            speaker_id = "unknown"
            return feats, torch.tensor(label, dtype=torch.long), torch.tensor(speaker_idx), speaker_id


def collate_fn_with_padding(batch):
    try:
        features, labels, speaker_idxs, speaker_ids = zip(*batch)
        
        # 检查并处理任何无效的特征
        valid_features = []
        valid_labels = []
        valid_speaker_ids = []
        valid_lengths = []
        
        for i, (feat, label, speaker_id) in enumerate(zip(features, labels, speaker_ids)):
            try:
                if torch.isnan(feat).any():
                    continue
                valid_features.append(feat)
                valid_labels.append(label)
                valid_speaker_ids.append(speaker_id)
                valid_lengths.append(feat.shape[0])
            except Exception:
                continue
        
        if not valid_features:
            # 创建一个伪批次
            dummy_feat = torch.zeros((1, features[0].shape[1]), dtype=torch.float32)
            dummy_label = torch.zeros(1, dtype=torch.long)
            dummy_padding_mask = torch.zeros((1, 1), dtype=torch.bool)
            return dummy_feat, dummy_label, dummy_padding_mask, ["unknown"]
            
        padded_feats = pad_sequence(valid_features, batch_first=True, padding_value=0.0)
        max_len = padded_feats.size(1)
        padding_mask = torch.arange(max_len).expand(len(valid_lengths), max_len) >= torch.tensor(valid_lengths).unsqueeze(1)
        
        return padded_feats, torch.stack(valid_labels), padding_mask, valid_speaker_ids
        
    except Exception:
        # 创建一个伪批次
        dummy_feat = torch.zeros((1, 32), dtype=torch.float32)  # 使用默认的d_model=32
        dummy_label = torch.zeros(1, dtype=torch.long)
        dummy_padding_mask = torch.zeros((1, 1), dtype=torch.bool)
        return dummy_feat, dummy_label, dummy_padding_mask, ["unknown"]