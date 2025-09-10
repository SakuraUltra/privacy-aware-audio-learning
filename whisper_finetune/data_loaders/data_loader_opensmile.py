# data_loader_opensmile.py
# 使用OpenSMILE特征替代mel-spectrogram的数据加载器

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import WhisperTokenizer
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import os
import sys

# 添加上级目录到路径以便导入config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from configs import config
except ImportError:
    # 如果相对导入失败，尝试绝对路径
    import importlib.util
    config_path = os.path.join(parent_dir, 'configs', 'config.py')
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

class OpenSMILEWhisperDataset(Dataset):
    """使用OpenSMILE特征的Whisper数据集"""
    def __init__(self, df, tokenizer, feature_dir=None, shared_scaler=None):
        self.df = df
        self.tokenizer = tokenizer
        self.feature_dir = feature_dir or "../data/extracted_features_train"
        
        # 使用共享的scaler，避免重复计算
        if shared_scaler is not None:
            self.scaler = shared_scaler
            self.normalize = True
        else:
            self.normalize = False
            self.scaler = None

    def _fit_scaler(self):
        """计算OpenSMILE特征的标准化参数"""
        all_features = []
        for idx in range(min(100, len(self.df))):  # 只用前100个样本计算标准化参数
            try:
                row = self.df.iloc[idx]
                feature_path = self._get_feature_path(row)
                if os.path.exists(feature_path):
                    feats_df = pd.read_csv(feature_path)
                    if feats_df.isnull().any().any():
                        feats_df = feats_df.fillna(0.0)
                    feats = feats_df.values
                    all_features.append(feats)
            except Exception as e:
                continue
        
        if all_features:
            all_features = np.vstack(all_features)
            self.scaler.fit(all_features)
        else:
            # 如果无法计算标准化参数，创建一个默认的
            self.scaler = None
            self.normalize = False

    def _get_feature_path(self, row):
        """根据音频路径获取对应的OpenSMILE特征路径"""
        audio_path = row["audio_path"]
        # 从音频路径提取文件名 (不含扩展名)
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
        feature_filename = f"{audio_filename}.csv"
        feature_path = os.path.join(self.feature_dir, feature_filename)
        return feature_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 加载OpenSMILE特征
        feature_path = self._get_feature_path(row)
        
        if not os.path.exists(feature_path):
            # 如果特征文件不存在，返回零特征
            print(f"Warning: Feature file not found: {feature_path}")
            opensmile_features = torch.zeros((100, 6373))  # 默认维度
        else:
            try:
                feats_df = pd.read_csv(feature_path)
                if feats_df.isnull().any().any():
                    feats_df = feats_df.fillna(0.0)
                opensmile_features = feats_df.values
                
                # 标准化
                if self.normalize and self.scaler is not None:
                    opensmile_features = self.scaler.transform(opensmile_features)
                    
                opensmile_features = torch.FloatTensor(opensmile_features)
                
            except Exception as e:
                print(f"Error loading feature file {feature_path}: {e}")
                opensmile_features = torch.zeros((100, 6373))

        # 2. 加载并处理文本
        text_path = row["text_path"]
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 使用tokenizer处理文本
        labels = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=False, 
            truncation=True, 
            max_length=config.MAX_TEXT_LEN
        ).input_ids[0]

        return {
            "input_features": opensmile_features,  # [seq_len, feature_dim]
            "labels": labels  # [text_len]
        }


class OpenSMILEWhisperDataModule(pl.LightningDataModule):
    """使用OpenSMILE特征的PyTorch Lightning数据模块"""
    def __init__(self, df: pd.DataFrame, batch_size: int, train_feature_dir=None, val_feature_dir=None):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.train_feature_dir = train_feature_dir or "../data/extracted_features_train"
        self.val_feature_dir = val_feature_dir or "../data/extracted_features_val"
        
        # 只需要tokenizer，不需要完整的processor
        self.tokenizer = WhisperTokenizer.from_pretrained(
            config.MODEL_NAME, 
            language=config.LANGUAGE, 
            task=config.TASK
        )

    def setup(self, stage=None):
        # 分割训练和验证数据集
        train_size = int(config.TRAIN_RATIO * len(self.df))
        val_size = len(self.df) - train_size
        
        # 创建训练和验证数据集
        train_df = self.df.iloc[:train_size].reset_index(drop=True)
        val_df = self.df.iloc[train_size:].reset_index(drop=True)
        
        self.train_dataset = OpenSMILEWhisperDataset(
            train_df, 
            self.tokenizer, 
            feature_dir=self.train_feature_dir
        )
        self.val_dataset = OpenSMILEWhisperDataset(
            val_df, 
            self.tokenizer, 
            feature_dir=self.val_feature_dir
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=2
        )

    def collate_fn(self, batch):
        """处理变长序列的批次填充"""
        # 收集特征和标签
        input_features = [item["input_features"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # 对OpenSMILE特征进行填充 (padding到最长序列)
        input_features_padded = pad_sequence(
            input_features, 
            batch_first=True, 
            padding_value=0.0
        )  # [batch_size, max_seq_len, feature_dim]
        
        # 对标签进行填充
        labels_padded = pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100  # Whisper使用-100作为ignore index
        )  # [batch_size, max_text_len]

        return {
            "input_features": input_features_padded,
            "labels": labels_padded
        }


def validate_opensmile_dataset(audio_dir, text_dir, feature_dir):
    """
    验证音频、文本和OpenSMILE特征数据集是否匹配
    """
    audio_speakers = sorted([d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))])
    text_speakers = sorted([d for d in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, d))])

    if audio_speakers != text_speakers:
        raise ValueError("音频和文本的说话人文件夹不匹配!")

    print(f"找到 {len(audio_speakers)} 个说话人文件夹。")

    all_files = []
    missing_features = []
    
    for speaker in audio_speakers:
        audio_files = sorted([f for f in os.listdir(os.path.join(audio_dir, speaker)) if f.endswith('.wav')])
        text_files = sorted([f for f in os.listdir(os.path.join(text_dir, speaker)) if f.endswith('.txt')])

        audio_basenames = [os.path.splitext(f)[0] for f in audio_files]
        text_basenames = [os.path.splitext(f)[0] for f in text_files]

        if audio_basenames != text_basenames:
            print(f"警告: 说话人 {speaker} 的音频/文本文件不完全匹配。")
            common_basenames = set(audio_basenames) & set(text_basenames)
            audio_files = [f"{bn}.wav" for bn in common_basenames]
            text_files = [f"{bn}.txt" for bn in common_basenames]

        for audio_file, text_file in zip(audio_files, text_files):
            basename = os.path.splitext(audio_file)[0]
            feature_file = f"{basename}.csv"
            feature_path = os.path.join(feature_dir, feature_file)
            
            if not os.path.exists(feature_path):
                missing_features.append(feature_path)
            
            all_files.append({
                "speaker": speaker,
                "audio_path": os.path.join(audio_dir, speaker, audio_file),
                "text_path": os.path.join(text_dir, speaker, text_file),
                "feature_path": feature_path
            })

    if missing_features:
        print(f"警告: 缺少 {len(missing_features)} 个OpenSMILE特征文件")
        print("前5个缺失文件:", missing_features[:5])

    df = pd.DataFrame(all_files)
    print(f"总计找到 {len(df)} 个音频/文本/特征文件匹配项。")
    return df
