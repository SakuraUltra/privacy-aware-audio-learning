# experiment/utils/whisper_mel_adapter.py
"""
适配现有 whisper_finetune 的 MEL 特征数据加载器
直接使用现有的数据结构和 padding 逻辑
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import WhisperProcessor, WhisperTokenizer
from typing import Optional, Tuple
from datasets import Dataset as HFDataset

# 添加 whisper_finetune 路径
whisper_finetune_path = "../whisper_finetune"
if whisper_finetune_path not in sys.path:
    sys.path.insert(0, whisper_finetune_path)

# 导入现有的数据加载器
from data_loaders.data_loader_opensmile import OpenSMILEWhisperDataset


class MELWhisperDataset(Dataset):
    """
    基于现有 OpenSMILEWhisperDataset 的 MEL 特征数据集
    复用现有的 padding 和处理逻辑
    """
    def __init__(self, df, tokenizer, mel_base_dir="audio_mel/data-mel"):
        self.df = df
        self.tokenizer = tokenizer
        self.mel_base_dir = mel_base_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载 MEL 特征
        feature_path = row['feature_path']
        # 处理相对路径
        if feature_path.startswith('data/'):
            feature_path = feature_path[5:]  # 去掉 'data/' 前缀
        full_feature_path = os.path.join(self.mel_base_dir, feature_path)
        
        try:
            # 读取 MEL 特征 CSV 文件
            mel_df = pd.read_csv(full_feature_path, header=None)
            mel_features = mel_df.values.astype(np.float32)
            # 转置为 (mel_bins, time_steps) 格式
            mel_features = mel_features.T
            mel_tensor = torch.from_numpy(mel_features)
        except Exception as e:
            print(f"Error loading MEL features from {full_feature_path}: {e}")
            # 返回默认的 MEL 特征
            mel_tensor = torch.zeros((80, 100), dtype=torch.float32)
        
        # 加载文本
        text_path = row['text_path']
        # 处理相对路径
        if text_path.startswith('../'):
            # 从 data-mel 目录向上到 experiment 目录，然后进入 data 目录
            experiment_dir = os.path.dirname(os.path.dirname(self.mel_base_dir))
            text_path = os.path.join(experiment_dir, text_path[3:])  # 去掉 '../' 前缀
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Error loading text from {text_path}: {e}")
            text = ""
        
        # 使用 tokenizer 处理文本（复用现有逻辑）
        labels = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=False, 
            truncation=True, 
            max_length=448  # Whisper 最大长度
        ).input_ids[0]
        
        return {
            "input_features": mel_tensor,
            "labels": labels,
            "speaker": row['speaker_id'],
            "text": text
        }


def create_mel_datasets_for_cv(
    csv_path: str = "audio_mel/data-mel/mel_dataset.csv",
    model_name: str = "openai/whisper-large-v3-turbo",
    language: str = "italian",
    task: str = "transcribe",
    max_samples: Optional[int] = None
) -> Tuple[WhisperProcessor, HFDataset]:
    """
    为交叉验证创建 MEL 特征数据集
    
    Args:
        csv_path: MEL 数据集 CSV 路径
        model_name: Whisper 模型名称
        language: 语言
        task: 任务
        max_samples: 最大样本数（用于测试）
    
    Returns:
        (processor, full_dataset)
    """
    print("=" * 60)
    print("Creating MEL datasets for Cross-Validation...")
    print(f"Model: {model_name}")
    print(f"Language: {language}")
    print(f"Task: {task}")
    print(f"CSV path: {csv_path}")
    print("=" * 60)
    
    # 创建 processor 和 tokenizer
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task=task
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language=language,
        task=task
    )
    
    # 读取 CSV
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
        print(f"Using first {max_samples} samples for testing")
    
    print(f"Processing {len(df)} samples...")
    
    # 创建 PyTorch Dataset
    pytorch_dataset = MELWhisperDataset(df, tokenizer)
    
    # 转换为 HuggingFace Dataset
    dataset_dict = {
        "input_features": [],
        "labels": [],
        "speaker": [],
        "text": []
    }
    
    successful_samples = 0
    for i in range(len(pytorch_dataset)):
        try:
            sample = pytorch_dataset[i]
            dataset_dict["input_features"].append(sample["input_features"])
            dataset_dict["labels"].append(sample["labels"])
            dataset_dict["speaker"].append(sample["speaker"])
            dataset_dict["text"].append(sample["text"])
            successful_samples += 1
            
            if successful_samples % 100 == 0:
                print(f"Processed {successful_samples} samples...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully processed {successful_samples} samples")
    
    # 创建 HuggingFace Dataset
    hf_dataset = HFDataset.from_dict(dataset_dict)
    
    print("✅ MEL datasets created successfully!")
    return processor, hf_dataset


def create_mel_train_eval_datasets(
    csv_path: str = "audio_mel/data-mel/mel_dataset.csv",
    model_name: str = "openai/whisper-large-v3-turbo",
    language: str = "italian",
    task: str = "transcribe",
    max_samples: Optional[int] = None,
    train_ratio: float = 0.8
) -> Tuple[WhisperProcessor, HFDataset, HFDataset]:
    """
    创建训练和验证数据集
    
    Returns:
        (processor, train_dataset, eval_dataset)
    """
    # 创建完整数据集
    processor, full_dataset = create_mel_datasets_for_cv(
        csv_path, model_name, language, task, max_samples
    )
    
    # 按说话人分割
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
    
    speakers = df['speaker_id'].unique()
    n_train_speakers = int(len(speakers) * train_ratio)
    
    train_speakers = speakers[:n_train_speakers]
    eval_speakers = speakers[n_train_speakers:]
    
    print(f"Train speakers: {len(train_speakers)}, Eval speakers: {len(eval_speakers)}")
    
    # 分割数据集
    train_indices = []
    eval_indices = []
    
    for idx, speaker in enumerate(full_dataset['speaker']):
        if speaker in train_speakers:
            train_indices.append(idx)
        else:
            eval_indices.append(idx)
    
    train_dataset = full_dataset.select(train_indices)
    eval_dataset = full_dataset.select(eval_indices)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Eval dataset: {len(eval_dataset)} samples")
    
    return processor, train_dataset, eval_dataset
