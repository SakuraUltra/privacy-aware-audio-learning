"""
数据处理工具
"""
import pandas as pd
from typing import Tuple, List
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import torch
import numpy as np


def validate_data_files(config) -> bool:
    """
    验证数据文件是否存在
    
    Args:
        config: 配置对象
    
    Returns:
        是否所有必需文件都存在
    """
    required_files = [config.data['speaker_folds_csv']]
    
    # 根据特征类型添加对应的数据文件
    dataset_config = config.get_dataset_config()
    if 'mel_dataset_csv' in dataset_config:
        required_files.append(dataset_config['mel_dataset_csv'])
    if 'all_expanded_features_csv' in dataset_config:
        required_files.append(dataset_config['all_expanded_features_csv'])
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print(f"✅ All required data files found")
    return True


def load_and_validate_data(config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载并验证数据
    
    Args:
        config: 配置对象
    
    Returns:
        (speaker_fold_df, full_df)
    """
    # 验证文件存在
    if not validate_data_files(config):
        raise FileNotFoundError("Required data files are missing")
    
    # 加载speaker fold信息
    speaker_fold_df = pd.read_csv(config.data['speaker_folds_csv'])
    speaker_fold_df['speaker_id'] = speaker_fold_df['speaker_id'].str.strip("'")
    
    # 根据特征类型加载数据
    dataset_config = config.get_dataset_config()
    if 'mel_dataset_csv' in dataset_config:
        full_df = pd.read_csv(dataset_config['mel_dataset_csv'])
        print(f"✅ Loaded MEL dataset: {len(full_df)} samples")
    else:
        full_df = pd.read_csv(dataset_config['all_expanded_features_csv'])
        print(f"✅ Loaded OpenSMILE dataset: {len(full_df)} samples")
    
    # 验证数据完整性
    print(f"✅ Speaker folds: {len(speaker_fold_df)} speakers across {speaker_fold_df['fold'].nunique()} folds")
    
    # 检查speaker_id是否匹配
    dataset_speakers = set(full_df['speaker_id'].unique())
    fold_speakers = set(speaker_fold_df['speaker_id'].unique())
    
    if not dataset_speakers.issubset(fold_speakers):
        missing_speakers = dataset_speakers - fold_speakers
        print(f"⚠️  Warning: {len(missing_speakers)} speakers in dataset but not in fold file")
    
    if not fold_speakers.issubset(dataset_speakers):
        missing_speakers = fold_speakers - dataset_speakers
        print(f"⚠️  Warning: {len(missing_speakers)} speakers in fold file but not in dataset")
    
    return speaker_fold_df, full_df


def split_data_by_folds(speaker_fold_df: pd.DataFrame, full_df: pd.DataFrame, 
                       val_fold_num: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根据fold划分训练和验证数据
    
    Args:
        speaker_fold_df: speaker fold信息
        full_df: 完整数据集
        val_fold_num: 验证fold编号
    
    Returns:
        (train_df, val_df)
    """
    # 获取验证集speaker
    val_speakers = speaker_fold_df[speaker_fold_df['fold'] == val_fold_num]['speaker_id'].tolist()
    
    # 获取训练集speaker
    all_folds = speaker_fold_df['fold'].unique()
    train_folds = [f for f in all_folds if f != val_fold_num]
    train_speakers = speaker_fold_df[speaker_fold_df['fold'].isin(train_folds)]['speaker_id'].tolist()
    
    # 划分数据
    train_df = full_df[full_df['speaker_id'].isin(train_speakers)].reset_index(drop=True)
    val_df = full_df[full_df['speaker_id'].isin(val_speakers)].reset_index(drop=True)
    
    print(f"  Train: {len(train_df)} samples from {len(train_speakers)} speakers")
    print(f"  Val: {len(val_df)} samples from {len(val_speakers)} speakers")
    
    return train_df, val_df

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Whisper/ASR 专用 Data Collator（兼容你当前项目的纯 .py 结构）
    期望每个样本包含：
      - "input_features": list 或 torch.Tensor，形状 ~ (80, T)
      - "labels": List[int] 或 torch.LongTensor
    可选：
      - "input_length": float/int（用于 group_by_length）

    注意：只返回 Whisper 模型需要的列，避免传递额外的列（如 speaker, text）
    """
    processor: Any  # WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        from torch.nn.utils.rnn import pad_sequence

        # 使用与 whisper_finetune 相同的 collate 逻辑
        # 收集特征和标签
        input_features = [item["input_features"] for item in features]
        labels = [item["labels"] for item in features]

        # 确保所有特征都是张量
        input_features_tensors = []
        for feat in input_features:
            if not isinstance(feat, torch.Tensor):
                feat = torch.as_tensor(feat)
            input_features_tensors.append(feat)

        # 对 MEL 特征进行填充 (padding到最长序列)
        input_features_padded = pad_sequence(
            input_features_tensors,
            batch_first=True,
            padding_value=0.0
        )  # [batch_size, max_seq_len, feature_dim] 或 [batch_size, feature_dim, max_seq_len]

        # 确保所有标签都是张量
        labels_tensors = []
        for lab in labels:
            if not isinstance(lab, torch.Tensor):
                lab = torch.as_tensor(lab)
            labels_tensors.append(lab)

        # 对标签进行填充
        labels_padded = pad_sequence(
            labels_tensors,
            batch_first=True,
            padding_value=-100  # Whisper使用-100作为ignore index
        )  # [batch_size, max_text_len]

        # 只返回 Whisper 模型期望的参数，避免传递额外的列
        batch = {
            "input_features": input_features_padded,
            "labels": labels_padded
        }

        return batch


def ensure_length_field_for_whisper(
    dataset,
    length_column_name: str = "input_length",
    feature_key: str = "input_features",
    as_seconds: bool = False,
    frames_per_second: float = 50.0,  # Whisper log-Mel 约 50 帧/秒
):
    """
    为 HuggingFace Dataset 补充长度字段（配合 TrainingArguments.group_by_length 使用）。
    - 若已有 length_column_name 则直接返回
    - 否则根据 'input_features' 的时间维推断长度
    仅在 whisper_lora 模式下的 HF Dataset 会用到；你当前的 pandas DataFrame 流程不受影响。
    """
    # 只有 HF Dataset 才有 .column_names / .map；pandas DataFrame 不会触发这里
    if hasattr(dataset, "column_names") and (length_column_name in dataset.column_names):
        return dataset

    def _length_mapper(ex):
        feat = ex.get(feature_key, None)
        if feat is None:
            ex[length_column_name] = 1.0
            return ex
        t_frames = int(torch.as_tensor(feat).shape[-1])  # 仅取末维长度
        ex[length_column_name] = float(t_frames / frames_per_second) if as_seconds else float(t_frames)
        return ex

    if hasattr(dataset, "map"):
        return dataset.map(_length_mapper, desc="Adding length field for group_by_length")