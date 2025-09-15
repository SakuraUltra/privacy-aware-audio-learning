"""
Audio MEL数据集类
与OpenSMILE数据集保持相同的接口，实现无缝切换
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from sklearn.preprocessing import StandardScaler

class AudioMelDataset(Dataset):
    """
    Audio MEL特征数据集
    与OpenSMILEAudioDataset保持相同的接口
    """
    
    def __init__(self,
                 data_source: Union[pd.DataFrame, str, Path],
                 from_df: bool = True,
                 normalize: bool = True,
                 config=None):
        """
        初始化MEL特征数据集

        Args:
            data_source: DataFrame或CSV文件路径
            from_df: 是否从DataFrame加载
            normalize: 是否标准化特征
            config: 配置对象，包含路径映射信息
        """
        self.normalize = normalize
        self.config = config

        if from_df and isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        else:
            self.df = pd.read_csv(data_source)

        self.df = self.df.reset_index(drop=True)

        # 验证必要的列
        required_cols = ['feature_path', 'speaker_id']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

        # 创建标签映射 (假设是H/P二分类)
        unique_speakers = self.df['speaker_id'].unique()
        self.label_mapping = self._create_label_mapping(unique_speakers)

        # 初始化标准化器（使用与OpenSMILEAudioDataset相同的方法）
        if self.normalize:
            self.scaler = StandardScaler()
            self._fit_scaler()

        print(f"AudioMelDataset初始化完成:")
        print(f"  - 样本数: {len(self.df)}")
        print(f"  - 说话人数: {len(unique_speakers)}")
        print(f"  - 标准化: {normalize}")
    
    def _fix_feature_path(self, feature_path):
        """根据配置修正特征文件路径"""
        # 如果路径已经存在，直接返回
        import os
        if os.path.exists(feature_path):
            return feature_path
        
        # 使用配置中的路径映射
        if self.config and 'feature_path_mappings' in self.config.DATA:
            for old_prefix, new_prefix in self.config.DATA['feature_path_mappings'].items():
                if feature_path.startswith(old_prefix):
                    return feature_path.replace(old_prefix, new_prefix, 1)
        
        # 如果没有配置，使用原来的硬编码逻辑作为fallback
        if feature_path.startswith('data/mel_features/'):
            return feature_path.replace('data/mel_features/', 'audio_mel/data-mel/mel_features/', 1)
        
        return feature_path

    def _load_text_for_slice(self, audio_path: str) -> str:
        """
        根据音频路径加载对应的文本内容
        
        Args:
            audio_path: 音频文件路径，例如：'/path/to/slices_64/01_CF56_1/01_CF56_1_slice01.wav'
        
        Returns:
            str: 对应的文本内容，如果找不到返回空字符串
        """
        try:
            from pathlib import Path
            
            # 从音频路径提取slice信息
            # 例如：'/path/to/slices_64/01_CF56_1/01_CF56_1_slice01.wav' -> '01_CF56_1_slice01'
            audio_path = Path(audio_path)
            slice_name = audio_path.stem  # 去掉扩展名
            
            # 提取speaker_id (例如：'01_CF56_1_slice01' -> '01_CF56_1')
            parts = slice_name.split('_')
            if len(parts) >= 3:
                speaker_id = '_'.join(parts[:3])  # 取前三部分作为speaker_id
                
                # 构建文本文件路径
                text_file_path = f"data/slices_txt_64/{speaker_id}/{slice_name}.txt"
                
                # 尝试读取文本文件
                if Path(text_file_path).exists():
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                        return text_content
                else:
                    print(f"Warning: Text file not found: {text_file_path}")
                    
        except Exception as e:
            print(f"Error loading text for {audio_path}: {e}")
        
        # 如果找不到对应文本，返回默认文本
        return "Audio content for privacy analysis."

    def _fit_scaler(self):
        """计算特征标准化参数（与OpenSMILEAudioDataset相同的方法）"""
        all_features = []
        corrupted_files = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            feature_path = self._fix_feature_path(row['feature_path'])
            try:
                feats_df = pd.read_csv(feature_path)

                # 检查数据类型和损坏的数据
                for col in feats_df.columns:
                    # 尝试转换为数值类型，捕获损坏的数据
                    try:
                        feats_df[col] = pd.to_numeric(feats_df[col], errors='coerce')
                    except Exception as e:
                        print(f"Warning: Column {col} in {feature_path} has invalid data: {e}")

                # 检查并填充 NaN 值
                if feats_df.isnull().any().any():
                    nan_count = feats_df.isnull().sum().sum()
                    if nan_count > 0:
                        print(f"Warning: Found {nan_count} NaN values in {feature_path}, filling with 0.0")
                    feats_df = feats_df.fillna(0.0)

                # 确保所有数据都是数值类型
                feats = feats_df.select_dtypes(include=[np.number]).values
                if feats.size > 0:
                    all_features.append(feats)
                else:
                    corrupted_files.append(feature_path)

            except Exception as e:
                print(f"Error loading {feature_path}: {e}")
                corrupted_files.append(feature_path)
                continue

        if corrupted_files:
            print(f"Warning: {len(corrupted_files)} corrupted MEL feature files found:")
            for f in corrupted_files[:5]:  # 只显示前5个
                print(f"  - {f}")
            if len(corrupted_files) > 5:
                print(f"  ... and {len(corrupted_files) - 5} more")

        if all_features:
            all_features = np.vstack(all_features)
            self.scaler.fit(all_features)
        else:
            print("Error: No valid MEL features found for scaler fitting!")
            raise ValueError("No valid MEL features available")

    def _create_label_mapping(self, speakers):
        """创建说话人到标签的映射"""
        # 从OpenSMILE数据集读取真实标签，而不是基于P/C推测
        try:
            import pandas as pd
            opensmile_df = pd.read_csv('data/all_expanded_features_fixed.csv')
            # 获取每个说话人的真实标签（取第一个出现的标签）
            speaker_to_label = opensmile_df.groupby('speaker_id')['label'].first().to_dict()

            label_map = {}
            missing_speakers = []

            for speaker in speakers:
                if speaker in speaker_to_label:
                    label_map[speaker] = speaker_to_label[speaker]
                else:
                    # 如果在OpenSMILE数据集中找不到该说话人，使用编号规则作为fallback
                    try:
                        speaker_num = int(speaker.split('_')[0])
                        if speaker_num <= 9:
                            label_map[speaker] = 0  # 编号01-09为标签0
                        else:
                            label_map[speaker] = 1  # 编号10+为标签1
                        missing_speakers.append(speaker)
                    except:
                        label_map[speaker] = 0  # 默认为0
                        missing_speakers.append(speaker)

            if missing_speakers:
                print(f"警告: {len(missing_speakers)} 个说话人在OpenSMILE数据集中未找到，使用编号规则分配标签:")
                print(f"  缺失的说话人: {missing_speakers[:5]}{'...' if len(missing_speakers) > 5 else ''}")

            print(f"标签映射统计: 标签0={sum(1 for v in label_map.values() if v == 0)}, 标签1={sum(1 for v in label_map.values() if v == 1)}")

            return label_map

        except Exception as e:
            print(f"警告: 无法从OpenSMILE数据集读取标签 ({e})，使用编号规则作为fallback")
            # Fallback: 使用编号规则
            label_map = {}
            for speaker in speakers:
                try:
                    speaker_num = int(speaker.split('_')[0])
                    if speaker_num <= 9:
                        label_map[speaker] = 0  # 编号01-09为标签0
                    else:
                        label_map[speaker] = 1  # 编号10+为标签1
                except:
                    label_map[speaker] = 0  # 默认为0
            return label_map
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            features: MEL特征 [time_steps, feature_dim]
            label: 标签 (0 or 1)
            speaker_id: 说话人ID
        """
        row = self.df.iloc[idx]
        try:
            # 加载MEL特征，使用配置化路径修正
            feature_path = self._fix_feature_path(row['feature_path'])

            # 读取特征文件并检查 NaN 值（与OpenSMILEAudioDataset相同）
            feats_df = pd.read_csv(feature_path)
            if feats_df.isnull().any().any():
                feats_df = feats_df.fillna(0.0)

            features = torch.tensor(feats_df.values, dtype=torch.float32)

            # 使用全局标准化（与OpenSMILEAudioDataset相同的方法）
            if self.normalize:
                features = torch.tensor(self.scaler.transform(features), dtype=torch.float32)

            # 获取标签
            speaker_id = row['speaker_id']
            label = self.label_mapping.get(speaker_id, 0)
            
            # 加载对应的文本（如果存在path列）
            text = ""
            if 'path' in row:
                text = self._load_text_for_slice(row['path'])
            
            return features, label, speaker_id, text
        except Exception as e:
            print(f"加载样本失败 {idx}: {e}")
            # 返回零特征（使用32维，与配置一致）
            features = torch.zeros((1, 32), dtype=torch.float32)
            return features, 0, "unknown", ""
    

    
    def get_stats(self):
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.df),
            'unique_speakers': self.df['speaker_id'].nunique(),
            'speakers_list': self.df['speaker_id'].unique().tolist(),
            'label_distribution': {}
        }
        
        # 计算标签分布
        for speaker in stats['speakers_list']:
            label = self.label_mapping[speaker]
            stats['label_distribution'][label] = stats['label_distribution'].get(label, 0) + \
                                               len(self.df[self.df['speaker_id'] == speaker])
        
        return stats


def collate_fn_mel_padding(batch):
    """
    MEL特征的collate函数，与OpenSMILE的collate_fn_with_padding保持兼容
    """
    features, labels, speaker_ids, texts = zip(*batch)

    # 找到最大序列长度
    max_len = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[1]

    # 填充特征
    padded_features = []
    padding_masks = []  # 改名为padding_masks以明确含义

    for f in features:
        seq_len = f.shape[0]

        # 填充到最大长度
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, feature_dim)
            padded_f = torch.cat([f, padding], dim=0)
            # 修复：使用与OpenSMILE相同的mask含义 (True=padding, False=valid)
            padding_mask = torch.cat([torch.zeros(seq_len), torch.ones(max_len - seq_len)]).bool()
        else:
            padded_f = f
            padding_mask = torch.zeros(seq_len).bool()  # 全部为False（有效）

        padded_features.append(padded_f)
        padding_masks.append(padding_mask)

    # 堆叠为batch
    features_batch = torch.stack(padded_features)  # [batch_size, max_len, feature_dim]
    labels_batch = torch.tensor(labels, dtype=torch.long)
    padding_masks_batch = torch.stack(padding_masks)  # [batch_size, max_len]

    # 返回5个元素以与训练循环兼容：features, labels, padding_masks, speaker_ids, texts
    return features_batch, labels_batch, padding_masks_batch, speaker_ids, texts


class MelDatasetConverter:
    """
    MEL数据集转换器
    将现有的数据集格式转换为MEL特征格式
    """
    
    @staticmethod
    def create_mel_dataset_from_audio(
        audio_dir: str,
        text_dir: str,
        speaker_folds_csv: str,
        output_dir: str,
        output_csv: str
    ) -> pd.DataFrame:
        """
        从音频目录创建MEL特征数据集
        
        Args:
            audio_dir: 音频文件目录
            text_dir: 文本文件目录  
            speaker_folds_csv: 说话人fold信息
            output_dir: 输出目录
            output_csv: 输出CSV路径
            
        Returns:
            mel_dataset_df: MEL特征数据集DataFrame
        """
        from .feature_extractor import MelFeatureExtractor, AudioToMelConverter
        
        # 初始化特征提取器
        extractor = MelFeatureExtractor()
        converter = AudioToMelConverter(extractor)
        
        # 转换数据集
        mel_df = converter.convert_dataset(audio_dir, text_dir, output_dir, output_csv)
        
        # 添加fold信息
        if Path(speaker_folds_csv).exists():
            folds_df = pd.read_csv(speaker_folds_csv)
            mel_df = mel_df.merge(folds_df, on='speaker_id', how='left')
        
        return mel_df
    
    @staticmethod
    def validate_mel_dataset(csv_path: str) -> bool:
        """验证MEL数据集的完整性"""
        try:
            df = pd.read_csv(csv_path)
            
            # 检查必要列
            required_cols = ['speaker_id', 'feature_path']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # 检查文件是否存在
            missing_files = []
            for _, row in df.iterrows():
                if not Path(row['feature_path']).exists():
                    missing_files.append(row['feature_path'])
            
            if missing_files:
                print(f"警告: {len(missing_files)} 个特征文件不存在")
                return False
            
            print(f"✅ MEL数据集验证通过: {len(df)} 个样本")
            return True
            
        except Exception as e:
            print(f"❌ MEL数据集验证失败: {e}")
            return False
