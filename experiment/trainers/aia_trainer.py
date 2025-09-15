"""
MIA训练器
专门用于训练成员推断攻击和属性推断攻击模型
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
import warnings
import pickle
import hashlib

from .base_trainer import BaseExperimentTrainer
from .core_trainer import CoreTrainer
from models.aia_models import create_attacker_model
from utils.representation_extractor import RepresentationExtractor
from data.demographic_extractor import DemographicExtractor


class MIADataset(Dataset):
    """MIA攻击数据集"""
    
    def __init__(
        self, 
        representations: np.ndarray, 
        labels: np.ndarray, 
        filenames: Optional[List[str]] = None,
        features: Optional[np.ndarray] = None
    ):
        self.representations = torch.FloatTensor(representations)
        self.labels = torch.LongTensor(labels)
        self.filenames = filenames or []
        self.features = torch.FloatTensor(features) if features is not None else None
    
    def __len__(self):
        return len(self.representations)
    
    def __getitem__(self, idx):
        # 在features-only模式下，只返回representations和labels
        return self.representations[idx], self.labels[idx]


class AIATrainer(BaseExperimentTrainer):
    """AIA攻击训练器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.representation_extractor = RepresentationExtractor(config.general['device'])
        self.demographic_extractor = DemographicExtractor()
        self.device = config.general['device']
        
        # 验证配置
        config.validate_config()
        
        print(f"AIA Trainer initialized for {config.aia_params['attack_type']} attack")
        print(f"Target model mode: {config.aia_params['target_model_mode']}")
        print(f"Feature type: {config.feature_type}")
    
    def get_cache_key(self) -> str:
        """生成缓存键值，基于关键参数"""
        # 只基于影响representations提取的参数，input_mode不影响representations提取
        key_components = [
            self.config.feature_type,
            self.config.aia_params['target_model_mode'], 
            str(sorted(self.config.aia_params['checkpoint_paths']))
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    def get_cache_path(self) -> str:
        """获取缓存文件路径 - Mix模式复用concatenation缓存"""
        input_mode = self.config.aia_params['input_mode']
        attack_type = self.config.aia_params['attack_type']
        
        # Mix模式使用concatenation的缓存（因为需要相同的数据：features + representations）
        cache_mode = 'concatenation' if input_mode == 'mix' else input_mode
        
        cache_dir = f".cache/{cache_mode}/{attack_type}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = self.get_cache_key()
        return f"{cache_dir}/representations_{cache_key}.pkl"
    
    def save_representations_cache(self, representations_dict: Dict[int, Tuple[np.ndarray, np.ndarray, List[str]]]):
        """保存表征到缓存"""
        cache_path = self.get_cache_path()
        cache_data = {
            'representations_dict': representations_dict,
            'config_info': {
                'feature_type': self.config.feature_type,
                'target_model_mode': self.config.aia_params['target_model_mode'],
                'checkpoint_paths': self.config.aia_params['checkpoint_paths']
            },
            'timestamp': time.time()
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✅ Representations cached to: {cache_path}")
    
    def load_representations_cache(self) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray, List[str]]]]:
        """从缓存加载表征"""
        cache_path = self.get_cache_path()
        
        if not os.path.exists(cache_path):
            print(f"🔍 No cache found at: {cache_path}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 验证缓存配置是否匹配
            config_info = cache_data.get('config_info', {})
            if (config_info.get('feature_type') == self.config.feature_type and
                config_info.get('target_model_mode') == self.config.aia_params['target_model_mode'] and
                config_info.get('checkpoint_paths') == self.config.aia_params['checkpoint_paths']):
                
                timestamp = cache_data.get('timestamp', 0)
                age_hours = (time.time() - timestamp) / 3600
                print(f"✅ Loaded representations from cache (age: {age_hours:.1f}h)")
                return cache_data['representations_dict']
            else:
                print(f"⚠️  Cache config mismatch, will regenerate")
                return None
                
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            return None
    
    def extract_target_representations(self) -> Dict[int, Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]]:
        """从目标模型提取表征 - 按正确的5-fold CV逻辑（支持缓存），包含原始features"""
        print("Extracting representations from target models using correct 5-fold CV...")
        
        # 尝试从缓存加载
        input_mode = self.config.aia_params['input_mode']
        cached_representations = self.load_representations_cache()
        if cached_representations is not None:
            if input_mode == 'mix':
                print("🚀 Mix mode: Using concatenation cache (features + representations), will apply mixing during processing!")
            else:
                print(f"🚀 Using cached representations for {input_mode} mode, skipping extraction!")
            return cached_representations
        
        print("💾 No valid cache found, extracting representations...")
        
        # 读取speaker fold映射
        fold_mapping = self.load_speaker_folds()
        
        representations_dict = {}
        
        # 对每个fold，用对应的checkpoint提取该fold的表征
        for fold_idx in range(5):
            checkpoint_path = self.config.aia_params['checkpoint_paths'][fold_idx]
            print(f"Processing fold {fold_idx+1} with checkpoint: {checkpoint_path}")
            
            # 提取该fold的所有表征（训练+测试）
            fold_representations = self.representation_extractor.extract_representations_from_checkpoint(
                checkpoint_path=checkpoint_path,
                feature_type=self.config.feature_type,
                model_mode=self.config.aia_params['target_model_mode'],
                layer_index=self.config.aia_params['representation_layer'],
                fold_filter=fold_idx + 1,  # fold编号从1开始
                fold_mapping=fold_mapping,
                attack_model_type=self.config.aia_params['attack_model_type'],
                include_features=True  # 启用features提取
            )
            
            if fold_representations is not None:
                representations_dict[fold_idx] = fold_representations
                print(f"Fold {fold_idx+1}: {len(fold_representations[0])} samples extracted")
            else:
                print(f"Warning: No representations extracted for fold {fold_idx+1}")
        
        # 保存到缓存
        if representations_dict:
            self.save_representations_cache(representations_dict)
        
        return representations_dict
    
    def load_speaker_folds(self) -> Dict[str, int]:
        """加载speaker fold映射"""
        import pandas as pd
        
        folds_path = 'data/speaker_folds.csv'
        if not os.path.exists(folds_path):
            raise FileNotFoundError(f"Speaker folds file not found: {folds_path}")
        
        df = pd.read_csv(folds_path)
        # 创建speaker_id到fold的映射
        fold_mapping = {}
        for _, row in df.iterrows():
            speaker_id = str(row['speaker_id']).strip("'")
            fold_mapping[speaker_id] = int(row['fold'])
        
        print(f"Loaded {len(fold_mapping)} speaker fold mappings")
        return fold_mapping
    
    def get_mel_data_directory(self) -> str:
        """获取MEL数据的实际目录路径"""
        if self.config.feature_type == 'mel':
            # MEL数据存储在分层目录结构中
            return 'audio_mel/data-mel/mel_features'
        else:
            # OpenSMILE数据存储在平铺结构中
            return self.config.get_dataset_config()['data_dir']
    
    def collect_mel_files(self, base_dir: str) -> List[str]:
        """收集所有MEL特征文件的路径"""
        import os
        mel_files = []
        
        # 遍历所有speaker目录
        for speaker_dir in os.listdir(base_dir):
            speaker_path = os.path.join(base_dir, speaker_dir)
            if os.path.isdir(speaker_path):
                # 收集该speaker的所有mel文件
                for filename in os.listdir(speaker_path):
                    if filename.endswith('_mel.csv'):
                        full_path = os.path.join(speaker_path, filename)
                        mel_files.append(full_path)
        
        return mel_files
    
    def prepare_demographic_labels(self) -> pd.DataFrame:
        """准备人口统计标签"""
        print("Preparing demographic labels...")
        
        # 检查是否已有标签文件
        labels_path = self.config.aia_params['demographic_labels_path']
        if os.path.exists(labels_path):
            print(f"Loading existing demographic labels from {labels_path}")
            return pd.read_csv(labels_path)
        
        # 创建新的标签文件 - 使用正确的数据目录
        data_dir = self.get_mel_data_directory()
        
        if self.config.feature_type == 'mel':
            # 对于MEL特征，收集所有文件路径
            mel_files = self.collect_mel_files(data_dir)
            # 从文件路径中提取文件名进行demographic解析
            filenames = [os.path.basename(f) for f in mel_files]
            demographic_df = self.demographic_extractor.extract_demographics_from_filenames(filenames)
            # 保存标签文件
            demographic_df.to_csv(labels_path, index=False)
            print(f"Saved demographic labels to {labels_path}")
        else:
            # 对于OpenSMILE特征，使用原有的目录扫描方法
            demographic_df = self.demographic_extractor.create_demographic_labels(
                data_dir=data_dir,
                output_path=labels_path
            )
        
        return demographic_df
    
    def create_attribute_labels(
        self,
        representations_dict: Dict[int, Tuple[np.ndarray, np.ndarray, List[str]]],
        demographic_df: pd.DataFrame,
        attribute_type: str
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        创建属性推断攻击的标签
        
        Args:
            representations_dict: 表征字典
            demographic_df: 人口统计信息DataFrame
            attribute_type: 属性类型 ('gender', 'age_level', 'education')
            
        Returns:
            字典：{fold_id: (representations, original_labels, attribute_labels)}
        """
        attribute_data = {}
        
        # 创建文件名到属性的映射
        filename_to_attr = {}
        for _, row in demographic_df.iterrows():
            filename = row['filename']
            if attribute_type == 'gender':
                attr_value = row['gender_encoded']
            elif attribute_type == 'age_level':
                attr_value = row['age_category']
            elif attribute_type == 'education':
                attr_value = row['education_level']
                # 过滤掉教育等级为None（即'X'）的样本
                if attr_value is None or pd.isna(attr_value):
                    continue
            else:
                raise ValueError(f"Unknown attribute type: {attribute_type}")
            
            filename_to_attr[filename] = attr_value
        
        # 对于教育等级攻击，打印过滤信息
        if attribute_type == 'education':
            total_samples = len(demographic_df)
            valid_samples = len(filename_to_attr)
            excluded_samples = total_samples - valid_samples
            print(f"Education attack: Excluded {excluded_samples} samples with unknown education level (X)")
            print(f"Valid samples for education attack: {valid_samples}")
        
        for fold_id, (representations, original_labels, filenames) in representations_dict.items():
            attribute_labels = []
            valid_indices = []
            
            for i, filename in enumerate(filenames):
                if filename in filename_to_attr:
                    attribute_labels.append(filename_to_attr[filename])
                    valid_indices.append(i)
            
            if not valid_indices:
                print(f"Warning: No valid attribute labels found for fold {fold_id}")
                continue
            
            # 只保留有效的样本
            valid_representations = representations[valid_indices]
            valid_original_labels = original_labels[valid_indices] if len(original_labels) > 0 else np.array([])
            valid_attribute_labels = np.array(attribute_labels)
            
            attribute_data[fold_id] = (valid_representations, valid_original_labels, valid_attribute_labels)
            
            print(f"Fold {fold_id}: {len(valid_indices)} samples with {attribute_type} labels")
        
        return attribute_data
    
    def create_fold_attack_datasets(
        self, 
        representations_dict: Dict[int, Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]],
        demographic_df: pd.DataFrame,
        attribute_type: str
    ) -> Dict[int, Tuple[MIADataset, MIADataset]]:
        """
        创建5-fold攻击数据集 - 支持不同输入模式
        每个fold：用其他4个fold训练攻击模型，在当前fold上测试
        
        Returns:
            字典：{fold_id: (train_dataset, test_dataset)}
        """
        print(f"Creating {attribute_type} attack datasets with correct 5-fold CV logic...")
        
        input_mode = self.config.aia_params['input_mode']
        print(f"🔧 INPUT MODE: {input_mode.upper()}")
        
        if input_mode == 'features_only':
            print("🔥 FEATURES-ONLY MODE: Using original features instead of representations")
        elif input_mode == 'representations_only':
            print("🧠 REPRESENTATIONS-ONLY MODE: Using learned representations only")
        elif input_mode == 'concatenation':
            print("🔗 CONCATENATION MODE: Using concatenated features and representations")
        elif input_mode == 'mix':
            alpha = self.config.aia_params['mix_alpha']
            print(f"🎯 MIX MODE: Normalized features * {alpha} + representations * {1-alpha}")
        
        fold_datasets = {}
        
        for test_fold in range(5):
            print(f"\nPreparing fold {test_fold+1} (test fold)...")
            
            # 收集训练数据（其他4个fold）和测试数据（当前fold）
            train_inputs = []
            train_labels = []
            test_inputs = []
            test_labels = []
            
            for fold_idx in range(5):
                print(f"  Processing fold {fold_idx+1} data...")
                if fold_idx not in representations_dict:
                    print(f"    Fold {fold_idx+1}: No data found, skipping")
                    continue
                    
                representations, original_labels, filenames, features = representations_dict[fold_idx]
                print(f"    Fold {fold_idx+1}: repr={representations.shape}, features={features.shape}, samples={len(filenames)}")
                
                # 检查数据是否可用
                has_representations = representations.size > 0
                has_features = features.size > 0
                
                # 检查并修复VIB模式的shape不匹配问题
                if has_representations and has_features:
                    if len(representations) != len(features):
                        print(f"    Warning: Shape mismatch - repr: {representations.shape}, features: {features.shape}")
                        # 如果representations数量是features的2倍，可能是VIB的采样问题，取前一半
                        if len(representations) == 2 * len(features):
                            print(f"    Fixing VIB sampling issue: taking first half of representations")
                            representations = representations[:len(features)]
                            # 同时需要调整filenames以保持一致
                            if len(filenames) == 2 * len(features):
                                filenames = filenames[:len(features)]
                        else:
                            print(f"    Skipping fold {fold_idx+1} due to unresolvable shape mismatch")
                            continue
                
                # 直接从speaker_id解析属性标签
                fold_attr_labels = []
                valid_indices = []
                print(f"    Parsing {len(filenames)} speaker IDs for {attribute_type}...")
                
                for i, speaker_id in enumerate(filenames):  # 这里实际上是speaker_ids
                    try:
                        if attribute_type == 'gender':
                            # 从speaker_id格式 "29_CF34_3" 中提取性别
                            gender = speaker_id.split('_')[1][1]  # C/P + M/F，取第二个字符
                            attr_value = 1 if gender == 'F' else 0
                        elif attribute_type == 'age_level':
                            # 从speaker_id中提取年龄并分类
                            age_str = speaker_id.split('_')[1][2:]  # 提取年龄部分
                            age = int(age_str)
                            if age <= 30:
                                attr_value = 0  # 年轻组
                            elif age <= 45:
                                attr_value = 1  # 中年组
                            else:
                                attr_value = 2  # 老年组
                        elif attribute_type == 'education':
                            # 从speaker_id中提取教育等级
                            edu_str = speaker_id.split('_')[2]
                            if edu_str.lower() == 'x':  # 未知教育等级
                                continue  # 跳过
                            else:
                                attr_value = int(edu_str) - 1  # 转换为0-based索引
                        else:
                            raise ValueError(f"Unknown attribute type: {attribute_type}")
                        
                        fold_attr_labels.append(attr_value)
                        valid_indices.append(i)
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Failed to parse speaker_id '{speaker_id}': {e}")
                        continue
                
                if not valid_indices:
                    print(f"Warning: No valid {attribute_type} labels for fold {fold_idx+1}")
                    continue
                
                # 准备输入数据根据输入模式
                valid_representations = representations[valid_indices] if has_representations else None
                valid_features = features[valid_indices] if has_features else None
                valid_attr_labels = np.array(fold_attr_labels)
                
                if input_mode == 'features_only':
                    # 只使用features（需要mean pooling对于MLP）
                    if valid_features is not None and has_features:
                        if self.config.aia_params['attack_model_type'] == 'mlp':
                            # MLP需要mean pooling
                            valid_inputs = np.mean(valid_features, axis=1)  # (N, 1000, 80) -> (N, 80)
                        else:
                            # Transformer可以处理序列
                            valid_inputs = valid_features  # (N, 1000, 80)
                    else:
                        print(f"Warning: No features available for fold {fold_idx+1}")
                        continue
                        
                elif input_mode == 'representations_only':
                    # 只使用representations
                    if valid_representations is not None and has_representations:
                        valid_inputs = valid_representations  # (N, repr_dim)
                    else:
                        print(f"Warning: No representations available for fold {fold_idx+1}")
                        continue
                        
                elif input_mode == 'concatenation':
                    # 拼接features和representations
                    print(f"    Starting concatenation mode for fold {fold_idx+1}...")
                    if valid_features is not None and valid_representations is not None and has_features and has_representations:
                        print(f"    Features shape: {valid_features.shape}, Repr shape: {valid_representations.shape}")
                        
                        if self.config.aia_params['attack_model_type'] == 'mlp':
                            # MLP: mean pool features then concatenate with pooled representations
                            print(f"    Computing mean pooling for features...")
                            pooled_features = np.mean(valid_features, axis=1)  # (N, 80)
                            print(f"    Pooled features shape: {pooled_features.shape}")
                            
                            # Pool representations if they have time dimension
                            if len(valid_representations.shape) == 3:
                                print(f"    Computing mean pooling for representations...")
                                pooled_representations = np.mean(valid_representations, axis=1)  # (N, 80)
                                print(f"    Pooled representations shape: {pooled_representations.shape}")
                            else:
                                pooled_representations = valid_representations  # Already pooled
                            
                            print(f"    Concatenating pooled features with representations...")
                            valid_inputs = np.concatenate([pooled_features, pooled_representations], axis=1)  # (N, 160)
                            print(f"    Concatenated shape: {valid_inputs.shape}")
                        else:
                            # Transformer: concatenate along feature dimension for each time step
                            print(f"    Broadcasting representations for transformer...")
                            
                            # If representations have time dimension, use as is; otherwise broadcast
                            if len(valid_representations.shape) == 3:
                                # Representations already have time dimension (N, seq_len, repr_dim)
                                if valid_representations.shape[1] == valid_features.shape[1]:
                                    # Same sequence length, use directly
                                    print(f"    Using time-sequence representations directly...")
                                    valid_inputs = np.concatenate([valid_features, valid_representations], axis=2)  # (N, 1000, 160)
                                else:
                                    # Different sequence lengths, use pooled representations and broadcast
                                    print(f"    Pooling and broadcasting representations...")
                                    pooled_repr = np.mean(valid_representations, axis=1)  # (N, 80)
                                    repr_broadcast = np.expand_dims(pooled_repr, axis=1)  # (N, 1, 80)
                                    repr_broadcast = np.repeat(repr_broadcast, valid_features.shape[1], axis=1)  # (N, 1000, 80)
                                    valid_inputs = np.concatenate([valid_features, repr_broadcast], axis=2)  # (N, 1000, 160)
                            else:
                                # Representations are pooled (N, repr_dim), need to broadcast
                                print(f"    Broadcasting pooled representations...")
                                repr_broadcast = np.expand_dims(valid_representations, axis=1)  # (N, 1, 80)
                                repr_broadcast = np.repeat(repr_broadcast, valid_features.shape[1], axis=1)  # (N, 1000, 80)
                                valid_inputs = np.concatenate([valid_features, repr_broadcast], axis=2)  # (N, 1000, 160)
                            
                            print(f"    Final concatenated shape: {valid_inputs.shape}")
                    else:
                        print(f"Warning: Missing data for concatenation in fold {fold_idx+1}")
                        continue
                        
                elif input_mode == 'mix':
                    # Mix模式：归一化后的features和representations加权混合
                    print(f"    Starting mix mode for fold {fold_idx+1}...")
                    if valid_features is not None and valid_representations is not None and has_features and has_representations:
                        alpha = self.config.aia_params['mix_alpha']
                        print(f"    Mix alpha: {alpha}")
                        print(f"    Features shape: {valid_features.shape}, Repr shape: {valid_representations.shape}")
                        
                        if self.config.aia_params['attack_model_type'] == 'mlp':
                            # MLP: 先mean pool features，然后归一化并混合
                            print(f"    Computing mean pooling for MLP...")
                            pooled_features = np.mean(valid_features, axis=1)  # (N, 80)
                            print(f"    Pooled features shape: {pooled_features.shape}")
                            
                            # Pool representations if they have time dimension
                            if len(valid_representations.shape) == 3:
                                print(f"    Computing mean pooling for representations...")
                                pooled_representations = np.mean(valid_representations, axis=1)  # (N, 80)
                                print(f"    Pooled representations shape: {pooled_representations.shape}")
                            else:
                                pooled_representations = valid_representations  # Already pooled
                            
                            print(f"    Computing L2 normalization...")
                            # L2归一化
                            pooled_features_norm = pooled_features / (np.linalg.norm(pooled_features, axis=1, keepdims=True) + 1e-8)
                            representations_norm = pooled_representations / (np.linalg.norm(pooled_representations, axis=1, keepdims=True) + 1e-8)
                            print(f"    Normalized shapes - features: {pooled_features_norm.shape}, repr: {representations_norm.shape}")
                            
                            print(f"    Computing weighted mix...")
                            # 加权混合
                            valid_inputs = alpha * pooled_features_norm + (1 - alpha) * representations_norm  # (N, 80)
                            print(f"    Mixed shape: {valid_inputs.shape}")
                        else:
                            # Transformer: 对每个时间步进行归一化和混合
                            print(f"    Processing transformer mix mode...")
                            
                            # Handle different representation shapes
                            if len(valid_representations.shape) == 3:
                                # Representations have time dimension
                                if valid_representations.shape[1] == valid_features.shape[1]:
                                    # Same sequence length, use directly
                                    print(f"    Using time-sequence representations for mix...")
                                    # L2归一化（沿着特征维度）
                                    features_norm = valid_features / (np.linalg.norm(valid_features, axis=2, keepdims=True) + 1e-8)  # (N, 1000, 80)
                                    repr_norm = valid_representations / (np.linalg.norm(valid_representations, axis=2, keepdims=True) + 1e-8)  # (N, 1000, 80)
                                    valid_inputs = alpha * features_norm + (1 - alpha) * repr_norm  # (N, 1000, 80)
                                else:
                                    # Different sequence lengths, pool and broadcast
                                    print(f"    Pooling representations for mix...")
                                    pooled_repr = np.mean(valid_representations, axis=1)  # (N, 80)
                                    repr_broadcast = np.expand_dims(pooled_repr, axis=1)  # (N, 1, 80)
                                    repr_broadcast = np.repeat(repr_broadcast, valid_features.shape[1], axis=1)  # (N, 1000, 80)
                                    
                                    # L2归一化
                                    features_norm = valid_features / (np.linalg.norm(valid_features, axis=2, keepdims=True) + 1e-8)
                                    repr_broadcast_norm = repr_broadcast / (np.linalg.norm(repr_broadcast, axis=2, keepdims=True) + 1e-8)
                                    valid_inputs = alpha * features_norm + (1 - alpha) * repr_broadcast_norm  # (N, 1000, 80)
                            else:
                                # Representations are pooled, need to broadcast
                                print(f"    Broadcasting pooled representations for transformer mix...")
                                repr_broadcast = np.expand_dims(valid_representations, axis=1)  # (N, 1, 80)
                                repr_broadcast = np.repeat(repr_broadcast, valid_features.shape[1], axis=1)  # (N, 1000, 80)
                                print(f"    Broadcasted repr shape: {repr_broadcast.shape}")
                                
                                print(f"    Computing L2 normalization for transformer...")
                                # L2归一化（沿着特征维度）
                                features_norm = valid_features / (np.linalg.norm(valid_features, axis=2, keepdims=True) + 1e-8)  # (N, 1000, 80)
                                repr_broadcast_norm = repr_broadcast / (np.linalg.norm(repr_broadcast, axis=2, keepdims=True) + 1e-8)  # (N, 1000, 80)
                                print(f"    Normalized shapes - features: {features_norm.shape}, repr: {repr_broadcast_norm.shape}")
                                
                                print(f"    Computing weighted mix...")
                                # 加权混合
                                valid_inputs = alpha * features_norm + (1 - alpha) * repr_broadcast_norm  # (N, 1000, 80)
                            
                            print(f"    Mixed shape: {valid_inputs.shape}")
                    else:
                        print(f"Warning: Missing data for mix mode in fold {fold_idx+1}")
                        continue
                else:
                    raise ValueError(f"Unknown input mode: {input_mode}")
                
                # 统计当前fold的属性分布
                unique_labels, counts = np.unique(valid_attr_labels, return_counts=True)
                print(f"  Fold {fold_idx+1} {attribute_type} distribution: {dict(zip(unique_labels, counts))}")
                
                print(f"    Processing input mode '{input_mode}' for fold {fold_idx+1}...")
                print(f"    Input shape: {valid_inputs.shape if valid_inputs is not None else 'None'}")
                print(f"    Label shape: {valid_attr_labels.shape}")
                
                # 分配到训练集或测试集
                if fold_idx == test_fold:
                    # 当前fold作为测试集
                    test_inputs.append(valid_inputs)
                    test_labels.append(valid_attr_labels)
                    print(f"  Test fold {fold_idx+1}: {len(valid_inputs)} samples ({input_mode})")
                else:
                    # 其他fold作为训练集
                    train_inputs.append(valid_inputs)
                    train_labels.append(valid_attr_labels)
                    print(f"  Train fold {fold_idx+1}: {len(valid_inputs)} samples ({input_mode})")
            
            # 合并训练数据和测试数据
            print(f"  Starting data concatenation...")
            print(f"  Train inputs count: {len(train_inputs)}, Test inputs count: {len(test_inputs)}")
            
            if train_inputs and test_inputs:
                print(f"  Concatenating train labels...")
                train_y = np.concatenate(train_labels)
                print(f"  Concatenating test labels...")
                test_y = np.concatenate(test_labels)
                print(f"  Labels concatenated - Train: {train_y.shape}, Test: {test_y.shape}")
                
                # 处理输入数据
                if train_inputs:
                    try:
                        print(f"  Stacking train inputs...")
                        train_X = np.vstack(train_inputs)
                        print(f"  Train inputs: {train_X.shape}")
                    except Exception as e:
                        print(f"  Warning: Failed to combine train inputs: {e}")
                        continue
                        
                if test_inputs:
                    try:
                        print(f"  Stacking test inputs...")
                        test_X = np.vstack(test_inputs)
                        print(f"  Test inputs: {test_X.shape}")
                    except Exception as e:
                        print(f"  Warning: Failed to combine test inputs: {e}")
                        continue
                
                print(f"  Final - Train: {len(train_X)}, Test: {len(test_X)}")
                print(f"  {input_mode.capitalize()}: Input dim = {train_X.shape[-1]}")
                
                # 创建数据集
                print(f"  Creating MIA datasets...")
                print(f"  Train data shape: {train_X.shape}, labels: {train_y.shape}")
                print(f"  Test data shape: {test_X.shape}, labels: {test_y.shape}")
                
                train_dataset = MIADataset(train_X, train_y)
                test_dataset = MIADataset(test_X, test_y)
                print(f"  Datasets created successfully")
                
                fold_datasets[test_fold] = (train_dataset, test_dataset)
                print(f"  Fold {test_fold+1} dataset saved to fold_datasets")
            else:
                print(f"Warning: No data available for fold {test_fold+1}")
        
        return fold_datasets    
    def create_filename_to_attribute_mapping(self, demographic_df: pd.DataFrame, attribute_type: str) -> Dict[str, int]:
        """创建文件名到属性值的映射"""
        filename_to_attr = {}
        for _, row in demographic_df.iterrows():
            filename = row['filename']
            if attribute_type == 'gender':
                attr_value = row['gender_encoded']
            elif attribute_type == 'age_level':
                attr_value = row['age_category']
            elif attribute_type == 'education':
                attr_value = row['education_level']
                # 过滤掉教育等级为None（即'X'）的样本
                if attr_value is None or pd.isna(attr_value):
                    continue
            else:
                raise ValueError(f"Unknown attribute type: {attribute_type}")
            
            filename_to_attr[filename] = attr_value
        
        return filename_to_attr
    
    def create_balanced_sampler(self, labels: torch.Tensor) -> WeightedRandomSampler:
        """创建平衡采样器"""
        # 计算每个类别的样本数
        class_counts = torch.bincount(labels)
        print(f"Class distribution: {class_counts.tolist()}")
        
        # 计算每个类别的权重（归一化的逆频率）
        total_samples = len(labels)
        num_classes = len(class_counts)
        
        # 权重 = 总样本数 / (类别数 * 该类别样本数)
        class_weights = total_samples / (num_classes * class_counts.float())
        
        # 为每个样本分配权重
        sample_weights = class_weights[labels]
        
        # 创建加权随机采样器
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"Class weights (normalized): {class_weights.tolist()}")
        print(f"Class weights sum: {class_weights.sum().item():.4f}")
        return sampler

    def train_single_fold_attacker(
        self, 
        fold_id: int, 
        train_dataset: MIADataset, 
        test_dataset: MIADataset
    ) -> Dict[str, float]:
        """训练单个fold的攻击模型"""
        print(f"\nTraining attacker for test fold {fold_id+1}")
        
        # 创建平衡采样器
        train_sampler = self.create_balanced_sampler(train_dataset.labels)
        
        # 创建数据加载器（使用平衡采样）
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.data['batch_size'],
            sampler=train_sampler  # 使用采样器，不能同时用shuffle
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.data['batch_size'], 
            shuffle=False
        )
        
        # 创建攻击模型
        attack_model_config = self.config.get_attack_model_config()
        
        # 计算特征维度（features-only模式）
        feature_dim = 0
        if hasattr(train_dataset, 'representations') and train_dataset.representations is not None:
            # 在features-only模式下，features数据存储在representations字段中
            sample_input = train_dataset.representations[0]
            input_mode = self.config.aia_params['input_mode']
            
            if sample_input.dim() == 2:  # (seq_len, feature_dim)
                if self.config.aia_params['attack_model_type'] == 'transformer':
                    # Transformer模型：使用完整时序特征
                    feature_dim = sample_input.shape[-1]
                    print(f"{input_mode.capitalize()} mode: Using {sample_input.shape} time-series features for Transformer")
                else:
                    # MLP模型：需要mean pooling
                    feature_dim = sample_input.shape[-1]  
                    print(f"{input_mode.capitalize()} mode: Using {sample_input.shape} time-series features (will be mean-pooled for MLP)")
            elif sample_input.dim() == 1:  # (feature_dim,)
                feature_dim = sample_input.shape[0]
                print(f"{input_mode.capitalize()} mode: Using {feature_dim}D flattened features")
            
        # 检查是否真的有features数据
        has_features = feature_dim > 0
        if not has_features:
            print("No features detected, this shouldn't happen in features-only mode!")
            feature_dim = 80  # fallback
        
        model = create_attacker_model(
            attack_type=self.config.aia_params['attack_type'],
            representation_dim=feature_dim,  # 使用feature_dim作为输入维度
            model_type=attack_model_config['model_type'],
            hidden_dims=attack_model_config.get('hidden_dims', [256, 128]),
            dropout=attack_model_config['dropout'],
            feature_dim=0,  # features-only模式下不需要额外的feature concatenation
            **{k: v for k, v in attack_model_config.items() 
               if k.startswith(('d_model', 'nhead', 'num_layers', 'dim_feedforward'))}
        ).to(self.device)
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.training['lr'],
            weight_decay=self.config.training.get('weight_decay', 1e-5)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Step-based训练参数（从配置读取）
        total_steps = self.config.training.get('total_steps', 300)
        eval_every_steps = self.config.training.get('eval_every_steps', 100)
        
        # 训练循环
        model.train()
        step = 0
        train_loader_iter = iter(train_loader)
        
        print(f"Starting step-based training: {total_steps} total steps, evaluate every {eval_every_steps} steps")
        
        while step < total_steps:
            step_loss = 0.0
            step_correct = 0
            step_total = 0
            
            # 每100步作为一个evaluation period
            for _ in range(min(eval_every_steps, total_steps - step)):
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    # 重新创建迭代器
                    train_loader_iter = iter(train_loader)
                    batch = next(train_loader_iter)
                
                # 解析batch数据 - features-only模式：只有input_features和labels
                input_features, labels = batch
                input_features, labels = input_features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Features-only模式：直接使用input_features
                outputs = model(input_features)
                    
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                step_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                step_total += labels.size(0)
                step_correct += (predicted == labels).sum().item()
                step += 1
            
            # 计算当前period的训练指标
            train_acc = 100 * step_correct / step_total if step_total > 0 else 0
            train_loss = step_loss / min(eval_every_steps, total_steps - (step - eval_every_steps))
            
            # 评估测试集
            test_metrics = self.evaluate_model(model, test_loader, has_features)
            test_acc = test_metrics['accuracy']
            
            print(f"Step {step:3d}/{total_steps}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.4f}, Test F1: {test_metrics['f1']:.4f}")
        
        # 返回最终结果（固定轮数）
        final_metrics = self.evaluate_model(model, test_loader, has_features)
        print(f"Test Fold {fold_id+1} Final Results: Acc={final_metrics['accuracy']:.4f}, "
              f"F1={final_metrics['f1']:.4f}, AUC={final_metrics.get('auc', 0):.4f}")
        
        return final_metrics
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, has_features: bool = False) -> Dict[str, float]:
        """评估模型性能"""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 解析batch数据 - features-only模式：只有input_features和labels
                input_features, labels = batch
                input_features, labels = input_features.to(self.device), labels.to(self.device)
                
                # Features-only模式：直接使用input_features
                outputs = model(input_features)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # 计算AUC（仅对二分类）
        if self.config.model['num_classes'] == 2:
            all_probs_array = np.array(all_probs)
            auc = roc_auc_score(all_labels, all_probs_array[:, 1])
            metrics['auc'] = auc
        
        return metrics
    
    def run_attack_experiment(self) -> Dict[str, float]:
        """运行完整的AIA攻击实验 - 正确的5-fold CV逻辑"""
        print(f"\n{'='*50}")
        print(f"Starting {self.config.aia_params['attack_type']} attack experiment")
        print(f"Target model: {self.config.aia_params['target_model_mode']}")
        print(f"Feature type: {self.config.feature_type}")
        print(f"Attack logic: 5-fold Cross-Validation")
        print(f"{'='*50}")
        
        # 1. 提取目标模型表征（按fold组织）
        representations_dict = self.extract_target_representations()
        
        # 2. 准备人口统计标签
        demographic_df = self.prepare_demographic_labels()
        
        # 3. 创建5-fold攻击数据集（正确的CV逻辑）
        fold_datasets = self.create_fold_attack_datasets(
            representations_dict, demographic_df, self.config.aia_params['attack_type']
        )
        
        # 4. 对每个fold执行攻击实验
        fold_results = {}
        for test_fold in range(5):
            if test_fold not in fold_datasets:
                print(f"\nSkipping fold {test_fold+1} - insufficient data")
                continue
                
            train_dataset, test_dataset = fold_datasets[test_fold]
            print(f"\n{'='*30}")
            print(f"Fold {test_fold+1}: Training attacker on 4 folds, testing on fold {test_fold+1}")
            print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
            print(f"{'='*30}")
            
            fold_results[test_fold] = self.train_single_fold_attacker(
                test_fold, train_dataset, test_dataset
            )
        
        # 5. 计算平均结果
        if fold_results:
            avg_results = self.compute_average_results(fold_results)
            self.print_final_results(avg_results)
            return avg_results
        else:
            print("No valid results obtained")
            return {}
    
    def compute_average_results(self, fold_results: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """计算所有fold的平均结果"""
        if not fold_results:
            return {}
        
        metrics = list(fold_results[list(fold_results.keys())[0]].keys())
        avg_results = {}
        
        for metric in metrics:
            values = [fold_results[fold][metric] for fold in fold_results]
            avg_results[f'{metric}_mean'] = np.mean(values)
            avg_results[f'{metric}_std'] = np.std(values)
        
        return avg_results
    
    def print_final_results(self, results: Dict[str, float]):
        """打印最终结果"""
        print(f"\n{'='*50}")
        print(f"Final Results ({self.config.aia_params['attack_type']} attack)")
        print(f"{'='*50}")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if 'auc_mean' in results:
            metrics.append('auc')
        
        for metric in metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in results and std_key in results:
                print(f"{metric.capitalize()}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")


def main():
    """测试函数"""
    from configs.aia_config import AIAConfig
    
    # 创建测试配置
    config = AIAConfig()
    config.feature_type = 'opensmile'
    config.set_attack_type('gender')
    config.set_target_model_mode('normal')
    config.set_checkpoint_paths([
        'checkpoints/normal_fold_0.pth',
        'checkpoints/normal_fold_1.pth',
        'checkpoints/normal_fold_2.pth',
        'checkpoints/normal_fold_3.pth',
        'checkpoints/normal_fold_4.pth',
    ])
    
    try:
        trainer = AIATrainer(config)
        results = trainer.run_attack_experiment()
        print("AIA experiment completed successfully")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()