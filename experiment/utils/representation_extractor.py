"""
表征提取工具
从训练好的模型中自动提取隐层表征，用于MIA攻击
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import DataLoader

from models.transformer import TransformerClassifier
from models.vib_model import VIBTransformer
from configs.config_factory import ConfigFactory
from data.unified_dataset import create_unified_dataloader


class RepresentationExtractor:
    """从训练好的模型中提取隐层表征的工具类"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}  # 缓存加载的模型
        self.representations_cache = {}  # 缓存提取的表征
    
    def load_model_from_checkpoint(
        self, 
        checkpoint_path: str, 
        feature_type: str,
        model_mode: str,
        **model_kwargs
    ) -> nn.Module:
        """
        从检查点加载模型
        
        Args:
            checkpoint_path: 模型检查点路径
            feature_type: 特征类型 ('opensmile' 或 'mel')
            model_mode: 模型训练模式 ('normal', 'dp', 'vib', 'mine')
            **model_kwargs: 额外的模型参数
            
        Returns:
            加载的模型
        """
        # 创建配置以获取模型参数
        config = ConfigFactory.create_config(feature_type, model_mode, **model_kwargs)
        
        # 根据模式创建模型
        if model_mode == 'vib':
            from models.vib_model import VIBConfig
            vib_cfg = VIBConfig()
            model = VIBTransformer(
                d_model=config.get_model_input_dim(),
                nhead=config.model['nhead'],
                num_layers=config.model['num_layers'],
                num_classes=config.model['num_classes'],
                dim_feedforward=config.model['dim_feedforward'],
                dropout=config.model.get('dropout', 0.1),
                vib_cfg=vib_cfg
            )
        else:
            # Normal、DP 和 MINE 模式使用相同的模型架构
            # MINE模式的检查点中主模型部分与normal/dp完全相同
            model = TransformerClassifier(
                d_model=config.get_model_input_dim(),
                nhead=config.model['nhead'],
                num_layers=config.model['num_layers'],
                num_classes=config.model['num_classes'],
                dim_feedforward=config.model['dim_feedforward'],
                dropout=config.model['dropout']
            )
        
        # 加载检查点 - 简化MINE模式处理
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                print("Loaded with weights_only=False")
            except Exception as e2:
                print(f"Failed to load checkpoint: {e2}")
                raise e2
        
        # 处理不同的检查点格式
        try:
            if 'main_model_state_dict' in checkpoint:
                # MINE模式的旧格式检查点
                state_dict = checkpoint['main_model_state_dict']
                print("Loaded from main_model_state_dict")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Loaded from model_state_dict")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Loaded from state_dict")
            else:
                # 对于直接保存的state_dict（新的MINE模式）
                state_dict = checkpoint
                print("Loaded as direct state_dict")
        except Exception as e:
            print(f"Error processing checkpoint format: {e}")
            # 如果处理checkpoint格式失败，尝试直接使用checkpoint作为state_dict
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
        
        # 处理MINE模式的状态字典加载
        if model_mode == 'mine':
            try:
                # 新的MINE模式：直接保存的encoder state_dict
                if isinstance(checkpoint, dict) and all(key.startswith(('encoder.', 'layers.', 'norm.', 'fc.')) for key in checkpoint.keys()):
                    # 这是encoder-only的state_dict，直接加载到model.encoder
                    model.encoder.load_state_dict(checkpoint, strict=False)
                    print("✅ Loaded MINE encoder-only checkpoint")
                elif 'main_model_state_dict' in checkpoint:
                    # 旧的MINE格式：包含完整模型的checkpoint
                    state_dict = checkpoint['main_model_state_dict']
                    model.load_state_dict(state_dict, strict=False)
                    print("✅ Loaded MINE checkpoint from main_model_state_dict")
                else:
                    # 尝试直接加载
                    model.load_state_dict(checkpoint, strict=False)
                    print("✅ Loaded MINE checkpoint directly")
            except Exception as e:
                print(f"❌ Failed to load MINE checkpoint: {e}")
                raise e
        else:
            # 处理其他模式（normal, dp, vib）的状态字典
            if model_mode == 'dp':
                # 对于DP模型，使用更宽松的加载方式
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("⚠️  DP model loaded with strict=False (some keys may be missing)")
                except Exception as e:
                    print(f"❌ Failed to load DP checkpoint: {e}")
                    # 尝试使用eval_encoder的参数
                    if any(key.startswith('eval_encoder.') for key in state_dict.keys()):
                        print("🔄 Trying to load from eval_encoder...")
                        eval_state_dict = {}
                        for key, value in state_dict.items():
                            if key.startswith('eval_encoder.'):
                                new_key = key.replace('eval_encoder.', 'encoder.')
                                eval_state_dict[new_key] = value
                        model.load_state_dict(eval_state_dict, strict=False)
                        print("✅ Loaded from eval_encoder parameters")
                    else:
                        raise e
            else:
                # 对于normal和vib模式，使用严格加载
                model.load_state_dict(state_dict)
                print(f"✅ Loaded {model_mode} checkpoint")
        
        model.to(self.device)
        model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        return model
    
    def extract_representations_from_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_index: int = -2,
        model_mode: str = 'normal',
        attack_model_type: str = 'mlp',
        include_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        从模型中提取表征
        
        Args:
            model: 已加载的模型
            dataloader: 数据加载器
            layer_index: 要提取的层索引（-1为输出层，-2为倒数第二层）
            model_mode: 模型模式，用于确定提取策略
            attack_model_type: 攻击模型类型 ('mlp' 或 'transformer')
            include_features: 是否同时返回原始features
            
        Returns:
            (representations, labels, filenames, features): 表征数组、标签数组、文件名列表、原始特征数组
        """
        representations = []
        labels = []
        filenames = []
        original_features = [] if include_features else None
        
        # 获取目标层用于hook
        target_layer = self._get_target_layer(model, layer_index)
        if target_layer is None:
            raise RuntimeError(f"Cannot find target layer at index {layer_index}")
            
        # 注册hook来提取中间层表征
        activations = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations.append(output[0].detach().cpu())
            else:
                activations.append(output.detach().cpu())
        
        hook = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    # 处理不同的批次格式
                    if len(batch) == 6:  # 新格式：(features, labels, padding_mask, speaker_ids, asr_labels, filenames)
                        features, batch_labels, padding_mask, speaker_ids, asr_labels, batch_filenames = batch
                    elif len(batch) == 5:  # MEL格式：(features, labels, padding_mask, speaker_ids, asr_labels)
                        features, batch_labels, padding_mask, speaker_ids, asr_labels = batch
                        batch_filenames = speaker_ids  # 使用speaker_ids作为filenames
                    elif len(batch) == 3:
                        features, batch_labels, batch_filenames = batch
                    else:
                        features, batch_labels = batch
                        batch_filenames = [f"sample_{i}" for i in range(len(batch_labels))]
                    
                    features = features.to(self.device)
                    
                    # 保存原始features（如果需要）
                    if include_features:
                        original_features.append(features.detach().cpu().numpy())
                    
                    # 直接获取pooled_output，避免时序复杂性
                    if model_mode == 'vib':
                        # VIB模型返回 (logits, pooled, vib_out)
                        outputs = model(features)
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            logits, pooled_output = outputs[0], outputs[1]
                            # 使用pooled_output作为representations
                            batch_representations = pooled_output.detach().cpu()
                            print(f"    VIB batch - features: {features.shape}, pooled: {pooled_output.shape}, labels: {len(batch_labels)}")
                            representations.append(batch_representations.numpy())
                            labels.extend(batch_labels.cpu().numpy() if torch.is_tensor(batch_labels) else batch_labels)
                            filenames.extend(batch_filenames)
                        else:
                            print(f"Warning: Unexpected VIB model output format: {type(outputs)}")
                            continue
                    else:
                        # 对于TransformerClassifier，根据攻击模型类型选择特征
                        if hasattr(model, 'encoder'):
                            if attack_model_type == 'transformer':
                                # Transformer攻击模型：使用encoder的时序输出
                                transformer_output = model.encoder(features)
                                # transformer_output shape: (batch_size, seq_len, d_model)
                                batch_representations = transformer_output.detach().cpu()
                                representations.append(batch_representations.numpy())
                                labels.extend(batch_labels.cpu().numpy() if torch.is_tensor(batch_labels) else batch_labels)
                                filenames.extend(batch_filenames)
                                continue  # 跳过hook处理
                            else:
                                # MLP攻击模型：使用pooled_output
                                outputs = model(features)
                                if isinstance(outputs, tuple):
                                    logits, pooled_output = outputs
                                    # pooled_output shape: (batch_size, d_model)
                                    batch_representations = pooled_output.detach().cpu()
                                    representations.append(batch_representations.numpy())
                                    labels.extend(batch_labels.cpu().numpy() if torch.is_tensor(batch_labels) else batch_labels)
                                    filenames.extend(batch_filenames)
                                    continue  # 跳过hook处理
                                else:
                                    logits = outputs
                        else:
                            outputs = model(features)
                            if isinstance(outputs, tuple):
                                logits, pooled_output = outputs
                            else:
                                logits = outputs
                    
                    # 处理提取到的表征
                    if activations:
                        batch_representations = activations[-1]  # 获取最新的激活
                        
                        # 处理不同的表征格式
                        if batch_representations.dim() == 3:
                            # (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
                            batch_representations = batch_representations.mean(dim=1)
                        elif batch_representations.dim() == 2:
                            # (batch, hidden_dim) - 已经是正确格式
                            pass
                        else:
                            batch_representations = batch_representations.flatten(1)
                        
                        representations.append(batch_representations.numpy())
                        labels.extend(batch_labels.cpu().numpy() if torch.is_tensor(batch_labels) else batch_labels)
                        filenames.extend(batch_filenames)
                        
                        activations.clear()  # 清理激活
        
        finally:
            hook.remove()  # 移除hook
        
        if representations:
            representations = np.vstack(representations)
            labels = np.array(labels)
            features_array = np.vstack(original_features) if include_features and original_features else np.array([])
            print(f"Extracted representations shape: {representations.shape}")
            if include_features and features_array.size > 0:
                print(f"Extracted features shape: {features_array.shape}")
            return representations, labels, filenames, features_array
        else:
            raise RuntimeError("Failed to extract representations")
    
    def _get_target_layer(self, model: nn.Module, layer_index: int) -> Optional[nn.Module]:
        """获取目标层用于hook - 简化版本，直接使用encoder输出"""
        # For all Transformer models, use the encoder output directly
        if hasattr(model, 'encoder'):
            return model.encoder
        elif hasattr(model, 'transformer'):
            return model.transformer
        elif hasattr(model, 'backbone'):
            return model.backbone
        else:
            raise ValueError(f"Cannot find encoder/transformer in model of type {type(model)}. "
                           f"Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")

    def extract_representations_from_checkpoint(
        self,
        checkpoint_path: str,
        feature_type: str,
        model_mode: str,
        layer_index: int = -2,
        fold_filter: Optional[int] = None,
        fold_mapping: Optional[Dict[str, int]] = None,
        attack_model_type: str = 'mlp',
        include_features: bool = True,
        **kwargs
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]]:
        """
        从单个检查点提取表征 - 支持fold过滤
        
        Returns:
            (representations, labels, filenames, features) 或 None
        """
        try:
            # 加载模型
            model = self.load_model_from_checkpoint(checkpoint_path, feature_type, model_mode, **kwargs)
            
            # 创建数据加载器
            config = ConfigFactory.create_config(feature_type, model_mode)
            _, eval_loader = create_unified_dataloader(
                config, 
                return_full_dataset=True,
                fold_filter=fold_filter,
                fold_mapping=fold_mapping
            )
            
            if eval_loader is None:
                return None
            
            # 提取表征
            return self.extract_representations_from_model(
                model, eval_loader, layer_index, model_mode, attack_model_type, include_features
            )
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def extract_representations_from_checkpoints(
        self,
        checkpoint_paths: List[str],
        feature_type: str,
        model_mode: str,
        layer_index: int = -2,
        data_dir: str = None,
        attack_model_type: str = 'mlp',
        include_features: bool = True,
        **kwargs
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]]:
        """
        从多个fold的检查点批量提取表征
        
        Args:
            checkpoint_paths: 检查点路径列表
            feature_type: 特征类型
            model_mode: 模型模式
            layer_index: 要提取的层索引
            data_dir: 数据目录
            attack_model_type: 攻击模型类型
            include_features: 是否包含原始features
            **kwargs: 额外参数
            
        Returns:
            字典：{fold_id: (representations, labels, filenames, features)}
        """
        all_representations = {}
        
        for fold_id, checkpoint_path in enumerate(checkpoint_paths):
            print(f"\nProcessing fold {fold_id}: {checkpoint_path}")
            
            # 检查检查点是否存在
            if not Path(checkpoint_path).exists():
                print(f"Warning: Checkpoint {checkpoint_path} not found, skipping")
                continue
            
            # 加载模型
            model = self.load_model_from_checkpoint(
                checkpoint_path, feature_type, model_mode, **kwargs
            )
            
            # 创建数据加载器
            # 这里我们需要为当前fold创建测试集
            if data_dir is None:
                if feature_type == 'opensmile':
                    data_dir = 'data/extracted_features_train'
                elif feature_type == 'mel':
                    data_dir = 'audio_mel/extracted_features'
            
            # 创建配置
            config = ConfigFactory.create_config(feature_type, model_mode, **kwargs)
            
            # 创建dataloader，这里使用所有数据（后续在MIA trainer中进行成员/非成员划分）
            dataloader = create_unified_dataloader(
                data_dir=data_dir,
                speaker_folds_csv=config.data['speaker_folds_csv'],
                feature_type=feature_type,
                current_fold=fold_id,
                batch_size=config.data['batch_size'],
                is_train=False,  # 使用测试模式，不进行数据增强
                return_filenames=True
            )
            
            # 提取表征
            result = self.extract_representations_from_model(
                model, dataloader, layer_index, model_mode, attack_model_type, include_features
            )
            
            if result:
                all_representations[fold_id] = result
            
            # 清理模型以释放内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return all_representations
    
    def save_representations(
        self, 
        representations_dict: Dict[int, Tuple[np.ndarray, np.ndarray, List[str]]],
        output_dir: str,
        prefix: str = "representations"
    ):
        """
        保存提取的表征到文件
        
        Args:
            representations_dict: 表征字典
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fold_id, (representations, labels, filenames) in representations_dict.items():
            # 保存为npz格式
            npz_file = output_path / f"{prefix}_fold_{fold_id}.npz"
            np.savez(
                npz_file,
                representations=representations,
                labels=labels,
                filenames=filenames
            )
            
            # 保存为CSV格式（便于查看和调试）
            csv_file = output_path / f"{prefix}_fold_{fold_id}.csv"
            df = pd.DataFrame(representations)
            df['label'] = labels
            df['filename'] = filenames
            df.to_csv(csv_file, index=False)
            
            print(f"Saved fold {fold_id} representations to {npz_file} and {csv_file}")
    
    def load_representations(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        从文件加载表征
        
        Args:
            file_path: npz文件路径
            
        Returns:
            (representations, labels, filenames)
        """
        data = np.load(file_path, allow_pickle=True)
        return data['representations'], data['labels'], data['filenames'].tolist()


def main():
    """测试函数"""
    extractor = RepresentationExtractor()
    
    # 测试配置
    checkpoint_paths = [
        'checkpoints/normal_fold_0.pth',
        'checkpoints/normal_fold_1.pth', 
        'checkpoints/normal_fold_2.pth',
        'checkpoints/normal_fold_3.pth',
        'checkpoints/normal_fold_4.pth',
    ]
    
    feature_type = 'opensmile'
    model_mode = 'normal'
    
    try:
        # 提取表征
        representations_dict = extractor.extract_representations_from_checkpoints(
            checkpoint_paths=checkpoint_paths,
            feature_type=feature_type,
            model_mode=model_mode,
            layer_index=-2
        )
        
        # 保存表征
        extractor.save_representations(
            representations_dict,
            output_dir='representations',
            prefix=f'{model_mode}_{feature_type}'
        )
        
        print(f"Successfully extracted representations from {len(representations_dict)} folds")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()