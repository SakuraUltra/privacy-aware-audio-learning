"""
MINE (Mutual Information Neural Estimation) 隐私保护配置类
"""

import torch
from typing import Dict, Any, Optional

try:
    from .base_config import BaseConfig
except ImportError:
    # 如果相对导入失败，尝试绝对导入（用于直接运行测试）
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from configs.base_config import BaseConfig


class MINEConfig(BaseConfig):
    """MINE隐私保护专用配置类"""
    
    def __init__(self):
        super().__init__()
        
        # MINE特定配置
        self.mine_params = {
            # MINE模型类型：'timeaware' 或 'standard'
            'mine_model_type': 'timeaware',
            
            # 隐私权重参数
            'privacy_weight': 0.2,  # gamma参数，控制隐私损失在总损失中的权重
            
            # MINE模型参数
            'input_dim': 2304,      # TimeAware MINE的联合表示维度
            'hidden_dim': 512,      # MLP隐藏层维度
            'ema_alpha': 0.99,      # 指数移动平均参数
            'mine_lr': 1e-4,        # MINE优化器学习率
            
            # TimeAware MINE特定参数
            'feature_dim': 768,     # 跨模态特征维度
            'audio_d_model': 80,    # 音频编码器d_model
            'audio_nhead': 8,       # 音频编码器注意力头数
            'audio_num_layers': 4,  # 音频编码器层数
            'audio_dim_feedforward': 512,  # 音频编码器前馈网络维度
            'audio_dropout': 0.3,   # 音频编码器dropout
            'target_dim': 768,      # 音频编码器目标输出维度
            
            # Standard MINE特定参数
            'standard_input_dim': 768,  # 每个模态的特征维度
            
            # 训练配置
            'mine_update_steps': 1,     # 每个batch的MINE更新步数
            'main_update_steps': 1,     # 每个batch的主模型更新步数
            'logging_steps': 50,        # 日志记录间隔
            
            # 任务类型
            'task_type': 'classification',  # 任务类型：'classification' 或 'regression'
        }
        
        # 模型配置
        self.model = {
            'nhead': 8,
            'num_layers': 4,
            'num_classes': 2,
            'dim_feedforward': 256,
            'dropout': 0.3,
            'input_dim': self.mine_params['input_dim'],  # MINE输入维度
            'hidden_dim': self.mine_params['hidden_dim'],  # MINE隐藏层维度
        }
        
        # 覆盖基类的一些默认配置
        self.experiment_mode = 'mine'
        self.training['lr'] = 1e-4          # 主模型学习率
        self.training['weight_decay'] = 1e-5
        self.training['logging_steps'] = 50
        self.data['batch_size'] = 32
        self.general['epochs'] = 5  # 改为5个epoch
        
        # 输出配置
        self.output = {
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
        }
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集特定配置"""
        if self.feature_type == 'opensmile':
            return {
                'all_expanded_features_csv': 'data/all_expanded_features_fixed.csv',
                'feature_type': 'opensmile',
            }
        elif self.feature_type == 'mel':
            return {
                'mel_dataset_csv': 'audio_mel/data-mel/mel_dataset.csv',
                'feature_type': 'mel',
                # 特征文件路径映射配置
                'feature_path_mappings': {
                    'data/mel_features/': 'audio_mel/data-mel/mel_features/',
                },
            }
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def get_dataset_class(self):
        """获取数据集类"""
        if self.feature_type == 'opensmile':
            from data.dataset import OpenSMILEAudioDataset
            return OpenSMILEAudioDataset
        elif self.feature_type == 'mel':
            from audio_mel.dataset import AudioMelDataset
            return AudioMelDataset
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def get_collate_fn(self):
        """获取collate函数"""
        if self.feature_type == 'opensmile':
            from data.dataset import collate_fn_with_padding
            return collate_fn_with_padding
        elif self.feature_type == 'mel':
            from audio_mel.dataset import collate_fn_mel_padding
            return collate_fn_mel_padding
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def get_model_input_dim(self) -> int:
        """获取模型输入维度"""
        if self.feature_type == 'opensmile':
            return 32  # OpenSMILE特征维度
        elif self.feature_type == 'mel':
            return 80  # MEL特征维度
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def set_mine_model_type(self, model_type: str):
        """设置MINE模型类型"""
        valid_types = ['timeaware', 'standard']
        if model_type not in valid_types:
            raise ValueError(f"Invalid MINE model type: {model_type}. Must be one of {valid_types}")
        self.mine_params['mine_model_type'] = model_type
        
        # 根据模型类型调整相关参数
        if model_type == 'standard':
            self.mine_params['input_dim'] = self.mine_params['standard_input_dim'] * 2  # 拼接两个模态
    
    def set_privacy_weight(self, weight: float):
        """设置隐私权重参数"""
        if weight < 0:
            raise ValueError("Privacy weight must be non-negative")
        self.mine_params['privacy_weight'] = weight
    
    def set_task_type(self, task_type: str):
        """设置任务类型"""
        valid_types = ['classification', 'regression']
        if task_type not in valid_types:
            raise ValueError(f"Invalid task type: {task_type}. Must be one of {valid_types}")
        self.mine_params['task_type'] = task_type
    
    def update_mine_params(self, **kwargs):
        """更新MINE参数"""
        for key, value in kwargs.items():
            if key in self.mine_params:
                self.mine_params[key] = value
            else:
                raise ValueError(f"Unknown MINE parameter: {key}")
    
    def update_audio_encoder_params(self, **kwargs):
        """更新音频编码器参数（仅对TimeAware MINE有效）"""
        audio_param_keys = [
            'audio_d_model', 'audio_nhead', 'audio_num_layers',
            'audio_dim_feedforward', 'audio_dropout', 'target_dim'
        ]
        for key, value in kwargs.items():
            param_key = f'audio_{key}' if not key.startswith('audio_') and key != 'target_dim' else key
            if param_key in audio_param_keys:
                self.mine_params[param_key] = value
            else:
                raise ValueError(f"Unknown audio encoder parameter: {key}")
    
    def get_mine_model_config(self) -> Dict[str, Any]:
        """获取MINE模型配置"""
        config = {
            'mine_model_type': self.mine_params['mine_model_type'],
            'input_dim': self.mine_params['input_dim'],
            'hidden_dim': self.mine_params['hidden_dim'],
            'lr': self.mine_params['mine_lr'],
            'ema_alpha': self.mine_params['ema_alpha'],
            'privacy_weight': self.mine_params['privacy_weight'],
            'device': str(self.general['device']),
        }
        
        # TimeAware MINE特定配置
        if self.mine_params['mine_model_type'] == 'timeaware':
            config.update({
                'd_model': self.mine_params['audio_d_model'],
                'nhead': self.mine_params['audio_nhead'],
                'num_layers': self.mine_params['audio_num_layers'],
                'dim_feedforward': self.mine_params['audio_dim_feedforward'],
                'dropout': self.mine_params['audio_dropout'],
                'target_dim': self.mine_params['target_dim'],
                'feature_dim': self.mine_params['feature_dim'],
            })
        
        # Standard MINE特定配置
        elif self.mine_params['mine_model_type'] == 'standard':
            config.update({
                'standard_input_dim': self.mine_params['standard_input_dim'],
            })
        
        return config
    
    def set_experiment_mode(self, mode: str):
        """重写基类方法，MINE配置固定为mine模式"""
        if mode != 'mine':
            raise ValueError("MINEConfig only supports 'mine' mode")
        self.experiment_mode = 'mine'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        d = super().to_dict()
        d['MINE_PARAMS'] = self.mine_params
        d['MINE_MODEL'] = self.get_mine_model_config()
        
        return d
    
    def validate_config(self):
        """验证配置的有效性"""
        # 检查MINE模型类型
        if self.mine_params['mine_model_type'] not in ['timeaware', 'standard']:
            raise ValueError(f"Invalid MINE model type: {self.mine_params['mine_model_type']}")
        
        # 检查隐私权重
        if self.mine_params['privacy_weight'] < 0:
            raise ValueError("Privacy weight must be non-negative")
        
        # 检查任务类型
        if self.mine_params['task_type'] not in ['classification', 'regression']:
            raise ValueError(f"Invalid task type: {self.mine_params['task_type']}")
        
        # 检查特征类型
        if self.feature_type not in ['opensmile', 'mel']:
            raise ValueError(f"Invalid feature type: {self.feature_type}")
        
        # TimeAware MINE特定验证
        if self.mine_params['mine_model_type'] == 'timeaware':
            if self.mine_params['audio_d_model'] % self.mine_params['audio_nhead'] != 0:
                raise ValueError("audio_d_model must be divisible by audio_nhead")
        
        print("MINE config validation passed")
        return True


def main():
    """测试函数"""
    import json
    
    # 测试OpenSMILE + TimeAware MINE
    print("Testing OpenSMILE + TimeAware MINE configuration:")
    config1 = MINEConfig()
    config1.feature_type = 'opensmile'
    config1.set_mine_model_type('timeaware')
    config1.set_privacy_weight(0.3)
    config1.validate_config()
    print(json.dumps(config1.get_mine_model_config(), indent=2))
    
    # 测试MEL + Standard MINE
    print("\nTesting MEL + Standard MINE configuration:")
    config2 = MINEConfig()
    config2.feature_type = 'mel'
    config2.set_mine_model_type('standard')
    config2.set_task_type('classification')
    config2.validate_config()
    print(json.dumps(config2.get_mine_model_config(), indent=2))
    
    print("\nFull config dictionary:")
    print(json.dumps(config2.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()