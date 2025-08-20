"""
基础配置类，定义所有配置的通用结构
"""
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseConfig(ABC):
    """基础配置类"""
    
    def __init__(self):
        self.experiment_mode = 'normal'  # 'normal', 'dp', 'vib'
        self.feature_type = 'opensmile'  # 'opensmile' or 'mel'
        self.vib_cfg = None  # VIBConfig对象，vib模式下使用

        # 通用配置
        self.general = {
            'seed': 666,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'num_folds': 5,
            'epochs': 10,
        }

        # 数据配置
        self.data = {
            'speaker_folds_csv': 'data/speaker_folds.csv',
            'batch_size': 32,
        }

        # 模型配置
        self.model = {
            'nhead': 8,
            'num_layers': 4,
            'num_classes': 2,
            'dim_feedforward': 256,
            'dropout': 0.3,
        }

        # 训练配置
        self.training = {
            'lr': 1e-4,
            'weight_decay': 1e-5,
        }

        # DP配置（仅在DP模式下使用）
        self.dp_params = {
            'target_epsilon': 8.0,
            'target_delta': 1e-5,
            'max_grad_norm': 1.2,
        }
    
        # VIB配置（仅在VIB模式下使用）
        self.vib_params = {
            'z_dim': 64,
            'beta': 1e-3,
            'mc_samples': 30,
        }
    @abstractmethod
    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集特定配置"""
        pass
    
    @abstractmethod
    def get_model_input_dim(self) -> int:
        """获取模型输入维度"""
        pass
    
    def update_dp_params(self, epsilon: Optional[float] = None, **kwargs):
        """更新DP参数"""
        if epsilon is not None:
            self.dp_params['target_epsilon'] = epsilon
        for key, value in kwargs.items():
            if key in self.dp_params:
                self.dp_params[key] = value

    def update_vib_params(self, z_dim: Optional[int]=None, beta: Optional[float]=None, mc_samples: Optional[int]=None):
        """更新VIB参数并创建VIBConfig对象"""
        if z_dim is not None:
            self.vib_params['z_dim'] = z_dim
        if beta is not None:
            self.vib_params['beta'] = beta
        if mc_samples is not None:
            self.vib_params['mc_samples'] = mc_samples

        # 创建VIBConfig对象
        from models.vib_model import VIBConfig
        self.vib_cfg = VIBConfig(
            z_dim=self.vib_params['z_dim'],
            beta=self.vib_params['beta'],
            mc_samples=self.vib_params['mc_samples']
        )

    def set_experiment_mode(self, mode: str):
        """设置实验模式"""
        if mode not in ['normal', 'dp', 'vib']:
            raise ValueError(f"Invalid experiment mode: {mode}")
        self.experiment_mode = mode
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，兼容原有代码"""
        d = {
            'EXPERIMENT_MODE': self.experiment_mode.upper(),
            'GENERAL': self.general,
            'DATA': {**self.data, **self.get_dataset_config()},
            'MODEL': {**self.model, 'd_model': self.get_model_input_dim()},
            'TRAINING': self.training,
            'DP_PARAMS': self.dp_params if self.experiment_mode == 'dp' else None,
            'VIB_PARAMS': self.vib_params if self.experiment_mode == 'vib' else None,
        }
        return d
    
    def __getattr__(self, name):
        """提供向后兼容性，允许直接访问配置项"""
        config_dict = self.to_dict()
        if name in config_dict:
            return config_dict[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
