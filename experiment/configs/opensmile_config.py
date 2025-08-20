"""
OpenSMILE特征配置
"""
from typing import Dict, Any
from .base_config import BaseConfig


class OpenSMILEConfig(BaseConfig):
    """OpenSMILE特征配置类"""
    
    def __init__(self):
        super().__init__()
        self.feature_type = 'opensmile'
        
        # OpenSMILE特定配置
        self.opensmile_config = {
            'feature_dim': 32,  # OpenSMILE特征维度
            'feature_type': 'opensmile',
        }
        
        # 更新模型配置
        self.model.update({
            'd_model': 32,  # OpenSMILE特征维度
        })
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """获取OpenSMILE数据集配置"""
        return {
            'all_expanded_features_csv': 'data/all_expanded_features_fixed.csv',
            'feature_type': 'opensmile',
        }
    
    def get_model_input_dim(self) -> int:
        """获取模型输入维度"""
        return self.opensmile_config['feature_dim']
    
    def get_dataset_class(self):
        """获取数据集类"""
        from data.dataset import OpenSMILEAudioDataset
        return OpenSMILEAudioDataset
    
    def get_collate_fn(self):
        """获取collate函数"""
        from data.dataset import collate_fn_with_padding
        return collate_fn_with_padding
