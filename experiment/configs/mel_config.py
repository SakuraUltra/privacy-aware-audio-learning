"""
MEL特征配置
"""
from typing import Dict, Any
from .base_config import BaseConfig


class MELConfig(BaseConfig):
    """MEL特征配置类"""
    
    def __init__(self):
        super().__init__()
        self.feature_type = 'mel'
        
        # MEL特定配置
        self.mel_config = {
            'feature_dim': 80,  # MEL特征维度
            'feature_type': 'mel',
            'model_name': 'openai/whisper-large-v3-turbo',
            'sample_rate': 16000,
            'target_dim': 32,  # 与OpenSMILE保持一致, deprecated
            'max_length': 1000,  # 最大序列长度
            'max_audio_length': 5.0,  # 最大音频长度（秒）
            'normalize_audio': True,  # 是否标准化音频
            'save_format': 'csv',  # 特征保存格式
            'feature_suffix': '_mel.csv',  # 特征文件后缀
            'normalize_features': True,  # 是否标准化特征
        }
        
        # 更新模型配置
        self.model.update({
            'd_model': 80,  # MEL特征维度
        })
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """获取MEL数据集配置"""
        return {
            'mel_dataset_csv': 'audio_mel/data-mel/mel_dataset.csv',
            'feature_type': 'mel',
            # 特征文件路径映射配置
            'feature_path_mappings': {
                'data/mel_features/': 'audio_mel/data-mel/mel_features/',
            },
        }
    
    def get_model_input_dim(self) -> int:
        """获取模型输入维度"""
        return self.mel_config['feature_dim']
    
    def get_dataset_class(self):
        """获取数据集类"""
        from audio_mel.dataset import AudioMelDataset
        return AudioMelDataset
    
    def get_collate_fn(self):
        """获取collate函数"""
        from audio_mel.dataset import collate_fn_mel_padding
        return collate_fn_mel_padding
