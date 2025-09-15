"""
AIA (Attribute Inference Attack) 配置类
"""

import json
from typing import Dict, Any, List, Optional
from .base_config import BaseConfig


class AIAConfig(BaseConfig):
    """AIA属性推断攻击专用配置类"""
    
    def __init__(self):
        super().__init__()
        
        # AIA特定配置
        self.aia_params = {
            # 攻击类型：'gender', 'age_level', 'education'
            'attack_type': 'gender',
            
            # 目标模型检查点路径（5个fold的路径列表）
            'checkpoint_paths': [],
            
            # 目标模型训练模式（用于确定模型架构）
            'target_model_mode': 'normal',  # 'normal', 'dp', 'vib'
            
            # 表征提取配置
            'representation_layer': -2,  # 从哪一层提取表征（-1为最后一层，-2为倒数第二层）
            'representation_dim': None,   # 自动从模型中推断
            
            # AIA模型配置
            'attack_model_hidden_dims': [64, 32],  # 进一步简化MLP
            'attack_model_dropout': 0.5,  # 保持较高dropout
            'attack_model_type': 'mlp',  # 使用简单MLP
            
            # Transformer攻击模型配置（仅当attack_model_type='transformer'时使用）
            'transformer_d_model': 64,   # 减少模型容量
            'transformer_nhead': 4,
            'transformer_num_layers': 2,  # 减少层数
            'transformer_dim_feedforward': 128,  # 减少前馈网络维度
            
            # 数据划分配置
            'test_ratio': 0.2,    # 测试集比例
            
            # 人口统计信息
            'demographic_labels_path': 'data/demographic_labels.csv',
            
            # 输入数据模式：'features_only', 'representations_only', 'concatenation', 'mix'
            'input_mode': 'features_only',
            
            # Mix模式参数：alpha * normalized_features + (1-alpha) * normalized_representations
            'mix_alpha': 0.5,
        }
        
        # 覆盖基类的一些默认配置
        self.experiment_mode = 'aia'
        self.training['lr'] = 1e-3  # AIA攻击模型学习率稍高
        self.training['weight_decay'] = 1e-3  # 增加权重衰减
        self.training['total_steps'] = 500  # 总训练步数
        self.training['eval_every_steps'] = 100  # 每100步评估一次
        self.data['batch_size'] = 32  # AIA训练批量可以更大
        self.general['epochs'] = 5   # 保留epochs配置用于兼容
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集特定配置"""
        if self.feature_type == 'opensmile':
            return {
                'data_dir': 'data/extracted_features_train',
                'num_features': 32,
            }
        elif self.feature_type == 'mel':
            return {
                'data_dir': 'audio_mel/extracted_features',
                'num_features': 1024,  # MEL特征维度
            }
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def get_model_input_dim(self) -> int:
        """获取模型输入维度（根据输入模式确定）"""
        input_mode = self.aia_params['input_mode']
        
        # 计算representation维度
        if self.aia_params['representation_dim'] is not None:
            repr_dim = self.aia_params['representation_dim']
        else:
            # 根据特征类型和目标模型模式推断表征维度
            if self.feature_type == 'opensmile':
                if self.aia_params['target_model_mode'] == 'vib':
                    repr_dim = self.vib_params.get('z_dim', 64)
                else:
                    repr_dim = 32  # opensmile特征维度
            elif self.feature_type == 'mel':
                if self.aia_params['target_model_mode'] == 'vib':
                    repr_dim = self.vib_params.get('z_dim', 64)
                else:
                    repr_dim = 80  # MEL Transformer模型的d_model维度
            else:
                repr_dim = 256  # 默认值
        
        # 计算feature维度
        if self.feature_type == 'opensmile':
            feature_dim = 32  # opensmile特征维度
        elif self.feature_type == 'mel':
            feature_dim = 80  # MEL特征维度（after mean pooling）
        else:
            feature_dim = 256  # 默认值
        
        # 根据输入模式返回正确的维度
        if input_mode == 'features_only':
            return feature_dim
        elif input_mode == 'representations_only':
            return repr_dim
        elif input_mode == 'concatenation':
            return feature_dim + repr_dim
        elif input_mode == 'mix':
            # Mix模式：归一化后的特征和表征具有相同维度
            return max(feature_dim, repr_dim)  # 取两者中的最大值作为混合后的维度
        else:
            raise ValueError(f"Unknown input mode: {input_mode}")
    
    def set_attack_type(self, attack_type: str):
        """设置攻击类型"""
        valid_types = ['gender', 'age_level', 'education']
        if attack_type not in valid_types:
            raise ValueError(f"Invalid attack type: {attack_type}. Must be one of {valid_types}")
        self.aia_params['attack_type'] = attack_type
        
        # 根据攻击类型调整输出类别数
        if attack_type == 'gender':
            self.model['num_classes'] = 2  # M/F
        elif attack_type == 'age_level':
            self.model['num_classes'] = 3  # 3个年龄层次
        elif attack_type == 'education':
            self.model['num_classes'] = 4  # 4个教育等级（1,2,3,4）
    
    def set_attack_model_type(self, model_type: str):
        """设置攻击模型类型"""
        valid_types = ['mlp', 'transformer']
        if model_type not in valid_types:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_types}")
        self.aia_params['attack_model_type'] = model_type
    
    def set_transformer_params(self, **kwargs):
        """设置Transformer参数"""
        transformer_keys = [
            'transformer_d_model', 'transformer_nhead', 
            'transformer_num_layers', 'transformer_dim_feedforward'
        ]
        for key, value in kwargs.items():
            if key in transformer_keys:
                self.aia_params[key] = value
            else:
                raise ValueError(f"Unknown transformer parameter: {key}")
    
    def set_checkpoint_paths(self, paths: List[str]):
        """设置目标模型检查点路径"""
        if not isinstance(paths, list) or len(paths) != 5:
            raise ValueError("checkpoint_paths must be a list of 5 paths for 5-fold CV")
        self.aia_params['checkpoint_paths'] = paths
    
    def set_target_model_mode(self, mode: str):
        """设置目标模型的训练模式"""
        valid_modes = ['normal', 'dp', 'vib', 'mine']
        if mode not in valid_modes:
            raise ValueError(f"Invalid target model mode: {mode}. Must be one of {valid_modes}")
        self.aia_params['target_model_mode'] = mode
    
    def set_input_mode(self, mode: str):
        """设置输入数据模式"""
        valid_modes = ['features_only', 'representations_only', 'concatenation', 'mix']
        if mode not in valid_modes:
            raise ValueError(f"Invalid input mode: {mode}. Must be one of {valid_modes}")
        self.aia_params['input_mode'] = mode
    
    def update_aia_params(self, **kwargs):
        """更新AIA参数"""
        for key, value in kwargs.items():
            if key in self.aia_params:
                self.aia_params[key] = value
            else:
                raise ValueError(f"Unknown AIA parameter: {key}")
    
    def set_experiment_mode(self, mode: str):
        """重写基类方法，AIA配置固定为aia模式"""
        if mode != 'aia':
            raise ValueError("AIAConfig only supports 'aia' mode")
        self.experiment_mode = 'aia'
    
    def get_attack_model_config(self) -> Dict[str, Any]:
        """获取攻击模型配置"""
        config = {
            'input_dim': self.get_model_input_dim(),
            'output_dim': self.model['num_classes'],
            'model_type': self.aia_params['attack_model_type'],
            'dropout': self.aia_params['attack_model_dropout'],
        }
        
        if self.aia_params['attack_model_type'] == 'mlp':
            config['hidden_dims'] = self.aia_params['attack_model_hidden_dims']
        elif self.aia_params['attack_model_type'] == 'transformer':
            config['d_model'] = self.aia_params['transformer_d_model']
            config['nhead'] = self.aia_params['transformer_nhead']
            config['num_layers'] = self.aia_params['transformer_num_layers']
            config['dim_feedforward'] = self.aia_params['transformer_dim_feedforward']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        d = super().to_dict()
        d['AIA_PARAMS'] = self.aia_params
        d['ATTACK_MODEL'] = self.get_attack_model_config()
        return d
    
    def validate_config(self):
        """验证配置的有效性"""
        # 检查检查点路径
        if not self.aia_params['checkpoint_paths']:
            raise ValueError("checkpoint_paths must be set for AIA attacks")
        
        if len(self.aia_params['checkpoint_paths']) != 5:
            raise ValueError("Must provide exactly 5 checkpoint paths for 5-fold CV")
        
        # 检查攻击类型
        if self.aia_params['attack_type'] not in ['gender', 'age_level', 'education']:
            raise ValueError(f"Invalid attack type: {self.aia_params['attack_type']}")
        
        # 检查目标模型模式
        if self.aia_params['target_model_mode'] not in ['normal', 'dp', 'vib', 'mine']:
            raise ValueError(f"Invalid target model mode: {self.aia_params['target_model_mode']}")
        
        print("AIA config validation passed")
        return True


def main():
    """测试函数"""
    config = AIAConfig()
    
    # 设置基本参数
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
    
    # 验证配置
    config.validate_config()
    
    print("Testing MLP configuration:")
    print(json.dumps(config.get_attack_model_config(), indent=2))
    
    # 测试Transformer配置
    config.set_attack_model_type('transformer')
    config.set_transformer_params(
        transformer_d_model=64,
        transformer_nhead=2,
        transformer_num_layers=1
    )
    
    print("\nTesting Transformer configuration:")
    print(json.dumps(config.get_attack_model_config(), indent=2))


if __name__ == "__main__":
    main()