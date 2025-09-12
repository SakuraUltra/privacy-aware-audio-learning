"""
配置工厂，用于创建不同类型的配置
"""
from typing import Optional
from .base_config import BaseConfig
from .opensmile_config import OpenSMILEConfig
from .mel_config import MELConfig
from .aia_config import AIAConfig


class ConfigFactory:
    """配置工厂类"""
    
    @staticmethod
    def create_config(
        feature_type: str,
        mode: str = 'normal',
        epsilon: Optional[float] = None,
        z_dim: Optional[int] = None,
        beta: Optional[float] = None,
        mc_samples: Optional[int] = None,
        **kwargs
    ) -> BaseConfig:
        """
        创建配置对象
        
        Args:
            feature_type: 特征类型 ('opensmile' 或 'mel')
            mode: 实验模式 ('normal', 'dp', 'vib', 'aia')
            epsilon: DP模式下的epsilon值
            z_dim: VIB模式下的隐变量维度
            beta: VIB模式下的KL正则化强度
            mc_samples: VIB模式下的MC采样次数
            **kwargs: 其他配置参数（AIA模式下包括checkpoint_paths, attack_type等）
        
        Returns:
            配置对象
        """
        # 创建对应的配置对象
        if mode.lower() == 'aia':
            # AIA模式使用专门的配置类
            config = AIAConfig()
            config.feature_type = feature_type.lower()
            
            # 设置AIA特定参数
            if 'attack_type' in kwargs:
                config.set_attack_type(kwargs['attack_type'])
            if 'checkpoint_paths' in kwargs:
                config.set_checkpoint_paths(kwargs['checkpoint_paths'])
            if 'target_model_mode' in kwargs:
                config.set_target_model_mode(kwargs['target_model_mode'])
            if 'attack_model_type' in kwargs:
                config.set_attack_model_type(kwargs['attack_model_type'])
            if 'input_mode' in kwargs:
                config.set_input_mode(kwargs['input_mode'])
            
            # 设置Transformer参数（如果使用）
            transformer_params = {k: v for k, v in kwargs.items() 
                                if k.startswith('transformer_')}
            if transformer_params:
                config.set_transformer_params(**transformer_params)
            
            # 更新其他AIA参数
            aia_params = {k: v for k, v in kwargs.items() 
                         if k in config.aia_params and k not in [
                             'attack_type', 'checkpoint_paths', 'target_model_mode', 
                             'attack_model_type', 'input_mode']}
            if aia_params:
                config.update_aia_params(**aia_params)
                
        elif feature_type.lower() == 'opensmile':
            config = OpenSMILEConfig()
        elif feature_type.lower() == 'mel':
            config = MELConfig()
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        # 对于非AIA模式，设置实验模式
        if mode.lower() != 'aia':
            # 设置实验模式
            config.set_experiment_mode(mode.lower())

            # 更新DP参数
            if mode.lower() == 'dp' and epsilon is not None:
                config.update_dp_params(epsilon=epsilon)

            # 原生支持VIBConfig对象
            if mode.lower() == 'vib':
                config.update_vib_params(z_dim=z_dim, beta=beta, mc_samples=mc_samples)
            
            # 更新其他参数
            for key, value in kwargs.items():
                if hasattr(config, key):
                    if isinstance(getattr(config, key), dict):
                        getattr(config, key).update(value)
                    else:
                        setattr(config, key, value)

        return config
    
    @staticmethod
    def get_available_feature_types():
        """获取可用的特征类型"""
        return ['opensmile', 'mel']
    
    @staticmethod
    def get_available_modes():
        """获取可用的实验模式"""
        return ['normal', 'dp', 'vib', 'aia']
