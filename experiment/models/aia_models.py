"""
AIA攻击模型
包含属性推断攻击的神经网络模型，支持MLP和Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class MLPAttacker(nn.Module):
    """通用的多层感知机攻击模型"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super(MLPAttacker, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU(inplace=True)
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 主要输入 (batch_size, input_dim) 或 (batch_size, seq_len, input_dim)
            features: 可选的原始features (batch_size, seq_len, feature_dim) 或 (batch_size, feature_dim)
                     如果提供，会进行mean pooling并与x concatenate
        
        Returns:
            输出logits (batch_size, output_dim)
        """
        # 处理时序输入
        if x.dim() == 3:
            # (batch_size, seq_len, feature_dim) -> (batch_size, feature_dim)
            x = x.mean(dim=1)
        elif x.dim() == 2:
            # (batch_size, feature_dim) - 已经是正确格式
            pass
        else:
            raise ValueError(f"Input must be 2D or 3D, got {x.dim()}D")
        
        if features is not None:
            # 对原始features进行mean pooling
            if features.dim() == 3:
                # (batch_size, seq_len, feature_dim) -> (batch_size, feature_dim)
                pooled_features = features.mean(dim=1)
            else:
                # (batch_size, feature_dim) - 已经是正确格式
                pooled_features = features
            
            # Concatenate pooled features with representations
            x = torch.cat([pooled_features, x], dim=1)
        
        return self.network(x)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerAttacker(nn.Module):
    """基于Transformer的攻击模型，适用于音频时序特征"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super(TransformerAttacker, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, features: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim) - 通常是encoder output
            mask: 注意力掩码 (batch_size, seq_len)
            features: 可选的原始features (batch_size, seq_len, feature_dim) 或 (batch_size, feature_dim)
                     如果提供，会与x在特征维度上concatenate
            
        Returns:
            输出logits (batch_size, output_dim)
        """
        # 处理输入维度
        if x.dim() == 2:
            # (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
            is_single_step = True
        else:
            is_single_step = False
        
        # 如果提供了原始features，进行concatenation
        if features is not None:
            if features.dim() == 2:
                # (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
                features = features.unsqueeze(1)
            elif features.dim() == 3:
                # (batch_size, seq_len, feature_dim) - 正确格式
                pass
            else:
                raise ValueError(f"Features must be 2D or 3D, got {features.dim()}D")
            
            # 确保序列长度匹配
            if features.shape[1] != x.shape[1]:
                if x.shape[1] == 1 and features.shape[1] > 1:
                    # 如果x是单步但features是多步，将x扩展到匹配features的长度
                    x = x.expand(-1, features.shape[1], -1)
                    is_single_step = False
                elif features.shape[1] == 1 and x.shape[1] > 1:
                    # 如果features是单步但x是多步，将features扩展到匹配x的长度
                    features = features.expand(-1, x.shape[1], -1)
                else:
                    # 序列长度不匹配，对features进行插值或截断
                    if features.shape[1] > x.shape[1]:
                        # 截断features
                        features = features[:, :x.shape[1], :]
                    else:
                        # 重复features的最后一个时间步
                        last_step = features[:, -1:, :]
                        repeat_count = x.shape[1] - features.shape[1]
                        repeated = last_step.repeat(1, repeat_count, 1)
                        features = torch.cat([features, repeated], dim=1)
            
            # 在特征维度上concatenate
            x = torch.cat([features, x], dim=2)
        
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=mask)  # (batch_size, seq_len, d_model)
        
        # 全局平均池化或取最后一个时间步
        if is_single_step:
            x = x.squeeze(1)  # (batch_size, d_model)
        else:
            # 使用全局平均池化聚合时序信息
            if mask is not None:
                # 考虑掩码的加权平均
                mask_expanded = (~mask).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                x = x.mean(dim=1)  # (batch_size, d_model)
        
        # 输出投影
        output = self.output_projection(x)  # (batch_size, output_dim)
        
        return output


class AttributeInferenceAttacker(MLPAttacker):
    """属性推断攻击模型（性别、年龄、教育等级）"""
    
    def __init__(
        self,
        representation_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        attribute_name: str = "unknown"
    ):
        super(AttributeInferenceAttacker, self).__init__(
            input_dim=representation_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            dropout=dropout
        )
        
        self.attribute_name = attribute_name
        self.num_classes = num_classes
    
    def predict_attribute(self, representations: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """预测属性"""
        logits = self.forward(representations, features)
        if self.num_classes == 2:
            return torch.sigmoid(logits)  # 二分类使用sigmoid
        else:
            return F.softmax(logits, dim=1)  # 多分类使用softmax


class GenderAttacker(AttributeInferenceAttacker):
    """性别推断攻击模型"""
    
    def __init__(
        self,
        representation_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        super(GenderAttacker, self).__init__(
            representation_dim=representation_dim,
            num_classes=2,  # M/F
            hidden_dims=hidden_dims,
            dropout=dropout,
            attribute_name="gender"
        )


class AgeAttacker(AttributeInferenceAttacker):
    """年龄层次推断攻击模型"""
    
    def __init__(
        self,
        representation_dim: int,
        num_age_categories: int = 5,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        super(AgeAttacker, self).__init__(
            representation_dim=representation_dim,
            num_classes=num_age_categories,
            hidden_dims=hidden_dims,
            dropout=dropout,
            attribute_name="age_category"
        )


class EducationAttacker(AttributeInferenceAttacker):
    """教育等级推断攻击模型"""
    
    def __init__(
        self,
        representation_dim: int,
        num_education_levels: int = 4,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        super(EducationAttacker, self).__init__(
            representation_dim=representation_dim,
            num_classes=num_education_levels,
            hidden_dims=hidden_dims,
            dropout=dropout,
            attribute_name="education_level"
        )


class TransformerGenderAttacker(TransformerAttacker):
    """基于Transformer的性别推断攻击模型"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super(TransformerGenderAttacker, self).__init__(
            input_dim=input_dim,
            output_dim=2,  # M/F
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )


class TransformerAgeAttacker(TransformerAttacker):
    """基于Transformer的年龄层次推断攻击模型"""
    
    def __init__(
        self,
        input_dim: int,
        num_age_categories: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super(TransformerAgeAttacker, self).__init__(
            input_dim=input_dim,
            output_dim=num_age_categories,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )


class TransformerEducationAttacker(TransformerAttacker):
    """基于Transformer的教育等级推断攻击模型"""
    
    def __init__(
        self,
        input_dim: int,
        num_education_levels: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super(TransformerEducationAttacker, self).__init__(
            input_dim=input_dim,
            output_dim=num_education_levels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )


class MultiTaskAttacker(nn.Module):
    """多任务攻击模型，同时进行多种属性推断"""
    
    def __init__(
        self,
        representation_dim: int,
        shared_hidden_dims: List[int] = [512, 256],
        task_hidden_dims: List[int] = [128],
        dropout: float = 0.3,
        tasks: Optional[List[str]] = None
    ):
        super(MultiTaskAttacker, self).__init__()
        
        self.tasks = tasks or ['membership', 'gender', 'age_category', 'education']
        
        # 共享特征提取器
        shared_layers = []
        prev_dim = representation_dim
        
        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*shared_layers)
        
        # 任务特定的头部
        self.task_heads = nn.ModuleDict()
        
        for task in self.tasks:
            if task == 'membership':
                num_classes = 2
            elif task == 'gender':
                num_classes = 2
            elif task == 'age_category':
                num_classes = 5
            elif task == 'education':
                num_classes = 4
            else:
                num_classes = 2  # 默认二分类
            
            # 任务特定的头部网络
            task_layers = []
            task_prev_dim = prev_dim
            
            for hidden_dim in task_hidden_dims:
                task_layers.extend([
                    nn.Linear(task_prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                task_prev_dim = hidden_dim
            
            task_layers.append(nn.Linear(task_prev_dim, num_classes))
            self.task_heads[task] = nn.Sequential(*task_layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, task: Optional[str] = None):
        """
        前向传播
        
        Args:
            x: 输入表征
            task: 指定任务，如果为None则返回所有任务的输出
            
        Returns:
            如果指定task，返回该任务的输出
            否则返回字典，包含所有任务的输出
        """
        shared_features = self.shared_encoder(x)
        
        if task is not None:
            if task not in self.task_heads:
                raise ValueError(f"Unknown task: {task}")
            return self.task_heads[task](shared_features)
        else:
            outputs = {}
            for task_name, head in self.task_heads.items():
                outputs[task_name] = head(shared_features)
            return outputs


def create_attacker_model(
    attack_type: str,
    representation_dim: int,
    model_type: str = 'mlp',  # 'mlp' or 'transformer'
    hidden_dims: List[int] = [256, 128],
    dropout: float = 0.3,
    feature_dim: int = 0,  # 原始特征维度，如果>0则启用feature concatenation
    **kwargs
) -> nn.Module:
    """
    工厂函数：根据攻击类型和模型类型创建相应的攻击模型
    
    Args:
        attack_type: 攻击类型 ('gender', 'age_level', 'education', 'multitask')
        representation_dim: 表征维度
        model_type: 模型类型 ('mlp' 或 'transformer')
        hidden_dims: 隐藏层维度列表（仅MLP使用）
        dropout: dropout率
        feature_dim: 原始特征维度，如果>0则启用feature concatenation
        **kwargs: 额外参数
        
    Returns:
        攻击模型
    """
    # 计算实际输入维度（考虑feature concatenation）
    if model_type == 'mlp':
        # MLP模式：pooled_features + pooled_representations
        actual_input_dim = representation_dim + feature_dim
    else:
        # Transformer模式：features和representations在特征维度concatenate
        actual_input_dim = representation_dim + feature_dim
    
    # Transformer模型参数
    d_model = kwargs.get('d_model', 128)
    nhead = kwargs.get('nhead', 4)
    num_layers = kwargs.get('num_layers', 2)
    dim_feedforward = kwargs.get('dim_feedforward', 256)
    
    if attack_type == 'gender':
        if model_type == 'transformer':
            return TransformerGenderAttacker(
                input_dim=actual_input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        else:
            return GenderAttacker(
                representation_dim=actual_input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
    elif attack_type == 'age_level':
        num_age_categories = kwargs.get('num_age_categories', 5)
        if model_type == 'transformer':
            return TransformerAgeAttacker(
                input_dim=actual_input_dim,
                num_age_categories=num_age_categories,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        else:
            return AgeAttacker(
                representation_dim=actual_input_dim,
                num_age_categories=num_age_categories,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
    elif attack_type == 'education':
        num_education_levels = kwargs.get('num_education_levels', 4)
        if model_type == 'transformer':
            return TransformerEducationAttacker(
                input_dim=actual_input_dim,
                num_education_levels=num_education_levels,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        else:
            return EducationAttacker(
                representation_dim=actual_input_dim,
                num_education_levels=num_education_levels,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
    elif attack_type == 'multitask':
        if model_type == 'transformer':
            raise NotImplementedError("Transformer multitask attacker not implemented yet")
        else:
            tasks = kwargs.get('tasks', ['gender', 'age_category', 'education'])
            return MultiTaskAttacker(
                representation_dim=actual_input_dim,
                shared_hidden_dims=hidden_dims,
                task_hidden_dims=kwargs.get('task_hidden_dims', [128]),
                dropout=dropout,
                tasks=tasks
            )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def main():
    """测试函数"""
    representation_dim = 32  # opensmile特征维度
    batch_size = 16
    seq_len = 100  # 音频时序长度
    
    print("Testing AIA Attack Models")
    print("=" * 50)
    
    # 测试输入：2D表征（批量，特征维度）
    test_input_2d = torch.randn(batch_size, representation_dim)
    print(f"2D input shape: {test_input_2d.shape}")
    
    # 测试输入：3D时序表征（批量，序列长度，特征维度）
    test_input_3d = torch.randn(batch_size, seq_len, representation_dim)
    print(f"3D input shape: {test_input_3d.shape}")
    
    print("\n1. Testing MLP-based Attackers")
    print("-" * 30)
    
    # 测试MLP性别攻击模型
    mlp_gender_model = create_attacker_model('gender', representation_dim, model_type='mlp')
    mlp_gender_output = mlp_gender_model(test_input_2d)
    print(f"MLP Gender output shape: {mlp_gender_output.shape}")
    
    # 测试MLP年龄攻击模型
    mlp_age_model = create_attacker_model('age_level', representation_dim, model_type='mlp')
    mlp_age_output = mlp_age_model(test_input_2d)
    print(f"MLP Age output shape: {mlp_age_output.shape}")
    
    print("\n2. Testing Transformer-based Attackers")
    print("-" * 30)
    
    # 测试Transformer性别攻击模型
    transformer_gender_model = create_attacker_model('gender', representation_dim, model_type='transformer')
    
    # 测试2D输入
    transformer_gender_output_2d = transformer_gender_model(test_input_2d)
    print(f"Transformer Gender (2D input) output shape: {transformer_gender_output_2d.shape}")
    
    # 测试3D时序输入
    transformer_gender_output_3d = transformer_gender_model(test_input_3d)
    print(f"Transformer Gender (3D input) output shape: {transformer_gender_output_3d.shape}")
    
    # 测试Transformer年龄攻击模型
    transformer_age_model = create_attacker_model('age_level', representation_dim, model_type='transformer')
    transformer_age_output = transformer_age_model(test_input_3d)
    print(f"Transformer Age output shape: {transformer_age_output.shape}")
    
    # 测试Transformer教育攻击模型
    transformer_edu_model = create_attacker_model('education', representation_dim, model_type='transformer')
    transformer_edu_output = transformer_edu_model(test_input_3d)
    print(f"Transformer Education output shape: {transformer_edu_output.shape}")
    
    print("\n3. Testing with custom Transformer parameters")
    print("-" * 30)
    
    # 测试自定义Transformer参数
    custom_transformer = create_attacker_model(
        'gender', 
        representation_dim, 
        model_type='transformer',
        d_model=64,
        nhead=2,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.2
    )
    custom_output = custom_transformer(test_input_3d)
    print(f"Custom Transformer output shape: {custom_output.shape}")
    
    print("\n4. Testing MultiTask Attacker")
    print("-" * 30)
    
    # 测试多任务攻击模型（仅MLP）
    multitask_model = create_attacker_model('multitask', representation_dim, model_type='mlp')
    multitask_outputs = multitask_model(test_input_2d)
    print(f"Multitask outputs: {list(multitask_outputs.keys())}")
    for task, output in multitask_outputs.items():
        print(f"  {task}: {output.shape}")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()