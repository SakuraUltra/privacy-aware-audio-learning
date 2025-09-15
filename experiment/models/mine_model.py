"""
MINE (Mutual Information Neural Estimation) 网络实现
用于估计和最小化声学特征与语义信息之间的互信息
遵循论文Algorithm 1的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from .transformer import TransformerEncoder, TransformerClassifier, pool

class AudioFeatureEncoder(nn.Module):
    """音频特征编码器：处理音频特征"""
    def __init__(self, 
                 d_model: int = 80,  # 修改为MEL特征的标准维度
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.3,
                 activation: str = 'gelu',
                 target_dim: int = 768):  # 添加目标维度参数
        super().__init__()
        self.d_model = d_model
        self.target_dim = target_dim
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # [batch_size, seq_len, hidden_size]
            norm_first=True    # Pre-LN架构，提高训练稳定性
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 特征归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 添加投影层将d_model维特征投影到target_dim维
        self.projection = nn.Linear(d_model, target_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 音频特征输入 [batch_size, seq_len, d_model]
        Returns:
            编码后的特征 [batch_size, seq_len, target_dim]
        """
        # 特征归一化
        x = self.layer_norm(x)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 投影到目标维度
        x = self.projection(x)
        
        return x


class CrossModalAlignment(nn.Module):
    """跨模态注意力对齐模块"""
    def __init__(self, feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
        
    def forward(self, Z, S):
        """
        跨模态注意力对齐
        Args:
            Z: 声学特征序列, shape [batch_size, T, d_model], T=1500
            S: 语义特征序列, shape [batch_size, M, d_model], M=15
        Returns:
            C: 对齐后的语义序列, shape [batch_size, T, d_model]
            attention_weights: 注意力权重, shape [batch_size, T, M]
        """
        # 确保输入张量需要梯度
        if not Z.requires_grad:
            Z = Z.detach().requires_grad_(True)
        if not S.requires_grad:
            S = S.detach().requires_grad_(True)
        
        # 1. 计算注意力分数 (Scaled Dot-Product Attention)
        scale_factor = torch.sqrt(torch.tensor(self.feature_dim, dtype=Z.dtype, device=Z.device))
        
        attention_scores = torch.bmm(Z, S.transpose(-2, -1)) / scale_factor
        
        # 2. 注意力权重归一化
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 3. 生成对齐后的语义序列
        C = torch.bmm(attention_weights, S)
    
        
        return C, attention_weights

class JointRepresentation(nn.Module):
    """联合表示构建模块"""
    def __init__(self, feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
    
    def forward(self, Z, C):
        """
        构建联合表示
        Args:
            Z: 声学特征序列, shape [batch_size, T, d_model]
            C: 对齐后的语义序列, shape [batch_size, T, d_model]
        Returns:
            h: 联合表示, shape [batch_size, 3*d_model]
        """
        # 确保输入张量需要梯度
        if not Z.requires_grad:
            Z = Z.detach().requires_grad_(True)
        if not C.requires_grad:
            C = C.detach().requires_grad_(True)
        
        # 1. 均值池化
        z_agg = torch.mean(Z, dim=1)  # [batch_size, d_model]
        c_agg = torch.mean(C, dim=1)  # [batch_size, d_model]
        
        # 2. 构建联合表示 h = [z_agg; c_agg; z_agg ⊙ c_agg]
        element_wise_prod = z_agg * c_agg
        h = torch.cat([z_agg, c_agg, element_wise_prod], dim=1)
        
        return h, (z_agg, c_agg)

class StatisticNetwork(nn.Module):
    """统计网络：将联合表示映射到标量统计量"""
    def __init__(self, input_dim: int = 2304, hidden_dim: int = 512):
        super().__init__()
        # 特征归一化
        self.bn = nn.BatchNorm1d(input_dim)
        
        # MLP网络
        self.net = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # 第二层
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # 输出层
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 对所有线性层添加谱归一化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.utils.spectral_norm(m)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: 联合表示 [batch_size, 2304]
        Returns:
            统计量 [batch_size, 1]
        """
        h = self.bn(h)
        return self.net(h)

class TimeAwareMINE(nn.Module):
    """TimeAware-MINE模型"""
    def __init__(self, 
                input_dim: int = 2304, 
                hidden_dim: int = 512,
                d_model: int = 768,
                nhead: int = 8,
                num_layers: int = 4,
                num_classes: int = 2,
                lr: float = 1e-4,
                ema_alpha: float = 0.99,
                privacy_gamma: float = 0.1,
                compress_beta: float = 0.01,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.ema_alpha = ema_alpha
        self.ema_mi = None
        
        # 编码器初始化
        self.text_encoder = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)
        
        # 使用默认参数初始化AudioFeatureEncoder，d_model=80
        self.audio_encoder = AudioFeatureEncoder().to(device)
            
        # 跨模态对齐和联合表示
        self.cross_modal_alignment = CrossModalAlignment(feature_dim=768).to(device)  # 使用768维特征
        self.joint_representation = JointRepresentation(feature_dim=768).to(device)  # 使用768维特征
        
        # 统计网络
        self.statistic_net = StatisticNetwork(input_dim, hidden_dim).to(device)
        
        # 设置MINE优化器
        self.mine_optimizer = torch.optim.Adam(
            list(self.statistic_net.parameters()) + 
            list(self.cross_modal_alignment.parameters()) +
            list(self.joint_representation.parameters()),
            lr=lr
        )
        
        # 将文本编码器设置为评估模式
        self.text_encoder.eval()
        
    def encode_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        编码MEL频谱图
        Args:
            mel_spec: MEL频谱图 [batch_size, n_mels, T]
        Returns:
            Z: 编码后的声学特征 [batch_size, T', d_model]
        """
        
        # 确保输入是浮点型
        if not mel_spec.is_floating_point():
            mel_spec = mel_spec.float()
        
        # 启用梯度计算
        with torch.set_grad_enabled(True):
            # 克隆输入并确保需要梯度
            mel_input = mel_spec.clone().requires_grad_(True)
            
            # 使用音频编码器
            self.audio_encoder.train(True)
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
            
            # 注意：Transformer期望的输入格式是 [batch_size, seq_len, features]
            # MEL特征的形状是 [batch_size, n_mels, T]，所以需要转置
            transposed_input = mel_input.transpose(1, 2)  # 转置以匹配Transformer输入格式
            
            Z = self.audio_encoder(transposed_input)
            
            # 确保输出需要梯度
            if not Z.requires_grad:
                Z = Z.detach().requires_grad_(True)
            
        return Z
    
    def encode_text(self, texts) -> torch.Tensor:
        """
        编码文本
        Args:
            texts: 单个文本字符串或文本列表
        Returns:
            S: 编码后的语义特征 [batch_size, M, d_model]
        """
        # 确保输入是列表形式
        if isinstance(texts, str):
            texts = [texts]
            
        # 使用BERT tokenizer进行批处理
        tokenizer = self.text_encoder.tokenizer
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # 获取token级别的语义嵌入，但不设置requires_grad
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 获取BERT模型
        bert_model = self.text_encoder._first_module().auto_model
        
        # 正向传播但不跟踪token ids的梯度
        with torch.set_grad_enabled(True):
            # 不对整数类型的token ids设置requires_grad
            outputs = bert_model(input_ids, attention_mask)
            S = outputs.last_hidden_state
            
            # 确保输出是浮点型并且需要梯度
            S = S.to(dtype=torch.float32)
            S = S.detach().requires_grad_(True)
        
        return S
    
    def process_mel_features(self, mel_path: str) -> torch.Tensor:
        """
        处理MEL特征文件
        Args:
            mel_path: MEL特征CSV文件路径
        Returns:
            mel_spec: 处理后的MEL特征 [1, n_mels, T]
        """
        # 加载MEL特征，确保使用float32类型
        mel_df = pd.read_csv(mel_path)
        mel_spec = torch.tensor(mel_df.values, dtype=torch.float32)
        print("[DEBUG Process] 初始mel_spec类型:", mel_spec.dtype)
        
        # 填充或截断到3000帧
        if mel_spec.shape[0] < 3000:
            padding = torch.zeros((3000 - mel_spec.shape[0], mel_spec.shape[1]), dtype=torch.float32)
            mel_spec = torch.cat([mel_spec, padding], dim=0)
        elif mel_spec.shape[0] > 3000:
            mel_spec = mel_spec[:3000, :]
        
        # 调整维度顺序：[sequence_length, feature_dim] -> [batch_size, feature_dim, sequence_length]
        mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)
        
        # 确保数据类型和设备正确
        mel_spec = mel_spec.to(self.device, dtype=torch.float32)
        print("[DEBUG Process] 最终mel_spec类型:", mel_spec.dtype)
        print("[DEBUG Process] 最终mel_spec设备:", mel_spec.device)
        
        return mel_spec
    
    def forward(self, input_features: torch.Tensor, text: str, mode: str = 'encoded') -> dict:
        """
        前向传播
        Args:
            input_features: 输入特征:
                - 如果mode='raw': MEL频谱图 [batch_size, n_mels, T]
                - 如果mode='encoded': 已编码的特征 [batch_size, d_model]
            text: 输入文本
            mode: 'raw'使用原始MEL特征,'encoded'使用已编码特征
        Returns:
            包含输出和损失信息的字典
        """
        # 1. 处理声学特征
        if mode == 'raw':
            self.audio_encoder.train()
            Z = self.encode_audio(input_features)
        else:  # mode == 'encoded'
            Z = input_features.unsqueeze(1)  # [batch_size, 1, d_model]
            
        S = self.encode_text(text)
        
        # 2. 跨模态对齐
        C, attention_weights = self.cross_modal_alignment(Z, S)
        
        # 3. 构建联合表示
        h, (z_agg, c_agg) = self.joint_representation(Z, C)
        
        outputs = {
            'Z': Z,
            'S': S,
            'C': C,
            'h': h,
            'z_agg': z_agg,
            'c_agg': c_agg,
            'attention_weights': attention_weights
        }
        
        # 直接返回必要的中间结果
        return outputs
    
    def compute_mutual_information(self, h_joint: torch.Tensor, h_marginal: torch.Tensor) -> torch.Tensor:
        """
        计算互信息估计
        Args:
            h_joint: 正样本对的联合表示 [batch_size, 2304]
            h_marginal: 负样本对的联合表示 [batch_size, 2304]
        Returns:
            mi_estimate: 互信息估计值
        """
        # 确保输入需要梯度
        if not h_joint.requires_grad:
            h_joint.requires_grad = True
        if not h_marginal.requires_grad:
            h_marginal.requires_grad = True
            
        # 计算统计量
        T_joint = self.statistic_net(h_joint)
        T_marginal = self.statistic_net(h_marginal)
        
        # 计算期望
        E_joint = T_joint.mean()
        
        # 使用 log_sum_exp 技巧来提高数值稳定性
        max_marginal = T_marginal.max().detach()
        E_marginal = max_marginal + torch.log(torch.exp(T_marginal - max_marginal).mean() + 1e-6)
        
        # 互信息估计
        mi_estimate = E_joint - E_marginal
        
        # 添加轻量级的正则化
        l2_reg = 0.001 * (T_joint.pow(2).mean() + T_marginal.pow(2).mean())
        
        return mi_estimate - l2_reg
    
    def create_marginal_samples(self, h: torch.Tensor, batch_size: int = 10) -> torch.Tensor:
        """
        从联合表示创建负样本batch
        Args:
            h: 原始联合表示 [1, 2304]
            batch_size: batch大小
        Returns:
            h_marginal: 负样本batch [batch_size, 2304]
        """
        # 复制到batch_size，并保留梯度信息
        # 修复：使用detach().requires_grad_(True)创建新的叶节点张量
        h_marginal = h.repeat(batch_size, 1).detach().requires_grad_(True)
        
        # 打乱最后1/3（元素乘积部分）
        prod_start = 2 * (h.shape[1] // 3)  # 元素乘积部分的起始位置
        with torch.no_grad():  # 打乱操作不需要梯度
            for i in range(batch_size):
                perm = torch.randperm(h.shape[1] // 3)
                h_marginal[i, prod_start:] = h_marginal[i, prod_start + perm]
        
        return h_marginal


    
    def update_estimator(self, h: torch.Tensor, batch_size: int = 10) -> dict:
        """
        更新MINE估计器参数
        Args:
            h: 联合表示 [1, 2304]
            batch_size: batch大小
        Returns:
            包含训练信息的字典
        """
        self.train()
        self.mine_optimizer.zero_grad()  # 使用mine_optimizer而不是optimizer
        
        # 创建正负样本batch
        h_joint = h.repeat(batch_size, 1)  # [batch_size, 2304]
        h_marginal = self.create_marginal_samples(h, batch_size)  # [batch_size, 2304]
        
        # 计算互信息估计
        mi_estimate = self.compute_mutual_information(h_joint, h_marginal)
        
        # 更新EMA
        with torch.no_grad():
            if self.ema_mi is None:
                self.ema_mi = mi_estimate.detach()
            else:
                self.ema_mi = self.ema_alpha * self.ema_mi + (1 - self.ema_alpha) * mi_estimate.detach()
        
        # 反向传播
        loss = -mi_estimate  # MINE训练目标是最大化互信息下界
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.mine_optimizer.step()  # 使用mine_optimizer而不是optimizer
        
        # 保存梯度计算前的值用于返回
        mi_val = mi_estimate.item()
        loss_val = loss.item()
        ema_val = self.ema_mi.item() if self.ema_mi is not None else 0.0
        
        return {
            'mi_estimate': mi_val,
            'ema_mi': ema_val,
            'loss': loss_val
        }

def create_timeaware_mine(input_dim: int = 2304, mlp_hidden: int = 512) -> TimeAwareMINE:
    """创建TimeAwareMINE实例"""
    return TimeAwareMINE(input_dim=input_dim, hidden_dim=mlp_hidden)
