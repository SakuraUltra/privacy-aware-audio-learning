"""
标准MINE (Mutual Information Neural Estimation)网络实现
用于估计和最小化声学特征与语义信息之间的互信息
遵循MINE论文的标准实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer


class AudioFeatureEncoder(nn.Module):
    """音频特征编码器：处理音频特征"""
    def __init__(self, 
                 d_model: int = 80,  # MEL特征的标准维度
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.3,
                 activation: str = 'gelu',
                 target_dim: int = 768):  
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
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 特征归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 投影层
        self.projection = nn.Linear(d_model, target_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 音频特征输入 [batch_size, seq_len, d_model]
        Returns:
            编码后的特征 [batch_size, seq_len, target_dim]
        """
        x = self.layer_norm(x)
        x = self.transformer(x)
        x = self.projection(x)
        return x


class StatisticNetwork(nn.Module):
    """统计网络：将联合表示映射到标量统计量
    标准MINE中的T(x,y)网络
    """
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512):
        super().__init__()
        
        # 特征归一化
        self.bn = nn.BatchNorm1d(input_dim)
        
        # MLP网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 谱归一化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.utils.spectral_norm(m)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        标准MINE的统计网络前向传播
        Args:
            x: 第一个变量的特征 [batch_size, dim_x]
            y: 第二个变量的特征 [batch_size, dim_y]
        Returns:
            统计量 [batch_size, 1]
        """
        # 拼接特征
        h = torch.cat([x, y], dim=1)
        h = self.bn(h)
        return self.net(h)


class StandardMINE(nn.Module):
    """标准MINE模型"""
    def __init__(self, 
                input_dim: int = 768,  # 每个模态的特征维度
                hidden_dim: int = 512,
                lr: float = 1e-4,
                ema_alpha: float = 0.99,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.ema_alpha = ema_alpha
        self.ema_mi = None
        
        # 编码器
        self.text_encoder = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)
        self.audio_encoder = AudioFeatureEncoder().to(device)
            
        # 统计网络(输入维度是两个模态特征的拼接)
        self.statistic_net = StatisticNetwork(input_dim * 2, hidden_dim).to(device)
        
        # 优化器
        self.mine_optimizer = torch.optim.Adam(
            list(self.statistic_net.parameters()), 
            lr=lr
        )
        
        # 文本编码器设置为评估模式
        self.text_encoder.eval()
        
    def encode_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """编码MEL频谱图"""
        if not mel_spec.is_floating_point():
            mel_spec = mel_spec.float()
        
        with torch.set_grad_enabled(True):
            mel_input = mel_spec.clone().requires_grad_(True)
            self.audio_encoder.train(True)
            
            # Transformer输入格式转换
            transposed_input = mel_input.transpose(1, 2)
            Z = self.audio_encoder(transposed_input)
            
            # 对序列维度取平均得到固定维度表示
            Z = torch.mean(Z, dim=1)  # [batch_size, d_model]
            
            if not Z.requires_grad:
                Z = Z.detach().requires_grad_(True)
            
        return Z
    
    def encode_text(self, texts) -> torch.Tensor:
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]
            
        tokenizer = self.text_encoder.tokenizer
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        bert_model = self.text_encoder._first_module().auto_model
        
        with torch.set_grad_enabled(True):
            outputs = bert_model(input_ids, attention_mask)
            # 取[CLS]token的表示作为整个句子的表示
            S = outputs.last_hidden_state[:, 0]
            
            S = S.to(dtype=torch.float32)
            S = S.detach().requires_grad_(True)
        
        return S

    def process_mel_features(self, mel_path: str) -> torch.Tensor:
        """处理MEL特征文件"""
        mel_df = pd.read_csv(mel_path)
        mel_spec = torch.tensor(mel_df.values, dtype=torch.float32)
        
        # 填充或截断到3000帧
        if mel_spec.shape[0] < 3000:
            padding = torch.zeros((3000 - mel_spec.shape[0], mel_spec.shape[1]), dtype=torch.float32)
            mel_spec = torch.cat([mel_spec, padding], dim=0)
        elif mel_spec.shape[0] > 3000:
            mel_spec = mel_spec[:3000, :]
        
        mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)
        mel_spec = mel_spec.to(self.device, dtype=torch.float32)
        
        return mel_spec
    
    def forward(self, input_features: torch.Tensor, text: str, mode: str = 'encoded') -> dict:
        """前向传播"""
        if mode == 'raw':
            self.audio_encoder.train()
            Z = self.encode_audio(input_features)
        else:
            Z = input_features
            
        S = self.encode_text(text)
        
        outputs = {
            'Z': Z,  # [batch_size, d_model]
            'S': S   # [batch_size, d_model]
        }
        
        return outputs
    
    def compute_mutual_information(self, x: torch.Tensor, y: torch.Tensor, y_shuffle: torch.Tensor = None) -> torch.Tensor:
        """
        计算标准MINE的互信息估计
        Args:
            x: 第一个变量的特征 [batch_size, dim_x]
            y: 第二个变量的特征 [batch_size, dim_y]
            y_shuffle: 可选的已打乱的y样本
        Returns:
            mi_estimate: 互信息估计值
        """
        if not x.requires_grad:
            x.requires_grad = True
        if not y.requires_grad:
            y.requires_grad = True
            
        # 如果没有提供打乱的y，则创建
        if y_shuffle is None:
            y_shuffle = y[torch.randperm(y.size(0))]
        
        # 计算联合分布和边缘分布的统计量
        t_xy = self.statistic_net(x, y)
        t_x_y = self.statistic_net(x, y_shuffle)
        
        # 计算互信息估计(MINE论文中的公式)
        E_xy = torch.mean(t_xy)
        E_x_y = torch.logsumexp(t_x_y, dim=0) - torch.log(torch.tensor(x.size(0), dtype=torch.float32))
        
        mi_estimate = E_xy - E_x_y
        
        # 添加轻量级的正则化
        l2_reg = 0.001 * (t_xy.pow(2).mean() + t_x_y.pow(2).mean())
        
        return mi_estimate - l2_reg
    
    def update_estimator(self, Z: torch.Tensor, S: torch.Tensor) -> dict:
        """
        更新MINE估计器参数
        Args:
            Z: 音频特征 [batch_size, dim]
            S: 文本特征 [batch_size, dim]
        Returns:
            包含训练信息的字典
        """
        self.train()
        self.mine_optimizer.zero_grad()
        
        # 计算互信息估计
        mi_estimate = self.compute_mutual_information(Z, S)
        
        # 更新EMA
        with torch.no_grad():
            if self.ema_mi is None:
                self.ema_mi = mi_estimate.detach()
            else:
                self.ema_mi = self.ema_alpha * self.ema_mi + (1 - self.ema_alpha) * mi_estimate.detach()
        
        # 反向传播(最大化互信息下界)
        loss = -mi_estimate
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.mine_optimizer.step()
        
        return {
            'mi_estimate': mi_estimate.item(),
            'ema_mi': self.ema_mi.item() if self.ema_mi is not None else 0.0,
            'loss': loss.item()
        }


def create_standard_mine(input_dim: int = 768, mlp_hidden: int = 512) -> StandardMINE:
    """创建StandardMINE实例"""
    return StandardMINE(input_dim=input_dim, hidden_dim=mlp_hidden)
