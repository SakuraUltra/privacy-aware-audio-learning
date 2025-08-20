import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPMultiheadAttention
import warnings
from torch.autograd import Function

# 直接在此文件中定义GradientReversalLayer，避免导入问题
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from DANN paper by Ganin et al.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        return -lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def pool(transformer_output, padding_mask=None):
    if padding_mask is not None:
        pooling_mask = ~padding_mask
        lengths = pooling_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_output = (transformer_output * pooling_mask.unsqueeze(-1)).sum(dim=1) / lengths
    else:
        pooled_output = transformer_output.mean(dim=1)
    pooled_output = torch.nan_to_num(pooled_output, nan=0.0)
    return pooled_output
    
class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:, :x.size(1), :]
        return self.dropout(x + pe)

class DPTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = DPMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # src_mask 传递给 attn_mask，src_key_padding_mask 直接传递，is_causal 仅为兼容PyTorch接口
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()  
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # src_mask 传递给 attn_mask，src_key_padding_mask 直接传递，is_causal 仅为兼容PyTorch接口
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=4, num_classes=2, dim_feedforward=128, dropout=0.3, dp_mode=False):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.dp_mode = dp_mode
        if dp_mode:
            encoder_layer = DPTransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                      dim_feedforward=dim_feedforward, dropout=dropout,
                                                      batch_first=True)
        else:
            encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, padding_mask=None):
        x = torch.nan_to_num(x, nan=0.0)
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return transformer_output

def sync_dp_to_normal_attention(dp_attn, normal_attn):
    """将DP注意力的分离权重同步到Normal注意力的合并权重"""
    with torch.no_grad():
        # 重组Q、K、V权重为合并格式
        q_weight = dp_attn.qlinear.weight  # [embed_dim, embed_dim]
        k_weight = dp_attn.klinear.weight  # [embed_dim, embed_dim]
        v_weight = dp_attn.vlinear.weight  # [embed_dim, embed_dim]

        # 合并为 [3*embed_dim, embed_dim]
        combined_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        normal_attn.in_proj_weight.copy_(combined_weight)

        # 同步bias（如果存在）
        if (hasattr(dp_attn, 'qlinear') and dp_attn.qlinear.bias is not None and
            hasattr(normal_attn, 'in_proj_bias') and normal_attn.in_proj_bias is not None):
            q_bias = dp_attn.qlinear.bias
            k_bias = dp_attn.klinear.bias
            v_bias = dp_attn.vlinear.bias
            combined_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            normal_attn.in_proj_bias.copy_(combined_bias)

        # 同步输出投影层
        if hasattr(dp_attn, 'out_proj') and hasattr(normal_attn, 'out_proj'):
            normal_attn.out_proj.weight.copy_(dp_attn.out_proj.weight)
            if (dp_attn.out_proj.bias is not None and
                normal_attn.out_proj.bias is not None):
                normal_attn.out_proj.bias.copy_(dp_attn.out_proj.bias)

class TransformerClassifier(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=4, num_classes=2, dim_feedforward=128, dropout=0.3, dp_mode=False):
        super().__init__()
        self.dp_mode = dp_mode
        self.encoder = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, num_classes=num_classes, dim_feedforward=dim_feedforward, dropout=dropout, dp_mode=dp_mode)
        self.classifier = nn.Linear(d_model, num_classes)

        # 为DP模式创建eval_encoder用于一致的评估
        if dp_mode:
            self.eval_encoder = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, num_classes=num_classes, dim_feedforward=dim_feedforward, dropout=dropout, dp_mode=False)
            self._sync_non_attention_weights()
        else:
            self.eval_encoder = None

    def _sync_non_attention_weights(self):
        """同步非注意力层的权重（初始化时）"""
        if self.eval_encoder is None:
            return

        with torch.no_grad():
            # 同步位置编码
            if hasattr(self.encoder.pos_encoder, 'pe') and hasattr(self.eval_encoder.pos_encoder, 'pe'):
                self.eval_encoder.pos_encoder.pe.copy_(self.encoder.pos_encoder.pe)

            # 同步每一层的非注意力权重
            for dp_layer, eval_layer in zip(self.encoder.transformer_encoder.layers, self.eval_encoder.transformer_encoder.layers):
                # 同步前馈网络权重
                eval_layer.linear1.weight.copy_(dp_layer.linear1.weight)
                eval_layer.linear1.bias.copy_(dp_layer.linear1.bias)
                eval_layer.linear2.weight.copy_(dp_layer.linear2.weight)
                eval_layer.linear2.bias.copy_(dp_layer.linear2.bias)

                # 同步LayerNorm权重
                eval_layer.norm1.weight.copy_(dp_layer.norm1.weight)
                eval_layer.norm1.bias.copy_(dp_layer.norm1.bias)
                eval_layer.norm2.weight.copy_(dp_layer.norm2.weight)
                eval_layer.norm2.bias.copy_(dp_layer.norm2.bias)

    def sync_eval_encoder(self):
        """将DP encoder的当前权重同步到eval encoder"""
        if self.eval_encoder is None:
            return

        with torch.no_grad():
            # 同步每一层的权重
            for dp_layer, eval_layer in zip(self.encoder.transformer_encoder.layers, self.eval_encoder.transformer_encoder.layers):
                # 同步注意力权重（关键部分）
                sync_dp_to_normal_attention(dp_layer.self_attn, eval_layer.self_attn)

                # 同步前馈网络权重
                eval_layer.linear1.weight.copy_(dp_layer.linear1.weight)
                eval_layer.linear1.bias.copy_(dp_layer.linear1.bias)
                eval_layer.linear2.weight.copy_(dp_layer.linear2.weight)
                eval_layer.linear2.bias.copy_(dp_layer.linear2.bias)

                # 同步LayerNorm权重
                eval_layer.norm1.weight.copy_(dp_layer.norm1.weight)
                eval_layer.norm1.bias.copy_(dp_layer.norm1.bias)
                eval_layer.norm2.weight.copy_(dp_layer.norm2.weight)
                eval_layer.norm2.bias.copy_(dp_layer.norm2.bias)

    def forward(self, x, padding_mask=None, use_eval_encoder=False):
        x = torch.nan_to_num(x, nan=0.0)

        # 在评估模式下且有eval_encoder时，使用eval_encoder
        if use_eval_encoder and self.eval_encoder is not None and not self.training:
            # 先同步权重
            self.sync_eval_encoder()
            transformer_output = self.eval_encoder(x, padding_mask=padding_mask)
        else:
            transformer_output = self.encoder(x, padding_mask=padding_mask)

        pooled_output = pool(transformer_output, padding_mask=padding_mask)
        logits = self.classifier(pooled_output)
        logits = torch.nan_to_num(logits, nan=0.0)
        return logits, pooled_output

class AdvASRTransformerClassifier(TransformerClassifier):
    def __init__(self, d_model=32, nhead=4, num_layers=4, num_classes=2, dim_feedforward=128, dropout=0.3, asr_vocab_size=50, lambda_adv=0.5):
        super().__init__()
        self.backbone = TransformerClassifier(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes, # 效用分类器的输出维度
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dp_mode=False # 对抗模式不使用Opacus的DP层
        )

        # 2. 定义梯度反转层
        self.grl = GradientReversalLayer(lambda_adv)

        # 3. 定义ASR对抗头
        # 它接收来自backbone的pooled_output作为输入
        self.adversary_head = nn.Linear(d_model, asr_vocab_size)

    def forward(self, x, padding_mask=None):
        # 从主干网络获取效用logits和中间特征Z
        utility_logits, pooled_output = self.backbone(x, padding_mask)

        # 对抗分支
        reversed_pooled_output = self.grl(pooled_output)
        asr_logits = self.adversary_head(reversed_pooled_output)

        return utility_logits, asr_logits