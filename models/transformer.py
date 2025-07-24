import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPMultiheadAttention
import warnings

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

class TransformerClassifier(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=4, num_classes=2, dim_feedforward=128, dropout=0.3, dp_mode=False):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.dp_mode = dp_mode
        if dp_mode:
            encoder_layer = DPTransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                      dim_feedforward=dim_feedforward, dropout=dropout,
                                                      batch_first=True)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=dim_feedforward, dropout=dropout,
                                                       batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        x = torch.nan_to_num(x, nan=0.0)
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            pooling_mask = ~padding_mask
            lengths = pooling_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled_output = (transformer_output * pooling_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled_output = transformer_output.mean(dim=1)
        pooled_output = torch.nan_to_num(pooled_output, nan=0.0)
        logits = self.classifier(pooled_output)
        logits = torch.nan_to_num(logits, nan=0.0)
        return logits, pooled_output