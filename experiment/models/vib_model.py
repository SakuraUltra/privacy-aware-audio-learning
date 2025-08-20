import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from models.transformer import TransformerEncoder, pool

@dataclass
class VIBConfig:
    z_dim: int = 64
    beta: float = 1e-3
    beta_warmup_epochs: int = 1
    utility: str = "ce"          # ce | mse
    eval_mode: str = "mean"      # mean | sample
    mc_samples: int = 30
    prior: str = "std_normal"
    nan_safe: bool = True

class VIBHead(nn.Module):
    def __init__(self, in_dim: int, z_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.logvar = nn.Linear(in_dim, z_dim)
        nn.init.xavier_uniform_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.xavier_uniform_(self.logvar.weight); nn.init.zeros_(self.logvar.bias)

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_diag_gauss_stdnormal(mu, logvar):
        return (0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - 1.0 - logvar, dim=1)).mean()

    def forward(self, h: torch.Tensor, sample: bool = True):
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar) if sample else mu
        kl = self.kl_diag_gauss_stdnormal(mu, logvar)
        return z, mu, logvar, kl

class VIBClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, cfg: VIBConfig):
        super().__init__()
        self.cfg = cfg
        self.head = VIBHead(in_dim, cfg.z_dim)
        self.utility_head = nn.Linear(cfg.z_dim, num_classes)
        nn.init.xavier_uniform_(self.utility_head.weight); nn.init.zeros_(self.utility_head.bias)

    def forward(self, h: torch.Tensor, sample: Optional[bool] = None):
        if sample is None:
            sample = (self.training or self.cfg.eval_mode == "sample")
        if self.cfg.nan_safe:
            h = torch.nan_to_num(h, nan=0.0)
        z, mu, logvar, kl = self.head(h, sample=sample)
        logits = self.utility_head(z)
        if self.cfg.nan_safe:
            logits = torch.nan_to_num(logits, nan=0.0)
        return {"logits": logits, "z": z, "mu": mu, "logvar": logvar, "kl": kl, "h": h}

    def compute_loss(self, h: torch.Tensor, y: torch.Tensor, beta: float):
        out = self.forward(h, sample=True)
        if self.cfg.utility == "ce":
            util = F.cross_entropy(out["logits"], y.long())
        elif self.cfg.utility == "mse":
            util = F.mse_loss(out["logits"], y)
        else:
            raise ValueError("utility must be 'ce' or 'mse'")
        loss = util + beta * out["kl"]
        return loss, {"util": util.detach(), "kl": out["kl"].detach(), "logits": out["logits"].detach()}

    @torch.no_grad()
    def mc_predict(self, h: torch.Tensor, mc: Optional[int] = None):
        mc = mc or self.cfg.mc_samples
        self.eval()
        pooled = pool(h)
        logits_list = []
        for _ in range(mc):
            out = self.forward(pooled, sample=True)
            logits_list.append(out["logits"])
        return torch.stack(logits_list, dim=0).mean(0)

class BetaWarmup:
    def __init__(self, target_beta: float, warmup_steps: int):
        self.target = float(target_beta)
        self.warmup_steps = int(max(0, warmup_steps))
    def __call__(self, global_step: int) -> float:
        if self.warmup_steps == 0:
            return self.target
        ratio = min(1.0, float(global_step) / float(self.warmup_steps))
        return self.target * ratio

class VIBTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, num_classes: int, dim_feedforward: int, dropout: float, vib_cfg: VIBConfig, dp_mode: bool = False):
        super().__init__()
        self.encoder = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, num_classes=num_classes, dim_feedforward=dim_feedforward, dropout=dropout, dp_mode=dp_mode)
        self.vib = VIBClassifier(in_dim=d_model, num_classes=num_classes, cfg=vib_cfg)

    def forward(self, x: torch.Tensor, padding_mask=None, sample: Optional[bool] = None):
        transformer_output = self.encoder(x, padding_mask=padding_mask)
        pooled = pool(transformer_output, padding_mask=padding_mask)
        vib_out = self.vib(pooled, sample=sample)
        return vib_out["logits"], pooled, vib_out

# 构建函数

def build_vib_model(base_config, vib_cfg_dict: Optional[Dict[str, Any]] = None, steps_per_epoch: Optional[int] = None) -> Tuple[VIBTransformer, BetaWarmup, VIBConfig]:
    vib_cfg = VIBConfig()
    if vib_cfg_dict:
        for k, v in vib_cfg_dict.items():
            if hasattr(vib_cfg, k):
                setattr(vib_cfg, k, v)
    warmup_steps = 0
    if steps_per_epoch is not None and vib_cfg.beta_warmup_epochs > 0:
        warmup_steps = steps_per_epoch * vib_cfg.beta_warmup_epochs
    beta_sched = BetaWarmup(vib_cfg.beta, warmup_steps)
    mcfg = base_config.MODEL
    model = VIBTransformer(d_model=mcfg['d_model'], nhead=mcfg['nhead'], num_layers=mcfg['num_layers'], num_classes=mcfg['num_classes'], dim_feedforward=mcfg['dim_feedforward'], dropout=mcfg['dropout'], vib_cfg=vib_cfg, dp_mode=(base_config.EXPERIMENT_MODE == 'DP'))
    return model, beta_sched, vib_cfg
