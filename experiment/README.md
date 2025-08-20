# Experiment Framework

这个文件夹包含了重构后的统一训练框架，支持不同的训练模式和特征类型。所有训练相关的代码都已整合到这个独立的experiment文件夹中。

## 文件夹结构

```
experiment/
├── configs/          # 配置管理
│   ├── base_config.py       # 基础配置类
│   ├── opensmile_config.py  # OpenSMILE特征配置
│   ├── mel_config.py        # MEL特征配置
│   └── config_factory.py    # 配置工厂
├── trainers/         # 训练器模块
│   ├── base_trainer.py      # 高级训练管理器（交叉验证、模型管理）
│   └── core_trainer.py      # 核心训练逻辑（epoch训练、评估）
├── models/           # 模型定义
│   └── transformer.py      # Transformer分类器
├── utils/            # 工具函数
│   ├── logging.py           # 日志工具
│   └── data_utils.py        # 数据处理工具
├── data/             # 数据处理模块
│   └── dataset.py           # 数据集类
├── audio_mel/        # MEL特征相关
│   ├── dataset.py           # MEL数据集
│   └── data-mel/            # MEL特征数据

├── train_unified.py  # 统一训练入口脚本
├── logs/             # 训练日志（自动创建）
└── README.md         # 说明文档
```

## 支持的训练模式

1. **Normal模式**: 标准训练
2. **DP模式**: 差分隐私训练
3. **VIB模式**: 变分信息瓶颈训练（可扩展）
4. **Adversarial模式**: 对抗训练（可扩展）

## 支持的特征类型

1. **OpenSMILE**: 传统音频特征
2. **MEL**: Whisper MEL特征

## 使用方法

### 在experiment文件夹内运行

```bash
# 进入experiment文件夹
cd experiment

# 激活conda环境
conda activate exp

# 运行OpenSMILE特征的普通训练
python train_unified.py --feature_type opensmile --mode normal

# 运行MEL特征的差分隐私训练
python train_unified.py --feature_type mel --mode dp --epsilon 8.0

# 运行MEL特征的VIB训练（使用默认参数）
python train_unified.py --feature_type mel --mode vib --epochs 50

# 运行VIB训练（自定义参数）
python train_unified.py --feature_type opensmile --mode vib \
    --z_dim 128 --beta 5e-3 --mc_samples 50 --epochs 20

# VIB参数说明：
# --z_dim: 隐变量维度（默认64），控制信息压缩程度，越小压缩越强
# --beta: KL正则化系数（默认1e-3），控制信息瓶颈强度，越大约束越强  
# --mc_samples: MC采样次数（默认30），评估时不确定性估计的采样数

# 运行OpenSMILE特征的对抗训练
python train_unified.py --feature_type opensmile --mode adversarial --epochs 30

# 查看所有可用参数
python train_unified.py --help
```

### 从项目根目录运行

```bash
# 使用shell脚本（推荐）
./run_unified_training.sh opensmile normal
./run_unified_training.sh mel dp --epsilon 8.0

# 或直接调用Python脚本
python experiment/train_unified.py --feature_type opensmile --mode normal
python experiment/train_unified.py --feature_type mel --mode vib \
    --z_dim 64 --beta 1e-3 --mc_samples 30 --epochs 50
```

## 重构的优势

1. **统一的入口点**: 一个脚本支持所有训练模式和特征类型
2. **模块化设计**: 清晰的文件夹结构，易于维护和扩展
3. **配置管理**: 统一的配置系统，支持不同模式的参数设置
4. **独立性**: experiment文件夹包含所有必要的依赖，可以独立运行
5. **向后兼容**: 保持与原有训练逻辑的兼容性

## 对比原有方式

### 原有方式（已废弃）
```bash
python main.py --em Normal
python main.py --em DP --epsilon 8.0
python main_mel.py --em Normal
python main_mel.py --em DP --epsilon 8.0
```

### 新的统一方式
```bash
python train_unified.py --feature_type opensmile --mode normal
python train_unified.py --feature_type opensmile --mode dp --epsilon 8.0
python train_unified.py --feature_type mel --mode normal
python train_unified.py --feature_type mel --mode dp --epsilon 8.0
python train_unified.py --feature_type mel --mode vib \
    --z_dim 128 --beta 5e-3 --mc_samples 50
```

## 开发指南

### 如何添加新的模型架构

如果你想添加新的模型架构（如VIB - Variational Information Bottleneck），请按照以下步骤：

#### 1. 创建新的模型文件

在 `models/` 目录下创建新的模型文件：

```python
# models/vib_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerClassifier

# 仅仅是一个示例，千万别照抄
class VIBModel(nn.Module):
    """Variational Information Bottleneck Model"""

    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, beta=1e-3):
        super(VIBModel, self).__init__()
        self.beta = beta

        # Encoder: maps input to latent distribution parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent distribution parameters
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Classifier: maps latent representation to predictions
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, padding_mask=None):
        """Forward pass with VIB loss computation"""
        # Encode to latent distribution
        mu, logvar = self.encode(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Classify
        logits = self.classifier(z)

        # Compute VIB loss components
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)

        return logits, {'kl_loss': kl_loss, 'beta': self.beta}
```

#### 2. 创建新的配置文件

在 `configs/` 目录下创建配置文件：

```python
# configs/vib_config.py
from .opensmile_config import OpenSMILEConfig
from .mel_config import MelConfig

class VIBConfig:
    def __init__(self, experiment_mode='vib', base_config='opensmile'):
        # 继承基础配置
        if base_config == 'opensmile':
            self.base_config = OpenSMILEConfig('normal')
        elif base_config == 'mel':
            self.base_config = MelConfig('normal')
        else:
            raise ValueError(f"Unsupported base config: {base_config}")

        # 复制基础配置
        self.__dict__.update(self.base_config.__dict__)

        # VIB specific parameters
        self.experiment_mode = experiment_mode
        self.model.update({
            'model_type': 'vib',
            'hidden_dim': 512,
            'latent_dim': 128,
            'beta': 1e-3,  # KL regularization weight
        })

        # Adjust training parameters for VIB
        self.training.update({
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'kl_warmup_epochs': 10,  # Gradually increase KL weight
        })
```

#### 3. 更新配置工厂

修改 `configs/config_factory.py` 以支持新模型：

```python
# configs/config_factory.py
from .vib_config import VIBConfig

def get_config(feature_type, experiment_mode='normal'):
    """Factory function to get appropriate config"""
    if experiment_mode == 'vib':
        # VIB可以应用于不同的特征类型
        if feature_type == 'opensmile':
            return VIBConfig(experiment_mode, base_config='opensmile')
        elif feature_type == 'mel':
            return VIBConfig(experiment_mode, base_config='mel')
        else:
            raise ValueError(f"VIB mode not supported for feature type: {feature_type}")
    elif feature_type == 'opensmile':
        return OpenSMILEConfig(experiment_mode)
    elif feature_type == 'mel':
        return MelConfig(experiment_mode)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
```

#### 4. 创建专用训练器（可选）

如果新模型需要特殊的训练逻辑，创建专用训练器：

```python
# trainers/vib_trainer.py
from .base_trainer import BaseTrainer
import torch.nn.functional as F

class VIBTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.kl_warmup_epochs = config.training.get('kl_warmup_epochs', 10)

    def create_model(self):
        """Create VIB model"""
        from models.vib_model import VIBModel

        model = VIBModel(
            input_dim=self.config.model['input_dim'],
            hidden_dim=self.config.model['hidden_dim'],
            latent_dim=self.config.model['latent_dim'],
            num_classes=self.config.model['num_classes'],
            beta=self.config.model['beta']
        )
        return model.to(self.config.general['device'])

    def compute_loss(self, logits, targets, model_outputs, epoch):
        """Compute VIB loss with KL warmup"""
        # Classification loss
        ce_loss = F.cross_entropy(logits, targets)

        # KL divergence loss with warmup
        kl_loss = model_outputs.get('kl_loss', 0)
        beta = model_outputs.get('beta', 1e-3)

        # Warmup schedule for KL weight
        kl_weight = min(1.0, epoch / self.kl_warmup_epochs) * beta

        total_loss = ce_loss + kl_weight * kl_loss

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item() if hasattr(kl_loss, 'item') else kl_loss,
            'kl_weight': kl_weight
        }
```

#### 5. 更新统一训练脚本

修改 `train_unified.py` 以支持新训练模式：

```python
# train_unified.py (添加到参数解析部分)
parser.add_argument('--mode', type=str,
                   choices=['normal', 'dp', 'vib', 'adversarial'],
                   default='normal',
                   help='Training mode to use')

# 在main函数中添加VIB支持
def main():
    # ... existing code ...

    if args.mode == 'vib':
        from trainers.vib_trainer import VIBTrainer
        trainer = VIBTrainer(config)
    elif args.mode == 'adversarial':
        from trainers.adversarial_trainer import AdversarialTrainer
        trainer = AdversarialTrainer(config)
    else:
        from trainers.base_trainer import BaseExperimentTrainer
        trainer = BaseExperimentTrainer(config)

    # ... rest of the code ...
```

### 如何添加新的训练方法

如果你想添加新的训练方法（如对抗训练、元学习等），请按照以下步骤：

#### 1. 继承BaseTrainer

```python
# trainers/adversarial_trainer.py
from .base_trainer import BaseTrainer
import torch
import torch.nn as nn

class AdversarialTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = config.adversarial.get('epsilon', 0.01)
        self.alpha = config.adversarial.get('alpha', 0.001)
        self.num_steps = config.adversarial.get('num_steps', 10)

    def adversarial_attack(self, x, y, model):
        """Generate adversarial examples using PGD"""
        x_adv = x.clone().detach()
        x_adv.requires_grad_(True)

        for _ in range(self.num_steps):
            logits, _ = model(x_adv)
            loss = nn.CrossEntropyLoss()(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + self.alpha * grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + delta
            x_adv = x_adv.detach()
            x_adv.requires_grad_(True)

        return x_adv

    def train_epoch(self, train_loader):
        """Training with adversarial examples"""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            x, y, mask = batch[0], batch[1], batch[2]
            x, y = x.to(self.device), y.to(self.device)

            # Generate adversarial examples
            x_adv = self.adversarial_attack(x, y, self.model)

            # Train on both clean and adversarial examples
            self.optimizer.zero_grad()

            # Clean loss
            logits_clean, _ = self.model(x)
            loss_clean = nn.CrossEntropyLoss()(logits_clean, y)

            # Adversarial loss
            logits_adv, _ = self.model(x_adv)
            loss_adv = nn.CrossEntropyLoss()(logits_adv, y)

            # Combined loss
            total_loss_batch = 0.5 * (loss_clean + loss_adv)
            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()

        return total_loss / len(train_loader)
```

#### 2. 添加相应配置

```python
# configs/base_config.py (添加到基础配置中)
class BaseConfig:
    def __init__(self, experiment_mode='normal'):
        # ... existing code ...

        # Adversarial training parameters
        self.adversarial = {
            'epsilon': 0.01,      # Attack strength
            'alpha': 0.001,       # Step size
            'num_steps': 10,      # Number of attack steps
        }
```

### 开发最佳实践

1. **保持接口一致性**: 新模型应该实现相同的forward接口
2. **配置驱动**: 所有超参数都应该通过配置文件管理
3. **模块化设计**: 将不同功能分离到不同的模块中
4. **测试驱动**: 为新功能编写单元测试
5. **文档完善**: 为新功能添加详细的文档和使用示例

### 使用新训练方法

添加完成后，你可以这样使用新的训练方法：

```bash
# 使用VIB训练方法处理MEL特征
python train_unified.py --feature_type mel --mode vib --epochs 50

# 使用VIB训练方法处理OpenSMILE特征
python train_unified.py --feature_type opensmile --mode vib --epochs 50

# 使用对抗训练处理OpenSMILE特征
python train_unified.py --feature_type opensmile --mode adversarial --epochs 30

# 使用对抗训练处理MEL特征
python train_unified.py --feature_type mel --mode adversarial --epochs 30
```

这个框架设计为高度可扩展的，你可以轻松地添加新的模型架构、训练方法和特征类型，而不需要修改核心训练逻辑。
