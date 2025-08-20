# DPIB-MI: 基于差分隐私和信息瓶颈的音频分类系统

## 📖 项目简介

DPIB-MI 是一个先进的音频分类深度学习框架，专注于隐私保护的音频特征分析。该项目结合了差分隐私(Differential Privacy)、变分信息瓶颈(Variational Information Bottleneck)等前沿技术，为音频分类任务提供了多种训练模式和特征提取方法。

### 🎯 主要特性

- **多种训练模式**: 支持普通训练、差分隐私训练、变分信息瓶颈训练和Whisper LoRA微调
- **双特征支持**: 同时支持OpenSMILE传统音频特征和Whisper MEL特征
- **隐私保护**: 集成Opacus差分隐私框架，保护训练数据隐私
- **信息理论**: 实现变分信息瓶颈，平衡模型性能与信息压缩
- **现代架构**: 基于Transformer架构的分类器
- **交叉验证**: 支持5折说话人感知的交叉验证
- **模块化设计**: 高度可扩展的框架结构

## 🏗️ 项目结构

```
DPIB-MI/
├── experiment/                    # 统一训练框架 (推荐使用)
│   ├── configs/                   # 配置管理
│   │   ├── base_config.py         # 基础配置类
│   │   ├── opensmile_config.py    # OpenSMILE特征配置
│   │   ├── mel_config.py          # MEL特征配置
│   │   ├── vib_config.py          # VIB训练配置
│   │   ├── whisper_lora_config.py # Whisper LoRA配置
│   │   └── config_factory.py      # 配置工厂
│   ├── trainers/                  # 训练器模块
│   │   ├── base_trainer.py        # 基础训练器
│   │   ├── core_trainer.py        # 核心训练逻辑
│   │   ├── vib_trainer.py         # VIB训练器
│   │   └── whisper_lora_trainer.py # Whisper LoRA训练器
│   ├── models/                    # 模型定义
│   │   ├── transformer.py         # Transformer分类器
│   │   ├── vib_model.py          # VIB模型
│   │   └── whisper_lora.py       # Whisper LoRA模型
│   ├── utils/                     # 工具函数
│   ├── data/                      # 数据处理模块
│   ├── train_unified.py           # 统一训练入口
│   └── README.md                  # 详细使用说明
├── data/                          # 数据文件
│   ├── all_expanded_features.csv  # 特征数据索引
│   ├── speaker_folds.csv          # 说话人折叠信息
│   ├── extracted_features_train/  # 训练特征文件
│   ├── extracted_features_val/    # 验证特征文件
│   ├── slices_64/                 # 音频切片
│   └── slices_txt_64/             # 转录文本
├── whisper_finetune/              # Whisper微调相关
├── audio_mel/                     # MEL特征处理
└── AudioWhisper_Train_v00001.ipynb # Whisper训练演示
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (GPU训练)

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd DPIB-MI

# 创建conda环境
conda create -n dpib-mi python=3.9
conda activate dpib-mi

# 安装PyTorch (根据你的CUDA版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install transformers
pip install opacus
pip install peft
pip install opensmile
pip install librosa
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
```

### 基本使用

#### 1. 使用统一训练框架 (推荐)

```bash
# 进入experiment目录
cd experiment

# OpenSMILE特征 + 普通训练
python train_unified.py --feature_type opensmile --mode normal --epochs 20

# MEL特征 + 差分隐私训练
python train_unified.py --feature_type mel --mode dp --epsilon 8.0 --epochs 15

# OpenSMILE特征 + VIB训练
python train_unified.py --feature_type opensmile --mode vib \
    --z_dim 128 --beta 5e-3 --mc_samples 50 --epochs 25

# Whisper LoRA微调
python train_unified.py --feature_type mel --mode whisper_lora \
    --epochs 10 --use_cv
```

#### 2. 参数说明

**基本参数:**
- `--feature_type`: 特征类型 (`opensmile` | `mel`)
- `--mode`: 训练模式 (`normal` | `dp` | `vib` | `whisper_lora`)
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率

**VIB参数:**
- `--z_dim`: 隐变量维度 (默认64)
- `--beta`: KL正则化强度 (默认1e-3)
- `--mc_samples`: MC采样次数 (默认30)

**差分隐私参数:**
- `--epsilon`: 隐私预算 (默认8.0)

## 🔬 支持的训练模式

### 1. 普通训练 (Normal)
标准的监督学习训练，使用交叉熵损失函数。

### 2. 差分隐私训练 (DP)
使用Opacus框架实现差分隐私训练，保护训练数据的隐私。
- 支持梯度裁剪和噪声注入
- 可配置隐私预算(epsilon)和delta值

### 3. 变分信息瓶颈训练 (VIB)
实现信息理论中的变分信息瓶颈原理：
- 学习压缩的表示，平衡预测性能和信息压缩
- 支持不确定性量化
- 可配置隐变量维度和KL正则化强度

### 4. Whisper LoRA微调
基于Whisper模型的低秩适应微调：
- 使用LoRA技术高效微调大型预训练模型
- 支持4bit/8bit量化训练
- 可配置LoRA参数和目标模块

## 📊 数据格式

### 特征数据
项目支持两种特征类型：

1. **OpenSMILE特征**: 传统音频特征，包含6373维特征向量
2. **MEL特征**: Whisper模型提取的MEL频谱特征

### 数据组织
- `all_expanded_features.csv`: 包含音频路径、标签、说话人ID和特征文件路径
- `speaker_folds.csv`: 5折交叉验证的说话人分组信息
- `extracted_features_*/`: 预提取的特征文件(CSV格式)

## 🎛️ 配置系统

项目采用模块化的配置系统，支持不同训练模式的参数配置：

- `BaseConfig`: 基础配置类，定义通用参数
- `OpenSMILEConfig`: OpenSMILE特征专用配置
- `MelConfig`: MEL特征专用配置
- `VIBConfig`: VIB训练专用配置
- `WhisperLoRAConfig`: Whisper LoRA专用配置

## 🔧 扩展开发

### 添加新的训练模式

1. 在`models/`目录下创建新的模型文件
2. 在`configs/`目录下创建对应的配置文件
3. 在`trainers/`目录下创建专用训练器
4. 更新`config_factory.py`和`train_unified.py`

详细的开发指南请参考 `experiment/README.md`。

## 📈 实验结果

项目支持详细的实验日志记录，包括：
- 训练/验证损失和准确率
- 混淆矩阵和分类报告
- 模型检查点保存
- TensorBoard可视化支持

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 预训练语音模型
- [Opacus](https://github.com/pytorch/opacus) - 差分隐私框架
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调
- [OpenSMILE](https://github.com/audeering/opensmile) - 音频特征提取

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至项目维护者

---

**注意**: 本项目仅用于研究目的，请确保在使用时遵守相关的数据隐私和伦理规范。
