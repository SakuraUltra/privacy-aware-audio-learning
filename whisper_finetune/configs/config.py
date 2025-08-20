"""
Whisper微调的配置参数
"""

# Whisper模型配置
MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "it"  # 意大利语
TASK = "transcribe"

# OpenSMILE特征配置
OPENSMILE_FEATURE_DIM = 32  # OpenSMILE特征维度

# 训练配置
BATCH_SIZE = 24  # 增加batch size从4到8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

# 数据配置
MAX_TEXT_LEN = 100  # 增加最大文本长度从50到100

# 验证配置
VAL_CHECK_INTERVAL = 1.0  # 每个epoch结束后验证
EARLY_STOPPING_PATIENCE = 5

# 设备配置
GRADIENT_CLIP_VAL = 1.0
ACCUMULATE_GRAD_BATCHES = 1  # 减少梯度累积从2到1，让实际batch size更大
PRECISION = 16