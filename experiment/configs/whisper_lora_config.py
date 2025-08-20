# experiment/configs/whisper_lora_config.py
from .base_config import BaseConfig

class WhisperLoRAConfig(BaseConfig):
    """
    Whisper + (Q)LoRA 训练的专用配置
    通过 train_unified.py --mode whisper_lora 使用，不影响你现有 normal/dp/vib/adversarial 流水线
    """
    def __init__(self):
        super().__init__()
        self.experiment_mode = 'whisper_lora'

        # ===== 模型相关（可按需调整）=====
        self.model = {
            'base_model': 'openai/whisper-large-v3-turbo',   # 使用 large-v3-turbo 模型
            'quantization': '4bit',                 # '4bit' | '8bit' | 'none'
            'attn_impl': 'sdpa',                    # 'sdpa'；如果环境支持可换 'flash_attention_2'
            'gradient_checkpointing': True,

            # LoRA 超参（balanced 起步）
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'target_modules': ['q_proj', 'v_proj'],  # 先用最省显存的组合
        }

        # ===== 训练超参（HF Seq2SeqTrainer 会读取）=====
        self.training.update({
            'lr': 1e-4,
            'warmup_ratio': 0.1,
            'epochs': 3,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'gradient_accumulation_steps': 2,
            'eval_steps': 500,
            'save_steps': 500,
            'logging_steps': 50,
            'generation_max_length': 128,
            'group_by_length': False,  # 暂时禁用以避免长度推断问题
            'length_column_name': 'input_length',   # 预处理时最好为每条样本写入
            'save_total_limit': 2,
        })

        # ===== 数据对接位（后续你把现有对象塞进来即可）=====
        self.data = {
            'language': None,            # 例如 'en'；用于 decoder prompt，可留 None
            'task': 'transcribe',
            'train_dataset': None,       # hf Dataset
            'eval_dataset': None,
            'processor': None,           # WhisperProcessor
            'data_collator': None,       # 可留空，Trainer 内部自动创建
            'compute_metrics': None,     # 函数 -> 返回 dict（如 {'wer': ...}）
        }

        # ===== 输出目录 =====
        self.general['output_dir'] = 'logs'

    def get_dataset_config(self):
        """获取 Whisper LoRA 数据集特定配置"""
        return {
            'feature_type': 'whisper_audio',  # 区别于 opensmile/mel
            'language': self.data.get('language', 'it'),
            'task': self.data.get('task', 'transcribe'),
        }

    def get_model_input_dim(self):
        """获取 Whisper 模型输入维度（MEL 特征维度）"""
        # Whisper 使用 80 维 MEL 特征
        return 80