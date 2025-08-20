# experiment/models/whisper_lora.py
import torch
from typing import Tuple, Optional, Dict, Any

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def _select_compute_dtype() -> torch.dtype:
    """
    A100/H100(>=SM80) 优先 bfloat16，其它 GPU 优先 float16；CPU 则 float32。
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def _get_bnb_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    """
    构造 bitsandbytes 的量化配置:
      - '4bit' -> NF4 + double quant + fp16/bf16 计算
      - '8bit' -> 8-bit 量化
      - 'none' -> 不量化（返回 None）
    """
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_select_compute_dtype(),
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _build_lora_config(cfg: Dict[str, Any]) -> LoraConfig:
    """
    从 config.model 生成 LoRA 配置。
    关键键：lora_r, lora_alpha, lora_dropout, target_modules

    注意：不设置 task_type 以避免 PEFT 包装器的 input_ids 错误
    参考：https://github.com/huggingface/peft/issues/1988
    """
    return LoraConfig(
        r=cfg.get("lora_r", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        # task_type="SEQ_2_SEQ_LM",  # 移除此行以避免 input_ids 错误
        inference_mode=False,
    )


def _maybe_set_decoder_prompt(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    language: Optional[str],
    task: Optional[str] = "transcribe",
):
    """
    可选：为推理/评估设置 decoder 强制前缀（语言与任务）。
    训练本身不强制需要；但在 eval/generate 时可减少语言漂移。
    """
    if language:
        try:
            forced_ids = processor.get_decoder_prompt_ids(language=language, task=task or "transcribe")
            # 保存在 generation_config，Trainer 预测时会用到
            model.generation_config.forced_decoder_ids = forced_ids
        except Exception as e:
            print(f"[WhisperLoRA] Skip setting decoder prompt: {e}")


def load_whisper_with_lora(cfg_model: Dict[str, Any],
                           language: Optional[str] = None,
                           task: Optional[str] = "transcribe"
                           ) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    核心入口：加载 Whisper + 挂载 (Q)LoRA，并返回 (model, processor)

    Args:
        cfg_model: 取自 config.model 的字典，需包含：
            - base_model: str (e.g., 'openai/whisper-small')
            - quantization: '4bit' | '8bit' | 'none'
            - attn_impl: 'sdpa' | 'flash_attention_2'（环境支持再用）
            - gradient_checkpointing: bool
            - lora_r / lora_alpha / lora_dropout / target_modules
        language: 可选，设置 decoder 提示语言（eval/infer 更稳定）
        task: 'transcribe' 或 'translate'
    """
    base = cfg_model.get("base_model", "openai/whisper-small")
    bnb_cfg = _get_bnb_config(cfg_model.get("quantization", "4bit"))
    torch_dtype = _select_compute_dtype()

    # 1) 加载底座模型
    model = WhisperForConditionalGeneration.from_pretrained(
        base,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=bnb_cfg,               # None 时不会触发量化
        attn_implementation=cfg_model.get("attn_impl", "sdpa"),
    )

    # 2) 梯度/检查点设置
    model.config.use_cache = False
    if cfg_model.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        # 使输入需要梯度，确保量化场景下仍可反传
        model.enable_input_require_grads()

    # 3) 若是 k-bit 量化，准备可训练层的前置处理
    if bnb_cfg is not None:
        model = prepare_model_for_kbit_training(model)

    # 4) 挂载 LoRA
    lora_cfg = _build_lora_config(cfg_model)
    model = get_peft_model(model, lora_cfg)

    # 5) 打印可训练参数占比，便于快速核对 LoRA 是否生效
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"[LoRA] Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100.0 * trainable / max(total, 1):.2f}%) | dtype={torch_dtype}")

    # 6) Processor（特征器+分词器）
    processor = WhisperProcessor.from_pretrained(base)

    # 7) （可选）为推理/评估设置 decoder 前缀
    _maybe_set_decoder_prompt(model, processor, language=language, task=task)

    return model, processor