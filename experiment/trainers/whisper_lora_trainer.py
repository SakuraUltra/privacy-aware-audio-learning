# experiment/trainers/whisper_lora_trainer.py
import torch
from typing import Optional, Dict, Any
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_utils import set_seed

from models.whisper_lora import load_whisper_with_lora
from utils.data_utils import (
    DataCollatorSpeechSeq2SeqWithPadding,
    ensure_length_field_for_whisper,
)


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """
    自定义 Seq2SeqTrainer，确保只传递 Whisper 模型期望的参数
    """

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        重写 compute_loss 方法，确保只传递模型期望的参数
        """
        # 只保留 Whisper 模型期望的参数
        filtered_inputs = {
            "input_features": inputs["input_features"],
            "labels": inputs.get("labels", None)
        }

        # 移除可能存在的额外参数
        if "attention_mask" in inputs:
            filtered_inputs["attention_mask"] = inputs["attention_mask"]

        # 调用模型
        if self.label_smoother is not None and "labels" in filtered_inputs:
            labels = filtered_inputs.pop("labels")
        else:
            labels = None

        outputs = model(**filtered_inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            if isinstance(outputs, dict):
                loss = outputs["loss"] if "loss" in outputs else None
            else:
                loss = outputs[0] if len(outputs) > 0 else None

        return (loss, outputs) if return_outputs else loss


class SavePeftModelOnly(TrainerCallback):
    """只保存 LoRA 适配器（PEFT 权重），不把底座一并保存。"""
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            kwargs["model"].save_pretrained(args.output_dir)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束再保存一次，确保最终权重在
        if state.is_world_process_zero:
            kwargs["model"].save_pretrained(args.output_dir)
        return control


class WhisperLoRATrainer:
    """
    入口：WhisperLoRATrainer(config).train()
    依赖：
      - config.model: 见 configs/whisper_lora_config.py
      - config.training: lr/epochs/batch_size 等
      - config.data: processor/train_dataset/eval_dataset/compute_metrics 可直接塞入
    """

    def __init__(self, config):
        self.config = config
        self.device = config.general.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 随机种子
        seed = config.general.get("seed", None)
        if seed is not None:
            set_seed(seed)

        # 1) 模型与 Processor
        self.model, default_processor = load_whisper_with_lora(
            cfg_model=config.model,
            language=config.data.get("language"),
            task=config.data.get("task", "transcribe"),
        )
        self.processor = config.data.get("processor") or default_processor

        # 2) 数据集检查 - 支持单次训练和交叉验证两种模式
        self.train_ds = config.data.get("train_dataset")
        self.eval_ds = config.data.get("eval_dataset")
        has_train_eval = self.train_ds is not None and self.eval_ds is not None
        has_full_dataset = config.data.get("full_dataset") is not None

        if not has_train_eval and not has_full_dataset:
            raise ValueError("[WhisperLoRA] 数据集未提供。"
                             "请在 config.data 中提供 train_dataset/eval_dataset 或 full_dataset。")

        # 如果是交叉验证模式（只有 full_dataset），跳过训练器初始化
        if has_full_dataset and not has_train_eval:
            print("Cross-validation mode detected. Trainer will be initialized during CV.")
            return

        # 3) 为 group_by_length 补齐长度字段（若缺失则自动写入）
        length_col = config.training.get("length_column_name", "input_length")
        self.train_ds = ensure_length_field_for_whisper(self.train_ds, length_col)
        self.eval_ds = ensure_length_field_for_whisper(self.eval_ds, length_col)

        # 4) collator & metrics
        self.data_collator = config.data.get("data_collator") or DataCollatorSpeechSeq2SeqWithPadding(self.processor)
        self.compute_metrics = config.data.get("compute_metrics", None)  # 可为空

        # 5) 训练参数
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        out_dir = config.general.get("output_dir", "outputs/whisper-lora")

        self.args = Seq2SeqTrainingArguments(
            output_dir=out_dir,
            learning_rate=config.training["lr"],
            warmup_ratio=config.training["warmup_ratio"],
            num_train_epochs=config.training["epochs"],
            per_device_train_batch_size=config.training["per_device_train_batch_size"],
            per_device_eval_batch_size=config.training["per_device_eval_batch_size"],
            gradient_accumulation_steps=config.training["gradient_accumulation_steps"],
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=config.training["eval_steps"],
            save_steps=config.training["save_steps"],
            logging_steps=config.training["logging_steps"],
            predict_with_generate=True,
            generation_max_length=config.training["generation_max_length"],
            group_by_length=config.training.get("group_by_length", True),
            length_column_name=length_col,
            fp16=not use_bf16,
            bf16=use_bf16,
            dataloader_num_workers=4,
            save_total_limit=config.training.get("save_total_limit", 2),
            report_to=["none"],  # 需要可改成 ["tensorboard"]
            gradient_checkpointing=config.model.get("gradient_checkpointing", True),
            label_smoothing_factor=0.1,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            ddp_find_unused_parameters=False,
            remove_unused_columns=True,  # 只保留模型需要的列
        )

        # 6) HF Trainer - 使用自定义的 WhisperSeq2SeqTrainer
        # tokenizer 传入 processor.tokenizer，有利于预测时的解码
        self.trainer = WhisperSeq2SeqTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            data_collator=self.data_collator,
            tokenizer=getattr(self.processor, "tokenizer", None),
            compute_metrics=self.compute_metrics,
        )
        self.trainer.add_callback(SavePeftModelOnly())

    def train(self):
        train_output = self.trainer.train()
        self.trainer.save_model(self.args.output_dir)
        return train_output

    def load_speaker_folds(self, fold_file_path="../data/speaker_folds.csv"):
        """加载预定义的说话人折分配"""
        import pandas as pd

        print(f"Loading speaker folds from {fold_file_path}...")

        # 读取 speaker folds
        folds_df = pd.read_csv(fold_file_path)

        # 清理 speaker_id 列（去掉引号）
        folds_df['speaker_id'] = folds_df['speaker_id'].str.strip("'")

        # 按 fold 分组
        speaker_folds = {}
        for fold_num in range(1, 6):
            speakers = folds_df[folds_df['fold'] == fold_num]['speaker_id'].tolist()
            speaker_folds[fold_num] = speakers
            print(f"Fold {fold_num}: {len(speakers)} speakers")

        return speaker_folds

    def run_cross_validation(self, n_folds=5):
        """使用预定义的说话人折进行 5-fold cross validation"""
        import numpy as np

        print("=" * 60)
        print("Starting 5-Fold Cross Validation for Whisper LoRA")
        print("=" * 60)

        # 1. 加载预定义的说话人折
        speaker_folds = self.load_speaker_folds()

        # 2. 获取完整数据集
        if 'full_dataset' not in self.config.data:
            raise ValueError("full_dataset not found in config.data. Please prepare the complete dataset first.")

        full_dataset = self.config.data['full_dataset']
        print(f"Total dataset size: {len(full_dataset)} samples")

        fold_results = []

        # 3. 遍历每一轮 (每个fold作为验证集)
        for val_fold_num in range(1, n_folds + 1):
            print(f"\n{'='*50}")
            print(f"Round {val_fold_num}/{n_folds}")
            print(f"Validation Fold: {val_fold_num}")
            train_folds = [f for f in range(1, n_folds + 1) if f != val_fold_num]
            print(f"Training Folds: {train_folds}")
            print(f"{'='*50}")

            # 4. 当前轮的验证说话人 = 当前fold的说话人
            val_speakers = speaker_folds[val_fold_num]

            # 5. 当前轮的训练说话人 = 其他4个fold的说话人
            train_speakers = []
            for train_fold_num in train_folds:
                train_speakers.extend(speaker_folds[train_fold_num])

            print(f"Train speakers: {len(train_speakers)}")
            print(f"Val speakers: {len(val_speakers)}")

            # 6. 根据说话人划分样本
            train_indices = []
            val_indices = []

            for idx, speaker in enumerate(full_dataset['speaker']):
                if speaker in train_speakers:
                    train_indices.append(idx)
                elif speaker in val_speakers:
                    val_indices.append(idx)

            print(f"Train samples: {len(train_indices)}")
            print(f"Val samples: {len(val_indices)}")

            # 7. 创建当前轮的数据集
            round_train_dataset = full_dataset.select(train_indices)
            round_val_dataset = full_dataset.select(val_indices)

            # 8. 训练当前轮
            round_result = self._train_single_round(
                round_train_dataset,
                round_val_dataset,
                val_fold_num
            )
            fold_results.append(round_result)

        # 9. 计算5轮的平均结果
        avg_results = self._compute_average_metrics(fold_results)

        print("\n" + "=" * 60)
        print("5-Fold Cross Validation Completed!")
        print("=" * 60)

        return avg_results

    def _train_single_round(self, train_dataset, val_dataset, round_num):
        """训练单个轮次"""
        print(f"Training round {round_num}...")

        # 更新配置中的数据集
        self.config.data['train_dataset'] = train_dataset
        self.config.data['eval_dataset'] = val_dataset

        # 重新初始化模型（每轮都用新的模型）
        self.model, self.processor = load_whisper_with_lora(
            cfg_model=self.config.model,
            language=self.config.data.get('language', 'italian')
        )

        # 重新处理数据集
        self.train_ds = ensure_length_field_for_whisper(train_dataset)
        self.eval_ds = ensure_length_field_for_whisper(val_dataset)
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)

        # 重新初始化训练参数（更新输出目录）
        round_output_dir = f"{self.config.general['output_dir']}/fold_{round_num}"

        self.args = Seq2SeqTrainingArguments(
            output_dir=round_output_dir,
            per_device_train_batch_size=self.config.training["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config.training["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.config.training["gradient_accumulation_steps"],
            learning_rate=self.config.training["lr"],
            num_train_epochs=self.config.training["epochs"],
            warmup_ratio=self.config.training.get("warmup_ratio", 0.1),
            eval_strategy="steps",
            eval_steps=self.config.training["eval_steps"],
            save_steps=self.config.training["save_steps"],
            logging_steps=self.config.training["logging_steps"],
            save_total_limit=self.config.training["save_total_limit"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            report_to=None,
            dataloader_num_workers=0,
            remove_unused_columns=True,  # 只保留模型需要的列
            label_smoothing_factor=0.1,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            ddp_find_unused_parameters=False,
        )

        # 重新初始化训练器 - 使用自定义的 WhisperSeq2SeqTrainer
        self.trainer = WhisperSeq2SeqTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            data_collator=self.data_collator,
            tokenizer=getattr(self.processor, "tokenizer", None),
            compute_metrics=getattr(self, 'compute_metrics', None),
        )
        self.trainer.add_callback(SavePeftModelOnly())

        # 训练
        train_output = self.trainer.train()

        # 评估
        eval_results = self.trainer.evaluate()

        # 保存当前轮的模型
        self.trainer.save_model(round_output_dir)

        print(f"Round {round_num} completed!")
        print(f"Results: {eval_results}")

        return eval_results

    def _compute_average_metrics(self, fold_results):
        """计算所有折的平均指标"""
        import numpy as np

        if not fold_results:
            return {}

        # 提取所有指标名称
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())

        avg_metrics = {}

        for metric in all_metrics:
            values = []
            for result in fold_results:
                if metric in result and isinstance(result[metric], (int, float)):
                    values.append(result[metric])

            if values:
                avg_metrics[f"avg_{metric}"] = np.mean(values)
                avg_metrics[f"std_{metric}"] = np.std(values)

        print("\n=== Cross-Validation Results ===")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")

        return avg_metrics