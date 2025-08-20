import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from transformers.modeling_outputs import BaseModelOutput
from torch.optim import AdamW
import evaluate

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import config

class OpenSMILEWhisperModel(pl.LightningModule):
    """
    使用OpenSMILE特征的Whisper模型 (最终修正版)
    """

    def __init__(self,
                 model_name=config.MODEL_NAME,
                 learning_rate=config.LEARNING_RATE,
                 opensmile_dim=config.OPENSMILE_FEATURE_DIM,
                 whisper_feature_dim=1280):
        super().__init__()
        self.save_hyperparameters()

        # 1. 加载预训练模型和分词器
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language=config.LANGUAGE, task=config.TASK)

        # 2. 冻结整个预训练的Whisper模型
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        print("✅ Pre-trained Whisper model has been frozen.")

        # 禁用缓存
        self.whisper_model.config.use_cache = False
        
        # 获取Whisper编码器期望的特征维度
        whisper_feature_dim = self.whisper_model.config.d_model
        print(f"Whisper model feature dimension: {whisper_feature_dim}")
        print(f"OpenSMILE feature dimension: {opensmile_dim}")

        # 3. 特征投影层 (您的自定义编码器)
        self.feature_projection = nn.Sequential(
            nn.Linear(opensmile_dim, whisper_feature_dim // 2),
            nn.LayerNorm(whisper_feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(whisper_feature_dim // 2, whisper_feature_dim),
            nn.LayerNorm(whisper_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 4. 自注意力层
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=whisper_feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 评估指标
        self.metric = evaluate.load("wer")

    def forward(self, input_features, labels=None):
        """
        【关键修正】统一的forward方法。
        根据是否提供 'labels' 参数，来决定是计算损失还是准备推理。
        """
        # 1. 投影特征
        projected_features = self.feature_projection(input_features)
        
        # 2. 应用自注意力
        enhanced_features, _ = self.feature_attention(
            projected_features, projected_features, projected_features
        )
        encoder_hidden_states = projected_features + enhanced_features

        # 如果提供了标签，则我们处于训练或验证（计算损失）模式
        if labels is not None:
            # 准备解码器输入
            decoder_input_ids = labels.clone()
            decoder_input_ids[decoder_input_ids == -100] = self.tokenizer.pad_token_id

            # 调用解码器
            decoder_outputs = self.whisper_model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True
            )
            
            # 使用预训练的lm_head
            lm_logits = self.whisper_model.proj_out(decoder_outputs.last_hidden_state)

            # 计算损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            return {'loss': loss, 'logits': lm_logits}
        
        # 如果没有提供标签，我们处于推理模式（为 `generate` 做准备）
        return {'encoder_hidden_states': encoder_hidden_states}

    def generate(self, input_features, **kwargs):
        """
        使用模型自带的generate函数
        """
        # 1. 调用forward方法（不带labels）来获取编码器状态
        forward_output = self.forward(input_features)
        encoder_hidden_states = forward_output['encoder_hidden_states']

        # 2. 将我们的特征包装成`BaseModelOutput`对象
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # 3. 调用Whisper内置的`generate`函数
        return self.whisper_model.generate(
            input_features=None,
            encoder_outputs=encoder_outputs,
            **kwargs
        )

    def training_step(self, batch, batch_idx):
        # 【修正】通过标准的 self() 调用统一的forward方法
        output = self(batch["input_features"], batch["labels"])
        loss = output['loss']
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 生成部分
        output_ids = self.generate(
            batch["input_features"],
            max_length=config.MAX_TEXT_LEN,
            forced_decoder_ids=self.tokenizer.get_decoder_prompt_ids(language=config.LANGUAGE, task=config.TASK)
        )
        
        labels = batch["labels"]
        
        # 解码
        predictions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        labels_cloned = labels.clone()
        labels_cloned[labels_cloned == -100] = self.tokenizer.pad_token_id
        references = self.tokenizer.batch_decode(labels_cloned, skip_special_tokens=True)
        
        # 计算WER
        wer = self.metric.compute(predictions=predictions, references=references)
        self.log("val/wer", wer, on_epoch=True, prog_bar=True, logger=True)

        # 损失计算部分
        val_output = self(batch["input_features"], batch["labels"])
        val_loss = val_output['loss']
        self.log("val/loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        
        # 打印样本
        if batch_idx == 0:
            print("\n--- Validation Samples ---")
            for i in range(min(4, len(predictions))):
                print(f"  [PRED] {predictions[i]}")
                print(f"  [REF ] {references[i]}")
                print("-" * 20)
            print("------------------------\n")

        return {"val_loss": val_loss, "wer": wer}

    def configure_optimizers(self):
        """
        只为需要训练的层创建优化器
        """
        trainable_params = list(self.feature_projection.parameters()) + list(self.feature_attention.parameters())
        
        print(f"✅ Optimizer configured. Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

        optimizer = AdamW(trainable_params, lr=self.hparams.learning_rate, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch"
            }
        }
