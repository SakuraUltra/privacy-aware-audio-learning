#!/bin/bash

# AIA攻击测试脚本 - 支持不同输入模式
# 用法: ./test_single_aia.sh [input_mode] [attack_type] [attack_model_type] [mix_alpha]
# 例子: ./test_single_aia.sh representations_only gender mlp
#      ./test_single_aia.sh concatenation age_level transformer
#      ./test_single_aia.sh features_only education
#      ./test_single_aia.sh mix gender mlp 0.3

echo "=========================================="
echo "AIA攻击流程测试 - 支持输入模式选择"
echo "=========================================="

# 激活conda环境
source /home/wxy/workspace/miniconda3/etc/profile.d/conda.sh
conda activate py311

# 设置基础路径
cd /home/wxy/projects/DPIB-MI/experiment

# 解析参数
INPUT_MODE=${1:-"features_only"}      # 默认为features_only
ATTACK_TYPE=${2:-"gender"}           # 默认为gender
ATTACK_MODEL_TYPE=${3:-"mlp"}        # 默认为mlp
MIX_ALPHA=${4:-"0.5"}               # 默认为0.5（仅用于mix模式）

echo "项目目录: $(pwd)"
echo "当前时间: $(date)"
echo "输入模式: $INPUT_MODE"
echo "攻击类型: $ATTACK_TYPE" 
echo "攻击模型: $ATTACK_MODEL_TYPE"
if [ "$INPUT_MODE" = "mix" ]; then
    echo "Mix Alpha: $MIX_ALPHA"
fi
echo

# 测试配置：MEL特征 + Normal模型 + 指定攻击类型 + 指定输入模式
echo "测试配置: mel + dp + $ATTACK_TYPE + $ATTACK_MODEL_TYPE + $INPUT_MODE"
echo "----------------------------------------"

# 构建checkpoint路径
checkpoint_paths="checkpoints/last_epoch_model_mel_dp_fold_1.pth,checkpoints/last_epoch_model_mel_dp_fold_2.pth,checkpoints/last_epoch_model_mel_dp_fold_3.pth,checkpoints/last_epoch_model_mel_dp_fold_4.pth,checkpoints/last_epoch_model_mel_dp_fold_5.pth"

echo "检查点路径: $checkpoint_paths"
echo

# 执行AIA攻击
CUDA_VISIBLE_DEVICES=5 python train_unified.py \
    --feature_type mel \
    --mode aia \
    --checkpoint_paths "$checkpoint_paths" \
    --attack_type "$ATTACK_TYPE" \
    --target_model_mode dp \
    --attack_model_type "$ATTACK_MODEL_TYPE" \
    --input_mode "$INPUT_MODE" \
    --mix_alpha "$MIX_ALPHA" \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-3

exit_code=$?

echo
echo "=========================================="
if [[ $exit_code -eq 0 ]]; then
    echo "✅ AIA攻击测试完成 ($INPUT_MODE模式, $ATTACK_TYPE攻击)"
else
    echo "❌ AIA攻击测试失败 (exit code: $exit_code)"
fi
echo "完成时间: $(date)"
echo "=========================================="
