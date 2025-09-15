#!/bin/bash

# MINE模型AIA攻击脚本
# 用法: ./mine_single_aia.sh [feature_type] [attack_type] [attack_model_type] [input_mode]
# 例子: ./mine_single_aia.sh opensmile gender mlp features_only
#      ./mine_single_aia.sh mel age_level transformer representations_only

echo "=========================================="
echo "MINE模型AIA攻击测试"
echo "=========================================="

# 激活conda环境
source /home/wxy/workspace/miniconda3/etc/profile.d/conda.sh
conda activate py311

# 设置基础路径
cd /home/wxy/projects/DPIB-MI/experiment

# 解析参数
FEATURE_TYPE=${1:-"mel"}        # 默认为opensmile
ATTACK_TYPE=${2:-"gender"}            # 默认为gender攻击
ATTACK_MODEL_TYPE=${3:-"mlp"}         # 默认为mlp攻击模型
INPUT_MODE=${4:-"features_only"}      # 默认为features_only模式

echo "项目目录: $(pwd)"
echo "当前时间: $(date)"
echo "特征类型: $FEATURE_TYPE"
echo "攻击类型: $ATTACK_TYPE"
echo "攻击模型: $ATTACK_MODEL_TYPE"
echo "输入模式: $INPUT_MODE"
echo

# 根据特征类型设置检查点路径
if [[ "$FEATURE_TYPE" == "opensmile" ]]; then
    CHECKPOINT_PATHS="checkpoints/last_epoch_model_opensmile_mine_fold_0.pth,checkpoints/last_epoch_model_opensmile_mine_fold_1.pth,checkpoints/last_epoch_model_opensmile_mine_fold_2.pth,checkpoints/last_epoch_model_opensmile_mine_fold_3.pth,checkpoints/last_epoch_model_opensmile_mine_fold_4.pth"
elif [[ "$FEATURE_TYPE" == "mel" ]]; then
    CHECKPOINT_PATHS="checkpoints/last_epoch_model_mel_timeaware_fold_1.pth,checkpoints/last_epoch_model_mel_timeaware_fold_2.pth,checkpoints/last_epoch_model_mel_timeaware_fold_3.pth,checkpoints/last_epoch_model_mel_timeaware_fold_4.pth,checkpoints/last_epoch_model_mel_timeaware_fold_5.pth"
else
    echo "❌ 不支持的特征类型: $FEATURE_TYPE"
    exit 1
fi

# 测试配置信息
echo "测试配置: $FEATURE_TYPE + mine + aia + $ATTACK_TYPE"
echo "配置详情:"
echo "  - 特征类型: $FEATURE_TYPE"
echo "  - 目标模型: MINE"
echo "  - 攻击类型: $ATTACK_TYPE"
echo "  - 攻击模型: $ATTACK_MODEL_TYPE"
echo "  - 输入模式: $INPUT_MODE"
echo "  - 检查点路径: $CHECKPOINT_PATHS"
echo "----------------------------------------"

# 执行AIA攻击
CUDA_VISIBLE_DEVICES=4 python train_unified.py \
    --feature_type "$FEATURE_TYPE" \
    --mode aia \
    --attack_type "$ATTACK_TYPE" \
    --target_model_mode mine \
    --attack_model_type "$ATTACK_MODEL_TYPE" \
    --input_mode "$INPUT_MODE" \
    --checkpoint_paths "$CHECKPOINT_PATHS" \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3

exit_code=$?

echo
echo "=========================================="
if [[ $exit_code -eq 0 ]]; then
    echo "✅ MINE模型AIA攻击完成"
    echo "   特征类型: $FEATURE_TYPE"
    echo "   攻击类型: $ATTACK_TYPE ($ATTACK_MODEL_TYPE)"
    echo "   输入模式: $INPUT_MODE"
else
    echo "❌ MINE模型AIA攻击失败 (exit code: $exit_code)"
fi
echo "完成时间: $(date)"
echo "=========================================="