#!/bin/bash

# MINE隐私保护训练脚本 - OpenSMILE特征
# 用法: ./mine_single_opensmile.sh [mine_model_type] [privacy_weight] [mine_lr]
# 例子: ./mine_single_opensmile.sh timeaware 0.3 1e-4
#      ./mine_single_opensmile.sh standard 0.2 5e-4

echo "=========================================="
echo "MINE隐私保护训练 - OpenSMILE特征"
echo "=========================================="

# 激活conda环境
source /home/wxy/workspace/miniconda3/etc/profile.d/conda.sh
conda activate py311

# 设置基础路径
cd /home/wxy/projects/DPIB-MI/experiment

# 解析参数
MINE_MODEL_TYPE=${1:-"timeaware"}    # 默认为timeaware
PRIVACY_WEIGHT=${2:-"0.3"}          # 默认隐私权重0.3
MINE_LR=${3:-"1e-4"}                # 默认MINE学习率1e-4

echo "项目目录: $(pwd)"
echo "当前时间: $(date)"
echo "MINE模型类型: $MINE_MODEL_TYPE"
echo "隐私权重: $PRIVACY_WEIGHT"
echo "MINE学习率: $MINE_LR"
echo

# 测试配置：OpenSMILE特征 + MINE隐私保护
echo "测试配置: opensmile + mine + $MINE_MODEL_TYPE"
echo "配置详情:"
echo "  - 特征类型: OpenSMILE"
echo "  - 训练模式: MINE"
echo "  - MINE模型: $MINE_MODEL_TYPE"
echo "  - 隐私权重: $PRIVACY_WEIGHT"
echo "  - MINE学习率: $MINE_LR"
echo "----------------------------------------"

# 执行MINE训练
CUDA_VISIBLE_DEVICES=5 python train_unified.py \
    --feature_type opensmile \
    --mode mine \
    --mine_model_type "$MINE_MODEL_TYPE" \
    --privacy_weight "$PRIVACY_WEIGHT" \
    --mine_lr "$MINE_LR" \
    --task_type classification \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4

exit_code=$?

echo
echo "=========================================="
if [[ $exit_code -eq 0 ]]; then
    echo "✅ MINE隐私保护训练完成 (OpenSMILE + $MINE_MODEL_TYPE)"
    echo "   Privacy Weight: $PRIVACY_WEIGHT"
    echo "   MINE LR: $MINE_LR"
else
    echo "❌ MINE隐私保护训练失败 (exit code: $exit_code)"
fi
echo "完成时间: $(date)"
echo "=========================================="