#!/bin/bash

# AIA攻击测试脚本 - 完整版本
# 测试所有三个属性：gender, age_level, education

echo "=========================================="
echo "AIA攻击完整测试 - 所有属性"
echo "=========================================="

# 激活conda环境
source /home/wxy/workspace/miniconda3/etc/profile.d/conda.sh
conda activate py311

# 设置基础路径
cd /home/wxy/projects/DPIB-MI/experiment

echo "项目目录: $(pwd)"
echo "当前时间: $(date)"
echo

# 构建checkpoint路径
checkpoint_paths="checkpoints/last_epoch_model_mel_normal_fold_1.pth,checkpoints/last_epoch_model_mel_normal_fold_2.pth,checkpoints/last_epoch_model_mel_normal_fold_3.pth,checkpoints/last_epoch_model_mel_normal_fold_4.pth,checkpoints/last_epoch_model_mel_normal_fold_5.pth"

echo "检查点路径: $checkpoint_paths"
echo

# 定义要测试的属性列表
attributes=("gender" "age_level" "education")

# 循环测试每个属性
for attr in "${attributes[@]}"; do
    echo "=========================================="
    echo "测试属性: $attr"
    echo "配置: mel + normal + $attr + mlp"
    echo "=========================================="
    
    # 执行AIA攻击
    CUDA_VISIBLE_DEVICES=5 python train_unified.py \
        --feature_type mel \
        --mode aia \
        --checkpoint_paths "$checkpoint_paths" \
        --attack_type "$attr" \
        --target_model_mode normal \
        --attack_model_type mlp \
        --batch_size 32 \
        --lr 1e-3
    
    exit_code=$?
    
    echo
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ $attr 攻击测试完成"
    else
        echo "❌ $attr 攻击测试失败 (exit code: $exit_code)"
        echo "停止后续测试"
        break
    fi
    echo
    echo "等待5秒后进行下一个测试..."
    sleep 5
done

echo "=========================================="
echo "所有AIA攻击测试完成"
echo "完成时间: $(date)"
echo "=========================================="