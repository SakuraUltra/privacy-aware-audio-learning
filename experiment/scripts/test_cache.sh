#!/bin/bash

# 缓存测试脚本
echo "========================================"
echo "AIA缓存系统测试"
echo "========================================"

cd /home/wxy/projects/DPIB-MI/experiment

# 清理现有缓存
echo "清理现有缓存..."
rm -rf .cache/gender/*.pkl
rm -rf .cache/age_level/*.pkl
rm -rf .cache/education/*.pkl

echo "缓存目录状态:"
find .cache/ -name "*.pkl" | wc -l
echo "pkl文件数量: $(find .cache/ -name "*.pkl" | wc -l)"

echo
echo "首次运行gender攻击 (应该提取表征并缓存)..."
echo "========================================"