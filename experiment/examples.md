# 使用示例

## 基本用法

### 1. OpenSMILE特征 + 普通训练
```bash
python train_unified.py --feature_type opensmile --mode normal
```

### 5. MEL特征 + VIB训练
```bash
python train_unified.py --feature_type mel --mode vib
```

### 6. OpenSMILE特征 + VIB训练
```bash
python train_unified.py --feature_type opensmile --mode vib
```
```

### 2. MEL特征 + 普通训练
```bash
python train_unified.py --feature_type mel --mode normal
```

### 3. OpenSMILE特征 + 差分隐私训练
```bash
python train_unified.py --feature_type opensmile --mode dp --epsilon 8.0
```

### 4. MEL特征 + 差分隐私训练
```bash
python train_unified.py --feature_type mel --mode dp --epsilon 10.0
```

### 5. MEL特征 + VIB训练
```bash
python train_unified.py --feature_type mel --mode vib --z_dim 64 --beta 1e-3 --mc_samples 30 --epochs 2
```

## 使用shell脚本

### 1. 查看帮助
```bash
./run_unified_training.sh --help
```

### 2. 运行OpenSMILE普通训练
```bash
./run_unified_training.sh opensmile normal
```

### 3. 运行MEL差分隐私训练
```bash
./run_unified_training.sh mel dp --epsilon 8.0
```

### 4. 自定义参数
```bash
./run_unified_training.sh opensmile normal --epochs 20 --batch_size 64 --lr 2e-4
```

## 高级参数

### 完整参数示例
```bash
python train_unified.py \
    --feature_type mel \
    --mode dp \
    --epsilon 8.0 \
    --epochs 15 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --num_layers 6 \
    --nhead 8 \
    --dropout 0.2 \
    --num_folds 5 \
    --seed 42
```

## 对比原有脚本

### 原有方式
```bash
# 原来需要分别运行不同的脚本
python main.py --em Normal
python main.py --em DP --epsilon 8.0
python main_mel.py --em Normal
python main_mel.py --em DP --epsilon 8.0
```

### 新的统一方式
```bash
# 现在只需要一个脚本，通过参数控制
python train_unified.py --feature_type opensmile --mode normal
python train_unified.py --feature_type opensmile --mode dp --epsilon 8.0
python train_unified.py --feature_type mel --mode normal
python train_unified.py --feature_type mel --mode dp --epsilon 8.0
```

## 日志文件

训练日志会自动保存到 `experiment/logs/` 目录下，文件名格式：
- `training_log_{feature_type}_{mode}_{timestamp}.txt`
- `training_log_{feature_type}_{mode}_eps{epsilon}_{timestamp}.txt` (DP模式)

例如：
- `training_log_opensmile_normal_20250811_143022.txt`
- `training_log_mel_dp_eps8.0_20250811_143022.txt`