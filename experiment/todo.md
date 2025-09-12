# AIA Implementation Todo List

## Phase 1: 基础设施建设

### 1.1 数据处理组件 
- [x] **人口统计信息提取器** (`data/demographic_extractor.py`) ✅
  - 从文件名解析性别（M/F）、年龄、教育等级
  - 年龄按统计分布分为5个层次分类
  - 生成人口统计标签映射表
  - 实现批量处理和缓存功能

### 1.2 表征提取工具
- [x] **自动化隐表征提取器** (`utils/representation_extractor.py`) ✅
  - 从5个fold的checkpoint自动加载模型
  - 支持Normal、DP、VIB三种训练模式
  - 支持opensmile和mel两种特征类型
  - 提取指定层的隐层表征
  - 实现批量处理和表征缓存功能

## Phase 2: AIA框架实现

### 2.1 配置系统
- [x] **AIA配置类** (`configs/aia_config.py`) ✅
  - 继承BaseConfig
  - 定义AIA特定参数（checkpoint路径、攻击类型等）
  - 支持不同属性攻击模式配置（gender, age_level, education）
  - 实现配置验证和自动推断功能

### 2.2 模型架构
- [x] **AIA攻击模型** (`models/aia_models.py`) ✅
  - 属性推断攻击器（性别、年龄层次、教育等级）
  - 轻量级MLP分类器架构
  - 多任务攻击模型支持
  - 模型工厂函数和权重初始化
  - **新增**：TransformerAttacker支持音频时序特征
    - 位置编码和轻量级Transformer架构
    - 支持2D/3D输入的自适应处理
    - TransformerGenderAttacker, TransformerAgeAttacker, TransformerEducationAttacker

### 2.3 训练器
- [x] **AIA训练器** (`trainers/aia_trainer.py`) ✅
  - 继承BaseExperimentTrainer
  - 处理隐表征输入和属性标签配对
  - 支持多种属性攻击类型的训练流程
  - 实现5折交叉验证和结果统计

## Phase 3: 系统集成

### 3.1 统一入口更新
- [x] **更新train_unified.py** ✅
  - 添加AIA模式支持
  - 新增命令行参数：
    - `--checkpoint_paths`: checkpoint路径列表
    - `--attack_type`: 属性攻击类型选择 (gender, age_level, education)
    - `--target_model_mode`: 目标模型训练模式选择
    - `--attack_model_type`: 攻击模型类型选择 (mlp, transformer)
    - Transformer相关参数支持
  - 集成AIA训练流程
  - 完善checkpoint路径验证

### 3.2 配置工厂更新
- [x] **更新ConfigFactory** (`configs/config_factory.py`) ✅
  - 在get_available_modes()中添加'aia'
  - 创建AIA配置对象的逻辑
  - 处理AIA特定参数，包括Transformer配置

## Phase 4: 实验验证和系统优化 - 完成 ✅

### 4.1 属性推断攻击实验
- [x] **性别预测攻击** ✅
  - 完成Normal、DP、VIB模型的性别信息泄露测试
  - 实现平衡采样和正则化优化
  - 建立合理的baseline效果

- [x] **年龄层次预测攻击** ✅ 
  - 优化年龄分层从5层改为3层（Young/Middle-aged/Senior）
  - 实现更合理的年龄划分：19-41, 42-53, 54+
  - 测试年龄信息的隐私泄露程度

- [x] **教育等级预测攻击** ✅
  - 评估4个教育等级的泄露情况
  - 处理类别不平衡问题
  - 量化隐私保护效果

### 4.2 系统性能优化
- [x] **表征提取优化** ✅
  - 修复特征维度匹配问题（MEL: 80维）
  - 实现MLP+pooled_output, Transformer+时序特征的正确配套
  - 优化VIBModel导入和模型创建逻辑

- [x] **训练流程优化** ✅
  - 实现step-based训练（300步总量，每100步评估）
  - 添加平衡采样解决类别不平衡（Gender: 72%F vs 28%M）
  - 修正class weights计算为归一化逆频率
  - 移除early stopping避免作弊嫌疑

- [x] **缓存系统实现** ✅
  - 实现智能缓存机制（.cache/{input_mode}/{attack_type}/）
  - 基于配置参数生成唯一缓存键（包含input_mode）
  - 自动验证缓存有效性和配置匹配
  - 大幅减少重复表征提取时间

### 4.3 配置系统完善
- [x] **训练参数配置化** ✅
  - 将step数量移至配置文件（total_steps: 300, eval_every_steps: 100）
  - 统一所有训练参数管理
  - 支持运行时参数调整

- [x] **脚本和工具** ✅
  - 创建完整测试脚本（test_all_aia.sh）
  - 更新test_single_aia.sh支持输入模式参数
  - 支持所有三个属性的自动化测试
  - 错误处理和进度显示

### 4.4 输入模式扩展 - 新增 ✅
- [x] **多输入模式支持** ✅
  - 实现features_only模式：仅使用原始特征
  - 实现representations_only模式：仅使用学习到的表征
  - 实现concatenation模式：拼接原始特征和表征
  - 针对MLP和Transformer模型的不同处理策略

- [x] **配置系统扩展** ✅
  - AIAConfig添加input_mode参数
  - 更新get_model_input_dim()方法支持不同模式
  - 修改缓存路径结构：`.cache/{input_mode}/{attack_type}/`
  - 更新train_unified.py和ConfigFactory支持新参数

### 4.5 结果分析和问题解决
- [x] **性能对比分析** 🔍
  - Features-only模式：Gender攻击达78.57% ± 5.70%准确率
  - Concatenation模式：性能下降至53.90% ± 4.71%
  - 发现原始特征对属性预测的强效性
  - 识别拼接方法可能存在的问题

- [ ] **表征质量优化** 🚧
  - 分析为什么representations-only效果不佳
  - 调试concatenation模式性能下降原因
  - 优化表征提取和使用策略
  - 实现更有效的特征融合方法

- [ ] **综合性能评估** 
  - 完成所有指标按5折交叉验证报告 `mean ± std` 格式
  - 比较不同输入模式和训练方法的隐私保护效果
  - 生成属性泄露分析报告

## Phase 5: 文档更新

### 5.1 使用文档
- [ ] **更新examples.md**
  - 添加AIA攻击实验示例
  - 提供完整的命令行使用说明

- [ ] **更新README.md** 
  - 新增AIA功能说明
  - 更新项目功能列表

### 5.2 代码文档
- [ ] **代码注释和文档字符串**
  - 为所有新增模块添加详细注释
  - 确保代码可维护性

## 实验执行示例

```bash
# 1. 训练目标模型（如果尚未训练）
python train_unified.py --feature_type opensmile --mode normal
python train_unified.py --feature_type opensmile --mode dp --epsilon 8.0
python train_unified.py --feature_type opensmile --mode vib --z_dim 64 --beta 1e-3

# 2. 执行不同输入模式的性别属性推断攻击
python train_unified.py --feature_type opensmile --mode aia \
    --checkpoint_paths checkpoints/normal_fold_0.pth,checkpoints/normal_fold_1.pth,... \
    --attack_type gender --input_mode features_only

python train_unified.py --feature_type opensmile --mode aia \
    --checkpoint_paths checkpoints/normal_fold_0.pth,checkpoints/normal_fold_1.pth,... \
    --attack_type gender --input_mode representations_only

python train_unified.py --feature_type opensmile --mode aia \
    --checkpoint_paths checkpoints/normal_fold_0.pth,checkpoints/normal_fold_1.pth,... \
    --attack_type gender --input_mode concatenation

# 3. 使用便捷测试脚本
./scripts/test_single_aia.sh representations_only gender mlp
./scripts/test_single_aia.sh concatenation age_level transformer
./scripts/test_single_aia.sh features_only education
```

## 关键实现要点

1. **保持统一框架**：所有功能通过train_unified.py执行
2. **自动化表征提取**：从checkpoint自动提取隐层表征
3. **年龄层次化**：将连续年龄转换为5个统计层次
4. **5折交叉验证**：确保实验结果的统计显著性
5. **属性隐私评估**：量化不同训练方法对属性信息的保护效果

## 当前状态
- [x] 项目结构分析完成
- [x] 实验计划制定完成
- [x] Phase 1-2 基础组件实现完成
- [x] Phase 3 系统集成完成
- [x] Phase 4 系统优化和缓存实现完成
- [ ] Phase 4 综合实验验证进行中

## 技术亮点

### 已完成功能
1. **智能表征提取**：自动从5-fold checkpoints提取隐层表征，支持缓存加速
2. **统计年龄分层**：优化为3个合理层次（Young: 19-41, Middle-aged: 42-53, Senior: 54+）
3. **双模态攻击模型**：MLP配套pooled_output，Transformer配套时序特征
4. **平衡采样系统**：解决严重类别不平衡问题，使用归一化逆频率权重
5. **step-based训练**：300步训练，每100步评估，避免epoch过拟合
6. **智能缓存机制**：按属性分类缓存，配置验证，大幅提升重复实验效率
7. **配置化参数管理**：所有训练参数可配置，支持运行时调整

### 核心实现
- **DemographicExtractor**: 从文件名解析性别、年龄、教育等级，优化年龄分层
- **RepresentationExtractor**: 批量提取目标模型隐表征，支持MLP/Transformer配套
- **CachingSystem**: 智能缓存系统，按MD5配置键管理表征缓存
- **BalancedSampling**: 解决类别不平衡的加权随机采样
- **AIATrainer**: 完整的属性推断攻击训练流程，step-based训练
- **AIAConfig**: 专门的AIA配置管理类，支持所有参数配置化

### 最新优化特性 - Phase 4.4 输入模式扩展
1. **多输入模式支持**: 
   - `features_only`: 仅使用原始MEL/OpenSMILE特征，性能优异（Gender: 78.57%）
   - `representations_only`: 仅使用学习到的表征，用于隐私评估
   - `concatenation`: 拼接特征和表征，需要进一步优化
2. **模式特定处理**: MLP使用mean-pooled特征，Transformer处理时序数据
3. **智能缓存系统**: 按`{input_mode}/{attack_type}`分层缓存，避免重复计算
4. **配置化参数**: 所有输入模式通过AIAConfig统一管理
5. **便捷测试脚本**: 支持参数化的test_single_aia.sh脚本
6. **性能对比框架**: 为不同输入模式的系统性比较奠定基础