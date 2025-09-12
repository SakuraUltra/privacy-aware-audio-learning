# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a unified machine learning framework for audio classification experiments supporting multiple training modes and feature types. The project focuses on differential privacy-based inference from biological and vocal features with support for:

- **Active Training Modes**: Normal, Differential Privacy (DP), Variational Information Bottleneck (VIB), Attribute Inference Attack (AIA)
- **Feature Types**: OpenSMILE traditional audio features, MEL spectrograms from Whisper

**IMPORTANT**: Ignore Adversarial (Adv) and LoRA-related code as these training modes are deprecated and no longer planned for use in this project.

## Key Commands

### Primary Training Command
```bash
# Main unified training script
python train_unified.py --feature_type <opensmile|mel> --mode <normal|dp|vib|aia> [options]

# Examples:
python train_unified.py --feature_type opensmile --mode normal
python train_unified.py --feature_type mel --mode dp --epsilon 8.0
python train_unified.py --feature_type mel --mode vib --z_dim 128 --beta 5e-3

# AIA (Attribute Inference Attack) examples:
python train_unified.py --feature_type opensmile --mode aia \
    --checkpoint_paths checkpoints/normal_fold_0.pth,checkpoints/normal_fold_1.pth,... \
    --attack_type gender --target_model_mode normal
    
python train_unified.py --feature_type mel --mode aia \
    --checkpoint_paths checkpoints/vib_fold_0.pth,checkpoints/vib_fold_1.pth,... \
    --attack_type age_level --target_model_mode vib --attack_model_type transformer
```

### Environment Setup
```bash
# Activate conda environment
conda activate exp

# Install dependencies
pip install -r requirements.txt
```

### Data Processing
```bash
# Check feature integrity
python data/feature_check.py

# Validate labels
python data/label-check.py
```

## Architecture Overview

### Core Components

1. **Configuration System** (`configs/`):
   - Factory pattern with `ConfigFactory` for creating mode-specific configurations
   - Base configuration class with specialized configs for each feature type
   - Supports runtime parameter updates for DP, VIB modes

2. **Training Framework** (`trainers/`):
   - `BaseExperimentTrainer`: Handles standard cross-validation training
   - `VIBTrainer`: Specialized trainer for Variational Information Bottleneck
   - `AIATrainer`: Specialized trainer for Attribute Inference Attacks
   - `CoreTrainer`: Low-level epoch training and evaluation logic

3. **Model Architecture** (`models/`):
   - `TransformerClassifier`: Main transformer-based classification model
   - `VIBModel`: Variational Information Bottleneck implementation
   - `AIA Attack Models`: MLP and Transformer-based attribute inference attackers

4. **Data Handling** (`data/`, `audio_mel/`):
   - Unified dataset interfaces for different feature types
   - Cross-validation splits based on speaker folds
   - Feature extraction and normalization utilities
   - Demographic information extraction from filename patterns

### Training Modes

- **Normal**: Standard cross-entropy training with transformer classifier
- **DP (Differential Privacy)**: Uses Opacus library for privacy-preserving training
- **VIB (Variational Information Bottleneck)**: Information-theoretic regularization approach
- **AIA (Attribute Inference Attack)**: Privacy evaluation through demographic attribute prediction

### Data Structure

- Audio features stored in CSV format under `data/extracted_features_train/`
- MEL spectrograms processed through Whisper preprocessing
- Cross-validation splits defined in `data/speaker_folds.csv`
- Training checkpoints saved to `checkpoints/` directory
- Training logs automatically saved to `logs/` with timestamps

## Development Patterns

### Code Development Guidelines

**CRITICAL REQUIREMENTS**:
1. **Maintain Program Simplicity**: Keep all code changes minimal and focused
2. **Unified Entry Point**: Only use `train_unified.py` as the single training entry point - do not create alternative training scripts
3. **Consistency**: Maintain the generality and consistency of `train_unified.py` across all training modes
4. **Documentation Updates**: Always update `README.md` and `examples.md` when making changes

### Result Reporting Requirements

**5-Fold Cross-Validation Results**: All results must be reported as `mean ± std` format in tables and logs. For example:
- Accuracy: 85.2 ± 2.1%
- F1-Score: 0.847 ± 0.023

Ensure both mean and standard deviation are calculated and displayed across all 5 folds for every metric.

### Adding New Training Modes
1. Create config class in `configs/` inheriting from `BaseConfig`
2. Implement specialized trainer in `trainers/` if needed
3. Update `ConfigFactory.get_available_modes()`
4. Add mode support in `train_unified.py`

### Adding New Feature Types
1. Create dataset class handling the new feature format
2. Add feature-specific configuration in `configs/`
3. Update `ConfigFactory.get_available_feature_types()`
4. Ensure proper collate function for DataLoader

### Model Modifications
- Models must implement forward pass returning `(logits, additional_outputs)`
- Additional outputs can contain mode-specific information (e.g., VIB KL loss)
- All models should handle optional `padding_mask` parameter

## File Structure Highlights

- `train_unified.py`: Single entry point for all training experiments
- `requirements.txt`: Complete dependency specification including PyTorch, Opacus, transformers
- `checkpoints/`: Pre-trained model weights for different configurations
- `logs/`: Detailed training logs with timestamps and mode information
- `examples.md`: Usage examples and parameter explanations

## Important Notes

- The project uses 5-fold cross-validation based on speaker identity
- Differential Privacy training requires careful batch size and gradient norm tuning
- VIB mode includes KL divergence warmup and Monte Carlo uncertainty estimation
- MEL features require Whisper model preprocessing pipeline

## Development Task Management

- **Current Status**: AIA (Attribute Inference Attack) implementation completed ✅
- Refer to `todo.md` for implementation details and progress tracking
- All Phase 1-3 AIA tasks have been successfully implemented
- System ready for Phase 4 experimental validation

## Future Development Plans

### Current Phase: AIA (Attribute Inference Attack) - COMPLETED ✅

**Status**: Implementation completed - all core functionality integrated

**Objective**: Implement attribute inference attacks to evaluate privacy leakage of demographic information (gender, age, education) in trained model representations and test the effectiveness of differential privacy and VIB approaches.

#### Successfully Implemented Features:
- ✅ **Automated Representation Extraction**: Extract hidden representations from trained model checkpoints (5-fold CV)
- ✅ **Demographic Information Parsing**: Extract age, gender, and education level from filename patterns (`编号_[PC][MF]年龄_教育等级_slice编号.csv`)
- ✅ **Multiple Attack Types**: Gender prediction, age-level prediction (5 categories), education prediction
- ✅ **Age Categorization**: Convert continuous ages into 5 statistical levels using quantiles
- ✅ **Dual Attack Models**: Both MLP and TransformerAttacker architectures for temporal audio features
- ✅ **Unified Framework Integration**: All attacks accessible through `train_unified.py` with new `--mode aia` option
- ✅ **Complete Configuration System**: AIAConfig with full parameter management
- ✅ **Validation Pipeline**: Comprehensive error checking and path validation
- ✅ **Multiple Input Modes**: Support for features-only, representations-only, and concatenation modes
- ✅ **Intelligent Caching**: Mode-specific caching system with path structure: `.cache/{input_mode}/{attack_type}/`

#### Usage Examples:
```bash
# Gender attack on Normal model with different input modes
python train_unified.py --mode aia --feature_type opensmile \
  --checkpoint_paths checkpoints/normal_fold_0.pth,checkpoints/normal_fold_1.pth,... \
  --attack_type gender --target_model_mode normal --input_mode features_only

# Age level attack using TransformerAttacker with representation-only input
python train_unified.py --mode aia --feature_type mel \
  --checkpoint_paths checkpoints/dp_fold_0.pth,checkpoints/dp_fold_1.pth,... \
  --attack_type age_level --target_model_mode dp --attack_model_type transformer \
  --input_mode representations_only

# Concatenation mode for enhanced attack performance
python train_unified.py --mode aia --feature_type mel \
  --checkpoint_paths checkpoints/vib_fold_0.pth,checkpoints/vib_fold_1.pth,... \
  --attack_type gender --target_model_mode vib --input_mode concatenation
```

### Next Phase: Experimental Validation and Analysis

**Objective**: Execute comprehensive AIA experiments and analyze privacy leakage across different training methods.

#### Phase 4: Experimental Execution
- **Gender Attack Experiments**: Test all model types (Normal, DP, VIB) with both feature types
- **Age Level Attack Experiments**: Evaluate 5-category age prediction across privacy methods
- **Education Level Attack Experiments**: Assess education information leakage
- **Comparative Analysis**: Generate detailed privacy protection effectiveness reports
- **Results Documentation**: Update examples.md and README.md with experimental findings

#### Implementation Strategy
- Execute experiments using completed AIA framework through `train_unified.py`
- Results must follow `mean ± std` reporting format across 5-fold validation
- Focus on quantifying attribute privacy leakage severity across different training paradigms
- Generate comprehensive privacy evaluation reports