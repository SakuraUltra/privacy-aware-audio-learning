# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a unified machine learning framework for audio classification experiments supporting multiple training modes and feature types. The project focuses on differential privacy-based inference from biological and vocal features with support for:

- **Active Training Modes**: Normal, Differential Privacy (DP), Variational Information Bottleneck (VIB)
- **Feature Types**: OpenSMILE traditional audio features, MEL spectrograms from Whisper

**IMPORTANT**: Ignore Adversarial (Adv) and LoRA-related code as these training modes are deprecated and no longer planned for use in this project.

## Key Commands

### Primary Training Command
```bash
# Main unified training script
python train_unified.py --feature_type <opensmile|mel> --mode <normal|dp|vib> [options]

# Examples:
python train_unified.py --feature_type opensmile --mode normal
python train_unified.py --feature_type mel --mode dp --epsilon 8.0
python train_unified.py --feature_type mel --mode vib --z_dim 128 --beta 5e-3
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
   - `CoreTrainer`: Low-level epoch training and evaluation logic

3. **Model Architecture** (`models/`):
   - `TransformerClassifier`: Main transformer-based classification model
   - `VIBModel`: Variational Information Bottleneck implementation

4. **Data Handling** (`data/`, `audio_mel/`):
   - Unified dataset interfaces for different feature types
   - Cross-validation splits based on speaker folds
   - Feature extraction and normalization utilities

### Training Modes

- **Normal**: Standard cross-entropy training with transformer classifier
- **DP (Differential Privacy)**: Uses Opacus library for privacy-preserving training
- **VIB (Variational Information Bottleneck)**: Information-theoretic regularization approach

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

## Future Development Plans

### Next Phase: Privacy Information Leakage Analysis

**Objective**: Detect and analyze privacy information leakage in learned hidden representations by training classifiers to predict demographic attributes.

#### Phase 1: Data Attribute Extraction
- **Data Naming Convention**: Extract demographic information from audio file naming patterns
  - Format example: `29_CF34_3` → Age: 29, Gender: F (Female), Age: 34, Education Level: 3
  - Parse and extract: Age, Gender (M/F), Education Level from filename patterns
- **Create Demographic Labels**: Build comprehensive demographic dataset with extracted attributes

#### Phase 2: Privacy Leakage Detection
- **Hidden Representation Analysis**: Extract hidden representations from trained models (Normal, DP, VIB)
- **Attribute Prediction**: Train separate classifiers to predict:
  - Age (regression/classification)
  - Gender (binary classification)  
  - Education Level (multi-class classification)
- **Privacy Evaluation**: Measure how well demographic attributes can be inferred from hidden representations
- **Comparison Across Methods**: Evaluate privacy protection effectiveness of DP and VIB vs Normal training

#### Implementation Strategy
- Maintain unified framework using `train_unified.py`
- Add new training modes for attribute prediction if needed
- Results must follow `mean ± std` reporting format across 5-fold validation
- Focus on quantifying privacy leakage severity across different training paradigms