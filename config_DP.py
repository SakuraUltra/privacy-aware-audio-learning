import torch

EXPERIMENT_MODE = 'DP'  # 可选值：'DP' 或 'NORMAL'

GENERAL = {
    'seed': 10086,
    'device': torch.device("cuda"),
    'num_folds': 5,
    'epochs': 5,
}

DATA = {
    'speaker_folds_csv': 'data/speaker_folds.csv',
    'all_expanded_features_csv': 'data/all_expanded_features_fixed.csv',
    'batch_size': 32,
}

MODEL = {
    'd_model': 32,
    'nhead': 8,
    'num_layers': 4,
    'num_classes': 2,
    'dim_feedforward': 256,
    'dropout': 0.3,
}

TRAINING = {
    'lr': 3e-4,
}

DP_PARAMS = {
    'target_epsilon': 100,
    'target_delta': 1e-5,
    'max_grad_norm': 1.2,
}