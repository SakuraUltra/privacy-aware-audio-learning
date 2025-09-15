#!/usr/bin/env python3
"""
统一训练入口脚本 - 项目根目录版本
支持不同的特征类型和训练模式
"""
import sys
import argparse
from pathlib import Path

from configs.config_factory import ConfigFactory
from trainers.base_trainer import BaseExperimentTrainer
from trainers.vib_trainer import VIBTrainer
from trainers.aia_trainer import AIATrainer
from trainers.mine_privacy_trainer import MINEPrivacyTrainer
from utils.logging import setup_experiment_logging
from utils.whisper_mel_adapter import create_mel_datasets_for_cv, create_mel_train_eval_datasets


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Unified Audio Classification Training")
    # 基本参数
    parser.add_argument('--feature_type', type=str, default='opensmile',
                        choices=['opensmile', 'mel'],
                        help="Feature type: 'opensmile' or 'mel'")
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'dp', 'vib', 'aia', 'mine'],
                        help="Training mode: 'normal', 'dp' (Differential Privacy), 'vib' (Variational Information Bottleneck), 'aia' (Attribute Inference Attack), or 'mine' (MINE Privacy)")

    # VIB相关参数
    parser.add_argument('--z_dim', type=int, default=64, help="VIB latent dimension")
    parser.add_argument('--beta', type=float, default=1e-3, help="VIB KL regularization strength")
    parser.add_argument('--mc_samples', type=int, default=30, help="VIB MC sampling times")

    # DP相关参数
    parser.add_argument('--epsilon', type=float, default=8.0,
                        help="Epsilon value for Differential Privacy (only used in DP mode)")

    # AIA相关参数
    parser.add_argument('--checkpoint_paths', type=str, default=None,
                        help="Comma-separated list of 5 checkpoint paths for AIA attacks")
    parser.add_argument('--attack_type', type=str, default='gender',
                        choices=['gender', 'age_level', 'education'],
                        help="AIA attack type: 'gender', 'age_level', or 'education'")
    parser.add_argument('--target_model_mode', type=str, default='normal',
                        choices=['normal', 'dp', 'vib', 'mine'],
                        help="Target model training mode for AIA attacks")
    parser.add_argument('--attack_model_type', type=str, default='mlp',
                        choices=['mlp', 'transformer'],
                        help="Attack model architecture: 'mlp' or 'transformer'")
    parser.add_argument('--input_mode', type=str, default='features_only',
                        choices=['features_only', 'representations_only', 'concatenation', 'mix'],
                        help="Input data mode: 'features_only', 'representations_only', 'concatenation', or 'mix'")
    parser.add_argument('--mix_alpha', type=float, default=0.5,
                        help="Alpha parameter for mix mode: alpha * features + (1-alpha) * representations")
    
    # Transformer攻击模型参数
    parser.add_argument('--transformer_d_model', type=int, default=128,
                        help="Transformer d_model for AIA attacks")
    parser.add_argument('--transformer_nhead', type=int, default=4,
                        help="Transformer number of heads for AIA attacks")
    parser.add_argument('--transformer_num_layers', type=int, default=2,
                        help="Transformer number of layers for AIA attacks")
    parser.add_argument('--transformer_dim_feedforward', type=int, default=256,
                        help="Transformer feedforward dimension for AIA attacks")

    # MINE相关参数
    parser.add_argument('--mine_model_type', type=str, default='timeaware',
                        choices=['timeaware', 'standard'],
                        help="MINE model type: 'timeaware' or 'standard'")
    parser.add_argument('--privacy_weight', type=float, default=0.2,
                        help="Privacy weight gamma for MINE training")
    parser.add_argument('--mine_lr', type=float, default=1e-4,
                        help="Learning rate for MINE optimizer")
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help="Task type for MINE training")
    
    # MINE音频编码器参数（仅对timeaware模式有效）
    parser.add_argument('--audio_d_model', type=int, default=80,
                        help="Audio encoder d_model for TimeAware MINE")
    parser.add_argument('--audio_nhead', type=int, default=8,
                        help="Audio encoder number of heads for TimeAware MINE")
    parser.add_argument('--audio_num_layers', type=int, default=4,
                        help="Audio encoder number of layers for TimeAware MINE")
    parser.add_argument('--target_dim', type=int, default=768,
                        help="Audio encoder target dimension for TimeAware MINE")

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")

    return parser.parse_args()


def validate_aia_arguments(args):
    """验证AIA相关参数"""
    if args.mode == 'aia':
        if args.checkpoint_paths is None:
            raise ValueError("--checkpoint_paths is required for AIA mode")
        
        # 解析checkpoint路径
        checkpoint_paths = [path.strip() for path in args.checkpoint_paths.split(',')]
        if len(checkpoint_paths) != 5:
            raise ValueError("Must provide exactly 5 checkpoint paths for AIA attacks")
        
        # 验证checkpoint文件是否存在
        for path in checkpoint_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        return checkpoint_paths
    return None


def main():
    """主函数"""
    args = parse_arguments()
    
    # 验证AIA参数
    checkpoint_paths = validate_aia_arguments(args)
    
    # 设置日志
    log_file = setup_experiment_logging(args.feature_type, args.mode, args.epsilon if args.mode == 'dp' else None)
    
    try:
        print("=" * 60)
        print(f"Starting Unified Training Framework")
        print(f"Feature Type: {args.feature_type.upper()}")
        print(f"Training Mode: {args.mode.upper()}")

        if args.mode == 'dp':
            print(f"DP Epsilon: {args.epsilon}")
            print("=" * 60)
        elif args.mode == 'vib':
            print(f"VIB Latent Dimension: {args.z_dim}")
            print(f"VIB Beta: {args.beta}")
            print(f"VIB MC Samples: {args.mc_samples}")
            print("=" * 60)
        elif args.mode == 'aia':
            print(f"AIA Attack Type: {args.attack_type.upper()}")
            print(f"Target Model Mode: {args.target_model_mode.upper()}")
            print(f"Attack Model Type: {args.attack_model_type.upper()}")
            print(f"Number of Checkpoints: {len(checkpoint_paths)}")
            print("=" * 60)
        elif args.mode == 'mine':
            print(f"MINE Model Type: {args.mine_model_type.upper()}")
            print(f"Privacy Weight: {args.privacy_weight}")
            print(f"MINE Learning Rate: {args.mine_lr}")
            print(f"Task Type: {args.task_type.upper()}")
            print("=" * 60)
        else:
            print("Running in Normal mode.")
            print("=" * 60)

        # 使用ConfigFactory创建配置
        if args.mode == 'aia':
            # AIA模式需要额外的参数
            config = ConfigFactory.create_config(
                feature_type=args.feature_type,
                mode=args.mode,
                attack_type=args.attack_type,
                checkpoint_paths=checkpoint_paths,
                target_model_mode=args.target_model_mode,
                attack_model_type=args.attack_model_type,
                input_mode=args.input_mode,
                mix_alpha=args.mix_alpha,
                transformer_d_model=args.transformer_d_model,
                transformer_nhead=args.transformer_nhead,
                transformer_num_layers=args.transformer_num_layers,
                transformer_dim_feedforward=args.transformer_dim_feedforward,
            )
        elif args.mode == 'mine':
            # MINE模式需要额外的参数
            config = ConfigFactory.create_config(
                feature_type=args.feature_type,
                mode=args.mode,
                mine_model_type=args.mine_model_type,
                privacy_weight=args.privacy_weight,
                mine_lr=args.mine_lr,
                task_type=args.task_type,
                # 音频编码器参数
                audio_d_model=args.audio_d_model,
                audio_nhead=args.audio_nhead,
                audio_num_layers=args.audio_num_layers,
                target_dim=args.target_dim,
            )
        else:
            # 其他模式使用标准参数
            config = ConfigFactory.create_config(
                feature_type=args.feature_type,
                mode=args.mode,
                epsilon=args.epsilon if args.mode == 'dp' else None,
                z_dim=args.z_dim if args.mode == 'vib' else None,
                beta=args.beta if args.mode == 'vib' else None,
                mc_samples=args.mc_samples if args.mode == 'vib' else None,
            )
        
        # 更新配置参数
        if args.mode != 'aia':
            # 非AIA模式配置更新
            config.general.update({
                'epochs': args.epochs,
            })
            config.data.update({
                'batch_size': args.batch_size,
            })
            config.training.update({
                'lr': args.lr,
            })
        # AIA模式的配置参数已经在创建时设置
        
        # 根据模式选择合适的训练器
        if args.mode == 'vib':
            print("Using VIBTrainer for VIB mode...")
            trainer = VIBTrainer(config)
            avg_metrics = trainer.run_cross_validation()
        elif args.mode == 'aia':
            print("Using AIATrainer for AIA mode...")
            trainer = AIATrainer(config)
            avg_metrics = trainer.run_attack_experiment()
        elif args.mode == 'mine':
            print("Using MINEPrivacyTrainer for MINE mode...")
            trainer = MINEPrivacyTrainer(config)
            avg_metrics = trainer.run_cross_validation()
        else:
            print("Using BaseExperimentTrainer for Normal/DP modes...")
            trainer = BaseExperimentTrainer(config)
            avg_metrics = trainer.run_cross_validation()
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Log file: {log_file}")
        print("=" * 60)
        
        return avg_metrics
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()