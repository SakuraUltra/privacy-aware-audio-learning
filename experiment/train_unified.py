#!/usr/bin/env python3
"""
统一训练入口脚本 - 项目根目录版本
支持不同的特征类型和训练模式
"""
import sys
import argparse
from pathlib import Path


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Unified Audio Classification Training")
    # 基本参数
    parser.add_argument('--feature_type', type=str, default='opensmile',
                        choices=['opensmile', 'mel'],
                        help="Feature type: 'opensmile' or 'mel'")
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'dp', 'vib', 'whisper_lora'],
                        help="Training mode: 'normal', 'dp' (Differential Privacy), 'vib' (Variational Information Bottleneck), or 'whisper_lora' (legacy, unmaintained)")

    parser.add_argument('--smoke-test', action='store_true',
                        help="CPU model self-check on synthetic features; normal/vib only, no dataset or downloads")

    # VIB相关参数
    parser.add_argument('--z_dim', type=int, default=64, help="VIB latent dimension")
    parser.add_argument('--beta', type=float, default=1e-3, help="VIB KL regularization strength")
    parser.add_argument('--mc_samples', type=int, default=30, help="VIB MC sampling times")

    # DP相关参数
    parser.add_argument('--epsilon', type=float, default=8.0,
                        help="Epsilon value for Differential Privacy (only used in DP mode)")

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")

    parser.add_argument('--use_cv', action='store_true',
                        help='Use 5-fold cross validation for whisper_lora mode')

    args = parser.parse_args()
    if args.smoke_test and args.mode not in ('normal', 'vib'):
        parser.error('--smoke-test supports normal/vib only; it does not validate DP training')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch_size must be positive')
    return args


def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        if args.smoke_test:
            from utils.smoke import run_model_smoke_test
            run_model_smoke_test(args.feature_type, args.mode)
            return 0

        # Load ML dependencies only after parsing --help / validating arguments.
        from configs.config_factory import ConfigFactory
        from trainers.base_trainer import BaseExperimentTrainer
        from trainers.vib_trainer import VIBTrainer
        from utils.logging import setup_experiment_logging

        log_file = setup_experiment_logging(args.feature_type, args.mode, args.epsilon if args.mode == 'dp' else None)
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
        elif args.mode == 'whisper_lora':
            print("Running Whisper LoRA Fine-tuning mode.")
            print("=" * 60)
        else:
            print("Running in Normal mode.")
            print("=" * 60)

        if args.mode == 'whisper_lora':
            # Historical LoRA dependencies must not block active training modes.
            from trainers.whisper_lora_trainer import WhisperLoRATrainer
            from configs.whisper_lora_config import WhisperLoRAConfig
            from utils.whisper_mel_adapter import create_mel_datasets_for_cv, create_mel_train_eval_datasets
            # Whisper LoRA 使用专门的配置
            config = WhisperLoRAConfig()
        else:
            # 其他模式使用 ConfigFactory
            config = ConfigFactory.create_config(
                feature_type=args.feature_type,
                mode=args.mode,
                epsilon=args.epsilon if args.mode == 'dp' else None,
                z_dim=args.z_dim if args.mode == 'vib' else None,
                beta=args.beta if args.mode == 'vib' else None,
                mc_samples=args.mc_samples if args.mode == 'vib' else None,
            )
        
        # 更新配置参数
        if args.mode == 'whisper_lora':
            # Whisper LoRA 配置更新
            config.general.update({
                'epochs': args.epochs,
            })
            config.data.update({
                'batch_size': args.batch_size,
            })
            config.training.update({
                'lr': args.lr,
            })
        else:
            # 其他模式配置更新
            config.general.update({
                'epochs': args.epochs,
            })
            config.data.update({
                'batch_size': args.batch_size,
            })
            config.training.update({
                'lr': args.lr,
            })
        
        Path("checkpoints").mkdir(parents=True, exist_ok=True)

        # 根据模式选择合适的训练器
        if args.mode == 'vib':
            print("Using VIBTrainer for VIB mode...")
            trainer = VIBTrainer(config)
            avg_metrics = trainer.run_cross_validation()
        elif args.mode == 'whisper_lora':
            print("Using WhisperLoRATrainer for Whisper LoRA mode...")

            if args.use_cv:
                print("Preparing complete MEL dataset for 5-Fold Cross Validation...")

                # 准备完整数据集用于交叉验证
                processor, full_dataset = create_mel_datasets_for_cv(
                    model_name=config.model['base_model'],
                    language=config.data.get('language', 'italian'),
                    task=config.data.get('task', 'transcribe'),
                    max_samples=500 if args.epochs == 1 else None  # 测试模式使用500个样本确保多个说话人
                )

                # 将完整数据集注入到配置中
                config.data['processor'] = processor
                config.data['full_dataset'] = full_dataset

                print(f"✅ Complete dataset prepared: {len(full_dataset)} samples")

                # 创建训练器并运行交叉验证
                trainer = WhisperLoRATrainer(config)
                avg_metrics = trainer.run_cross_validation(n_folds=5)

            else:
                print("Preparing MEL datasets for single training...")

                # 准备训练和验证数据集
                processor, train_dataset, eval_dataset = create_mel_train_eval_datasets(
                    model_name=config.model['base_model'],
                    language=config.data.get('language', 'italian'),
                    task=config.data.get('task', 'transcribe'),
                    max_samples=500 if args.epochs == 1 else None  # 测试模式使用500个样本确保多个说话人
                )

                # 将数据集注入到配置中
                config.data['processor'] = processor
                config.data['train_dataset'] = train_dataset
                config.data['eval_dataset'] = eval_dataset

                print(f"✅ Datasets prepared: {len(train_dataset)} train, {len(eval_dataset)} eval samples")

                # 创建训练器并进行单次训练
                trainer = WhisperLoRATrainer(config)
                trainer.train()
                avg_metrics = None
        else:
            print("Using BaseExperimentTrainer for Normal/DP mode...")
            trainer = BaseExperimentTrainer(config)
            avg_metrics = trainer.run_cross_validation()
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Log file: {log_file}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
