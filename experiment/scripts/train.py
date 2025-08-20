#!/usr/bin/env python3
"""
统一训练入口脚本
支持不同的特征类型和训练模式
"""
import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiment.configs.config_factory import ConfigFactory
from experiment.trainers.base_trainer import BaseExperimentTrainer
from experiment.utils.logging import setup_experiment_logging


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Unified Audio Classification Training")
    
    # 基本参数
    parser.add_argument('--feature_type', type=str, default='opensmile',
                        choices=['opensmile', 'mel'],
                        help="Feature type: 'opensmile' or 'mel'")
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'dp'],
                        help="Training mode: 'normal' or 'dp' (Differential Privacy)")
    
    # DP相关参数
    parser.add_argument('--epsilon', type=float, default=8.0,
                        help="Epsilon value for Differential Privacy (only used in DP mode)")
    parser.add_argument('--delta', type=float, default=1e-5,
                        help="Delta value for Differential Privacy")
    parser.add_argument('--max_grad_norm', type=float, default=1.2,
                        help="Maximum gradient norm for DP")
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument('--seed', type=int, default=666,
                        help="Random seed")
    
    # 模型参数
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument('--nhead', type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout rate")
    
    # 其他参数
    parser.add_argument('--num_folds', type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use ('cuda', 'cpu', or 'auto')")
    parser.add_argument('--log_dir', type=str, default='experiment/logs',
                        help="Directory to save logs")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
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
        
        # 创建配置
        config = ConfigFactory.create_config(
            feature_type=args.feature_type,
            mode=args.mode,
            epsilon=args.epsilon if args.mode == 'dp' else None
        )
        
        # 更新配置参数
        config.general.update({
            'epochs': args.epochs,
            'seed': args.seed,
            'num_folds': args.num_folds,
        })
        
        if args.device != 'auto':
            import torch
            config.general['device'] = torch.device(args.device)
        
        config.data.update({
            'batch_size': args.batch_size,
        })
        
        config.training.update({
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        })
        
        config.model.update({
            'num_layers': args.num_layers,
            'nhead': args.nhead,
            'dropout': args.dropout,
        })
        
        if args.mode == 'dp':
            config.dp_params.update({
                'target_delta': args.delta,
                'max_grad_norm': args.max_grad_norm,
            })
        
        # 创建训练器并开始训练
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
