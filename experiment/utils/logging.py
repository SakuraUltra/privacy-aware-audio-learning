"""
日志工具
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Tee:
    """将输出同时写入到多个文件对象"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def setup_experiment_logging(feature_type: str, mode: str, epsilon: Optional[float] = None) -> str:
    """
    设置实验日志
    
    Args:
        feature_type: 特征类型
        mode: 训练模式
        epsilon: DP模式下的epsilon值
    
    Returns:
        日志文件路径
    """
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == 'dp' and epsilon is not None:
        log_filename = f"training_log_{feature_type}_{mode}_eps{epsilon}_{timestamp}.txt"
    else:
        log_filename = f"training_log_{feature_type}_{mode}_{timestamp}.txt"
    
    log_file_path = log_dir / log_filename
    
    # 设置日志输出
    logfile = open(log_file_path, "w", encoding='utf-8')
    sys.stdout = Tee(sys.stdout, logfile)
    
    print(f"Logging to: {log_file_path}")
    
    return str(log_file_path)


def close_logging():
    """关闭日志文件"""
    if hasattr(sys.stdout, 'files'):
        for f in sys.stdout.files:
            if f != sys.__stdout__:
                f.close()
        sys.stdout = sys.__stdout__
