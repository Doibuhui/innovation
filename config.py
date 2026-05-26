"""
故障诊断系统 - 配置文件
纯超参数配置，数据集信息由系统自动检测
"""
import os
from datetime import datetime


class Config:
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # 路径配置
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')

    # 图像参数
    IMG_SIZE = 224

    # 训练参数（全部自动，用户只需调 epoch）
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-3
    LAMBDA_PHYSICS = 0.1

    # 模型参数
    CONV_TYPE = 'fourier'
    USE_MHA = True

    # 设备
    DEVICE = 'cuda'

    # 随机种子
    SEED = 42

    # 动态属性（由 app.py 在运行时注入）
    IN_CHANNELS = None
    NUM_CLASSES = None
    CLASS_NAMES = None
    CURRENT_DATASET = None

    @classmethod
    def get_checkpoint_path(cls, dataset_name):
        """获取模型保存路径"""
        safe_name = dataset_name.replace("-", "_").replace(" ", "_")
        return os.path.join(cls.CHECKPOINT_DIR, f'{safe_name}_best.pth')

    @classmethod
    def generate_checkpoint_path(cls, dataset_name):
        """生成带时间戳的模型保存路径"""
        safe_name = dataset_name.replace("-", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(cls.CHECKPOINT_DIR, f'{safe_name}_{timestamp}_best.pth')
