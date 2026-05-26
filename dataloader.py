"""
数据加载模块
支持自动检测通道数、类别数
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os
import h5py


def _load_mat_file(mat_path):
    """加载 .mat 文件，自动兼容 v7.3 (HDF5) 和旧版格式"""
    try:
        data = loadmat(mat_path)
        if len([k for k in data if not k.startswith('__')]) > 0:
            return data, 'scipy'
    except NotImplementedError:
        pass

    f = h5py.File(mat_path, 'r')
    data = {}
    for key in f.keys():
        if not key.startswith('#'):
            data[key] = np.array(f[key])
    f.close()
    return data, 'h5py'


def detect_dataset_info(data_path):
    """自动检测数据集信息：通道数、类别数、图像尺寸、样本数"""
    train_path = os.path.join(data_path, 'train_data.mat')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_path}")

    data, reader = _load_mat_file(train_path)

    # 获取图像数据
    X = data.get('X_train')
    if X is None:
        raise ValueError(f"train_data.mat 中未找到 X_train")

    y = data.get('y_train')
    if y is None:
        raise ValueError(f"train_data.mat 中未找到 y_train")

    y = y.flatten()

    # 判断维度格式并统一为 (N, C, H, W)
    if X.ndim == 4:
        n_labels = len(y)
        if X.shape[0] == n_labels:
            # 已经是 (N, C, H, W)
            pass
        elif X.shape[-1] == n_labels:
            # (H, W, C, N) → (N, C, H, W)
            X = np.transpose(X, (3, 2, 0, 1))
        else:
            raise ValueError(f"无法推断数据维度: X.shape={X.shape}, n_labels={n_labels}")
    elif X.ndim == 3:
        # (H, W, N) → (N, 1, H, W)
        X = np.transpose(X, (2, 0, 1))
        X = X[:, np.newaxis, :, :]

    # 提取信息
    in_channels = X.shape[1]
    img_h, img_w = X.shape[2], X.shape[3]
    num_classes = len(np.unique(y))
    train_samples = len(y)

    # 验证集和测试集样本数
    val_samples = 0
    test_samples = 0

    val_path = os.path.join(data_path, 'val_data.mat')
    if os.path.exists(val_path):
        val_data, _ = _load_mat_file(val_path)
        y_val = val_data.get('y_val')
        if y_val is not None:
            val_samples = len(y_val.flatten())

    test_path = os.path.join(data_path, 'test_data.mat')
    if os.path.exists(test_path):
        test_data, _ = _load_mat_file(test_path)
        y_test = test_data.get('y_test')
        if y_test is not None:
            test_samples = len(y_test.flatten())

    # 类别名
    class_names = None
    if 'class_names' in data:
        raw = data['class_names']
        try:
            # 处理嵌套数组结构
            flat = raw.flatten()
            class_names = []
            for n in flat:
                if hasattr(n, 'flatten'):
                    # 内层还是数组，再展平
                    inner = n.flatten()
                    for item in inner:
                        class_names.append(str(item).strip())
                else:
                    class_names.append(str(n).strip())
        except Exception:
            class_names = None

    return {
        'in_channels': int(in_channels),
        'num_classes': int(num_classes),
        'class_names': class_names,
        'train_samples': int(train_samples),
        'val_samples': int(val_samples),
        'test_samples': int(test_samples),
        'image_size': f"{img_h}x{img_w}",
        'image_shape': (int(in_channels), int(img_h), int(img_w)),
    }


def save_class_names(data_path, class_names):
    """将类别名保存到 train_data.mat"""
    train_path = os.path.join(data_path, 'train_data.mat')

    # 读取现有数据
    data, reader = _load_mat_file(train_path)

    # 添加 class_names
    data['class_names'] = np.array(class_names, dtype=object)

    # 重新保存
    save_dict = {}
    for key, val in data.items():
        if not key.startswith('__'):
            save_dict[key] = val

    save_path = train_path
    h5py.File(save_path, 'w').close()  # 清空文件
    import scipy.io as sio
    sio.savemat(save_path, save_dict, do_compression=True)


class CWTDataset(Dataset):
    """CWT时频图数据集 - 简化版（数据已是 NCHW 格式）"""

    def __init__(self, mat_path, in_channels=1, transform=None):
        data, reader = _load_mat_file(mat_path)

        # 获取数据和标签
        if 'X_train' in data:
            self.images = data['X_train']
            self.labels = data['y_train'].flatten()
        elif 'X_val' in data:
            self.images = data['X_val']
            self.labels = data['y_val'].flatten()
        elif 'X_test' in data:
            self.images = data['X_test']
            self.labels = data['y_test'].flatten()
        else:
            raise ValueError(f"无法识别的数据格式: {mat_path}, keys: {list(data.keys())}")

        self.labels = self.labels.flatten()

        # 维度处理
        if self.images.ndim == 4:
            n_labels = len(self.labels)
            if self.images.shape[0] == n_labels:
                # 已经是 (N, C, H, W)
                pass
            elif self.images.shape[-1] == n_labels:
                # (H, W, C, N) → (N, C, H, W)
                self.images = np.transpose(self.images, (3, 2, 0, 1))
            else:
                raise ValueError(f"无法推断数据维度: {self.images.shape}")
        elif self.images.ndim == 3:
            # (H, W, N) → (N, 1, H, W)
            self.images = np.transpose(self.images, (2, 0, 1))
            self.images = self.images[:, np.newaxis, :, :]

        # 通道数适配
        current_channels = self.images.shape[1]
        if current_channels != in_channels:
            if in_channels == 1 and current_channels > 1:
                self.images = np.mean(self.images, axis=1, keepdims=True)
            elif in_channels == 3 and current_channels == 1:
                self.images = np.repeat(self.images, 3, axis=1)
            elif in_channels == 3 and current_channels > 3:
                self.images = self.images[:, :3, :, :]

        # 归一化
        if self.images.dtype == np.uint8:
            self.images = self.images.astype(np.float32) / 255.0
        else:
            self.images = self.images.astype(np.float32)

        self.labels = self.labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)


class TrainAugmentation:
    """数据增强"""
    def __init__(self, hflip_p=0.5, vflip_p=0.3, noise_std=0.01, noise_p=0.3):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.noise_std = noise_std
        self.noise_p = noise_p

    def __call__(self, x):
        if np.random.rand() < self.hflip_p:
            x = x.flip(-1)
        if np.random.rand() < self.vflip_p:
            x = x.flip(-2)
        if np.random.rand() < self.noise_p:
            x = x + torch.randn_like(x) * self.noise_std
        return x


def get_dataloaders(config, data_path):
    """获取训练、验证、测试数据加载器"""
    train_path = os.path.join(data_path, 'train_data.mat')
    val_path = os.path.join(data_path, 'val_data.mat')
    test_path = os.path.join(data_path, 'test_data.mat')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_path}")

    train_dataset = CWTDataset(train_path, in_channels=config.IN_CHANNELS, transform=TrainAugmentation())

    val_dataset = None
    if os.path.exists(val_path):
        val_dataset = CWTDataset(val_path, in_channels=config.IN_CHANNELS)

    test_dataset = None
    if os.path.exists(test_path):
        test_dataset = CWTDataset(test_path, in_channels=config.IN_CHANNELS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader
