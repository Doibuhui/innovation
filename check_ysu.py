import numpy as np
import os

def load_data(path):
    from scipy.io import loadmat
    try:
        return loadmat(path)
    except NotImplementedError:
        import h5py
        f = h5py.File(path, 'r')
        data = {k: np.array(f[k]) for k in f.keys() if not k.startswith('#')}
        f.close()
        return data

print('='*60)
print('YSU_V QUICK ANALYSIS')
print('='*60)

data_path = 'ysu_v0hp'

# 加载数据
train_d = load_data(os.path.join(data_path, 'train_data.mat'))
val_d = load_data(os.path.join(data_path, 'val_data.mat'))
test_d = load_data(os.path.join(data_path, 'test_data.mat'))

X_train = train_d['X_train']
y_train = train_d['y_train'].flatten()
X_val = val_d['X_val']
y_val = val_d['y_val'].flatten()
X_test = test_d['X_test']
y_test = test_d['y_test'].flatten()

print(f'Train: {X_train.shape} -> {len(y_train)} samples')
print(f'Val:   {X_val.shape} -> {len(y_val)} samples')
print(f'Test:  {X_test.shape} -> {len(y_test)} samples')

# 检查格式
# MATLAB 保存的是 (H,W,C,N) 还是 (N,C,H,W)?
# 如果 shape[0] == 224 且 shape[-1] == N，说明是 HWCN
def check_format(X, n, name):
    if X.ndim == 4:
        if X.shape[0] == n:
            print(f'{name}: (N,C,H,W) format, N={n}')
        elif X.shape[-1] == n:
            print(f'{name}: (H,W,C,N) format, N={n}')
        else:
            print(f'{name}: UNKNOWN format, shape={X.shape}, n_labels={n}')

check_format(X_train, len(y_train), 'Train')
check_format(X_val, len(y_val), 'Val')
check_format(X_test, len(y_test), 'Test')

# 样本数
print(f'\nTrain per class: {dict(zip(*np.unique(y_train, return_counts=True)))}')
print(f'Val per class:   {dict(zip(*np.unique(y_val, return_counts=True)))}')
print(f'Test per class:  {dict(zip(*np.unique(y_test, return_counts=True)))}')

# 总计
total_per_class = len(y_train) + len(y_val) + len(y_test)
print(f'\nTotal per class: {total_per_class // 4}')

# 预期: signal_len=122881, step=512, window=1024
# total_segments = floor((122881-1024)/512)+1 = 239
# train = round(239*0.7) = 167
# val = round(239*0.15) = 36
# test = 239-167-36 = 36
print(f'\nExpected (step=512): 239 per class -> 167/36/36')
print(f'Actual: {len(y_train)//4}/{len(y_val)//4}/{len(y_test)//4}')

# 检查是否有已训练的模型
import glob
ckpts = glob.glob(os.path.join('checkpoints', '*ysu*'))
if ckpts:
    print(f'\nFound YSU checkpoints: {ckpts}')
    import torch
    for ck in ckpts:
        checkpoint = torch.load(ck, map_location='cpu', weights_only=False)
        print(f'  {os.path.basename(ck)}: val_acc={checkpoint.get("val_acc", "N/A")}')
else:
    print('\nNo YSU checkpoints found')

# 检查 ClassMatrices 中每个类有多少文件
mat_dir = os.path.join(data_path, 'ClassMatrices')
if os.path.exists(mat_dir):
    print('\n--- ClassMatrices ---')
    for cls in sorted(os.listdir(mat_dir)):
        cls_path = os.path.join(mat_dir, cls)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path) if f.endswith('.mat')]
            print(f'{cls}: {len(files)} files')
