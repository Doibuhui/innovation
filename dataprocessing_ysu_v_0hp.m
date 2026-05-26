% dataprocessing_ysu_v_0hp.m
% 功能：YSU_V 数据集 CWT 预处理（8通道 → PCA 3通道 → RGB）
%   1. 加载 4 类故障 .mat 文件（每类 8 通道）
%   2. PCA 降维：8通道 → 3通道
%   3. 先按时间顺序 70/15/15 切分（防止数据泄露）
%   4. 对各段独立做 Morlet CWT
%   5. 3通道堆叠为 RGB 图像
%   6. 保存为 NCHW 格式（N, 3, 224, 224）

clear; clc; close all;

%% ========== 参数设置 ==========
window_len = 1024;
step = 512;              % 50% 重叠，增加样本量
sampling_rate = 12000;
wavelet = 'amor';
voices_per_octave = 48;
img_size = 224;

train_ratio = 0.7;
val_ratio = 0.15;
test_ratio = 0.15;

%% ========== 路径 ==========
data_dir = 'C:\Users\ClearNight\Desktop\YSU_V';
output_root = 'C:\Users\ClearNight\Desktop\innovation\ysu_v0hp';
mat_dir = fullfile(output_root, 'ClassMatrices');

%% ========== 类别映射 ==========
class_files = {
    'Ball_0.mat',   0;
    'Inner_0.mat',  1;
    'Narmal_0.mat', 2;
    'Outer_0.mat',  3;
};

num_classes = size(class_files, 1);
num_channels = 3;  % PCA 后 3 通道

%% ========== 1. 生成 CWT（先切分信号再 CWT）==========
fprintf('========== 按时间顺序切分信号并生成 CWT ==========\n');

for i = 0:num_classes-1
    mkdir(fullfile(mat_dir, ['class_', num2str(i)]));
end

X_train_cells = {}; y_train_list = [];
X_val_cells   = {}; y_val_list   = [];
X_test_cells  = {}; y_test_list  = [];

for k = 1:num_classes
    filename = class_files{k, 1};
    class_label = class_files{k, 2};
    file_path = fullfile(data_dir, filename);

    if ~exist(file_path, 'file')
        warning('文件不存在: %s，跳过', file_path);
        continue;
    end

    raw = load(file_path);
    signal = raw.data;

    % 确保 (样本, 通道) 格式
    if size(signal, 1) < size(signal, 2)
        signal = signal';
    end

    sig_len = size(signal, 1);
    num_ch = size(signal, 2);

    fprintf('[类别 %d] %s: 信号长度=%d, 通道数=%d -> PCA 3通道\n', ...
            class_label, filename, sig_len, num_ch);

    % 计算可切段数
    total_segments = floor((sig_len - window_len) / step) + 1;
    n_train_seg = round(total_segments * train_ratio);
    n_val_seg   = round(total_segments * val_ratio);
    n_test_seg  = total_segments - n_train_seg - n_val_seg;

    train_end_idx = n_train_seg * step;
    val_end_idx   = (n_train_seg + n_val_seg) * step;

    % PCA 仅在训练段上拟合，然后应用到全部
    signal_train = signal(1:train_end_idx, :);
    [coeff, ~, ~, ~, explained] = pca(signal_train);
    fprintf('  PCA 前3主成分解释方差: %.1f%%\n', sum(explained(1:3)));

    % 用训练集的 PCA 基变换全部信号
    signal_pca_all = signal * coeff(:, 1:3);

    % 归一化参数仅在训练段计算
    pca_train = signal_train * coeff(:, 1:3);
    ch_mins = zeros(1, 3);
    ch_maxs = zeros(1, 3);
    for ch = 1:3
        ch_mins(ch) = min(pca_train(:, ch));
        ch_maxs(ch) = max(pca_train(:, ch));
    end

    % 用训练集的 min/max 归一化全部信号
    signal_norm = zeros(size(signal_pca_all));
    for ch = 1:3
        if ch_maxs(ch) - ch_mins(ch) > 0
            signal_norm(:, ch) = (signal_pca_all(:, ch) - ch_mins(ch)) / (ch_maxs(ch) - ch_mins(ch));
        end
    end

    fprintf('  总可切=%d段, 训练=%d, 验证=%d, 测试=%d\n', total_segments, n_train_seg, n_val_seg, n_test_seg);

    % --- 训练段 ---
    for s = 1:n_train_seg
        start_idx = (s-1) * step + 1;
        seg = signal_norm(start_idx : start_idx + window_len - 1, :);

        cwt_stack = zeros(img_size, img_size, num_channels);
        for ch = 1:num_channels
            [cfs, ~] = cwt(seg(:, ch), wavelet, sampling_rate, 'VoicesPerOctave', voices_per_octave);
            mag = abs(cfs);
            mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));
            cwt_stack(:, :, ch) = imresize(mag_norm, [img_size, img_size]);
        end
        X_train_cells{end+1} = cwt_stack;
        y_train_list(end+1) = class_label;
    end

    % --- 验证段 ---
    offset_val = n_train_seg * step;
    for s = 1:n_val_seg
        start_idx = offset_val + (s-1) * step + 1;
        seg = signal_norm(start_idx : start_idx + window_len - 1, :);

        cwt_stack = zeros(img_size, img_size, num_channels);
        for ch = 1:num_channels
            [cfs, ~] = cwt(seg(:, ch), wavelet, sampling_rate, 'VoicesPerOctave', voices_per_octave);
            mag = abs(cfs);
            mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));
            cwt_stack(:, :, ch) = imresize(mag_norm, [img_size, img_size]);
        end
        X_val_cells{end+1} = cwt_stack;
        y_val_list(end+1) = class_label;
    end

    % --- 测试段 ---
    offset_test = (n_train_seg + n_val_seg) * step;
    for s = 1:n_test_seg
        start_idx = offset_test + (s-1) * step + 1;
        seg = signal_norm(start_idx : start_idx + window_len - 1, :);

        cwt_stack = zeros(img_size, img_size, num_channels);
        for ch = 1:num_channels
            [cfs, ~] = cwt(seg(:, ch), wavelet, sampling_rate, 'VoicesPerOctave', voices_per_octave);
            mag = abs(cfs);
            mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));
            cwt_stack(:, :, ch) = imresize(mag_norm, [img_size, img_size]);
        end
        X_test_cells{end+1} = cwt_stack;
        y_test_list(end+1) = class_label;
    end
end

%% ========== 2. 合并、各集合内打乱 ==========
fprintf('\n========== 合并与划分 ==========\n');

X_train = cat(4, X_train_cells{:});
y_train = y_train_list(:);
X_val   = cat(4, X_val_cells{:});
y_val   = y_val_list(:);
X_test  = cat(4, X_test_cells{:});
y_test  = y_test_list(:);

rng(42);
idx_train = randperm(size(X_train, 4));
X_train = X_train(:, :, :, idx_train);
y_train = y_train(idx_train);

idx_val = randperm(size(X_val, 4));
X_val = X_val(:, :, :, idx_val);
y_val = y_val(idx_val);

idx_test = randperm(size(X_test, 4));
X_test = X_test(:, :, :, idx_test);
y_test = y_test(idx_test);

%% ========== 3. 转为 NCHW 格式并保存 ==========
X_train = permute(X_train, [4, 3, 1, 2]);
X_val   = permute(X_val,   [4, 3, 1, 2]);
X_test  = permute(X_test,  [4, 3, 1, 2]);

fprintf('  训练集: %d 样本\n', size(X_train, 1));
fprintf('  验证集: %d 样本\n', size(X_val, 1));
fprintf('  测试集: %d 样本\n', size(X_test, 1));
fprintf('  通道数: %d (PCA)\n', num_channels);
fprintf('  类别数: %d\n', num_classes);
fprintf('  图像尺寸: %dx%d\n', img_size, img_size);

save(fullfile(output_root, 'train_data.mat'), 'X_train', 'y_train', '-v7.3');
save(fullfile(output_root, 'val_data.mat'),   'X_val',   'y_val',   '-v7.3');
save(fullfile(output_root, 'test_data.mat'),  'X_test',  'y_test',  '-v7.3');

fprintf('\n========== 完成 ==========\n');
