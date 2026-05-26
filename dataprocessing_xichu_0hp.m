% dataprocessing_xichu_0hp.m
% 功能：CWRU 西储大学 0HP DE端数据 CWT 预处理
%   1. 加载 CWRU .mat 文件 (DE_time, 0HP)
%   2. 先按时间顺序 70/15/15 切分原始信号（防止数据泄露）
%   3. 对各段独立做 Morlet CWT
%   4. 平衡样本
%   5. 保存为 train_data.mat, val_data.mat, test_data.mat（NCHW格式）

clear; clc; close all;

%% ========== 参数设置 ==========
window_len = 1024;
step = 512;              % 50% 重叠，增加样本量
sampling_rate = 12000;
wavelet = 'amor';
voices_per_octave = 48;
target_num = 118;
img_size = 224;

train_ratio = 0.7;
val_ratio = 0.15;
test_ratio = 0.15;

%% ========== 路径 ==========
data_dir = 'C:\Users\ClearNight\Desktop\10series\CWRU0HP';
output_root = 'C:\Users\ClearNight\Desktop\innovation\xichu0hp';
mat_dir = fullfile(output_root, 'ClassMatrices');

%% ========== 标签映射 ==========
label_map = {
    '97',  0;   % Normal
    '105', 1;   % IR007
    '169', 2;   % IR014
    '209', 3;   % IR021
    '130', 4;   % OR007
    '197', 5;   % OR014
    '234', 6;   % OR021
    '118', 7;   % Ball007
    '185', 8;   % Ball014
    '222', 9;   % Ball021
};

%% ========== 1. 生成 CWT ==========
fprintf('========== 按时间顺序切分信号并生成 CWT ==========\n');

for i = 0:9
    mkdir(fullfile(mat_dir, ['class_', num2str(i)]));
end

X_train_cells = {}; y_train_list = [];
X_val_cells   = {}; y_val_list   = [];
X_test_cells  = {}; y_test_list  = [];

for k = 1:size(label_map, 1)
    filename = [label_map{k, 1}, '.mat'];
    class_label = label_map{k, 2};
    file_path = fullfile(data_dir, filename);

    if ~exist(file_path, 'file')
        warning('文件不存在: %s，跳过', file_path);
        continue;
    end

    data_struct = load(file_path);
    var_names = fieldnames(data_struct);
    de_var = '';
    for v = 1:length(var_names)
        if contains(var_names{v}, 'DE_time')
            de_var = var_names{v};
            break;
        end
    end
    if isempty(de_var)
        warning('未找到DE_time: %s', filename);
        continue;
    end
    signal = data_struct.(de_var);
    sig_len = length(signal);

    total_segments = floor((sig_len - window_len) / step) + 1;

    n_train_seg = round(total_segments * train_ratio);
    n_val_seg   = round(total_segments * val_ratio);
    n_test_seg  = total_segments - n_train_seg - n_val_seg;

    fprintf('[类别 %d] %s: 信号长度=%d, 总可切=%d段, 训练=%d, 验证=%d, 测试=%d\n', ...
            class_label, filename, sig_len, total_segments, n_train_seg, n_val_seg, n_test_seg);

    % --- 训练段 (前 70% 的时间) ---
    for s = 1:n_train_seg
        start_idx = (s-1) * step + 1;
        seg = signal(start_idx : start_idx + window_len - 1);
        [cfs, ~] = cwt(seg, wavelet, sampling_rate, 'VoicesPerOctave', voices_per_octave);
        mag = abs(cfs);
        mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));
        img = imresize(mag_norm, [img_size img_size]);
        X_train_cells{end+1} = img;
        y_train_list(end+1) = class_label;
    end

    % --- 验证段 (中间 15% 的时间) ---
    offset_val = n_train_seg * step;
    for s = 1:n_val_seg
        start_idx = offset_val + (s-1) * step + 1;
        seg = signal(start_idx : start_idx + window_len - 1);
        [cfs, ~] = cwt(seg, wavelet, sampling_rate, 'VoicesPerOctave', voices_per_octave);
        mag = abs(cfs);
        mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));
        img = imresize(mag_norm, [img_size img_size]);
        X_val_cells{end+1} = img;
        y_val_list(end+1) = class_label;
    end

    % --- 测试段 (后 15% 的时间) ---
    offset_test = (n_train_seg + n_val_seg) * step;
    for s = 1:n_test_seg
        start_idx = offset_test + (s-1) * step + 1;
        seg = signal(start_idx : start_idx + window_len - 1);
        [cfs, ~] = cwt(seg, wavelet, sampling_rate, 'VoicesPerOctave', voices_per_octave);
        mag = abs(cfs);
        mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));
        img = imresize(mag_norm, [img_size img_size]);
        X_test_cells{end+1} = img;
        y_test_list(end+1) = class_label;
    end
end

%% ========== 2. 合并、打乱（仅在各自集合内打乱）==========
fprintf('\n========== 合并与划分 ==========\n');

X_train = cat(4, X_train_cells{:});
y_train = y_train_list(:);
X_val   = cat(4, X_val_cells{:});
y_val   = y_val_list(:);
X_test  = cat(4, X_test_cells{:});
y_test  = y_test_list(:);

% 各集合内随机打乱（不打乱跨集合）
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
fprintf('  通道数: 1\n');
fprintf('  类别数: 10\n');
fprintf('  图像尺寸: %dx%d\n', img_size, img_size);

save(fullfile(output_root, 'train_data.mat'), 'X_train', 'y_train', '-v7.3');
save(fullfile(output_root, 'val_data.mat'),   'X_val',   'y_val',   '-v7.3');
save(fullfile(output_root, 'test_data.mat'),  'X_test',  'y_test',  '-v7.3');

fprintf('\n========== 完成 ==========\n');
