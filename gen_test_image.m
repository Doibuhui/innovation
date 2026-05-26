% gen_test_image.m
% 从 CWRU 0HP 原始信号生成独立测试图（供诊断页面使用）
% 改 class_id 选不同故障，改 start_offset 选信号中的位置

clear; clc;

%% ========== 参数 ==========
signal_path = "C:\Users\ClearNight\Desktop\10series\CWRU0HP\97.mat";  % 改文件名选故障
class_id = 4;               % 故障类别ID
start_offset = 115000;      % 信号中的起始位置（选训练集未覆盖的区域）

window_len = 1024;
sampling_rate = 12000;
wavelet = 'amor';
voices_per_octave = 48;

%% ========== 加载信号 ==========
data = load(signal_path);
var_names = fieldnames(data);
de_var = '';
for v = 1:length(var_names)
    if contains(var_names{v}, 'DE_time')
        de_var = var_names{v};
        break;
    end
end
signal = data.(de_var);

%% ========== CWT ==========
seg = signal(start_offset : start_offset + window_len - 1);
[cfs, ~] = cwt(seg, wavelet, sampling_rate, 'VoicesPerOctave', voices_per_octave);
mag = abs(cfs);
mag_norm = (mag - min(mag(:))) / (max(mag(:)) - min(mag(:)));
img = imresize(mag_norm, [224, 224]);

%% ========== 保存 ==========
out_path = 'C:\Users\ClearNight\Desktop\innovation\test\matlab_test.png';
imwrite(img, out_path);
fprintf('已保存: %s\n', out_path);
fprintf('类别ID: %d\n', class_id);
