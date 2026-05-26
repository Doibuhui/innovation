"""
工具函数
"""
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
import plotly.express as px


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config):
    """获取设备"""
    if config.DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        return device, f"GPU: {gpu_name}"
    return torch.device('cpu'), "CPU"


def plot_training_history(history):
    """绘制训练历史曲线（支持实时更新）"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history['val_loss'] and any(v > 0 for v in history['val_loss']):
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy 曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if history['val_acc'] and any(v > 0 for v in history['val_acc']):
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names):
    """绘制混淆矩阵"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        title="Confusion Matrix",
        height=500,
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig


def plot_prediction_probabilities(prob_dict):
    """绘制预测概率柱状图"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(prob_dict.keys()),
            y=list(prob_dict.values()),
            marker_color=px.colors.qualitative.Set2[:len(prob_dict)]
        )
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400
    )
    return fig


def plot_sample_images(images, labels, class_names, num_samples=4):
    """绘制样本图像"""
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, num_samples,
                            figsize=(num_samples*3, num_classes*3))

    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for i, class_name in enumerate(class_names):
        class_indices = np.where(labels == i)[0]
        selected = np.random.choice(class_indices, min(num_samples, len(class_indices)), replace=False)

        for j, idx in enumerate(selected):
            if j < num_samples:
                img = images[idx]
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                elif img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))

                ax = axes[i, j]
                ax.imshow(img, cmap='jet' if img.ndim == 2 else None)
                ax.set_title(f'{class_name}' if j == 0 else '')
                ax.axis('off')

    plt.tight_layout()
    return fig


def format_class_report(report_text):
    """格式化分类报告为 Markdown"""
    lines = report_text.strip().split('\n')
    md = "```\n" + report_text + "\n```"
    return md
