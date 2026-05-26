"""
故障诊断系统 - 可视化界面
流程：检测数据集 → 输入类别名 → 训练 → 诊断
启动命令: python app.py
访问地址: http://127.0.0.1:7860
"""
import gradio as gr
import numpy as np
import torch
import os
from PIL import Image

from config import Config
from models import PINNFaultDiagnosis
from trainer import Trainer
from utils import set_seed, get_device, plot_training_history, plot_confusion_matrix, plot_prediction_probabilities


# ==================== 全局状态 ====================
app_state = {
    "model": None,
    "device": None,
    "device_info": "",
    "config": None,
    "train_loader": None,
    "val_loader": None,
    "test_loader": None,
    "history": None,
    "dataset_info": None,
    "data_path": None,
    "is_training": False,
}


# ==================== 数据集检测 ====================
def detect_dataset(data_path):
    """检测已有数据集"""
    if not data_path or not os.path.exists(data_path):
        return "请选择有效的数据集文件夹", None, gr.update()

    try:
        from dataloader import detect_dataset_info
        info = detect_dataset_info(data_path)
        app_state["dataset_info"] = info
        app_state["data_path"] = data_path

        info_text = f"""
| 项目 | 值 |
|------|-----|
| 通道数 | {info['in_channels']} |
| 类别数 | {info['num_classes']} |
| 图像尺寸 | {info['image_size']} |
| 训练样本 | {info['train_samples']} |
| 验证样本 | {info['val_samples']} |
| 测试样本 | {info['test_samples']} |
        """

        # 默认类别名
        defaults = []
        for i in range(info['num_classes']):
            if info['class_names'] and i < len(info['class_names']):
                defaults.append(info['class_names'][i])
            else:
                defaults.append(f"Class_{i}")
        default_str = ", ".join(defaults)

        return "检测成功！", info_text, default_str

    except Exception as e:
        return f"检测失败: {str(e)}", None, gr.update()


def save_class_names(class_name_str):
    """保存类别名"""
    if not app_state["dataset_info"]:
        return "**请先检测数据集**"

    if not class_name_str or not class_name_str.strip():
        return "**请输入类别名**"

    names = [n.strip() for n in class_name_str.split(',') if n.strip()]
    num_classes = app_state["dataset_info"]["num_classes"]

    if len(names) != num_classes:
        return f"**错误**: 类别名数量({len(names)})与类别数({num_classes})不匹配，请重新输入"

    app_state["dataset_info"]["class_names"] = names

    try:
        from dataloader import save_class_names as save_names
        save_names(app_state["data_path"], names)
    except Exception as e:
        pass

    return f"**已保存 {len(names)} 个类别名**: {', '.join(names)}"


# ==================== 训练 ====================
def start_training(epochs, progress=gr.Progress()):
    """开始训练"""
    if not app_state["dataset_info"]:
        return "请先检测数据集", None, None, ""

    if app_state["is_training"]:
        return "训练进行中...", None, None, ""

    try:
        app_state["is_training"] = True

        info = app_state["dataset_info"]
        config = Config()
        config.IN_CHANNELS = info["in_channels"]
        config.NUM_CLASSES = info["num_classes"]
        config.CLASS_NAMES = info.get("class_names", [f"Class_{i}" for i in range(info["num_classes"])])
        config.EPOCHS = int(epochs)
        config.CURRENT_DATASET = os.path.basename(app_state["data_path"])

        app_state["config"] = config
        set_seed(config.SEED)

        # 创建模型
        model = PINNFaultDiagnosis(
            in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
            conv_type=config.CONV_TYPE,
            use_mha=config.USE_MHA
        )
        device, device_info = get_device(config)
        model.to(device)
        app_state["model"] = model
        app_state["device"] = device
        app_state["device_info"] = device_info

        # 加载数据
        from dataloader import get_dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(config, app_state["data_path"])
        app_state["train_loader"] = train_loader
        app_state["val_loader"] = val_loader
        app_state["test_loader"] = test_loader

        trainer = Trainer(model, config, train_loader, val_loader)

        # 实时回调
        loss_fig = None
        cm_fig = None
        status_text = ""

        def progress_callback(cb):
            nonlocal loss_fig, cm_fig, status_text
            epoch = cb['epoch']
            total = cb['total_epochs']
            progress(epoch / total, desc=f"Epoch {epoch}/{total}")

            history = cb['history']
            loss_fig = plot_training_history(history)

            status_text = (
                f"| Epoch | {epoch}/{total} |\n"
                f"|------|-----|\n"
                f"| Train Loss | {cb['train_loss']:.4f} |\n"
                f"| Train Acc | {cb['train_acc']:.2f}% |\n"
                f"| Val Loss | {cb['val_loss']:.4f} |\n"
                f"| Val Acc | {cb['val_acc']:.2f}% |\n"
                f"| Best Val | {cb['best_val_acc']:.2f}% (Ep.{cb['best_epoch']}) |\n"
            )

        history = trainer.train(progress_callback=progress_callback)
        app_state["history"] = history

        results = trainer.evaluate(test_loader)

        if len(results['confusion_matrix']) > 0:
            cm_fig = plot_confusion_matrix(results['confusion_matrix'], config.CLASS_NAMES)

        save_path = config.get_checkpoint_path(config.CURRENT_DATASET)

        result_text = (
            f"| 指标 | 值 |\n|------|----|\n"
            f"| Best Epoch | {trainer.best_epoch} |\n"
            f"| Best Val Acc | {trainer.best_val_acc:.2f}% |\n"
            f"| Test Acc | {results['accuracy']:.2f}% |\n"
            f"| 模型路径 | `{save_path}` |\n\n"
            f"**分类报告**\n```\n{results['report']}\n```"
        )

        app_state["is_training"] = False
        progress(1.0, desc="完成")
        return result_text, loss_fig, cm_fig, status_text

    except Exception as e:
        app_state["is_training"] = False
        import traceback
        return f"训练失败: {str(e)}\n{traceback.format_exc()}", None, None, ""


# ==================== 加载模型 ====================
def get_checkpoint_list():
    checkpoint_dir = Config.CHECKPOINT_DIR
    if not os.path.exists(checkpoint_dir):
        return []
    return [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]


def load_model(model_path):
    if not model_path:
        return "请选择模型文件"

    try:
        device, device_info = get_device(Config())
        checkpoint = torch.load(model_path, map_location=device)

        in_channels = checkpoint.get('in_channels', 1)
        num_classes = checkpoint.get('num_classes', 10)
        class_names = checkpoint.get('class_names', None)
        val_acc = checkpoint.get('val_acc', 'N/A')

        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        model = PINNFaultDiagnosis(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_type=Config.CONV_TYPE,
            use_mha=Config.USE_MHA
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        app_state["model"] = model
        app_state["device"] = device
        app_state["device_info"] = device_info

        config = Config()
        config.IN_CHANNELS = in_channels
        config.NUM_CLASSES = num_classes
        config.CLASS_NAMES = class_names
        app_state["config"] = config

        return (
            f"| 项目 | 值 |\n|------|----|\n"
            f"| 设备 | {device_info} |\n"
            f"| 通道数 | {in_channels} |\n"
            f"| 类别数 | {num_classes} |\n"
            f"| Val Acc | {val_acc}% |\n"
            f"| 类别 | {', '.join(class_names)} |\n"
        )
    except Exception as e:
        return f"加载失败: {str(e)}"


# ==================== 诊断 ====================
def predict(image):
    if image is None:
        return "请上传图片", None

    if app_state["model"] is None:
        return "请先训练或加载模型", None

    try:
        config = app_state["config"]
        model = app_state["model"]
        device = app_state["device"]
        class_names = config.CLASS_NAMES

        model.eval()

        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0

            in_ch = config.IN_CHANNELS
            if image.ndim == 2:
                image = np.expand_dims(image, 0) if in_ch == 1 else np.stack([image]*3, axis=0)
            elif image.ndim == 3:
                if image.shape[2] == 3:
                    image = np.transpose(image, (2, 0, 1))
                    if in_ch == 1:
                        image = np.mean(image, axis=0, keepdims=True)
                elif image.shape[2] == 1:
                    image = np.transpose(image, (2, 0, 1))
                    if in_ch == 3:
                        image = np.repeat(image, 3, axis=0)
                elif image.shape[0] in (1, 3):
                    if in_ch == 1 and image.shape[0] == 3:
                        image = np.mean(image, axis=0, keepdims=True)
                    elif in_ch == 3 and image.shape[0] == 1:
                        image = np.repeat(image, 3, axis=0)

            image = torch.from_numpy(image).float().unsqueeze(0)

        image = image.to(device)

        with torch.no_grad():
            logits, _ = model(image)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        pred_name = class_names[pred]
        prob_dict = {name: float(probs[0][i]) for i, name in enumerate(class_names)}
        prob_fig = plot_prediction_probabilities(prob_dict)

        result = f"| 项目 | 值 |\n|------|-----|\n| 预测类别 | **{pred_name}** |\n| 置信度 | {conf:.1%} |\n"
        return result, prob_fig

    except Exception as e:
        return f"预测失败: {str(e)}", None


# ==================== 界面 ====================
with gr.Blocks(title="故障诊断系统", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 智能故障诊断系统\n基于物理信息神经网络（PINN）的轴承故障诊断")

    with gr.Tabs():

        # ===== Tab 1: 训练 =====
        with gr.Tab("1. 训练模型"):

            gr.Markdown("### 第一步：选择数据集")
            gr.Markdown("选择 CWT 预处理后的数据集文件夹（包含 train_data.mat, val_data.mat, test_data.mat）")

            with gr.Row():
                data_path = gr.Textbox(label="数据集路径", placeholder="如 C:\\Users\\xxx\\CWT_1HPMatrices")
                detect_btn = gr.Button("检测数据集", variant="primary")

            with gr.Row():
                detect_status = gr.Markdown("")
                dataset_info = gr.Markdown("")

            gr.Markdown("### 第二步：输入类别名")
            gr.Markdown("按标签顺序输入，逗号分隔。如：Normal, IR007, IR014")

            class_names_input = gr.Textbox(label="类别名称", placeholder="Normal, IR007, IR014, ...")
            save_names_btn = gr.Button("确认类别名")
            names_status = gr.Markdown("")

            gr.Markdown("### 第三步：训练")

            with gr.Row():
                epochs = gr.Slider(5, 200, value=20, step=5, label="训练轮数")
                train_btn = gr.Button("开始训练", variant="primary", size="lg")

            train_status = gr.Markdown("")

            with gr.Row():
                with gr.Column():
                    loss_plot = gr.Plot(label="Loss 曲线")
                with gr.Column():
                    cm_plot = gr.Plot(label="混淆矩阵")

            train_result = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### 加载已有模型")

            with gr.Row():
                model_dropdown = gr.Dropdown(choices=get_checkpoint_list(), label="模型文件", allow_custom_value=True)
                refresh_btn = gr.Button("刷新")
                load_btn = gr.Button("加载模型")

            load_status = gr.Markdown("")

            # 事件绑定
            detect_btn.click(detect_dataset, inputs=[data_path], outputs=[detect_status, dataset_info, class_names_input])
            save_names_btn.click(save_class_names, inputs=[class_names_input], outputs=[names_status])
            train_btn.click(start_training, inputs=[epochs], outputs=[train_result, loss_plot, cm_plot, train_status])
            refresh_btn.click(lambda: gr.Dropdown(choices=get_checkpoint_list()), outputs=[model_dropdown])
            load_btn.click(load_model, inputs=[model_dropdown], outputs=[load_status])


        # ===== Tab 2: 诊断 =====
        with gr.Tab("2. 故障诊断"):
            gr.Markdown("### 上传 CWT 时频图进行诊断")

            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="上传图片", type="numpy", height=300)
                    pred_btn = gr.Button("开始诊断", variant="primary", size="lg")

                with gr.Column():
                    pred_result = gr.Markdown("")
                    prob_plot = gr.Plot(label="概率分布")

            pred_btn.click(predict, inputs=[img_input], outputs=[pred_result, prob_plot])


if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
