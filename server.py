"""
FastAPI backend for PINN Fault Diagnosis System
REST API + WebSocket for real-time training updates
"""
import asyncio
import json
import os
import uuid
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from io import BytesIO

from config import Config
from models import PINNFaultDiagnosis
from trainer import Trainer
from utils import set_seed, get_device
from dataloader import detect_dataset_info, save_class_names, get_dataloaders

app = FastAPI(title="PINN Fault Diagnosis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    "class_names": None,
}

ws_clients: dict[str, WebSocket] = {}


class DetectRequest(BaseModel):
    data_path: str


class ClassNamesRequest(BaseModel):
    data_path: str
    class_names: str


class TrainRequest(BaseModel):
    data_path: str
    class_names: str
    epochs: int = 20


class LoadModelRequest(BaseModel):
    model_path: str


@app.get("/api/health")
async def health():
    return {"status": "ok", "training": app_state["is_training"]}


@app.post("/api/detect")
async def detect_dataset(req: DetectRequest):
    if not req.data_path or not os.path.exists(req.data_path):
        raise HTTPException(status_code=400, detail="无效的数据路径")

    try:
        info = detect_dataset_info(req.data_path)
        app_state["dataset_info"] = info
        app_state["data_path"] = req.data_path

        defaults = []
        for i in range(info["num_classes"]):
            if info["class_names"] and i < len(info["class_names"]):
                defaults.append(info["class_names"][i])
            else:
                defaults.append(f"Class_{i}")

        return {
            "success": True,
            "info": {
                "in_channels": info["in_channels"],
                "num_classes": info["num_classes"],
                "image_size": info["image_size"],
                "train_samples": info["train_samples"],
                "val_samples": info["val_samples"],
                "test_samples": info["test_samples"],
            },
            "default_class_names": ", ".join(defaults),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/class-names")
async def set_class_names(req: ClassNamesRequest):
    if not req.class_names or not req.class_names.strip():
        raise HTTPException(status_code=400, detail="需要输入类别名")

    names = [n.strip() for n in req.class_names.split(",") if n.strip()]

    if app_state["dataset_info"]:
        num_classes = app_state["dataset_info"]["num_classes"]
        if len(names) != num_classes:
            raise HTTPException(
                status_code=400,
                detail=f"需要 {num_classes} 个类别名，但只提供了 {len(names)} 个",
            )
        app_state["dataset_info"]["class_names"] = names

    app_state["class_names"] = names

    try:
        if app_state["data_path"]:
            save_class_names(app_state["data_path"], names)
    except Exception:
        pass

    return {"success": True, "class_names": names}


@app.get("/api/checkpoints")
async def list_checkpoints():
    checkpoint_dir = Config.CHECKPOINT_DIR
    if not os.path.exists(checkpoint_dir):
        return {"checkpoints": []}
    files = []
    for f in sorted(os.listdir(checkpoint_dir), reverse=True):
        if f.endswith(".pth"):
            full_path = os.path.join(checkpoint_dir, f)
            mtime = os.path.getmtime(full_path)
            from datetime import datetime
            dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            files.append({"name": f, "path": full_path, "created_at": dt})
    return {"checkpoints": files}


@app.post("/api/load-model")
async def load_model(req: LoadModelRequest):
    if not req.model_path or not os.path.exists(req.model_path):
        raise HTTPException(status_code=400, detail="无效的模型路径")

    try:
        device, device_info = get_device(Config())
        checkpoint = torch.load(req.model_path, map_location=device)

        in_channels = checkpoint.get("in_channels", 1)
        num_classes = checkpoint.get("num_classes", 10)
        class_names = checkpoint.get("class_names", None)
        val_acc = checkpoint.get("val_acc", "N/A")
        timestamp = checkpoint.get("timestamp", "未知")

        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        model = PINNFaultDiagnosis(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_type=Config.CONV_TYPE,
            use_mha=Config.USE_MHA,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = num_params * 4 / (1024 * 1024)

        app_state["model"] = model
        app_state["device"] = device
        app_state["device_info"] = device_info

        config = Config()
        config.IN_CHANNELS = in_channels
        config.NUM_CLASSES = num_classes
        config.CLASS_NAMES = class_names
        app_state["config"] = config
        app_state["class_names"] = class_names

        return {
            "success": True,
            "info": {
                "device": device_info,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "val_acc": val_acc,
                "class_names": class_names,
                "num_params": num_params,
                "trainable_params": num_params,
                "model_size_mb": round(model_size_mb, 2),
                "timestamp": timestamp,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if app_state["model"] is None:
        raise HTTPException(status_code=400, detail="未加载模型")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image).astype(np.float32)

        if image_np.max() > 1.0:
            image_np = image_np / 255.0

        config = app_state["config"]
        in_ch = config.IN_CHANNELS

        if image_np.ndim == 2:
            image_np = np.expand_dims(image_np, 0) if in_ch == 1 else np.stack([image_np] * 3, axis=0)
        elif image_np.ndim == 3:
            if image_np.shape[2] == 3:
                image_np = np.transpose(image_np, (2, 0, 1))
                if in_ch == 1:
                    image_np = np.mean(image_np, axis=0, keepdims=True)
            elif image_np.shape[2] == 1:
                image_np = np.transpose(image_np, (2, 0, 1))
                if in_ch == 3:
                    image_np = np.repeat(image_np, 3, axis=0)

        image_tensor = torch.from_numpy(image_np).float().unsqueeze(0)
        image_tensor = image_tensor.to(app_state["device"])

        model = app_state["model"]
        model.eval()
        class_names = config.CLASS_NAMES

        with torch.no_grad():
            logits, _ = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        prob_dict = {name: float(probs[0][i]) for i, name in enumerate(class_names)}

        return {
            "success": True,
            "prediction": {
                "class_name": class_names[pred],
                "class_index": pred,
                "confidence": conf,
                "probabilities": prob_dict,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    ws_clients[client_id] = websocket

    try:
        data = await websocket.receive_text()
        req = json.loads(data)

        data_path = req.get("data_path")
        class_names_str = req.get("class_names", "")
        epochs = int(req.get("epochs", 20))

        if app_state["is_training"]:
            await websocket.send_json({"type": "error", "message": "训练正在进行中"})
            await websocket.close()
            return

        if not data_path or not os.path.exists(data_path):
            await websocket.send_json({"type": "error", "message": "无效的数据路径"})
            await websocket.close()
            return

        app_state["is_training"] = True
        info = detect_dataset_info(data_path)
        app_state["dataset_info"] = info
        app_state["data_path"] = data_path

        names = [n.strip() for n in class_names_str.split(",") if n.strip()]
        if not names:
            names = [f"Class_{i}" for i in range(info["num_classes"])]
        if len(names) != info["num_classes"]:
            names = [f"Class_{i}" for i in range(info["num_classes"])]

        app_state["class_names"] = names
        info["class_names"] = names

        config = Config()
        config.IN_CHANNELS = info["in_channels"]
        config.NUM_CLASSES = info["num_classes"]
        config.CLASS_NAMES = names
        config.EPOCHS = epochs
        config.CURRENT_DATASET = os.path.basename(data_path)
        save_path = Config.generate_checkpoint_path(config.CURRENT_DATASET)
        config.SAVE_PATH = save_path

        app_state["config"] = config
        set_seed(config.SEED)

        model = PINNFaultDiagnosis(
            in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
            conv_type=config.CONV_TYPE,
            use_mha=config.USE_MHA,
        )
        device, device_info = get_device(config)
        model.to(device)
        app_state["model"] = model
        app_state["device"] = device
        app_state["device_info"] = device_info

        train_loader, val_loader, test_loader = get_dataloaders(config, data_path)
        app_state["train_loader"] = train_loader
        app_state["val_loader"] = val_loader
        app_state["test_loader"] = test_loader

        trainer = Trainer(model, config, train_loader, val_loader)

        import queue
        import concurrent.futures
        msg_queue = queue.Queue()

        def progress_callback(cb):
            msg = {
                "type": "epoch",
                "epoch": cb["epoch"],
                "total_epochs": cb["total_epochs"],
                "train_loss": cb["train_loss"],
                "train_acc": cb["train_acc"],
                "val_loss": cb["val_loss"],
                "val_acc": cb["val_acc"],
                "best_val_acc": cb["best_val_acc"],
                "best_epoch": cb["best_epoch"],
                "history": {
                    "train_loss": cb["history"]["train_loss"],
                    "train_acc": cb["history"]["train_acc"],
                    "val_loss": cb["history"]["val_loss"],
                    "val_acc": cb["history"]["val_acc"],
                },
            }
            msg_queue.put(msg)

        async def drain_queue():
            while not msg_queue.empty():
                try:
                    msg = msg_queue.get_nowait()
                    await websocket.send_json(msg)
                except queue.Empty:
                    break
                except WebSocketDisconnect:
                    break

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            train_future = loop.run_in_executor(pool, lambda: trainer.train(progress_callback=progress_callback))
            while not train_future.done():
                await asyncio.sleep(0.5)
                await drain_queue()
            await drain_queue()
            history = train_future.result()
        app_state["history"] = history

        results = trainer.evaluate(test_loader)
        cm = results["confusion_matrix"].tolist() if len(results["confusion_matrix"]) > 0 else []

        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = num_params * 4 / (1024 * 1024)

        done_msg = {
            "type": "done",
            "best_epoch": trainer.best_epoch,
            "best_val_acc": trainer.best_val_acc,
            "test_accuracy": results["accuracy"],
            "confusion_matrix": cm,
            "class_names": names,
            "report": results["report"],
            "save_path": save_path,
            "model_info": {
                "num_params": num_params,
                "trainable_params": trainable_params,
                "model_size_mb": round(model_size_mb, 2),
                "in_channels": config.IN_CHANNELS,
                "num_classes": config.NUM_CLASSES,
                "conv_type": config.CONV_TYPE,
                "use_mha": config.USE_MHA,
                "epochs": config.EPOCHS,
            },
        }
        await websocket.send_json(done_msg)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        app_state["is_training"] = False
        ws_clients.pop(client_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
