"""
训练模块 - 支持每 epoch 实时回调
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime


class Trainer:
    """PINN训练器"""

    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS
        )

        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS}')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, physics_features = self.model(images)

            cls_loss = self.criterion(logits, labels)
            physics_loss = self.model.physics_loss(physics_features)

            loss = cls_loss + self.config.LAMBDA_PHYSICS * physics_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        if self.val_loader is None:
            return 0.0, 0.0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, progress_callback=None):
        """完整训练流程"""
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        for epoch in range(self.config.EPOCHS):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 更新最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                save_path = getattr(self.config, 'SAVE_PATH', self.config.get_checkpoint_path(self.config.CURRENT_DATASET))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'num_classes': self.config.NUM_CLASSES,
                    'in_channels': self.config.IN_CHANNELS,
                    'class_names': self.config.CLASS_NAMES,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }, save_path)

            # 每 epoch 回调
            if progress_callback:
                callback_data = {
                    'epoch': epoch + 1,
                    'total_epochs': self.config.EPOCHS,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': self.best_val_acc,
                    'best_epoch': self.best_epoch,
                    'history': history,
                }
                progress_callback(callback_data)

        return history

    @torch.no_grad()
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        if test_loader is None:
            return {
                'accuracy': 0.0,
                'predictions': np.array([]),
                'labels': np.array([]),
                'confusion_matrix': np.array([]),
                'report': '无测试数据'
            }

        for images, labels in test_loader:
            images = images.to(self.device)

            logits, _ = self.model(images)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = np.mean(all_preds == all_labels) * 100

        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(all_labels, all_preds)

        class_names = self.config.CLASS_NAMES
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.config.NUM_CLASSES)]

        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            digits=4,
            zero_division=0
        )

        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'confusion_matrix': cm,
            'report': report
        }
