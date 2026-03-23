import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision.models import resnet50,resnet18,efficientnet_b0,vit_b_16
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score
)
import logging
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import argparse
@dataclass
class Config:

    data_root: str = r"./data/cornealconfocal"
    work_root: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),"classification_outputs")
    
    batch_size: int = 32
    epochs: int = 100
    lr: float = 5e-5
    num_workers: int = 2
    seed: int = 42
    
    model_list = []
    image_size: Tuple[int, int] = (384, 384) 
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    def __init__(self, args):
        self.data_root = args.data_root
        self.work_root = args.output
        self.model_list = [args.model_type]
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.seed = args.seed

def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CorneaDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, root_dir: str, target_size: Tuple[int, int], transform: Optional[T.Compose] = None):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.transform = transform
        self.samples =[]
        self.class_to_idx = {}
        
        self._load_metadata()

    def _load_metadata(self) -> None:
        class_id = 0
        for major in ["Normal", "Lesion"]:
            major_dir = self.root_dir / major
            if not major_dir.exists():
                continue
            
            for cls_dir in sorted(p for p in major_dir.iterdir() if p.is_dir()):
                class_name = f"{major}_{cls_dir.name}"
                self.class_to_idx[class_name] = class_id
                
                img_dir = cls_dir / "Images"
                if not img_dir.exists():
                    continue
                    
                for img_path in img_dir.iterdir():
                    if img_path.suffix.lower() in self.IMG_EXTS:
                        self.samples.append((str(img_path), class_id))
                class_id += 1
                
        self.classes = list(self.class_to_idx.keys())
        self.labels = np.array([s[1] for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        
        arr = np.array(img)
        h, w = self.target_size
        arr = arr[:h, :w]
        img = Image.fromarray(arr)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
            
        return img, label, path

def get_model(cfg, model_type: str, num_classes: int) -> nn.Module:
    model_type = model_type.lower()
    
    if model_type == "resnet18":
        model = resnet18(weights='DEFAULT')
        w = model.conv1.weight.data
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = w.mean(dim=1, keepdim=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_type == "resnet50":
        model = resnet50(weights='DEFAULT')
        w = model.conv1.weight.data
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = w.mean(dim=1, keepdim=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_type == "efficientnetb0":
        model = efficientnet_b0(weights='DEFAULT')
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1, out_channels=old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None)
        )
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias
        model.features[0][0] = new_conv
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_type == "vit":
        model = vit_b_16(weights='DEFAULT')
        w = model.conv_proj.weight.data  # shape: (768, 3, 16, 16)
        new_conv = nn.Conv2d(
            1, model.conv_proj.out_channels,
            kernel_size=model.conv_proj.kernel_size,
            stride=model.conv_proj.stride,
            padding=model.conv_proj.padding,
            bias=model.conv_proj.bias is not None
        )
        new_conv.weight.data = w.mean(dim=1, keepdim=True)
        if model.conv_proj.bias is not None:
            new_conv.bias.data = model.conv_proj.bias.data.clone()
        model.conv_proj = new_conv
        # 修改分类头
        in_features = model.heads.head.in_features  # 默认 768
        model.heads.head = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return model.to(cfg.device)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    results =[]
    num_classes = len(class_names)
    y_true_onehot = np.eye(num_classes)[y_true]

    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        
        try:
            auc_score = roc_auc_score(y_true_onehot[:, c], y_proba[:, c])
        except ValueError:
            auc_score = np.nan
            
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        
        try:
            kappa = cohen_kappa_score(y_true_bin, y_pred_bin)
        except Exception:
            kappa = np.nan
            
        results.append({
            "Class": class_names[c],
            "Accuracy": acc,
            "AUC": auc_score,
            "Precision": prec,
            "Recall (Sens)": rec,
            "Specificity": spec,
            "F1_Score": f1,
            "Kappa": kappa
        })
        
    return pd.DataFrame(results)

class Trainer:
    def __init__(self, model: nn.Module, model_name: str, config: Config, logger: logging.Logger):
        self.model = model.to(config.device)
        self.model_name = model_name
        self.cfg = config
        self.logger = logger
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )
        
        self.save_dir = Path(config.work_root) / model_name / "results"
        self.ckpt_dir = Path(config.work_root) / model_name / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_model_path = self.ckpt_dir / "best_model.pth"

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for xb, yb, _ in tqdm(loader, desc="Training", leave=False):
            xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)
            self.optimizer.zero_grad()
            
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        for xb, yb, _ in loader:
            xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            
        return total_loss / len(loader.dataset)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.logger.info(f"--- Starting training for {self.model_name} ---")
        best_val_loss = float("inf")
        
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Epoch {epoch:03d}/{self.cfg.epochs} | "
                             f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(f"  --> Saved new best model (Val Loss: {best_val_loss:.4f})")

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, class_names: List[str]) -> None:
        self.logger.info(f"--- Evaluating {self.model_name} ---")
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.cfg.device))
        self.model.eval()
        
        all_probs, all_preds, all_labels, all_paths = [], [],[],[]
        
        for xb, yb, paths in tqdm(val_loader, desc="Evaluating"):
            xb = xb.to(self.cfg.device)
            logits = self.model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_labels.append(yb.numpy())
            all_paths.extend(paths)
            
        probs = np.vstack(all_probs)
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        results_df = pd.DataFrame({
            "Filename": all_paths,
            "GT_Label": labels,
            "Pred_Label": preds,
            "GT_Name": [class_names[idx] for idx in labels],
            "Pred_Name": [class_names[idx] for idx in preds],
        })
        for i, cname in enumerate(class_names):
            results_df[f"Prob_{cname}"] = probs[:, i]
            
        pred_csv = self.save_dir / "validation_predictions.csv"
        results_df.to_csv(pred_csv, index=False)
        self.logger.info(f"Saved predictions to {pred_csv}")
        
        metrics_df = compute_metrics(labels, preds, probs, class_names)
        metrics_csv = self.save_dir / "per_class_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        self.logger.info(f"\nPer-class Metrics:\n{metrics_df.to_string()}")

def main(args):
    cfg = Config(args)
    seed_everything(cfg.seed)
    
    main_logger = setup_logger(os.path.join(cfg.work_root, "logs"), "main_pipeline")
    main_logger.info("Initializing Pipeline...")
    main_logger.info(f"Device: {cfg.device}")

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize([0.5], [0.25])
    ])
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.25])
    ])

    main_logger.info("Loading dataset and performing stratified split...")
    full_dataset = CorneaDataset(cfg.data_root, target_size=cfg.image_size)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=cfg.seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(full_dataset)), full_dataset.labels))

    train_set = Subset(CorneaDataset(cfg.data_root, cfg.image_size, transform=train_transform), train_idx)
    val_set = Subset(CorneaDataset(cfg.data_root, cfg.image_size, transform=eval_transform), val_idx)
    
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    for model_name in cfg.model_list:
        model_logger = setup_logger(os.path.join(cfg.work_root, model_name, "logs"), f"{model_name}_train")
        
        try:
            model = get_model(cfg, model_name, num_classes)
            trainer = Trainer(model, model_name, cfg, model_logger)

            if model_name == "vit":
                transform_train = T.Compose([
                    T.Resize((224, 224)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.25])
                ])

                transform_eval = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.25])
                ])

                train_set = Subset(CorneaDataset(cfg.data_root, cfg.image_size, transform=transform_train), train_idx)
                val_set = Subset(CorneaDataset(cfg.data_root, cfg.image_size, transform=transform_eval), val_idx)
                
                train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
                val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

            
            trainer.fit(train_loader, val_loader)
            trainer.evaluate(val_loader, class_names)
            
        except Exception as e:
            model_logger.error(f"Error occurred while processing {model_name}: {e}", exc_info=True)
            
    main_logger.info("Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="./classification_outputs")
    parser.add_argument("--model_type", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main(args)