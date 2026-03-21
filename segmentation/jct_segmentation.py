
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from sklearn.manifold import TSNE
import torch.optim as optim
from sklearn.model_selection import train_test_split
from dataset.cornealconfocal import cornealconfocal as SegDataset
from model.unet import UNet
from model.nestedunet import NestedUNet
from model.resunet import ResUNet

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_split_csv(file_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_list = [os.path.basename(f) for f in file_list]
    df = pd.DataFrame(file_list, columns=['filename'])
    df.to_csv(save_path, index=False)
    print(f"Split saved to: {save_path}")

def dice_coef(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    dice = (2 * intersection + eps) / union
    return dice.mean()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (2, 3) # 在 H, W 维度求和
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

def train(model, dataset_path, batch_size=8, epochs=100, lr=5e-5, seed=42, save_path="result"):
    print(f"==================== Start Training ====================\n")
    
    # 1. 固定种子
    seed_everything(seed)

    class_name = os.path.basename(dataset_path)
    img_dir = os.path.join(dataset_path, "Images")
    lbl_dir = os.path.join(dataset_path, "Annotations")
    
    os.makedirs(save_path, exist_ok=True)
    split_dir = os.path.join(save_path, "splits")
    
    all_files = sorted(os.listdir(img_dir))
    all_files = np.array(all_files)
    
    train_files, val_files = train_test_split(
        all_files, test_size=0.3, random_state=seed, shuffle=True
    )

    test_files = val_files

    print(f"Total: {len(all_files)}")
    print(f"Train: {len(train_files)} ({(len(train_files)/len(all_files))*100:.1f}%)")
    print(f"Test:  {len(test_files)} ({(len(test_files)/len(all_files))*100:.1f}%)")

    save_split_csv(train_files, os.path.join(split_dir, "train_split.csv"))
    save_split_csv(val_files,   os.path.join(split_dir, "val_split.csv"))
    save_split_csv(test_files,  os.path.join(split_dir, "test_split.csv"))

    train_ds = SegDataset(train_files, img_dir, lbl_dir)
    val_ds   = SegDataset(val_files,   img_dir, lbl_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "unet":
        model = UNet(n_classes=1,input_channels=1).to(device)
    elif model == "nestedunet":
        model = NestedUNet(n_classes=1,input_channels=1).to(device)
    elif model == "resunet":
        model = ResUNet(n_classes=1,input_channels=1).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    warmup_epochs = 10
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.1)

    max_val_dice = 0.0
    best_model_path = os.path.join(save_path, f"best_{class_name}.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_dice = 0

        for img, lbl in train_loader:
            img, lbl = img.to(device), lbl.to(device)

            pred = model(img)
            loss = criterion(pred, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coef(pred, lbl).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                pred = model(img)

                val_loss += criterion(pred, lbl).item()
                val_dice += dice_coef(pred, lbl).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        if val_dice > max_val_dice:
            max_val_dice = val_dice
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}: New best Dice {val_dice:.4f} saved.")

        print(f"Epoch {epoch+1}/{epochs} | lr={optimizer.param_groups[0]['lr']:.6f} | "
              f"T_Loss {train_loss:.4f} T_Dice {train_dice:.4f} | "
              f"V_Loss {val_loss:.4f} V_Dice {val_dice:.4f}")

    print(f"train completed, best model save to: {best_model_path}")
    return best_model_path, os.path.join(split_dir, "test_split.csv")


def inference(model ,model_path, dataset_path, test_csv_path, save_dir):
    img_dir = os.path.join(dataset_path, "img")
    lbl_dir = os.path.join(dataset_path, "label")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "unet":
        model = UNet(n_classes=1,input_channels=1)
    elif model == "resunet":
        model = ResUNet(n_classes=1,input_channels=1)
    elif model == "nestedunet":
        model = NestedUNet(n_classes=1,input_channels=1)
    else:
        raise ValueError(f"Unsupported model type: {model}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"failed to load model from dict: {e}")
        return

    model.to(device)
    model.eval()

    to_tensor = T.ToTensor()
    if not os.path.exists(test_csv_path):
        print(f"Error: Test split file not found at {test_csv_path}")
        return

    df = pd.read_csv(test_csv_path)
    files = df['filename'].tolist()

    print(f"start inference， {len(files)} images in total...")

    dice_scores = []

    for name in tqdm(files, desc="Inference"):
        try:
            img_path = os.path.join(img_dir, name)
            lbl_path = os.path.join(lbl_dir, name)

            img = Image.open(img_path).convert("L")
            lbl = Image.open(lbl_path).convert("L")
        except Exception as e:
            print(f"Skip {name}, {e}")
            continue

        img_crop = img.crop((0, 0, 384, 384))
        lbl_crop = lbl.crop((0, 0, 384, 384))

        img_tensor = to_tensor(img_crop).unsqueeze(0).to(device)
        lbl_tensor = (to_tensor(lbl_crop) > 0.5).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_sigmoid = torch.sigmoid(pred)
            pred_mask = (pred_sigmoid > 0.5).float()

        intersection = (pred_mask * lbl_tensor).sum()
        union = pred_mask.sum() + lbl_tensor.sum() + 1e-6
        dice_value = float((2 * intersection / union).cpu().item())
        dice_scores.append(dice_value)

    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    print(f"\nInference completed! mean Dice: {avg_dice:.4f}")


if __name__ == "__main__":
    class_name = ["BK","ESCD","FECD","LC","FK"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="./segmentation_outputs")
    parser.add_argument("--model_type", type=str, default="unet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # DATA_ROOT = "/home/u22515072/a_eye_confocal/data/cornealconfocal/Lesion"
    # WORK_ROOT = "/home/u22515072/a_eye_confocal/segmentation1"
    # MODEL_TYPE = "nestedunet" 
    # SAVE_ROOT = os.path.join(WORK_ROOT, f"{MODEL_TYPE}")
    try:
        DATA_ROOT = args.data_root
        WORK_ROOT = args.output
        MODEL_TYPE = args.model_type
        SAVE_ROOT = os.path.join(WORK_ROOT, f"{MODEL_TYPE}")
        epochs = args.epochs
        batch_size = args.batch_size
        seed = args.seed
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        exit(1)
    print("==================== model : {} ====================".format(MODEL_TYPE))
    for cls in class_name:
        print("==================== Processing Class: {} ====================".format(cls))
        cls_path = os.path.join(DATA_ROOT, cls)
        save_path = os.path.join(SAVE_ROOT, cls)
        best_model, test_csv = train(
        model = MODEL_TYPE,
        dataset_path=cls_path, 
        batch_size=batch_size, 
        epochs=epochs, 
        seed=seed, 
        save_path=save_path
         )
        inference(
        model = MODEL_TYPE,
        model_path=best_model,
        dataset_path=cls_path,
        test_csv_path=test_csv,
        save_dir=save_path
    )
