from sklearn.metrics import roc_curve, auc
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

class cornealconfocal(Dataset):
    def __init__(self, file_list, img_dir, lbl_dir):
        self.file_list = file_list
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        try:
            img = Image.open(os.path.join(self.img_dir, name)).convert("L")
            lbl = Image.open(os.path.join(self.lbl_dir, name)).convert("L")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            new_idx = (idx + 1) % len(self.file_list)
            return self.__getitem__(new_idx)

        img = img.crop((0, 0, 384, 384))
        lbl = lbl.crop((0, 0, 384, 384))

        img = self.to_tensor(img)
        lbl = self.to_tensor(lbl)
        lbl = (lbl > 0.5).float()

        return img, lbl
