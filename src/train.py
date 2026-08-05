"""
Train two small CNNs (PyTorch) on the synthetic dataset:
  - EyeCNN   : open  vs closed
  - MouthCNN : yawn vs no_yawn

Uses torch / torchvision / numpy / pandas / scikit-learn / matplotlib only.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------

class ImageFolderDS(Dataset):
    """Lazily maps (path -> image) without loading everything into RAM."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2_imread(row["path"])
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(row["label"], dtype=torch.long)


def cv2_imread(path):
    import cv2
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def make_df(folders, labels, val_frac=0.2, seed=42):
    rows = []
    for folder, label in zip(folders, labels):
        for name in os.listdir(folder):
            if name.endswith(".png"):
                rows.append({"path": os.path.join(folder, name), "label": label})
    df = pd.DataFrame(rows)
    rng = np.random.RandomState(seed)
    split_flag = rng.rand(len(df))
    df["split"] = np.where(split_flag < val_frac, "val", "train")
    return df[df["split"] == "train"], df[df["split"] == "val"]


def build_dfs(kind):
    if kind == "eye":
        folders = ["data/eyes/open", "data/eyes/closed"]
        labels = [0, 1]
        names = ["open", "closed"]
    else:
        folders = ["data/mouth/yawn", "data/mouth/no_yawn"]
        labels = [0, 1]
        names = ["yawn", "no_yawn"]
    train, val = make_df(folders, labels)
    return train, val, names


TRAIN_TFM = T.Compose([
    T.ToTensor(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomRotation(8),
    T.RandomAffine(0, translate=(0.05, 0.05)),
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
VAL_TFM = T.Compose([
    T.ToTensor(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """Compact CNN: 2 conv blocks + dropout + 2 fully-connected layers."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 6 * 6, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------------

def train_one(model, loader, opt, criterion):
    model.train()
    total, correct, running = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        running += loss.item() * len(xb)
        total += len(xb)
        correct += (out.argmax(1) == yb).sum().item()
    return running / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels, total, correct, running = [], [], 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        running += nn.functional.cross_entropy(out, yb).item() * len(xb)
        total += len(xb)
        correct += (out.argmax(1) == yb).sum().item()
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(yb.cpu().numpy())
    return (running / total, correct / total, np.array(preds), np.array(labels))


def fit(kind, out_path, plot_path):
    train_df, val_df, names = build_dfs(kind)
    train_ds = ImageFolderDS(train_df, transform=TRAIN_TFM)
    val_ds = ImageFolderDS(val_df, transform=VAL_TFM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"\n=== {kind.upper()} | train={len(train_df)} val={len(val_df)} ===")

    model = SmallCNN(2).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    hist = {"tl": [], "ta": [], "vl": [], "va": []}
    for ep in range(1, EPOCHS + 1):
        tl, ta = train_one(model, train_loader, opt, criterion)
        vl, va, _, _ = evaluate(model, val_loader)
        sched.step()
        hist["tl"].append(tl); hist["ta"].append(ta)
        hist["vl"].append(vl); hist["va"].append(va)
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:02d} | train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va:.3f}")

    torch.save(model.state_dict(), out_path)

    # history plot
    plt.figure(figsize=(7, 3))
    plt.plot(hist["tl"], label="train loss")
    plt.plot(hist["vl"], label="val loss")
    plt.plot(hist["ta"], label="train acc")
    plt.plot(hist["va"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("value"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"{kind} CNN training"); plt.tight_layout()
    plt.savefig(plot_path, dpi=110)
    plt.close()

    # evaluation on the val split
    _, _, preds, labels = evaluate(model, val_loader)
    report = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    cm = confusion_matrix(labels, preds)
    print(f"val accuracy {report['accuracy']:.4f} | precision {report['precision']:.4f} | "
          f"recall {report['recall']:.4f} | f1 {report['f1']:.4f}")
    print("confusion matrix (rows=true, cols=pred):\n", cm)

    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{kind} confusion matrix"); plt.tight_layout()
    plt.savefig(plot_path.replace("_history", "_cm"))
    plt.close()
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["eye", "mouth", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    EPOCHS = args.epochs

    results = {}
    if args.kind in ("eye", "all"):
        results["eye"] = fit("eye", "models/eye_cnn.pth", "models/eye_history.png")
    if args.kind in ("mouth", "all"):
        results["mouth"] = fit("mouth", "models/mouth_cnn.pth", "models/mouth_history.png")
    print("\nFINAL SUMMARY")
    for k, r in results.items():
        print(f"{k}: acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
              f"rec={r['recall']:.3f} f1={r['f1']:.3f}")
