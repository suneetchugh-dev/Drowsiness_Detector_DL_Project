"""
Train the eye CNN on the MRL Eye Dataset (closed vs open), then it becomes
the base model for fine-tuning on the user's real calibration crops.

MRL filename format: s<subject>_<seq>_<state>_<reflections>_<lighting>_<sensor>_<...>.png
    state = 0 (closed) or 1 (open)

Performance: images are read + resized ONCE and cached as a uint8 ndarray
(locally, outside OneDrive), then random transforms run on the cached tensor
each epoch - this avoids the cv2.imread + ToTensor bottleneck (~25x faster).

Usage:
    python src/train_mrl.py
    -> saves models/eye_cnn_mrl.pth (used by finetune.py when present)
"""

import os
import random
import tempfile
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from train import SmallCNN, DEVICE, IMG_SIZE, BATCH_SIZE, LR

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MRL_ROOT = "data/mrl/mrlEyes_2018_01"
N_PER_CLASS = 18000
VAL_FRAC = 0.15
EPOCHS = 15

CACHE_DIR = os.path.join(tempfile.gettempdir(), "opencode", "mrl_cache")
CACHE_TRAIN = os.path.join(CACHE_DIR, "train.npz")
CACHE_VAL = os.path.join(CACHE_DIR, "val.npz")

AUG_TFM = T.Compose([
    T.RandomRotation(10),
    T.RandomAffine(0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
    T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.35)], p=0.8),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
VAL_TFM = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def _parse(path):
    return 0 if int(os.path.basename(path).split("_")[2]) == 0 else 1


def build_mrl_df(n_per_class=18000, val_frac=0.15):
    rows = []
    for sub in sorted(os.listdir(MRL_ROOT)):
        d = os.path.join(MRL_ROOT, sub)
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            if n.endswith(".png"):
                rows.append({"path": os.path.join(d, n)})
    df = pd.DataFrame(rows)
    df["label"] = df["path"].map(_parse)
    df = df[df.label.isin((0, 1))]
    print(f"MRL total: {len(df)} (closed={int((df.label==0).sum())}, "
          f"open={int((df.label==1).sum())})")

    per = {0: df[df.label == 0].sample(n_per_class, random_state=42),
           1: df[df.label == 1].sample(min(n_per_class, int((df.label==1).sum())),
                                       random_state=42)}
    df = pd.concat([per[0], per[1]], ignore_index=True)
    train, val = train_test_split(df, test_size=val_frac, stratify=df["label"],
                                  random_state=42)
    print(f"sampled: train={len(train)} val={len(val)}")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _to_array(df):
    import cv2
    arr = np.empty((len(df), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(df["path"]):
        img = cv2.imread(p)
        if img is None:
            arr[i] = 128
        else:
            arr[i] = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return arr, df["label"].to_numpy(np.int64)


def _to_tensor(arr, labels):
    return (torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0),
            torch.from_numpy(labels))


def get_cached(df, cache_path):
    """Read+resize once, cache as .npz, then return float32 tensors."""
    if os.path.exists(cache_path):
        print(f"loading cache: {cache_path}")
        z = np.load(cache_path)
        return _to_tensor(z["arr"], z["labels"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"preprocessing {len(df)} images once ...")
    t0 = time.time()
    arr, labels = _to_array(df)
    np.savez_compressed(cache_path, arr=arr, labels=labels)
    print(f"  cached in {time.time()-t0:.0f}s -> {cache_path}")
    return _to_tensor(arr, labels)


def make_loader(x, y, transform, batch_size=BATCH_SIZE):
    """Yield (transformed_batch, labels) without a DataLoader."""
    n = len(x)
    while True:
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            sel = idx[i:i + batch_size]
            b = x[sel].clone()
            if transform is not None:
                b = transform(b)
            yield b.to(DEVICE), y[sel].to(DEVICE)


def run_epoch(model, x, y, transform, opt, criterion, train=True):
    model.train() if train else model.eval()
    loader = make_loader(x, y, transform)
    total, correct, running, n = 0, 0, 0.0, len(x)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.set_grad_enabled(train):
        for _ in range(n_batches):
            xb, yb = next(loader)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            running += loss.item() * len(xb)
            total += len(xb)
            correct += (out.argmax(1) == yb).sum().item()
    return running / total, correct / total


def main():
    train_df, val_df = build_mrl_df(N_PER_CLASS, VAL_FRAC)
    x_train, y_train = get_cached(train_df, CACHE_TRAIN)
    x_val, y_val = get_cached(val_df, CACHE_VAL)
    print(f"train tensors: {x_train.shape} | val: {x_val.shape}")

    model = SmallCNN(2).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, x_train, y_train, AUG_TFM, opt, criterion, train=True)
        vl, va = run_epoch(model, x_val, y_val, VAL_TFM, opt, criterion, train=False)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:02d} | train {tl:.4f}/{ta:.3f} | val {vl:.4f}/{va:.3f} "
              f"(best {best_acc:.3f}) | {time.time()-t0:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), "models/eye_cnn_mrl.pth")

    model.eval()
    with torch.no_grad():
        out = model(VAL_TFM(x_val).to(DEVICE))
        preds = out.argmax(1).cpu().numpy()
        labels = y_val.numpy()
    print(f"\nMRL val: acc {accuracy_score(labels, preds):.4f} "
          f"| precision {precision_score(labels, preds):.3f} "
          f"| recall {recall_score(labels, preds):.3f} "
          f"| f1 {f1_score(labels, preds):.3f}")
    print("saved -> models/eye_cnn_mrl.pth")


if __name__ == "__main__":
    main()
