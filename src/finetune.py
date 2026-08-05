"""
Fine-tune the CNNs on REAL calibration crops captured from the webcam.

Strategy (transfer learning to avoid overfitting on ~40 real samples):
  * start from the synthetic-trained weights (general eye/mouth knowledge)
  * mix in the real calibration crops, repeated REPEAT_REAL times
  * low learning rate + weight decay
  * a held-out 20% of real crops monitors generalization (best model saved)

Usage:
    python src/collect_calibration.py     # 1) capture your face (guided)
    python src/finetune.py                # 2) fine-tune + save real weights

Saved to models/eye_cnn_real.pth and models/mouth_cnn_real.pth, which the
detector automatically prefers over the synthetic-only models.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from train import (ImageFolderDS, build_dfs, TRAIN_TFM, VAL_TFM, SmallCNN,
                   train_one, evaluate, DEVICE, BATCH_SIZE, LR)

CAL_ROOT = "data/calibration"
REPEAT_REAL = 4          # repeats of each real crop inside the training set
FINETUNE_EPOCHS = 8
FINETUNE_LR = 2e-4
WEIGHT_DECAY = 1e-4


def make_cal_df(kind):
    """DataFrame of UNIQUE real calibration crops + their label.

    Filters by filename suffix so eye crops (_L/_R) never leak into the mouth
    dataset and vice versa (older captures stored eye crops in every folder)."""
    rows = []
    if kind == "eye":
        classes = [("eye_open", 0), ("eye_closed", 1)]
        is_target = lambda n: n.endswith(("_L.png", "_R.png"))
    else:
        classes = [("mouth_yawn", 1), ("mouth_no_yawn", 0)]
        is_target = lambda n: n.endswith(".png") and not n.endswith(("_L.png", "_R.png"))
    for folder, label in classes:
        path = os.path.join(CAL_ROOT, folder)
        if not os.path.isdir(path):
            continue
        for n in os.listdir(path):
            if is_target(n):
                rows.append({"path": os.path.join(path, n), "label": label})
    return pd.DataFrame(rows)


def finetune(kind, out_path, pretrained_path):
    synth_train, synth_val, names = build_dfs(kind)
    cal = make_cal_df(kind)
    print(f"=== FINETUNE {kind.upper()} | synthetic={len(synth_train)} "
          f"real unique={len(cal)} (x{REPEAT_REAL}) ===")
    if len(cal) == 0:
        print("no calibration data found - skipping")
        return

    # hold out 20% of REAL crops for monitoring generalization
    cal_train, cal_val = train_test_split(
        cal, test_size=0.2, stratify=cal["label"], random_state=42)
    cal_train = pd.concat([cal_train] * REPEAT_REAL, ignore_index=True)
    mixed = pd.concat([synth_train, cal_train], ignore_index=True).sample(frac=1, random_state=42)
    mixed.reset_index(drop=True, inplace=True)

    train_ds = ImageFolderDS(mixed, transform=TRAIN_TFM)
    val_ds = ImageFolderDS(cal_val, transform=VAL_TFM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # transfer learning: start from the synthetic model
    model = SmallCNN(2)
    model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for ep in range(1, FINETUNE_EPOCHS + 1):
        tl, ta = train_one(model, train_loader, opt, criterion)
        vl, va, _, _ = evaluate(model, val_loader)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:02d} | train {tl:.4f}/{ta:.3f} | real-val {vl:.4f}/{va:.3f} (best {best_acc:.3f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_path)

    _, _, preds, labels = evaluate(model, val_loader)
    print(f"best real-val accuracy {accuracy_score(labels, preds):.4f} "
          f"| precision {precision_score(labels, preds, zero_division=0):.3f} "
          f"| recall {recall_score(labels, preds, zero_division=0):.3f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    eye_base = "models/eye_cnn_mrl.pth" if os.path.exists("models/eye_cnn_mrl.pth") \
        else "models/eye_cnn.pth"
    print(f"eye base model: {eye_base}")
    finetune("eye", "models/eye_cnn_real.pth", eye_base)
    finetune("mouth", "models/mouth_cnn_real.pth", "models/mouth_cnn.pth")
    print("done")
