"""
Train a glasses-vs-no-glasses classifier on the REAL calibration eye crops.

Labels come from the '_g' tag that collect_calibration adds when it detects
glasses (see src/collect_calibration.py / src/finetune.py). Both 'eye_open'
and 'eye_closed' crops are used, so the classifier must recognise the frame
regardless of open/closed eyes.

Run:  python src/train_glasses.py
Saves: models/glasses_cnn.pth   (SmallCNN, 2 classes: 0=no glasses, 1=glasses)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from finetune import make_cal_df, FT_TRAIN_TFM, FT_VAL_TFM
from train import ImageFolderDS, SmallCNN, train_one, evaluate, DEVICE, BATCH_SIZE

GLASSES_EPOCHS = 15
GLASSES_LR = 2e-4
GLASSES_OUT = "models/glasses_cnn.pth"


def main():
    cal = make_cal_df("eye")  # eye_open + eye_closed crops, with 'glasses' column
    cal["label"] = cal["glasses"]  # classifier label: 1=glasses, 0=no glasses
    cal = cal.reset_index(drop=True)
    print(f"glasses crops: total={len(cal)} "
          f"no-glasses={int((cal['label'] == 0).sum())} glasses={int(cal['label'].sum())}")

    train, val = train_test_split(cal, test_size=0.2, stratify=cal["label"],
                                  random_state=42)
    print(f"train={len(train)} (glasses {int(train['label'].sum())}) | "
          f"val={len(val)} (glasses {int(val['label'].sum())})")

    # oversample the minority class inside the training set for balance
    neg = train[train["label"] == 0]
    pos = train[train["label"] == 1]
    times = int(np.ceil(len(neg) / max(1, len(pos))))
    mixed = pd.concat([neg, pd.concat([pos] * times, ignore_index=True)],
                      ignore_index=True).sample(frac=1, random_state=42)

    train_ds = ImageFolderDS(mixed, transform=FT_TRAIN_TFM)
    val_ds = ImageFolderDS(val, transform=FT_VAL_TFM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # transfer-learning start from the synthetic eye model (eye-specific features)
    model = SmallCNN(2)
    base = "models/eye_cnn_mrl.pth" if os.path.exists("models/eye_cnn_mrl.pth") \
        else "models/eye_cnn.pth"
    if os.path.exists(base):
        model.load_state_dict(torch.load(base, map_location="cpu"))
        print("init from", base)
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=GLASSES_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for ep in range(1, GLASSES_EPOCHS + 1):
        tl, ta = train_one(model, train_loader, opt, criterion)
        vl, va, _, _ = evaluate(model, val_loader)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:02d} | train {tl:.4f}/{ta:.3f} | val {vl:.4f}/{va:.3f} "
              f"(best {best_acc:.3f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), GLASSES_OUT)

    _, _, preds, labels = evaluate(model, val_loader)
    print(f"best val accuracy {accuracy_score(labels, preds):.4f} "
          f"| precision {precision_score(labels, preds, zero_division=0):.3f} "
          f"| recall {recall_score(labels, preds, zero_division=0):.3f}")
    print(f"saved -> {GLASSES_OUT}")


if __name__ == "__main__":
    main()
