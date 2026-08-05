"""Measure per-model CNN probabilities on ALL calibration crops.

Runs the real (fine-tuned) EyeCNN/MouthCNN over every calibration image and
prints P(open) / P(yawn) distributions per class plus best decision thresholds.
This is the ground-truth of what each model achieves on real crops.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "calibration")


def load(kind):
    d = os.path.join(BASE, kind)
    imgs = []
    for fn in os.listdir(d):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            imgs.append(cv2.imread(os.path.join(d, fn)))
    return [i for i in imgs if i is not None]


def best_threshold(a, b):
    vals = np.concatenate([a, b])
    lo, hi = vals.min(), vals.max()
    best_t, best_acc = lo, 0.0
    for t in np.linspace(lo, hi, 200):
        acc = (np.mean(a >= t) + np.mean(b < t)) / 2.0
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def report(probs_a, probs_b, label_a, label_b):
    print(f"\n== {label_a} (n={len(probs_a)}) vs {label_b} (n={len(probs_b)}) ==")
    print(f"  {label_a}: mean={probs_a.mean():.3f} p05={np.percentile(probs_a,5):.3f} "
          f"p25={np.percentile(probs_a,25):.3f} median={np.median(probs_a):.3f} "
          f"p75={np.percentile(probs_a,75):.3f} p95={np.percentile(probs_a,95):.3f}")
    print(f"  {label_b}: mean={probs_b.mean():.3f} p05={np.percentile(probs_b,5):.3f} "
          f"p25={np.percentile(probs_b,25):.3f} median={np.median(probs_b):.3f} "
          f"p75={np.percentile(probs_b,75):.3f} p95={np.percentile(probs_b,95):.3f}")
    t, acc = best_threshold(probs_a, probs_b)
    print(f"  best threshold (prob>{t:.3f} -> {label_a}): accuracy={acc:.3f}")


def main():
    det = DrowsinessDetector(use_yolo=False)
    print("using real models:", det.using_real_model)

    eye_open = [det._cnn_probs(det.eye_net, i)[0] for i in load("eye_open")]
    eye_closed = [det._cnn_probs(det.eye_net, i)[0] for i in load("eye_closed")]
    report(np.array(eye_open), np.array(eye_closed), "eye_open", "eye_closed")

    mouth_yawn = [det._cnn_probs(det.mouth_net, i)[1] for i in load("mouth_yawn")]
    mouth_no_yawn = [det._cnn_probs(det.mouth_net, i)[1] for i in load("mouth_no_yawn")]
    report(np.array(mouth_yawn), np.array(mouth_no_yawn), "mouth_yawn", "mouth_no_yawn")


if __name__ == "__main__":
    main()
