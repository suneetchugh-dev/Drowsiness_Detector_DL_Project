"""Measure pixel-heuristic features over the calibration crops per class.

Prints dark-fraction / contrast distributions for eye_open vs eye_closed and
mouth_yawn vs mouth_no_yawn, plus the best separating thresholds, so the
detector thresholds can be tuned on real data.
"""

import os
import numpy as np
import cv2

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "calibration")


def load(kind):
    d = os.path.join(BASE, kind)
    imgs = []
    for fn in os.listdir(d):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            imgs.append(cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE))
    return [i for i in imgs if i is not None]


def eye_feats(img):
    h = img.shape[0]
    band = img[int(h * 0.30):int(h * 0.85), :]
    dark = float((band < 60).mean())
    contrast = float(band.std())
    return dark, contrast


def mouth_feats(img):
    h = img.shape[0]
    band = img[int(h * 0.15):int(h * 0.85), :]
    dark = float((band < 70).mean())
    contrast = float(band.std())
    return dark, contrast


def best_threshold(a, b, name):
    vals = np.array([v for v, _ in a + b])
    lo, hi = vals.min(), vals.max()
    best_t, best_acc = lo, 0.0
    for t in np.linspace(lo, hi, 200):
        acc = (np.mean([v >= t for v, _ in a]) +
               np.mean([v < t for v, _ in b])) / 2.0
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def report(open_imgs, closed_imgs, feats, label_a, label_b):
    a = [feats(i) for i in open_imgs]
    b = [feats(i) for i in closed_imgs]
    print(f"\n== {label_a} (n={len(a)}) vs {label_b} (n={len(b)}) ==")
    for name in ("dark", "contrast"):
        va = np.array([v for v, _ in a]) if name == "dark" else np.array([c for _, c in a])
        vb = np.array([v for v, _ in b]) if name == "dark" else np.array([c for _, c in b])
        t, acc = best_threshold(
            list(zip(va, va)), list(zip(vb, vb)), name)
        print(f"{name:10s} {label_a}: mean={va.mean():.3f} p05={np.percentile(va,5):.3f} "
              f"p95={np.percentile(va,95):.3f} | {label_b}: mean={vb.mean():.3f} "
              f"p05={np.percentile(vb,5):.3f} p95={np.percentile(vb,95):.3f} "
              f"| best_thr={t:.3f} acc={acc:.2f}")


def main():
    eye_open = load("eye_open")
    eye_closed = load("eye_closed")
    mouth_yawn = load("mouth_yawn")
    mouth_no_yawn = load("mouth_no_yawn")
    report(eye_open, eye_closed, eye_feats, "eye_open", "eye_closed")
    report(mouth_yawn, mouth_no_yawn, mouth_feats, "mouth_yawn", "mouth_no_yawn")


if __name__ == "__main__":
    main()
