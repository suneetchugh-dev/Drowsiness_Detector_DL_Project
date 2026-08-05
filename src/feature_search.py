"""Brute-force search for pixel features that separate the calibration classes.

For the eye: tries various horizontal bands and dark/bright thresholds to find
the best open-vs-closed discriminator.
For the mouth: same for yawn-vs-no_yawn.

Prints the best few features so we can pick robust ones.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "calibration")


def load(kind):
    d = os.path.join(BASE, kind)
    imgs = []
    for fn in os.listdir(d):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            imgs.append(cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE))
    return [i for i in imgs if i is not None]


def feats(img):
    """Return dict of candidate features for a crop."""
    h, w = img.shape
    out = {}
    for a, b in [(0.15, 0.85), (0.25, 0.75), (0.30, 0.85), (0.35, 0.70),
                 (0.40, 0.65), (0.45, 0.60)]:
        band = img[int(h * a):int(h * b), :]
        if band.size == 0:
            continue
        out[f"dark60_{a:.2f}-{b:.2f}"] = float((band < 60).mean())
        out[f"dark70_{a:.2f}-{b:.2f}"] = float((band < 70).mean())
        out[f"bright180_{a:.2f}-{b:.2f}"] = float((band > 180).mean())
        out[f"contrast_{a:.2f}-{b:.2f}"] = float(band.std())
    return out


def evaluate(imgs_a, imgs_b, label_a):
    """For each feature, find best threshold + accuracy; return sorted list."""
    fa = [feats(i) for i in imgs_a]
    fb = [feats(i) for i in imgs_b]
    keys = fa[0].keys()
    results = []
    for k in keys:
        va = np.array([d[k] for d in fa])
        vb = np.array([d[k] for d in fb])
        lo, hi = min(va.min(), vb.min()), max(va.max(), vb.max())
        best_t, best_acc = lo, 0.0
        for t in np.linspace(lo, hi, 300):
            acc = (np.mean(va >= t) + np.mean(vb < t)) / 2.0
            if acc > best_acc:
                best_acc, best_t = acc, t
        results.append((best_acc, best_t, k, va.mean(), vb.mean()))
    results.sort(reverse=True)
    print(f"\n== {label_a}: best features ==")
    for acc, t, k, ma, mb in results[:8]:
        print(f"  acc={acc:.3f} thr={t:.3f} {k:28s} classA_mean={ma:.3f} classB_mean={mb:.3f}")


def main():
    eye_open = load("eye_open")
    eye_closed = load("eye_closed")
    mouth_yawn = load("mouth_yawn")
    mouth_no_yawn = load("mouth_no_yawn")
    evaluate(eye_open, eye_closed, "eye_open vs eye_closed")
    evaluate(mouth_yawn, mouth_no_yawn, "mouth_yawn vs mouth_no_yawn")


if __name__ == "__main__":
    main()
