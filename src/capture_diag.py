"""Diagnose glasses failure: save live eye (L/R) and mouth crops per pose and
report per-eye-side P(open) and mouth P(yawn)+contrast.

Stages (5s each, camera shows a countdown):
    1) eyes OPEN (natural/relaxed)
    2) eyes CLOSED
    3) mouth NEUTRAL (relaxed, teeth not visible)
    4) YAWN (big yawn)

Crops are saved to _live_crops/diag/{open,closed,neutral,yawn}/ for visual
inspection. Press 'q' to abort.

    python src/capture_diag.py
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector
from collect_calibration import FACE_CASCADE, crop_regions

DURATION = 5.0
SAMPLE_EVERY = 0.15
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "_live_crops", "diag")


def capture(stage_label, det, cap, key):
    start = time.time()
    last = 0.0
    frames = 0
    while time.time() - start < DURATION:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
        if len(faces):
            f = faces[0]
            fx, fy, fw, fh = f
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            now = time.time()
            if now - last > SAMPLE_EVERY:
                last = now
                le, re, mouth = crop_regions(frame, f)
                tag = f"{key}_{frames:03d}"
                if le.size:
                    cv2.imwrite(os.path.join(OUT, key, f"L_{tag}.png"), le)
                if re.size:
                    cv2.imwrite(os.path.join(OUT, key, f"R_{tag}.png"), re)
                if mouth.size:
                    cv2.imwrite(os.path.join(OUT, key, f"M_{tag}.png"), mouth)
                frames += 1
        cv2.putText(frame, f"{stage_label} ... {int(DURATION - (time.time() - start))}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("diag", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def report():
    det = DrowsinessDetector(use_yolo=False)
    print("\n--- eye P(open) per side ---")
    for key in ("open", "closed"):
        for side in ("L", "R"):
            ps = []
            d = os.path.join(OUT, key)
            if not os.path.isdir(d):
                continue
            for n in sorted(os.listdir(d)):
                if not n.startswith(side + "_"):
                    continue
                img = cv2.imread(os.path.join(d, n))
                ps.append(det._cnn_probs(det.eye_net, img)[0])
            ps = np.array(ps)
            if len(ps):
                print(f"{key:7s} {side}: n={len(ps):3d} p05={np.percentile(ps,5):.3f} "
                      f"p25={np.percentile(ps,25):.3f} median={np.median(ps):.3f} "
                      f"p75={np.percentile(ps,75):.3f}")
    print("\n--- mouth P(yawn) + contrast ---")
    for key in ("neutral", "yawn"):
        ps, cs = [], []
        d = os.path.join(OUT, key)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if not n.startswith("M_"):
                continue
            img = cv2.imread(os.path.join(d, n))
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h = g.shape[0]
            band = g[int(h * 0.25):int(h * 0.75), :]
            ps.append(det._cnn_probs(det.mouth_net, img)[1])
            cs.append(float(band.std()))
        ps, cs = np.array(ps), np.array(cs)
        if len(ps):
            print(f"{key:7s}: n={len(ps):3d} p_yawn median={np.median(ps):.3f} "
                  f"(p25={np.percentile(ps,25):.3f}, p75={np.percentile(ps,75):.3f}) "
                  f"| contrast median={np.median(cs):.1f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    for key in ("open", "closed", "neutral", "yawn"):
        os.makedirs(os.path.join(OUT, key), exist_ok=True)
    cap = cv2.VideoCapture(0)
    print("STAGE 1/4: relax, look at the camera with eyes OPEN (natural, not squinting)")
    if not capture("eyes OPEN (natural)", None, cap, "open"):
        return
    print("STAGE 2/4: now close both eyes")
    if not capture("eyes CLOSED", None, cap, "closed"):
        return
    print("STAGE 3/4: neutral mouth - lips relaxed, teeth hidden")
    if not capture("mouth NEUTRAL", None, cap, "neutral"):
        return
    print("STAGE 4/4: big YAWN")
    if not capture("YAWN", None, cap, "yawn"):
        return
    cap.release()
    cv2.destroyAllWindows()
    report()
    print(f"\ncrops saved -> {OUT}")


if __name__ == "__main__":
    main()
