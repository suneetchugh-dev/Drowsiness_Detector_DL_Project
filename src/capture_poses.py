"""Capture LIVE labeled eye poses (open then closed) and report the CNN
P(open) distribution for each, plus pixel features, to tune live decisions.

    python src/capture_poses.py

Holds eyes OPEN for 3s, then CLOSED for 3s. Prints stats at the end.
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector
from collect_calibration import FACE_CASCADE, crop_regions

DURATION = 3.0
SAMPLE_EVERY = 0.15


def wait_for_face(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
        if len(faces):
            return
        cv2.putText(frame, "waiting for face ...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("poses", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit


def capture(stage_label, det, cap, key, samples):
    start = time.time()
    last = 0.0
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
                le, re, _ = crop_regions(frame, f)
                for crop in (le, re):
                    if crop.size:
                        p = det._cnn_probs(det.eye_net, crop)[0]
                        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        h = g.shape[0]
                        band = g[int(h * 0.15):int(h * 0.85), :]
                        samples[key].append({"p_open": float(p),
                                             "dark": float((band < 60).mean()),
                                             "contrast": float(band.std())})
        cv2.putText(frame, f"{stage_label} ... {int(DURATION - (time.time() - start))}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("poses", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def show(name, arr):
    a = np.array([s[name] for s in arr])
    print(f"  {name:9s} n={len(a)} mean={a.mean():.3f} p05={np.percentile(a,5):.3f} "
          f"p25={np.percentile(a,25):.3f} median={np.median(a):.3f} "
          f"p75={np.percentile(a,75):.3f} p95={np.percentile(a,95):.3f}")


def main():
    det = DrowsinessDetector(use_yolo=False)
    cap = cv2.VideoCapture(0)
    samples = {"open": [], "closed": []}
    print("Position your face in the camera, then it will start...")
    wait_for_face(cap)
    print("Hold eyes WIDE OPEN for 3s...")
    if not capture("eyes OPEN", det, cap, "open", samples):
        return
    print("Now hold eyes CLOSED for 3s...")
    if not capture("eyes CLOSED", det, cap, "closed", samples):
        return
    cap.release()
    cv2.destroyAllWindows()
    for key in ("open", "closed"):
        print(f"\n== live {key} (n={len(samples[key])}) ==")
        for name in ("p_open", "dark", "contrast"):
            show(name, samples[key])


if __name__ == "__main__":
    main()
