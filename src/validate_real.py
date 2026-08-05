"""
Capture FRESH real samples from the webcam and evaluate the fine-tuned
detector on them (honest accuracy on data NOT used for training).

Run:  python src/validate_real.py
Follow the on-screen prompts (~20 s).
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector
from collect_calibration import FACE_CASCADE, crop_regions

DURATION = 2.0
STAGES = [("eye_open", "eyes open"), ("eye_closed", "eyes closed"),
          ("mouth_yawn", "mouth wide open"), ("mouth_no_yawn", "mouth relaxed")]


def capture_stage(stage, key, det):
    cap = cv2.VideoCapture(0)
    start = None
    last = 0.0
    results = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
        if not len(faces):
            if start is not None:
                break
            cv2.putText(frame, "waiting for your face ...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.imshow("validation - follow the prompts", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        f = faces[0]
        fx, fy, fw, fh = f
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        if start is None:
            start = time.time()
        now = time.time()
        if now - start < DURATION:
            if now - last > 0.15:
                last = now
                le, re, mouth = crop_regions(frame, f)
                if key == "eye":
                    for crop in (le, re):
                        if crop.size:
                            pred, _ = det._predict(det.eye_net, crop, "eye")
                            results.append(pred)
                else:
                    if mouth.size:
                        pred, _ = det._predict(det.mouth_net, mouth, "mouth")
                        results.append(pred)
        cv2.putText(frame, f"{stage} ... {int(DURATION - (now - start))}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("validation - follow the prompts", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(results)


def main():
    det = DrowsinessDetector(use_yolo=False)
    expected = {"eye_open": 0, "eye_closed": 1, "mouth_yawn": 1, "mouth_no_yawn": 0}
    key = {"eye_open": "eye", "eye_closed": "eye",
           "mouth_yawn": "mouth", "mouth_no_yawn": "mouth"}

    print("Face the camera. Follow the 4 short prompts.")
    time.sleep(2)
    accs = {}
    for stage, label in STAGES:
        print(f"-> {label} ...")
        preds = capture_stage(stage, key[stage], det)
        if len(preds) == 0:
            print(f"   no face captured for {stage}")
            continue
        acc = float((preds == expected[stage]).mean())
        accs[stage] = (acc, len(preds))
        print(f"   {stage}: accuracy={acc:.2f} ({len(preds)} samples)")
        time.sleep(0.5)

    if accs:
        overall = np.mean([a for a, _ in accs.values()])
        print(f"\nOVERALL accuracy on fresh real data: {overall:.2f}")
    else:
        print("\nno data captured - make sure your face is visible and re-run")


if __name__ == "__main__":
    main()
