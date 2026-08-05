"""
Debug one stage: capture crops live and print CNN prob / heuristic / fused score
per sample, so we can diagnose classification failures.

    python src/debug_stage.py mouth_yawn     (or eye_open / eye_closed / mouth_no_yawn)

Hold the pose for ~4 s.
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector
from collect_calibration import FACE_CASCADE, crop_regions

DURATION = 4.0
EXPECT = {"eye_open": (0, "eye"), "eye_closed": (1, "eye"),
          "mouth_yawn": (1, "mouth"), "mouth_no_yawn": (0, "mouth")}


def main(stage):
    if stage not in EXPECT:
        print("usage: debug_stage.py <eye_open|eye_closed|mouth_yawn|mouth_no_yawn>")
        return
    expected, key = EXPECT[stage]
    det = DrowsinessDetector(use_yolo=False)
    cap = cv2.VideoCapture(0)
    start = None
    last = 0.0
    print(f"hold pose '{stage}' ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
        if not len(faces):
            if start is not None:
                break
            cv2.putText(frame, "waiting for face ...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.imshow("debug", frame)
            cv2.waitKey(1)
            continue
        f = faces[0]
        fx, fy, fw, fh = f
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        if start is None:
            start = time.time()
        now = time.time()
        if now - start < DURATION and now - last > 0.15:
            last = now
            le, re, mouth = crop_regions(frame, f)
            crops = (le, re) if key == "eye" else (mouth,)
            for crop in crops:
                if crop.size == 0:
                    continue
                net = det.eye_net if key == "eye" else det.mouth_net
                prob = det._cnn_probs(net, crop)
                pred, fused = det._predict(net, crop, key)
                if key == "eye":
                    cnn_open = float(prob[0])
                    mark = "OK " if pred == expected else "XX "
                    print(f"{mark}cnn_open={cnn_open:.2f} fused={fused:.2f} "
                          f"-> {'OPEN' if pred == 0 else 'CLOSED'}")
                else:
                    cnn_yawn = float(prob[1])
                    mark = "OK " if pred == expected else "XX "
                    print(f"{mark}cnn_yawn={cnn_yawn:.2f} fused={fused:.2f} "
                          f"-> {'YAWN' if pred == 1 else 'NO_YAWN'}")
        cv2.putText(frame, f"{stage} ... {int(DURATION - (now - start))}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("debug", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eye_open")
