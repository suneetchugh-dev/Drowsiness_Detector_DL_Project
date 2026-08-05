"""Capture a few live crops to files so we can visually compare with the
calibration crops (check for crop misalignment / lighting differences).

Press q or wait: captures left/right eye when a face appears.
"""

import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_calibration import FACE_CASCADE, crop_regions

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_live_crops")
os.makedirs(OUT, exist_ok=True)


def save(fname, img):
    cv2.imwrite(os.path.join(OUT, fname), img)
    print("saved", fname, img.shape)


def main():
    cap = cv2.VideoCapture(0)
    print("face the camera...")
    saved = set()
    while len(saved) < 3:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
        if len(faces):
            f = faces[0]
            fx, fy, fw, fh = f
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            le, re, mouth = crop_regions(frame, f)
            if le.size and "le" not in saved:
                save("live_le.png", le); saved.add("le")
            if re.size and "re" not in saved:
                save("live_re.png", re); saved.add("re")
            if mouth.size and "mouth" not in saved:
                save("live_mouth.png", mouth); saved.add("mouth")
        cv2.putText(frame, f"saved {len(saved)}/3", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
