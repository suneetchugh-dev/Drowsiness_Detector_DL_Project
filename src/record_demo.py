"""Record a short raw webcam video for offline testing with detect_video.

    python src/record_demo.py           # records 20s to demo/raw_demo.mp4

Follow the on-screen prompts: eyes open -> eyes closed -> yawn -> relaxed.
Only records after a face is detected; press 'q' to stop early.
"""

import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_calibration import FACE_CASCADE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo")
DURATION = 20.0


def main():
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("webcam not available")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path = os.path.join(OUT, "raw_demo.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"recording {DURATION:.0f}s -> {path}")
    start = time.time()
    while time.time() - start < DURATION:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
        if len(faces):
            f = faces[0]
            cv2.rectangle(frame, (f[0], f[1]), (f[0] + f[2], f[1] + f[3]),
                          (0, 255, 0), 2)
        left = int(DURATION - (time.time() - start))
        cv2.putText(frame, f"recording ... {left}s (q=quit)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(frame)
        cv2.imshow("record demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("done ->", path)


if __name__ == "__main__":
    main()
