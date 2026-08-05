"""
Guided webcam calibration - collects REAL eye/mouth crops from the user's face.

Run:  python src/collect_calibration.py

The tool shows on-screen prompts with a countdown:
  1. EYES OPEN   (3 s) -> crops saved as 'open'
  2. EYES CLOSED (3 s) -> crops saved as 'closed'
  3. YAWN        (3 s) -> crops saved as 'yawn'
  4. MOUTH RELAXED (3s) -> crops saved as 'no_yawn'

Cropped regions are stored under data/calibration/<class>/ for later
fine-tuning of the CNNs on real data.
"""

import os
import time
import cv2

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

OUT = "data/calibration"
DURATION = 3.0          # seconds per stage
SAMPLE_EVERY = 0.12     # seconds between saved crops
MIN_FACE = 90


def crop_regions(frame, face):
    fx, fy, fw, fh = face
    y1, y2 = fy + int(fh * 0.15), fy + int(fh * 0.55)
    lx, lw = fx + int(fw * 0.05), int(fw * 0.42)
    rx = fx + int(fw * 0.53)
    left_eye = frame[y1:y2, lx:lx + lw]
    right_eye = frame[y1:y2, rx:rx + lw]
    my1, my2 = fy + int(fh * 0.55), fy + int(fh * 0.97)
    mx, mw = fx + int(fw * 0.22), int(fw * 0.56)
    mouth = frame[my1:my2, mx:mx + mw]
    return left_eye, right_eye, mouth


def collect(stage, instruction, count, face, glasses=False):
    """Stage: 'eye_open' | 'eye_closed' | 'mouth_yawn' | 'mouth_no_yawn'.

    With glasses=True the saved crops get a '_g' suffix, which marks them as
    "captured with glasses" so finetune.py can train the glasses-specific eye
    model without mixing appearances (and train_glasses.py can build a
    glasses-vs-no-glasses classifier)."""
    suffix = "_g" if glasses else ""
    folder = os.path.join(OUT, stage)
    os.makedirs(folder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("webcam not available")
    start = time.time()
    last = 0.0
    saved = 0
    while time.time() - start < DURATION:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE, MIN_FACE))
        f = faces[0] if len(faces) else face
        fx, fy, fw, fh = f
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        now = time.time()
        if f is not None and now - last > SAMPLE_EVERY:
            last = now
            le, re, mouth = crop_regions(frame, f)
            if stage.startswith("eye"):
                for side, crop in (("L", le), ("R", re)):
                    if crop.size:
                        cv2.imwrite(os.path.join(folder, f"real_{count:05d}_{side}{suffix}.png"), crop)
            else:  # mouth stages: only the mouth crop (keeps datasets unpolluted)
                if mouth.size:
                    cv2.imwrite(os.path.join(folder, f"real_{count:05d}{suffix}.png"), mouth)
            saved += 1
            count += 1
        left = int(DURATION - (time.time() - start))
        cv2.putText(frame, f"STAGE: {stage}  ({left}s)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, instruction, (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"saved: {saved}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("calibration - follow the prompts", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return count


def next_start_count():
    """Highest existing sample index + 1 so re-calibration appends, never
    overwrites earlier sessions (e.g. with vs without glasses)."""
    nums = []
    for folder in ("eye_open", "eye_closed", "mouth_yawn", "mouth_no_yawn"):
        d = os.path.join(OUT, folder)
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            if n.startswith("real_"):
                try:
                    nums.append(int(n.split("_")[1].split(".")[0]))
                except (IndexError, ValueError):
                    pass
    return max(nums) + 1 if nums else 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glasses", action="store_true",
                    help="tag crops with '_g' (capture while wearing glasses)")
    args = ap.parse_args()

    print("Make sure your face is fully visible, well-lit and centered.")
    if args.glasses:
        print("GLASSES MODE: keep your glasses ON for the whole run.")
    time.sleep(2)
    # warm-up / find the face
    cap = cv2.VideoCapture(0)
    face = None
    for _ in range(40):
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE, MIN_FACE))
        if len(faces):
            face = faces[0]
            break
    cap.release()
    if face is None:
        print("Could not detect your face. Move into frame and re-run.")
        return

    count = next_start_count()
    print(f"appending after sample index {count - 1}")
    stages = [
        ("eye_open", "keep eyes NATURALLY open - do NOT widen them"),
        ("eye_closed", "gently close both eyes, relax"),
        ("mouth_yawn", "open mouth WIDE like a big yawn"),
        ("mouth_no_yawn", "relax mouth, lips lightly together (no smile)"),
    ]
    for stage, instruction in stages:
        print(f"-> {stage} ...")
        count = collect(stage, instruction, count, face, glasses=args.glasses)
        time.sleep(0.6)
    print("calibration complete")


if __name__ == "__main__":
    main()
