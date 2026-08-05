"""
Driver Drowsiness & Distraction Detector — inference engine.

Shared by all three input modes:
  - live webcam
  - uploaded image file
  - uploaded video file

Components (all inside the allowed library set):
  * OpenCV Haar cascades  -> face / eye region localisation
  * PyTorch CNNs          -> eye open/closed + mouth yawn/no-yawn classification
  * YOLOv8 (ultralytics)  -> driver (person) presence verification
  * numpy / cv2           -> temporal smoothing, overlays, drawing
"""

import os
import time
import numpy as np
import cv2
import torch

try:
    import winsound
except ImportError:
    winsound = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

EYE_WEIGHTS = os.path.join(MODEL_DIR, "eye_cnn.pth")
MOUTH_WEIGHTS = os.path.join(MODEL_DIR, "mouth_cnn.pth")
EYE_WEIGHTS_REAL = os.path.join(MODEL_DIR, "eye_cnn_real.pth")
MOUTH_WEIGHTS_REAL = os.path.join(MODEL_DIR, "mouth_cnn_real.pth")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov8n.pt")

IMG_SIZE = 48

# ---- tuned alert thresholds -------------------------------------------------
CLOSED_FRAMES_ALERT = 3     # consecutive frames both eyes closed -> DROWSY
YAWN_FRAMES_ALERT = 2       # consecutive frames yawning           -> YAWNING
SMOOTH_WIN = 5              # majority-vote window for eye/mouth states
MOUTH_OPEN_CONTRAST = 18.0  # band std below which the mouth is considered CLOSED
                            # (cal crops: yawn median 20.4, no_yawn median 16.1)


class SmallCNN(torch.nn.Module):
    """Must match the architecture used in training (src/train.py)."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128 * 6 * 6, 128), torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DrowsinessDetector:
    """One object handles detection for webcam, image and video modes."""

    def __init__(self, device=None, use_yolo=True, yolo_interval=3,
                 closed_frames=CLOSED_FRAMES_ALERT, yawn_frames=YAWN_FRAMES_ALERT,
                 smooth_win=SMOOTH_WIN, cnn_weight=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_yolo = use_yolo
        self.yolo_interval = yolo_interval
        self.closed_frames = closed_frames
        self.yawn_frames = yawn_frames
        self.smooth_win = smooth_win
        self.beep_alerts = True
        self._prev_drowsy = False
        # per-image mean/std normalisation makes the CNNs invariant to webcam
        # auto-exposure; the fine-tuned models are trained with this transform.
        self.standardize_crops = True

        # 1) Haar cascades shipped with opencv-python
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_eye.xml"))

        # 2) trained CNNs (real fine-tuned weights take priority over synthetic)
        self.eye_net = self._load_cnn(EYE_WEIGHTS_REAL if os.path.exists(EYE_WEIGHTS_REAL)
                                      else EYE_WEIGHTS)
        self.mouth_net = self._load_cnn(MOUTH_WEIGHTS_REAL if os.path.exists(MOUTH_WEIGHTS_REAL)
                                        else MOUTH_WEIGHTS)
        self.using_real_model = os.path.exists(EYE_WEIGHTS_REAL)
        # Fusion: each decision = cnn_weight * CNN + (1-cnn_weight) * heuristic.
        # Thresholds/weights were tuned on the user's real calibration crops:
        #   eye    CNN alone separates open/closed at ~0.88, pixel features only ~0.72
        #   mouth  CNN ~0.83, contrast heuristic ~0.72
        # so the fine-tuned CNN dominates, with the heuristic as a guard.
        if cnn_weight is not None:
            self.eye_cnn_weight = self.mouth_cnn_weight = cnn_weight
        elif self.using_real_model:
            self.eye_cnn_weight = 0.60
            self.mouth_cnn_weight = 0.70
        else:
            # synthetic-only CNN has little transfer to real faces -> heuristic leads
            self.eye_cnn_weight = self.mouth_cnn_weight = 0.25

        # 3) YOLOv8 person detector (optional)
        self.yolo = None
        if use_yolo and os.path.exists(YOLO_WEIGHTS):
            from ultralytics import YOLO
            self.yolo = YOLO(YOLO_WEIGHTS)
        elif use_yolo:
            print("[detector] yolov8n.pt not found - YOLO disabled")

        # frame-to-frame state
        self.frame_idx = 0
        self._closed_run = 0
        self._yawn_run = 0
        self._eye_hist = []
        self._mouth_hist = []
        self._last_face = None
        self._dbg_frame = None
        self._last_mouth_contrast = None

    # ------------------------------------------------------------------ setup
    def _load_cnn(self, path):
        net = SmallCNN(2)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path, map_location=self.device))
            net.to(self.device).eval()
        else:
            print(f"[detector] missing weights {path} - CNN not loaded")
        return net

    def _cnn_probs(self, net, crop_bgr):
        """Raw softmax probabilities [p_open, p_closed] / [p_no_yawn, p_yawn].

        Crops are per-image standardised (subtract mean, divide by std) before
        the CNN. This makes the network invariant to webcam auto-exposure /
        lighting changes (live crops measured ~3x brighter than calibration),
        which caused open eyes to be read as closed in live light."""
        if crop_bgr is None or crop_bgr.size == 0 or net is None:
            return None
        img = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.standardize_crops:
            img = (img - img.mean()) / (img.std() + 1e-6)
        else:
            img = (img - 0.5) / 0.5
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.softmax(net(x), 1)[0]
        return prob.cpu().numpy()

    def _predict(self, net, crop_bgr, kind):
        """Resize + normalise a BGR crop and run the binary CNN.

        With fine-tuned (real) weights the CNN alone is the most reliable signal
        (tuned on the user's calibration crops, ~0.87 eyes / ~0.83 mouth), so it
        decides directly at a standard 0.5 threshold. The pixel-heuristic fusion
        below is only used as a fallback for the synthetic-only weights, where the
        CNN does not transfer to real faces."""
        if crop_bgr is None or crop_bgr.size == 0 or net is None:
            return None, None
        prob = self._cnn_probs(net, crop_bgr)
        if prob is None:
            return None, None
        if self.using_real_model:
            if kind == "eye":
                return (0 if prob[0] >= 0.5 else 1), float(prob[0])
            # mouth: only report a yawn when the mouth actually looks open.
            # the yawn CNN can fire on a closed mouth (noisy training labels),
            # so gate it on the mouth-openness contrast heuristic. A wide-open
            # mouth is bright inside (teeth) -> high std; a closed mouth shows
            # only a thin dark line -> low std.
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            band = gray[int(gray.shape[0] * 0.25):int(gray.shape[0] * 0.75), :]
            contrast = float(band.std())
            is_open = contrast >= MOUTH_OPEN_CONTRAST
            self._last_mouth_contrast = contrast
            return (1 if (is_open and prob[1] >= 0.5) else 0), float(prob[1])

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        if kind == "eye":
            # best pixel discriminator found on real crops: dark fraction of a
            # wide band (open 0.887 vs closed 0.832 mean); weak but directional
            band = gray[int(h * 0.15):int(h * 0.85), :]
            dark = float((band < 60).mean())
            h_open = float(np.clip((dark - 0.84) / 0.12, 0.0, 1.0))
            fused = self.eye_cnn_weight * float(prob[0]) + (1 - self.eye_cnn_weight) * h_open
            return (0 if fused >= 0.5 else 1), fused
        else:  # mouth
            # best pixel discriminator: contrast of the middle band
            # (yawn 29.9 vs no_yawn 16.5 mean); a wide-open mouth is bright inside
            band = gray[int(h * 0.25):int(h * 0.75), :]
            contrast = float(band.std())
            h_yawn = float(np.clip((contrast - 20) / 20, 0.0, 1.0))
            fused = self.mouth_cnn_weight * float(prob[1]) + (1 - self.mouth_cnn_weight) * h_yawn
            return (1 if fused >= 0.5 else 0), fused

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _majority(hist):
        if not hist:
            return None
        return 1 if sum(hist) > len(hist) // 2 else 0

    def _reset_runs(self):
        self._closed_run = 0
        self._yawn_run = 0

    # -------------------------------------------------------------- main pass
    def process_frame(self, frame_bgr):
        """Annotate one frame. Returns (annotated_frame, status_dict)."""
        self.frame_idx += 1
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        status = {
            "driver_present": False, "face": False,
            "eye_left": None, "eye_right": None, "mouth": None,
            "drowsy": False, "yawning": False,
            "alert": "SAFE",
        }

        # ---- 1) YOLO: is a person (driver) in the frame? (every Nth frame)
        yolo_person = False
        if self.yolo is not None and (self.frame_idx % self.yolo_interval == 0):
            res = self.yolo(frame_bgr, verbose=False)[0]
            for box in res.boxes:
                if box.cls.item() == 0 and box.conf.item() > 0.5:
                    yolo_person = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 200, 0), 1)
                    cv2.putText(frame_bgr, "driver (YOLO)", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    break

        # ---- 2) face localisation.
        # IMPORTANT: identical cascade settings to collect_calibration.py
        # (minNeighbors=5, minSize=90) so live crops match the calibration
        # crops the CNNs were trained on. No looser fallback: a differently
        # sized face box would misplace the eye/mouth crops.
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        # Only run eye/mouth classification when a face is detected in the
        # CURRENT frame. A stale last-known box may contain background, so it
        # must never be used for drowsy/yawn decisions.
        if len(faces):
            face = faces[0]
            self._last_face = face
        else:
            face = None

        if face is not None:
            fx, fy, fw, fh = face
            status["face"] = True
            status["driver_present"] = True

            # geometric eye strip (upper ~45% of the face)
            eye_strip_y1 = fy + int(fh * 0.15)
            eye_strip_y2 = fy + int(fh * 0.55)
            lx, lw = fx + int(fw * 0.05), int(fw * 0.42)
            rx = fx + int(fw * 0.53)
            left_eye = frame_bgr[eye_strip_y1:eye_strip_y2, lx:lx + lw]
            right_eye = frame_bgr[eye_strip_y1:eye_strip_y2, rx:rx + lw]

            # mouth region (generous: a wide yawn can grow outside a tight box)
            my1, my2 = fy + int(fh * 0.55), fy + int(fh * 0.97)
            mx, mw = fx + int(fw * 0.22), int(fw * 0.56)
            mouth = frame_bgr[my1:my2, mx:mx + mw]

            eL, eLc = self._predict(self.eye_net, left_eye, "eye")
            eR, eRc = self._predict(self.eye_net, right_eye, "eye")
            mS, mSc = self._predict(self.mouth_net, mouth, "mouth")
            status["eye_left"] = "closed" if eL == 1 else "open"
            status["eye_right"] = "closed" if eR == 1 else "open"
            status["mouth"] = "yawn" if mS == 1 else "no_yawn"

            # diagnostics (used by the tuning script; no effect on behaviour)
            self._dbg_frame = {
                "frame": self.frame_idx,
                "eL": eL, "eR": eR, "mS": mS,
                "eLc": float(eLc) if eLc is not None else None,
                "eRc": float(eRc) if eRc is not None else None,
                "mSc": float(mSc) if mSc is not None else None,
                "mouth_contrast": getattr(self, "_last_mouth_contrast", None),
                "closed_run": self._closed_run, "yawn_run": self._yawn_run,
            }

            # temporal smoothing via majority vote (drowsy requires BOTH eyes
            # closed - a single misclassified/blinking eye must not alarm)
            self._eye_hist.append(1 if (eL == 1 and eR == 1) else 0)
            self._mouth_hist.append(mS)
            if len(self._eye_hist) > self.smooth_win:
                self._eye_hist.pop(0)
            if len(self._mouth_hist) > self.smooth_win:
                self._mouth_hist.pop(0)
            closed_now = self._majority(self._eye_hist) == 1
            yawn_now = self._majority(self._mouth_hist) == 1

            self._closed_run = self._closed_run + 1 if closed_now else 0
            self._yawn_run = self._yawn_run + 1 if yawn_now else 0

            # draw overlays
            color = (0, 255, 0)
            if self._closed_run >= self.closed_frames:
                status["drowsy"] = True
                color = (0, 0, 255)
            elif self._yawn_run >= self.yawn_frames:
                status["yawning"] = True
                color = (0, 165, 255)

            cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), color, 2)
            cv2.rectangle(frame_bgr, (lx, eye_strip_y1), (lx + lw, eye_strip_y2), (255, 255, 255), 1)
            cv2.rectangle(frame_bgr, (rx, eye_strip_y1), (rx + lw, eye_strip_y2), (255, 255, 255), 1)
            cv2.rectangle(frame_bgr, (mx, my1), (mx + mw, my2), (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"eyes: L {status['eye_left']} / R {status['eye_right']}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"mouth: {status['mouth']}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # no face detected - nothing to classify this frame
            status["driver_present"] = yolo_person

        # ---- 3) final alert level
        if status["drowsy"]:
            status["alert"] = "DROWSY"
        elif status["yawning"]:
            status["alert"] = "YAWNING"
        else:
            status["alert"] = "SAFE"

        self._draw_status(frame_bgr, status)
        return frame_bgr, status

    def _draw_status(self, frame_bgr, status):
        h, w = frame_bgr.shape[:2]
        txt = status["alert"]
        color = {"SAFE": (0, 255, 0), "DROWSY": (0, 0, 255),
                 "YAWNING": (0, 165, 255)}[txt]
        cv2.rectangle(frame_bgr, (w // 2 - 140, 12), (w // 2 + 140, 52), (0, 0, 0), -1)
        cv2.putText(frame_bgr, f"STATE: {txt}", (w // 2 - 110, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if winsound is not None and self.beep_alerts and status["drowsy"] \
                and not self._prev_drowsy:
            winsound.Beep(900, 250)
        self._prev_drowsy = status["drowsy"]


# ----------------------------------------------------------------------------
# public entry points for the three input modes
# ----------------------------------------------------------------------------

def detect_image(detector, image_path, output_path=None):
    """Classify a single uploaded image. Returns annotated BGR frame + status."""
    detector.beep_alerts = False
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    annotated, status = detector.process_frame(img)
    if output_path:
        cv2.imwrite(output_path, annotated)
    return annotated, status


def detect_video(detector, video_path, output_path=None, show=False, max_frames=None):
    """Process an uploaded video file frame by frame."""
    detector.beep_alerts = False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    stats = {"frames": 0, "drowsy": 0, "yawn": 0}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, status = detector.process_frame(frame)
        stats["frames"] += 1
        for key, status_key in (("drowsy", "drowsy"), ("yawn", "yawning")):
            if status.get(status_key):
                stats[key] += 1
        if out is not None:
            out.write(frame)
        if show:
            cv2.imshow("result", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames and stats["frames"] >= max_frames:
            break
    cap.release()
    if out is not None:
        out.release()
    if show:
        cv2.destroyAllWindows()
    return stats


def detect_webcam(detector, camera=0):
    """Live webcam loop. Press 'q' to quit."""
    detector.beep_alerts = True
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("webcam not available")
    print("live detection started - press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, status = detector.process_frame(frame)
        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["webcam", "image", "video"])
    ap.add_argument("--input", help="image/video path (not needed for webcam)")
    ap.add_argument("--output", help="where to save the annotated file")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-yolo", action="store_true")
    args = ap.parse_args()

    det = DrowsinessDetector(use_yolo=not args.no_yolo)
    if args.mode == "webcam":
        detect_webcam(det, camera=args.camera)
    elif args.mode == "image":
        annotated, status = detect_image(det, args.input, args.output)
        print(status)
        if args.output:
            print(f"saved -> {args.output}")
    else:
        stats = detect_video(det, args.input, args.output, show=False)
        print(stats)
        if args.output:
            print(f"saved -> {args.output}")
