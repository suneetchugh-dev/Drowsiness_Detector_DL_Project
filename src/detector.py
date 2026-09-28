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

try:
    from crops import detect_face, localize_regions, crop_regions
except ImportError:
    from src.crops import detect_face, localize_regions, crop_regions

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

EYE_WEIGHTS_MRL = os.path.join(MODEL_DIR, "eye_cnn_mrl.pth")
EYE_WEIGHTS_SYNTH = os.path.join(MODEL_DIR, "eye_cnn.pth")
EYE_WEIGHTS = EYE_WEIGHTS_MRL if os.path.exists(EYE_WEIGHTS_MRL) else EYE_WEIGHTS_SYNTH
MOUTH_WEIGHTS = os.path.join(MODEL_DIR, "mouth_cnn.pth")
EYE_WEIGHTS_REAL = os.path.join(MODEL_DIR, "eye_cnn_real.pth")
EYE_WEIGHTS_GLASSES = os.path.join(MODEL_DIR, "eye_cnn_glasses.pth")
MOUTH_WEIGHTS_REAL = os.path.join(MODEL_DIR, "mouth_cnn_real.pth")
GLASSES_WEIGHTS = os.path.join(MODEL_DIR, "glasses_cnn.pth")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov8n.pt")

IMG_SIZE = 48

# ---- tuned alert thresholds -------------------------------------------------
CLOSED_FRAMES_ALERT = 3     # consecutive frames both eyes closed -> DROWSY
YAWN_FRAMES_ALERT = 4       # consecutive frames yawning           -> YAWNING
TIRED_FRAMES_ALERT = 6      # consecutive frames of tired eyes      -> TIRED
SMOOTH_WIN = 5              # majority-vote window for eye/mouth states
MOUTH_OPEN_CONTRAST = 18.0  # band std for telemetry / diagnostics
MOUTH_YAWN_THRESH = 0.60    # CNN p(yawn) threshold (CNN-only gate)
EYE_CLOSED_P = 0.34         # per-eye p_open below this -> eye CLOSED
EYE_OPEN_P = 0.70           # per-eye p_open at/above this -> clearly OPEN
                            # in-between the eye is TIRED (confidence band)
STATE_HOLD_FRAMES = 12      # frames a candidate alert must persist before it is
                            # committed (debounce, ~400ms at 30fps). Stops the
                            # model from flickering between states when unsure.

EYE_OPEN, EYE_TIRED, EYE_CLOSED = 0, 1, 2
EYE_LABELS = {EYE_OPEN: "open", EYE_TIRED: "tired", EYE_CLOSED: "closed"}


class SmallCNN(torch.nn.Module):
    """Must match the architecture used in training (src/train.py)."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
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
                 tired_frames=TIRED_FRAMES_ALERT, smooth_win=SMOOTH_WIN,
                 state_hold_frames=STATE_HOLD_FRAMES, mouth_yawn_thresh=MOUTH_YAWN_THRESH,
                 cnn_weight=None, glasses_mode=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_yolo = use_yolo
        self.yolo_interval = yolo_interval
        self.closed_frames = closed_frames
        self.yawn_frames = yawn_frames
        self.tired_frames = tired_frames
        self.smooth_win = smooth_win
        self.state_hold_frames = state_hold_frames
        self.mouth_yawn_thresh = mouth_yawn_thresh
        self.beep_alerts = True
        self._prev_drowsy = False
        # glasses_mode: None = auto-detect every frame; 0/1 = force the
        # no-glasses / glasses eye model once (chosen at startup), skipping the
        # per-frame glasses re-check entirely.
        self.forced_glasses = glasses_mode
        # per-image mean/std normalisation makes the CNNs invariant to webcam
        # auto-exposure.
        self.standardize_crops = True

        # 1) Haar cascades shipped with opencv-python
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_eye.xml"))

        # 2) trained CNNs: primary base eye model is MRL Eye dataset (84,898 images)
        self.eye_net = self._load_cnn(EYE_WEIGHTS)
        self.eye_net_glasses = self._load_cnn(EYE_WEIGHTS_GLASSES) \
            if os.path.exists(EYE_WEIGHTS_GLASSES) else None
        self.glasses_net = self._load_cnn(GLASSES_WEIGHTS) \
            if os.path.exists(GLASSES_WEIGHTS) else None
        self.glasses_ok = self.glasses_net is not None and self.eye_net_glasses is not None
        self.mouth_net = self._load_cnn(MOUTH_WEIGHTS)
        self.using_real_model = False
        if cnn_weight is not None:
            self.eye_cnn_weight = self.mouth_cnn_weight = cnn_weight
        else:
            self.eye_cnn_weight = 0.85
            self.mouth_cnn_weight = 0.85

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
        self._tired_run = 0
        self._eye_hist = []
        self._mouth_hist = []
        self._glasses_hist = []
        self.glasses_now = 0
        self._last_face = None
        self._dbg_frame = None
        self._last_mouth_contrast = None
        # debounced alert state (candidate holds for state_hold_frames)
        self._alert_now = None
        self._cand_label = None
        self._cand_count = 0

    # ------------------------------------------------------------------ setup
    def _load_cnn(self, path):
        if not path or not os.path.exists(path):
            print(f"[detector] missing weights {path} - CNN not loaded")
            return None
        state = torch.load(path, map_location=self.device)
        num_classes = 2
        for key in ("classifier.4.weight", "classifier.3.weight", "classifier.4.bias"):
            if key in state:
                num_classes = state[key].shape[0]
                break
        net = SmallCNN(num_classes=num_classes)
        net.load_state_dict(state)
        net.to(self.device).eval()
        net.num_classes = num_classes
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

    def _glasses_probs(self, crop_bgr):
        """P(glasses) from the glasses-vs-no-glasses CNN (or None if absent)."""
        if crop_bgr is None or crop_bgr.size == 0 or self.glasses_net is None:
            return None
        prob = self._cnn_probs(self.glasses_net, crop_bgr)
        return None if prob is None else float(prob[1])

    def _predict(self, net, crop_bgr, kind):
        """Resize + normalise a BGR crop and run the CNN.

        For eyes:
          - If a 3-class model is loaded, returns (pred_class, prob_array).
          - If a 2-class model is loaded, returns (pred_bin, p_open).
        For mouth:
          - Uses a CNN-only threshold (p_yawn >= mouth_yawn_thresh) without
            gating on the contrast heuristic, eliminating false negatives on
            darker or lower-contrast yawns.
        """
        if crop_bgr is None or crop_bgr.size == 0 or net is None:
            return None, None
        prob = self._cnn_probs(net, crop_bgr)
        if prob is None:
            return None, None

        if kind == "eye":
            num_classes = getattr(net, "num_classes", 2)
            if num_classes == 3:
                pred_class = int(np.argmax(prob))
                return pred_class, prob
            else:
                p_open = float(prob[0])
                pred_bin = 0 if p_open >= 0.5 else 1
                return pred_bin, p_open

        # mouth: compute contrast for telemetry / diag
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        band = gray[int(gray.shape[0] * 0.25):int(gray.shape[0] * 0.75), :]
        self._last_mouth_contrast = float(band.std())

        if self.using_real_model:
            p_yawn = float(prob[1])
            is_yawn = 1 if p_yawn >= self.mouth_yawn_thresh else 0
            return is_yawn, p_yawn

        # synthetic fallback
        contrast = self._last_mouth_contrast
        h_yawn = float(np.clip((contrast - 20) / 20, 0.0, 1.0))
        fused = self.mouth_cnn_weight * float(prob[1]) + (1 - self.mouth_cnn_weight) * h_yawn
        return (1 if fused >= 0.5 else 0), fused

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _majority(hist):
        if not hist:
            return None
        return 1 if sum(hist) > len(hist) // 2 else 0

    @staticmethod
    def _eye_class(eye_res):
        """Map eye prediction (3-class output or 2-class p_open) to OPEN / TIRED / CLOSED.

        3-class model: directly returns predicted class 0 (open), 1 (tired), 2 (closed).
        2-class model: maps openness probability via confidence bands:
          - p_open >= EYE_OPEN_P (0.70) -> OPEN
          - p_open < EYE_CLOSED_P (0.34) -> CLOSED
          - in-between -> TIRED (droopy / ambiguous)
        """
        if isinstance(eye_res, (int, np.integer)):
            return int(eye_res)
        if isinstance(eye_res, (list, tuple, np.ndarray)):
            if len(eye_res) == 3:
                return int(np.argmax(eye_res))
            eye_res = eye_res[0]
        p_open = float(eye_res)
        if p_open >= EYE_OPEN_P:
            return EYE_OPEN
        if p_open < EYE_CLOSED_P:
            return EYE_CLOSED
        return EYE_TIRED

    @classmethod
    def _combine_eyes(cls, eL_res, eR_res):
        """Per-frame eye verdict from both eye results.

        DROWSY needs BOTH eyes closed; TIRED needs at least one eye reading
        tired while neither is actually closed. Otherwise the frame counts as
        open (a single blinking / squinting eye must never alarm)."""
        lc, rc = cls._eye_class(eL_res), cls._eye_class(eR_res)
        if lc == EYE_CLOSED and rc == EYE_CLOSED:
            return EYE_CLOSED
        if lc != EYE_CLOSED and rc != EYE_CLOSED and (lc == EYE_TIRED or rc == EYE_TIRED):
            return EYE_TIRED
        return EYE_OPEN

    @classmethod
    def _majority_code(cls, hist):
        """Most frequent eye class in the smoothing window (ties -> OPEN)."""
        if not hist:
            return EYE_OPEN
        counts = {}
        for x in hist:
            counts[x] = counts.get(x, 0) + 1
        best = max(counts.values())
        candidates = [k for k, v in counts.items() if v == best]
        return EYE_OPEN if EYE_OPEN in candidates else candidates[0]

    def _reset_runs(self):
        self._closed_run = 0
        self._yawn_run = 0
        self._tired_run = 0

    def _resolve_alert(self, desired):
        """Debounce helper: a new alert only commits after state_hold_frames
        of continuous support, so an unsure model cannot flicker between
        states. NO FACE (context loss) and the first frame commit instantly."""
        if desired == "NO FACE":
            self._alert_now = "NO FACE"
            self._cand_label = None
            self._cand_count = 0
            return "NO FACE"
        if self._alert_now is None or self._alert_now == "NO FACE":
            self._alert_now = desired
            self._cand_label = None
            self._cand_count = 0
            return desired
        if desired == self._alert_now:
            self._cand_label = None
            self._cand_count = 0
            return desired
        if desired == self._cand_label:
            self._cand_count += 1
        else:
            self._cand_label = desired
            self._cand_count = 1
        if self._cand_count >= max(1, self.state_hold_frames):
            self._alert_now = desired
            self._cand_label = None
            self._cand_count = 0
            return desired
        return self._alert_now

    # -------------------------------------------------------------- main pass
    def process_frame(self, frame_bgr):
        """Annotate one frame. Returns (annotated_frame, status_dict)."""
        self.frame_idx += 1
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        status = {
            "driver_present": False, "face": False,
            "eye_left": None, "eye_right": None, "mouth": None,
            "drowsy": False, "yawning": False, "tired": False,
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

        # ---- 2) face localisation via shared pitch-robust cascades
        face = detect_face(gray)

        if face is not None:
            fx, fy, fw, fh = map(int, face)
            face = (fx, fy, fw, fh)
            self._last_face = face
            status["face"] = True
            status["driver_present"] = True

            # Localize and crop regions using shared crops logic
            le_box, re_box, mouth_box = localize_regions(frame_bgr, face)
            left_eye, right_eye, mouth = crop_regions(frame_bgr, face)

            # glasses state: fixed at startup (no re-check) or auto-detected
            # every frame via the glasses-vs-no-glasses classifier
            if self.forced_glasses is not None:
                self.glasses_now = 1 if self.forced_glasses else 0
            else:
                gp_l = self._glasses_probs(left_eye)
                gp_r = self._glasses_probs(right_eye)
                if gp_l is not None and gp_r is not None:
                    gp = 0.5 * (gp_l + gp_r)
                    self._glasses_hist.append(1 if gp >= 0.5 else 0)
                    if len(self._glasses_hist) > max(2 * self.smooth_win, 5):
                        self._glasses_hist.pop(0)
                    self.glasses_now = 1 if sum(self._glasses_hist) > len(self._glasses_hist) // 2 \
                        else 0
            eye_net = self.eye_net_glasses if (self.glasses_now and self.eye_net_glasses is not None) \
                else self.eye_net
            eL, eLc = self._predict(eye_net, left_eye, "eye")
            eR, eRc = self._predict(eye_net, right_eye, "eye")
            mS, mSc = self._predict(self.mouth_net, mouth, "mouth")

            # three-way per-eye label: open / tired / closed
            status["eye_left"] = EYE_LABELS[self._eye_class(eLc)]
            status["eye_right"] = EYE_LABELS[self._eye_class(eRc)]
            status["mouth"] = "yawn" if mS == 1 else "no_yawn"

            # diagnostics (used by the tuning script; no effect on behaviour)
            self._dbg_frame = {
                "frame": self.frame_idx,
                "glasses": self.glasses_now if self.glasses_ok else None,
                "eL": eL, "eR": eR, "mS": mS,
                "eLc": eLc, "eRc": eRc,
                "mSc": float(mSc) if mSc is not None else None,
                "mouth_contrast": getattr(self, "_last_mouth_contrast", None),
                "closed_run": self._closed_run, "yawn_run": self._yawn_run,
                "tired_run": self._tired_run,
            }

            # temporal smoothing via majority vote (drowsy requires BOTH eyes
            # closed - a single misclassified/blinking eye must not alarm)
            self._eye_hist.append(self._combine_eyes(eLc, eRc))
            self._mouth_hist.append(mS)
            if len(self._eye_hist) > self.smooth_win:
                self._eye_hist.pop(0)
            if len(self._mouth_hist) > self.smooth_win:
                self._mouth_hist.pop(0)
            eye_state = self._majority_code(self._eye_hist)
            closed_now = eye_state == EYE_CLOSED
            tired_now = eye_state == EYE_TIRED
            yawn_now = self._majority(self._mouth_hist) == 1

            self._closed_run = self._closed_run + 1 if closed_now else 0
            self._yawn_run = self._yawn_run + 1 if yawn_now else 0
            # tired (eyes only, no yawn) requires its own sustained streak
            self._tired_run = self._tired_run + 1 if (tired_now and not yawn_now) else 0

            # draw overlays
            color = (0, 255, 0)
            if self._closed_run >= self.closed_frames:
                status["drowsy"] = True
                color = (0, 0, 255)
            elif self._yawn_run >= self.yawn_frames:
                status["yawning"] = True
                color = (0, 165, 255)
            elif self._tired_run >= self.tired_frames:
                status["tired"] = True
                color = (0, 210, 255)

            lx, ly, lw, lh = le_box
            rx, ry, rw, rh = re_box
            mx, my, mw, mh = mouth_box
            cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), color, 2)
            cv2.rectangle(frame_bgr, (lx, ly), (lx + lw, ly + lh), (255, 255, 255), 1)
            cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)
            cv2.rectangle(frame_bgr, (mx, my), (mx + mw, my + mh), (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"eyes: L {status['eye_left']} / R {status['eye_right']}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"mouth: {status['mouth']}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if self.glasses_ok:
                tag = " (fixed)" if self.forced_glasses is not None else ""
                cv2.putText(frame_bgr, f"glasses: {'yes' if self.glasses_now else 'no'}{tag}",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # no face in the frame -> make NO prediction at all
            status["driver_present"] = yolo_person
            status["alert"] = "NO FACE"
            self._reset_runs()
            self._eye_hist.clear()
            self._mouth_hist.clear()
            self._dbg_frame = None

        # ---- 3) final alert level.
        # Priority: NO FACE > DROWSY > YAWNING > TIRED > SAFE.
        # A yawn in progress never resolves to SAFE.
        desired = "SAFE"
        if status["alert"] == "NO FACE":
            desired = "NO FACE"
        elif status["drowsy"]:
            desired = "DROWSY"
        elif status["yawning"]:
            desired = "YAWNING"
        elif status["tired"]:
            desired = "TIRED"

        # debounce: a candidate alert must survive state_hold_frames before the
        # display switches, so an unsure model cannot flicker state to state.
        status["alert"] = self._resolve_alert(desired)
        if status["yawning"] and status["alert"] == "SAFE":
            status["alert"] = "YAWNING"
        status["drowsy"] = status["alert"] == "DROWSY"
        status["yawning"] = status["alert"] == "YAWNING"
        status["tired"] = status["alert"] == "TIRED"

        self._draw_status(frame_bgr, status)
        return frame_bgr, status

    def _draw_status(self, frame_bgr, status):
        h, w = frame_bgr.shape[:2]
        txt = status["alert"]
        color = {"SAFE": (0, 255, 0), "DROWSY": (0, 0, 255),
                 "YAWNING": (0, 165, 255), "TIRED": (0, 210, 255),
                 "NO FACE": (220, 220, 220)}[txt]
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
    # one frame only: commit its verdict immediately (no debounce wait)
    detector.state_hold_frames = 1
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


def detect_webcam(detector, camera=0, fullscreen=False, window_name="Driver Drowsiness Detection"):
    """Live webcam loop. Press 'q' to quit.

    The window is resizable (WINDOW_NORMAL) by default; pass fullscreen=True
    to open maximised. Keys:
      q / ESC  -> quit
      f        -> toggle fullscreen
    """
    detector.beep_alerts = True
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("webcam not available")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        try:
            cv2.resizeWindow(window_name, 960, 640)
        except cv2.error:
            pass

    print("live detection started - press 'q' to quit, 'f' to toggle fullscreen")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, status = detector.process_frame(frame)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            prop = cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN) != cv2.WINDOW_FULLSCREEN \
                else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
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
