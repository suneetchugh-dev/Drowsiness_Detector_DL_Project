# Driver Drowsiness Detection

A real-time system that monitors a driver's face from the webcam (or an uploaded
image/video) and warns when the driver is **drowsy** or **yawning**. Built with
PyTorch, OpenCV, scikit-learn and YOLOv8 — using only the standard workshop
library set.

![state badge](https://img.shields.io/badge/status-working-brightgreen)

## Features

- **Live webcam detection** with an annotated OpenCV window, embedded Tkinter GUI, and audible alert when DROWSY is detected
- **Upload an image** or **upload a video** for offline analysis (annotated copy saved)
- **One-click launcher**: run `CLI_Runner.bat` or `Drowsiness_Detector_GUI.bat` to launch the unified suite
- **Two tiny CNN classifiers** (`EyeCNN`, `MouthCNN`) running at >30 FPS on CPU/laptop
- **Cascade-localized pitch-robust crops** (`crops.py`): tracks actual eye and smile positions with geometric fallbacks
- **3-Class eye classifier support** (`open`, `tired`, `closed`) with graceful 2-class confidence band fallback
- **Live Diagnostics HUD** (`live_diag.py`): real-time gauge meters for `p_open`, `p_yawn`, contrast, and detector state
- **Safety gates**: classification only runs on active faces, and yawning uses CNN-only probability thresholds that never false-clear to SAFE during an active yawn
- Optional **YOLOv8** person check confirms a driver is in the frame

## Alert states

| State | Meaning |
|---|---|
| SAFE | Eyes open (neutral or smiling), looking at the road |
| TIRED | Eyes droopy / partially closed but not shut (eyes-only, no yawn) |
| DROWSY | Both eyes closed for several consecutive frames (PERCLOS-style) |
| YAWNING | Mouth open wide for several consecutive frames |

A state must persist briefly (~0.4 s) before the display switches, so the
model cannot flicker between states when it is unsure. A single blinking or
squinting eye never triggers an alarm.

## Quick start

1. Install dependencies:

   ```bash
   pip install numpy pandas opencv-python matplotlib scikit-learn torch torchvision ultralytics pillow
   ```

2. **Run GUI**:
   ```bash
   python src/gui.py
   ```

3. **Run Live Diagnostics**:
   ```bash
   python src/live_diag.py
   ```

4. **Run CLI modes**:

   ```bash
   python src/detector.py webcam
   python src/detector.py image --input samples/sample_face.jpg --output outputs/annotated_image.jpg
   python src/detector.py video --input demo/raw_demo.mp4 --output demo/annotated_demo.mp4
   ```

> **Live webcam:** press `q` inside the video window or click **Stop Detection** to stop.

## How it works

```
frame ──► Haar cascade (face) ──► cascade-localised eye/mouth crops (crops.py)
                                  │
   YOLOv8 person check (optional) │
                                  ▼
                   EyeCNN ──► open / tired / closed
                   MouthCNN ─► yawn / no yawn (CNN-only probability gate)
                                  │
                   temporal counters (debounced) ─► SAFE / TIRED / DROWSY / YAWNING
```

- `crops.py` — shared pitch-robust eye/mouth region localisation and cropping
- `detector.py` — the detection engine (`DrowsinessDetector`, all input modes)
- `live_diag.py` — guided live telemetry measurement of `p_open`, `p_yawn`, contrast, and states
- `train.py` — trains the CNNs on synthetic datasets
- `train_mrl.py` — trains the eye CNN on the MRL Eye Dataset (GPU, cached pipeline)
- `finetune.py` — fine-tunes the 2-class / 3-class eye CNNs + mouth CNN on real calibration crops
- `train_glasses.py` — trains the glasses-vs-no-glasses classifier (`glasses_cnn.pth`)
- `collect_calibration.py` — guided capture of eyes (`open`, `tired`, `closed`) and mouth (`yawn`, `no_yawn`)
- `generate_data.py` — generates the synthetic eye/mouth dataset
- `menu.py` — interactive console menu
- `gui.py` — flat monochromatic Tkinter GUI with embedded live feed, mode selection, and screen fill light

## Results

- **Eye CNN** (MRL): **99.8%** accuracy on 5,400 held-out real eye images
- **Fine-tuned on user crops**: **100%** held-out accuracy for both CNNs
- **Live test**: correct SAFE / TIRED / DROWSY / YAWNING detection
- **Demo video** (`demo/annotated_demo.mp4`): 585 frames → DROWSY 345, YAWNING 65

## Calibration (optional, recommended)

The detector automatically prefers the fine-tuned `*_real3.pth` / `*_real.pth` models when they
exist. To create them for your own face:

```bash
# without glasses (guides through open, tired, closed, yawn, relaxed)
python src/collect_calibration.py

# if you wear glasses, run the capture again with them ON:
python src/collect_calibration.py --glasses

# retrain models:
python src/finetune.py          # creates eye_cnn_real.pth, eye_cnn_real3.pth, etc.
python src/train_glasses.py     # creates glasses_cnn.pth
```

At runtime the detector classifies each frame as glasses / no-glasses and feeds
the eye crop to the matching model, so neither appearance's weights are diluted
by the other.

## Project structure

```
.
├── src/                 # all code (training, detection, GUI, calibration, diagnostics)
├── models/              # CNN weights (+ yolov8n.pt)
├── data/                # datasets — git-ignored (generated/downloaded/personal)
├── demo/                # recorded demo videos & diagnostics snapshots
├── samples/             # sample media
├── Driver_Drowsiness_Detection.ipynb
├── Drowsiness_Detector_GUI.bat
├── CLI_Runner.bat
└── .gitignore
```

## Notes & limitations

- `data/`, `demo/`, `samples/`, `outputs/` and calibration captures are
  intentionally **not** in the repository (personal data / large downloadable
  datasets). Regenerate with the scripts above.
- Detection accuracy depends on lighting; the GUI includes a screen fill-light toggle for low-light conditions.
- Real-time performance is >30 FPS on typical CPU hardware.
