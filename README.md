# Driver Drowsiness Detection

A real-time system that monitors a driver's face from the webcam (or an uploaded
image/video) and warns when the driver is **drowsy** or **yawning**. Built with
PyTorch, OpenCV, scikit-learn and YOLOv8 — using only the standard workshop
library set.

![state badge](https://img.shields.io/badge/status-working-brightgreen)

## Features

- **Live webcam detection** with an annotated OpenCV window and an audible beep
  when DROWSY is detected
- **Upload an image** or **upload a video** for offline analysis (annotated copy saved)
- **One-click launcher**: double-click `Drowsiness_Detector_Runner.bat` to open
  a console menu — **1. Launch GUI**, **2. Live Webcam**, **3. Upload Video**,
  **4. Upload Image**, **5. Quit**. After a mode finishes it returns to the menu
  so you can re-run without restarting; press `q` to quit.
- **Two tiny CNN classifiers** (`EyeCNN`, `MouthCNN`) that run at >30 FPS on a laptop
- Eye CNN trained on the public **MRL Eye Dataset** (84,898 real eye images)
  and fine-tuned on your own webcam crops for personal accuracy
- **Safety gates**: classification only runs when a face is in the current frame,
  and a yawn is only reported when the mouth actually looks open
- Optional **YOLOv8** person check confirms a driver is in the frame

## Alert states

| State | Meaning |
|---|---|
| SAFE | Eyes open, looking at the road |
| DROWSY | Both eyes closed for several consecutive frames (PERCLOS-style) |
| YAWNING | Mouth open wide for several consecutive frames |

## Quick start

1. Install dependencies:

   ```bash
   pip install numpy pandas opencv-python matplotlib scikit-learn torch torchvision ultralytics
   ```

2. **Easiest way** — double-click `Drowsiness_Detector_Runner.bat` and pick a
   mode from the console menu (GUI is option 1).

3. Or run from the notebook `Driver_Drowsiness_Detection.ipynb` (has upload
   widgets for image/video and a live webcam cell).

4. Or run the command line:

   ```bash
   python src/detector.py webcam
   python src/detector.py image --input samples/sample_face.jpg --output outputs/annotated_image.jpg
   python src/detector.py video --input demo/raw_demo.mp4 --output demo/annotated_demo.mp4
   ```

> **Live webcam:** press `q` inside the video window to stop.

## How it works

```
frame ──► Haar cascade (face) ──► geometric eye/mouth crops
                                  │
   YOLOv8 person check (optional) │
                                  ▼
                   EyeCNN ──► open / closed
                   MouthCNN ─► yawn / no yawn (gated by mouth-openness)
                                  │
                   temporal counters ──► DROWSY / YAWNING / SAFE
```

- `detector.py` — the detection engine (`DrowsinessDetector`, all three modes)
- `train.py` — trains the CNNs on the synthetic dataset
- `train_mrl.py` — trains the eye CNN on the MRL Eye Dataset (GPU, cached pipeline)
- `finetune.py` — fine-tunes the eye CNNs (glasses / no-glasses) + mouth CNN on your real calibration crops
- `train_glasses.py` — trains the glasses-vs-no-glasses classifier (`glasses_cnn.pth`)
- `collect_calibration.py` — guided capture of your eyes/mouth (~30 s); `--glasses` tags the crops
- `generate_data.py` — generates the synthetic eye/mouth dataset
- `menu.py` — console menu behind `Drowsiness_Detector_Runner.bat`
- `gui.py` — Tkinter launcher (one of the menu options)

## Results

- **Eye CNN** (MRL): **99.8%** accuracy on 5,400 held-out real eye images
- **Fine-tuned on the user's crops**: **100%** held-out accuracy for both CNNs
- **Live test**: correct SAFE / DROWSY / YAWNING detection
- **Demo video** (`demo/annotated_demo.mp4`): 585 frames → DROWSY 345,
  YAWNING 65

## Calibration (optional, recommended)

The detector automatically prefers the fine-tuned `*_real.pth` models when they
exist. To create them for your own face:

```bash
# without glasses
python src/collect_calibration.py
# if you also wear glasses, run the same capture again with them ON (same
# lighting/position), using the --glasses flag so the crops get tagged:
python src/collect_calibration.py --glasses

# then retrain everything:
python src/finetune.py          # eye_cnn_real.pth (no glasses) + eye_cnn_glasses.pth + mouth_cnn_real.pth
python src/train_glasses.py     # glasses_cnn.pth (routes the eye model at runtime)
```

At runtime the detector classifies each frame as glasses / no-glasses and feeds
the eye crop to the matching model, so neither appearance's weights are diluted
by the other. The launcher also asks once at startup whether you are wearing
glasses: answer **y**/**n** to fix the model (no per-frame re-check) or **s** to
skip and auto-detect every frame. Capturing both appearances in the *same
session/lighting* matters: the classifier must learn the frame, not the lighting.
If you never wear glasses, skip the `--glasses` run and the detector just uses
the no-glasses model.

## Project structure

```
.
├── src/                 # all code (training, detection, GUI, calibration)
├── models/              # CNN weights (+ yolov8n.pt)
├── data/                # datasets — git-ignored (generated/downloaded/personal)
├── demo/                # recorded demo videos — git-ignored
├── samples/             # sample media — git-ignored
├── Driver_Drowsiness_Detection.ipynb
├── Drowsiness_Detector_Runner.bat
└── .gitignore
```

## Notes & limitations

- `data/`, `demo/`, `samples/`, `outputs/` and calibration captures are
  intentionally **not** in the repository (personal data / large downloadable
  datasets). Regenerate with the scripts above.
- Detection accuracy depends on lighting; very dark scenes are handled via
  brightness/contrast augmentation at training time but can still be hard.
- Real-time performance depends on webcam auto-exposure.
