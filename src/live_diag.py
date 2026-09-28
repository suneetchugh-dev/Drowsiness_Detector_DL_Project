"""Guided Live Measurement and Diagnostics Tool.

Measures real-time eye openness (p_open), mouth yawn probability (p_yawn),
band contrast, and glasses classification directly from the live webcam.

Usage:
    python src/live_diag.py

Keyboard Controls:
    q / ESC : Quit
    g       : Cycle glasses mode (Auto -> On -> Off)
    s       : Save diagnostic snapshot (image + telemetry)
    r       : Reset telemetry history
"""

import os
import sys
import time
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector, EYE_OPEN, EYE_TIRED, EYE_CLOSED, EYE_LABELS
from crops import detect_face, localize_regions, crop_regions


def draw_meter(frame, x, y, w, h, val, thresh=0.5, label="", color=(0, 255, 0), thresh_color=(0, 0, 255)):
    """Draw a horizontal gauge/progress bar with a threshold line."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill_w = int(max(0.0, min(1.0, val)) * w)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    
    # threshold tick
    tx = x + int(thresh * w)
    cv2.line(frame, (tx, y - 2), (tx, y + h + 2), thresh_color, 2)
    
    # text
    txt = f"{label}: {val:.2f}"
    cv2.putText(frame, txt, (x + w + 10, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)


def main():
    print("=" * 65)
    print("  Driver Drowsiness Detection - Live Diagnostics & Telemetry")
    print("=" * 65)
    print("Controls:")
    print("  [q / ESC] Quit")
    print("  [g]       Toggle glasses mode (Auto / On / Off)")
    print("  [s]       Save diagnostic snapshot")
    print("  [r]       Reset stats")
    print("=" * 65)

    det = DrowsinessDetector(use_yolo=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not available.")
        return

    win_name = "Live Measurement & Diagnostics"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1080, 720)

    snap_idx = 0
    os.makedirs("demo/diag_snapshots", exist_ok=True)

    glasses_modes = [None, 1, 0]  # Auto, On, Off
    mode_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, status = det.process_frame(frame.copy())
        h, w = frame.shape[:2]

        # Diagnostics HUD overlay panel
        panel_w = 340
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        px = w - panel_w + 15
        py = 30

        cv2.putText(frame, "TELEMETRY HUD", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        py += 30

        # State
        state_txt = status["alert"]
        st_color = {"SAFE": (0, 255, 0), "DROWSY": (0, 0, 255), "YAWNING": (0, 165, 255),
                    "TIRED": (0, 210, 255), "NO FACE": (200, 200, 200)}.get(state_txt, (255, 255, 255))
        cv2.putText(frame, f"State: {state_txt}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, st_color, 2)
        py += 25

        # Glasses
        g_mode_str = "Auto" if det.forced_glasses is None else ("On" if det.forced_glasses else "Off")
        g_det_str = "Yes" if det.glasses_now else "No"
        cv2.putText(frame, f"Glasses Mode: {g_mode_str} (Det: {g_det_str})", (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        py += 30

        dbg = det._dbg_frame or {}

        # Left Eye
        eLc = dbg.get("eLc", 0.0)
        eL_val = float(eLc[0]) if isinstance(eLc, (list, tuple, np.ndarray)) else float(eLc)
        draw_meter(frame, px, py, 120, 12, eL_val, thresh=0.5, label=f"L_eye ({status['eye_left']})",
                   color=(0, 220, 0) if status['eye_left'] == 'open' else (0, 100, 255))
        py += 24

        # Right Eye
        eRc = dbg.get("eRc", 0.0)
        eR_val = float(eRc[0]) if isinstance(eRc, (list, tuple, np.ndarray)) else float(eRc)
        draw_meter(frame, px, py, 120, 12, eR_val, thresh=0.5, label=f"R_eye ({status['eye_right']})",
                   color=(0, 220, 0) if status['eye_right'] == 'open' else (0, 100, 255))
        py += 30

        # Mouth Yawn Prob
        mSc = dbg.get("mSc", 0.0)
        mSc_val = float(mSc) if mSc is not None else 0.0
        yawn_color = (0, 140, 255) if status['mouth'] == 'yawn' else (0, 220, 0)
        draw_meter(frame, px, py, 120, 12, mSc_val, thresh=det.mouth_yawn_thresh,
                   label=f"p(yawn) [{status['mouth']}]", color=yawn_color)
        py += 24

        # Mouth Contrast
        contrast = dbg.get("mouth_contrast")
        contrast_val = float(contrast) if contrast is not None else 0.0
        draw_meter(frame, px, py, 120, 12, min(contrast_val / 40.0, 1.0), thresh=18.0 / 40.0,
                   label=f"Contrast ({contrast_val:.1f})", color=(200, 200, 100))
        py += 30

        # Run counters
        cv2.putText(frame, f"Closed streak : {dbg.get('closed_run', 0)} / {det.closed_frames}",
                    (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        py += 20
        cv2.putText(frame, f"Yawn streak   : {dbg.get('yawn_run', 0)} / {det.yawn_frames}",
                    (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        py += 20
        cv2.putText(frame, f"Tired streak  : {dbg.get('tired_run', 0)} / {det.tired_frames}",
                    (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        py += 35

        # Feature boxes on image
        if det._last_face is not None:
            fx, fy, fw, fh = det._last_face
            le, re, mouth = localize_regions(frame, det._last_face)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), st_color, 2)
            cv2.rectangle(frame, (le[0], le[1]), (le[0] + le[2], le[1] + le[3]), (255, 255, 0), 1)
            cv2.rectangle(frame, (re[0], re[1]), (re[0] + re[2], re[1] + re[3]), (255, 255, 0), 1)
            cv2.rectangle(frame, (mouth[0], mouth[1]), (mouth[0] + mouth[2], mouth[1] + mouth[3]), (0, 255, 255), 1)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("g"):
            mode_idx = (mode_idx + 1) % len(glasses_modes)
            det.forced_glasses = glasses_modes[mode_idx]
            print(f"[live_diag] Glasses mode set to: {g_mode_str}")
        elif key == ord("r"):
            det._reset_runs()
            print("[live_diag] Streak counters reset.")
        elif key == ord("s"):
            snap_idx += 1
            snap_path = f"demo/diag_snapshots/diag_snap_{snap_idx:03d}.png"
            cv2.imwrite(snap_path, frame)
            print(f"[live_diag] Snapshot saved -> {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[live_diag] Session finished.")


if __name__ == "__main__":
    main()
