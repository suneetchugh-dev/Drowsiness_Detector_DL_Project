"""Console menu for Driver Drowsiness Detection.

Run it via Drowsiness_Detector.bat (or directly: python -u src/menu.py).
Options: Launch GUI / Live Webcam / Upload Video / Upload Image / Quit.
After a mode finishes you return here, so you can re-run without restarting.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector, detect_image, detect_video, detect_webcam

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(BASE, "demo")
os.makedirs(OUT_DIR, exist_ok=True)

MENU = """\
========================================
     Driver Drowsiness Detection
========================================
  1. Launch GUI
  2. Run Live Webcam
  3. Upload Video
  4. Upload Image
  5. Quit
----------------------------------------
"""

GLASSES_LABEL = {0: "no glasses (fixed)", 1: "glasses (fixed)", None: "auto-detect"}


def ask_glasses():
    """Ask once whether the driver wears glasses, so we don't have to re-check
    every frame. Returns 1 (glasses), 0 (no glasses) or None (auto-detect)."""
    print("\nAre you wearing glasses?")
    print("   y - yes  -> use the glasses model (no per-frame re-check)")
    print("   n - no   -> use the normal model")
    print("   s - skip -> auto-detect glasses every frame")
    while True:
        ans = input("Your choice [y/n/s]: ").strip().lower()
        if ans in ("y", "yes"):
            return 1
        if ans in ("n", "no"):
            return 0
        if ans in ("s", "skip", ""):
            return None
        print("Please enter 'y', 'n' or 's'.")


def pick_file(title, kinds):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=kinds)
    root.destroy()
    return path


def run_gui(glasses_mode):
    print("Launching GUI ... close the window to come back to this menu.")
    import gui
    gui.main(glasses_mode)


def run_webcam(det):
    print("Live webcam detection started - press 'q' in the video window to stop.")
    detect_webcam(det)
    print("Webcam session finished.")


def run_video(det):
    path = pick_file("Select a video", [("Video files", "*.mp4 *.avi *.mov")])
    if not path:
        print("No file selected - returning to menu.")
        return
    out = os.path.join(OUT_DIR, "annotated_" + os.path.basename(path))
    print(f"Processing video: {path}")
    stats = detect_video(det, path, out)
    print(f"Done: {stats['frames']} frames | drowsy={stats['drowsy']} "
          f"| yawn={stats['yawn']}")
    print(f"Annotated video saved -> {out}")
    os.startfile(out)


def run_image(det):
    path = pick_file("Select an image", [("Image files", "*.png *.jpg *.jpeg")])
    if not path:
        print("No file selected - returning to menu.")
        return
    out = os.path.join(OUT_DIR, "annotated_" + os.path.basename(path))
    print(f"Processing image: {path}")
    _, status = detect_image(det, path, out)
    print(f"Result: {status}")
    print(f"Annotated image saved -> {out}")
    os.startfile(out)


def main():
    glasses_mode = ask_glasses()
    det = DrowsinessDetector(use_yolo=False, glasses_mode=glasses_mode)
    print(f"\nGlasses mode: {GLASSES_LABEL[glasses_mode]}")
    while True:
        print(MENU)
        opt = input("Choose an option (1-5, or press 'q' to quit): ").strip().lower()
        if opt in ("q", "5"):
            print("Goodbye.")
            return
        if opt == "1":
            run_gui(glasses_mode)
        elif opt == "2":
            run_webcam(det)
        elif opt == "3":
            run_video(det)
        elif opt == "4":
            run_image(det)
        else:
            print("Invalid option.")
            continue
        input("\nPress Enter to return to the menu, or type 'q' and Enter to quit: ")


if __name__ == "__main__":
    main()
