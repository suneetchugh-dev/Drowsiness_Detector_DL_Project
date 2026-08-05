"""Tkinter launcher for the Drowsiness & Distraction Detection system.

Three modes: live webcam, upload a video, upload an image.
"""

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector, detect_image, detect_video, detect_webcam

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(BASE, "demo")
os.makedirs(OUT_DIR, exist_ok=True)


class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.det = DrowsinessDetector(use_yolo=False)
        self.q = queue.Queue()

        root.title("Driver Drowsiness & Distraction Detection")
        root.geometry("560x420")
        root.resizable(False, False)

        title = tk.Label(root, text="Driver Drowsiness & Distraction Detection",
                         font=("Segoe UI", 14, "bold"))
        title.pack(pady=12)

        info = tk.Label(root, text="Select an input mode below.", fg="#444")
        info.pack(pady=2)

        frame = ttk.Frame(root)
        frame.pack(pady=18)

        ttk.Button(frame, text="\u25b6  Live Webcam",
                   command=self.start_webcam, width=28).pack(pady=6)
        ttk.Button(frame, text="\u2191  Upload Video",
                   command=self.start_video, width=28).pack(pady=6)
        ttk.Button(frame, text="\U0001f5bc  Upload Image",
                   command=self.start_image, width=28).pack(pady=6)

        self.status = tk.Label(root, text="Ready.", fg="green",
                               font=("Segoe UI", 10), justify="left", anchor="w")
        self.status.pack(fill="x", padx=16, pady=8)
        self.log = tk.Text(root, height=7, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", padx=16, pady=(0, 10))

        ttk.Button(root, text="Quit", command=root.destroy).pack(pady=(0, 10))

        self.root.after(100, self._poll)

    # ------------------------------------------------------------ helpers
    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _set_status(self, msg, color="green"):
        self.status.config(text=msg, fg=color)

    def _worker(self, fn):
        def run():
            try:
                fn()
            except Exception as e:
                self.q.put(("error", str(e)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._set_status(payload)
                elif kind == "error":
                    self._set_status("Error: " + payload, "red")
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------ modes
    def start_webcam(self):
        self._set_status("Starting webcam ... (press 'q' in the video window to stop)",
                         "blue")
        self._worker(lambda: self._run_webcam())

    def _run_webcam(self):
        self._log("live webcam detection started")
        detect_webcam(self.det)
        self.q.put(("status", "Webcam session finished."))
        self.q.put(("log", "webcam stopped"))

    def start_video(self):
        path = filedialog.askopenfilename(
            title="Select a video", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not path:
            return
        out = os.path.join(OUT_DIR, "annotated_" + os.path.basename(path))
        self._set_status("Processing video ...", "blue")
        self._log(f"video: {path}\noutput: {out}")
        self._worker(lambda: self._run_video(path, out))

    def _run_video(self, path, out):
        stats = detect_video(self.det, path, out)
        msg = f"Video done: {stats['frames']} frames | drowsy={stats['drowsy']} " \
              f"| yawn={stats['yawn']} | distracted={stats['distracted']}"
        self._log(msg)
        self.q.put(("status", f"Saved annotated video -> {out}"))
        self.q.put(("log", "opening result ..."))
        os.startfile(out)

    def start_image(self):
        path = filedialog.askopenfilename(
            title="Select an image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not path:
            return
        out = os.path.join(OUT_DIR, "annotated_" + os.path.basename(path))
        self._set_status("Processing image ...", "blue")
        self._log(f"image: {path}\noutput: {out}")
        self._worker(lambda: self._run_image(path, out))

    def _run_image(self, path, out):
        annotated, status = detect_image(self.det, path, out)
        self._log(f"image result: {status}")
        self.q.put(("status", f"Saved annotated image -> {out}"))
        self.q.put(("log", "opening result ..."))
        os.startfile(out)


def main():
    root = tk.Tk()
    DrowsinessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
