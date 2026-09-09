"""Embedded Tkinter GUI for the Driver Drowsiness Detection system.

The live webcam feed is embedded directly inside the Tkinter window (via PIL/
ImageTk) rather than shown in a separate OpenCV window, so we can overlay a
modern, sleek in-feed "Stop Detection" control.

Input modes
  - Live webcam (embedded, resizable, in-feed stop button)
  - Upload a video
  - Upload an image

Resolution control
  - A modern segmented Eye-model toggle (Auto / On / Off). Defaults to Auto.

The UI uses a flat, monochromatic, high-contrast professional theme.
"""

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DrowsinessDetector, detect_image, detect_video

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(BASE, "demo")
os.makedirs(OUT_DIR, exist_ok=True)

GLASSES_MODE = {"auto": None, "on": 1, "off": 0}

APP_FONT = ("Segoe UI", 10)
APP_FONT_BOLD = ("Segoe UI", 10, "bold")
TITLE_FONT = ("Segoe UI", 17, "bold")
MONO_FONT = ("Consolas", 9)

FEED_MAX_W = 880
FEED_MAX_H = 560
WELCOME_TEXT = ("Start the webcam to see the live detection feed.\n"
                "Use Stop Detection to end the session.")


class MonochromeTheme:
    """Centralised monochromatic palette (grayscale)."""

    BG = "#F7F7F7"
    SURFACE = "#FFFFFF"
    FG = "#1B1B1B"
    MUTED = "#6E6E6E"
    BORDER = "#D9D9D9"
    ACCENT = "#232323"
    ACCENT_ACTIVE = "#3E3E3E"
    FIELD_BG = "#EFEFEF"
    ERROR = "#8C1F28"
    SEL_BG = "#E4E4E4"
    DISABLED_FG = "#B0B0B0"
    WHITE = "#FFFFFF"
    FEED_BG = "#000000"


class StatusBar(tk.Frame):
    """Slim status bar with monochromatic styling."""

    def __init__(self, master, **kwargs):
        super().__init__(master, bg=MonochromeTheme.BG, **kwargs)
        self._label = tk.Label(
            self, text="Ready", font=APP_FONT, anchor="w",
            bg=MonochromeTheme.BG, fg=MonochromeTheme.MUTED,
            padx=14, pady=8,
        )
        self._label.pack(fill="x")

    def set(self, text, kind="info"):
        color = {
            "info": MonochromeTheme.MUTED,
            "ok": MonochromeTheme.FG,
            "error": MonochromeTheme.ERROR,
            "busy": MonochromeTheme.ACCENT,
        }.get(kind, MonochromeTheme.MUTED)
        self._label.config(text=text, fg=color)


class LogPanel(tk.Frame):
    """Read-only monospaced activity log.

    Each line is timestamped and colour-coded by severity:
      info  -> grey, ok -> dark (success), busy -> accent, error -> red.
    A slim header provides a collapse/expand toggle and a Clear action.
    """

    def __init__(self, master, height=7, **kwargs):
        super().__init__(master, bg=MonochromeTheme.SURFACE, **kwargs)
        self._collapsed = False

        header = tk.Frame(self, bg=MonochromeTheme.SURFACE)
        header.pack(fill="x", pady=(0, 4))
        self._toggle_btn = tk.Button(
            header, text="\u25bc  Output Log", font=APP_FONT_BOLD,
            relief="flat", bd=0, bg=MonochromeTheme.SURFACE,
            fg=MonochromeTheme.FG, activebackground=MonochromeTheme.SEL_BG,
            activeforeground=MonochromeTheme.FG, cursor="hand2",
            command=self._toggle)
        self._toggle_btn.pack(side="left")
        clear_btn = tk.Button(
            header, text="Clear", font=APP_FONT,
            relief="flat", bd=0, bg=MonochromeTheme.SURFACE,
            fg=MonochromeTheme.MUTED, activebackground=MonochromeTheme.SEL_BG,
            activeforeground=MonochromeTheme.FG, cursor="hand2",
            command=self.clear)
        clear_btn.pack(side="right")

        self._body = tk.Frame(self, bg=MonochromeTheme.SURFACE)
        self._body.pack(fill="both", expand=True)

        self._text = tk.Text(
            self._body, height=height, state="disabled",
            font=MONO_FONT, wrap="word",
            bg=MonochromeTheme.FIELD_BG, fg=MonochromeTheme.FG,
            relief="flat", padx=12, pady=8, borderwidth=0,
            highlightthickness=1, highlightbackground=MonochromeTheme.BORDER,
            highlightcolor=MonochromeTheme.BORDER,
        )
        scroll = ttk.Scrollbar(self._body, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=scroll.set)
        self._text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._text.tag_configure("time", foreground=MonochromeTheme.MUTED)
        self._text.tag_configure("info", foreground=MonochromeTheme.MUTED)
        self._text.tag_configure("ok", foreground=MonochromeTheme.FG)
        self._text.tag_configure("busy", foreground=MonochromeTheme.ACCENT)
        self._text.tag_configure("error", foreground=MonochromeTheme.ERROR)

    def _toggle(self):
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._body.pack_forget()
            self._toggle_btn.configure(text="\u25b6  Output Log")
        else:
            self._body.pack(fill="both", expand=True)
            self._toggle_btn.configure(text="\u25bc  Output Log")

    def append(self, msg, kind="info"):
        self._text.config(state="normal")
        stamp = time.strftime("%H:%M:%S")
        self._text.insert("end", "[%s] " % stamp, "time")
        self._text.insert("end", msg + "\n", kind)
        self._text.see("end")
        self._text.config(state="disabled")

    def clear(self):
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.config(state="disabled")


class SegmentedControl(tk.Frame):
    """Modern flat segmented control (single-select, non-wrapping).

    Renders as a row of connected toggle buttons. The active segment is shown
    with the accent (dark) fill; inactive segments use the light surface fill.
    """

    def __init__(self, master, options, variable, command=None, **kwargs):
        super().__init__(master, bg=MonochromeTheme.SURFACE, **kwargs)
        self._seg_options = list(options)
        self._variable = variable
        self._command = command
        self._buttons = []

        for i, (value, label) in enumerate(self._seg_options):
            btn = tk.Button(
                self, text=label, font=APP_FONT_BOLD,
                relief="flat", borderwidth=0, bd=0,
                padx=18, pady=6, cursor="hand2",
                highlightthickness=0, activebackground=MonochromeTheme.ACCENT_ACTIVE,
                activeforeground=MonochromeTheme.WHITE,
                command=lambda v=value: self._select(v),
            )
            btn.grid(row=0, column=i, sticky="nsew")
            self._buttons.append(btn)
            if i < len(self._seg_options) - 1:
                ttk.Separator(self, orient="vertical").grid(
                    row=0, column=i, sticky="ns")

        for c in range(len(self._seg_options)):
            self.columnconfigure(c, weight=1)
        self._refresh()

    def _select(self, value):
        if self._variable.get() != value:
            self._variable.set(value)
            self._refresh()
            if self._command:
                self._command()

    def _refresh(self):
        current = self._variable.get()
        for btn, (value, _) in zip(self._buttons, self._seg_options):
            if value == current:
                btn.configure(
                    bg=MonochromeTheme.ACCENT, fg=MonochromeTheme.WHITE,
                    activebackground=MonochromeTheme.ACCENT_ACTIVE)
            else:
                btn.configure(
                    bg=MonochromeTheme.SURFACE, fg=MonochromeTheme.FG,
                    activebackground=MonochromeTheme.SEL_BG,
                    activeforeground=MonochromeTheme.FG)

    def set(self, value):
        self._variable.set(value)
        self._refresh()

    def selected(self):
        return self._variable.get()


class FlatButton(tk.Button):
    """Uniform modern flat button (tk.Button so styling is fully controlled)."""

    def __init__(self, master, text, command=None, accent=False, width=None,
                 **kwargs):
        super().__init__(
            master, text=text, font=APP_FONT_BOLD,
            relief="flat", borderwidth=0, bd=0,
            padx=16, pady=9, cursor="hand2",
            highlightthickness=0,
            bg=MonochromeTheme.ACCENT if accent else MonochromeTheme.FIELD_BG,
            fg=MonochromeTheme.WHITE if accent else MonochromeTheme.FG,
            activebackground=(MonochromeTheme.ACCENT_ACTIVE
                              if accent else MonochromeTheme.SEL_BG),
            activeforeground=MonochromeTheme.WHITE if accent else MonochromeTheme.FG,
            disabledforeground=MonochromeTheme.DISABLED_FG,
            command=command,
        )
        if width:
            self.configure(width=width)

    def set_accent(self, accent):
        self.configure(
            bg=MonochromeTheme.ACCENT if accent else MonochromeTheme.FIELD_BG,
            fg=MonochromeTheme.WHITE if accent else MonochromeTheme.FG,
            activebackground=(MonochromeTheme.ACCENT_ACTIVE
                              if accent else MonochromeTheme.SEL_BG),
            activeforeground=(MonochromeTheme.WHITE if accent
                              else MonochromeTheme.FG))


def _powershell(args):
    """Run a PowerShell command silently. Returns stdout text (or None)."""
    if sys.platform != "win32":
        return None
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             args],
            capture_output=True, text=True, timeout=15,
            creationflags=flags)
        return proc.stdout.strip()
    except Exception:
        return None


def _screen_brightness_get():
    val = _powershell(
        "(Get-CimInstance -Namespace root/WMI -ClassName "
        "WmiMonitorBrightness).CurrentBrightness")
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _screen_brightness_set(pct):
    pct = max(0, min(100, int(pct)))
    _powershell(
        "(Get-WmiObject -Namespace root/WMI -Class "
        "WmiMonitorBrightnessMethods).WmiSetBrightness(1, %d)" % pct)


class DrowsinessApp:
    def __init__(self, root, glasses_mode=None):
        self.root = root
        self.root.title("Driver Drowsiness Detection")
        self.root.geometry("720x760")
        self.root.minsize(680, 720)
        self.root.configure(bg=MonochromeTheme.BG)

        self.force_glasses = glasses_mode
        self.glasses_var = tk.StringVar(value=self._glasses_default())
        self._flash_var = tk.StringVar(value="on")   # screen light default: on
        self._auto_checked = False                   # auto low-light evaluated?
        self.det = None
        self.q = queue.Queue()          # worker -> UI messages
        self.frame_q = queue.Queue()    # camera thread -> UI frames
        self.stop_event = threading.Event()
        self._cam = None
        self._cam_thread = None
        self._feed_img = None           # keep reference to PhotoImage
        self._feed_open = False
        self._fullscreen = False
        self._brightness_restore = None  # original screen brightness to restore

        self._build_topbar()
        self._build_settings()
        self._build_actions()
        self._build_feed_panel()
        self._build_log()
        self._build_statusbar()

        self.root.after(40, self._poll_frames)
        self.root.after(100, self._poll)
        self._init_detector()

    # ------------------------------------------------------------- top bar
    def _build_topbar(self):
        header = tk.Frame(self.root, bg=MonochromeTheme.BG)
        header.pack(fill="x", padx=20, pady=(20, 6))
        tk.Label(header, text="Driver Drowsiness Detection",
                 font=TITLE_FONT, fg=MonochromeTheme.FG,
                 bg=MonochromeTheme.BG).pack(anchor="w")
        tk.Label(header, text="Live monitoring and offline analysis",
                 font=APP_FONT, fg=MonochromeTheme.MUTED,
                 bg=MonochromeTheme.BG).pack(anchor="w", pady=(2, 0))
        ttk.Separator(self.root, orient="horizontal").pack(
            fill="x", padx=20, pady=(12, 0))

    # ---------------------------------------------------------- settings
    def _build_settings(self):
        card = tk.Frame(self.root, bg=MonochromeTheme.SURFACE,
                        highlightthickness=1,
                        highlightbackground=MonochromeTheme.BORDER)
        card.pack(fill="x", padx=20, pady=(16, 4))

        body = tk.Frame(card, bg=MonochromeTheme.SURFACE)
        body.pack(fill="x", padx=16, pady=12)

        tk.Label(body, text="Eye glasses model",
                 font=APP_FONT_BOLD, fg=MonochromeTheme.FG,
                 bg=MonochromeTheme.SURFACE).pack(anchor="w", pady=(0, 8))

        row = tk.Frame(body, bg=MonochromeTheme.SURFACE)
        row.pack(fill="x")

        self.btn_glasses = FlatButton(row, self._glasses_button_text(),
                                      command=self._toggle_glasses)
        self.btn_glasses.pack(side="left", fill="x", expand=True)

        tk.Label(row, text="  Tap to switch. Auto detects each frame; "
                           "On / Off fix the model at startup.",
                 font=("Segoe UI", 9), fg=MonochromeTheme.MUTED,
                 bg=MonochromeTheme.SURFACE).pack(side="left")

        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=(12, 12))

        tk.Label(body, text="Screen visibility light",
                 font=APP_FONT_BOLD, fg=MonochromeTheme.FG,
                 bg=MonochromeTheme.SURFACE).pack(anchor="w", pady=(0, 8))

        row2 = tk.Frame(body, bg=MonochromeTheme.SURFACE)
        row2.pack(fill="x")

        self.btn_flash_mode = FlatButton(row2, self._flash_button_text(),
                                         command=self._toggle_flash)
        self.btn_flash_mode.pack(side="left", fill="x", expand=True)

        tk.Label(row2, text="  Max screen brightness as a fill light. "
                            "Auto boosts only in low light.",
                 font=("Segoe UI", 9), fg=MonochromeTheme.MUTED,
                 bg=MonochromeTheme.SURFACE).pack(side="left")

    # ---------------------------------------------------------- glasses
    def _glasses_button_text(self):
        return "Eye glasses: %s" % self.glasses_var.get().capitalize()

    def _toggle_glasses(self):
        if self.glasses_var.get() == "auto":
            self.glasses_var.set("on")
        elif self.glasses_var.get() == "on":
            self.glasses_var.set("off")
        else:
            self.glasses_var.set("auto")
        self.btn_glasses.configure(text=self._glasses_button_text())
        self._on_glasses_change()

    # ------------------------------------------------------ screen light
    def _flash_button_text(self):
        return "Screen light: %s" % self._flash_var.get().capitalize()

    def _toggle_flash(self):
        if self._flash_var.get() == "on":
            self._flash_var.set("auto")
        elif self._flash_var.get() == "auto":
            self._flash_var.set("off")
        else:
            self._flash_var.set("on")
        self.btn_flash_mode.configure(text=self._flash_button_text())
        mode = self._flash_var.get()
        if self._feed_open:
            if mode == "on":
                self._raise_screen_brightness()
                self.screen_flash()
            elif mode == "off":
                self._restore_screen_brightness()
            else:
                self._auto_checked = False
            self._set_status("Screen light: %s (applied to live feed)" % mode,
                             "info")
        else:
            self._set_status("Screen light: %s (applies on webcam start)" % mode,
                             "info")

    # ---------------------------------------------------------- actions
    def _build_actions(self):
        actions = tk.Frame(self.root, bg=MonochromeTheme.BG)
        actions.pack(fill="x", padx=20, pady=(14, 6))

        grid = tk.Frame(actions, bg=MonochromeTheme.BG)
        grid.pack(fill="x")

        self.btn_webcam = FlatButton(grid, "Live Webcam", command=self.start_webcam,
                                     accent=True)
        self.btn_webcam.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.btn_video = FlatButton(grid, "Upload Video", command=self.start_video)
        self.btn_video.grid(row=0, column=1, sticky="ew", padx=(6, 6))

        self.btn_image = FlatButton(grid, "Upload Image", command=self.start_image)
        self.btn_image.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)

    # -------------------------------------------------------- feed panel
    def _build_feed_panel(self):
        self.panel = tk.Frame(self.root, bg=MonochromeTheme.BG)
        self.panel.pack(fill="both", expand=True, padx=20, pady=(10, 6))

        head = tk.Frame(self.panel, bg=MonochromeTheme.BG)
        head.pack(fill="x")
        tk.Label(head, text="Live Feed", font=APP_FONT_BOLD,
                 fg=MonochromeTheme.FG, bg=MonochromeTheme.BG).pack(side="left")

        view_controls = tk.Frame(head, bg=MonochromeTheme.BG)
        view_controls.pack(side="right")
        self.btn_flash = FlatButton(view_controls, "Screen Flash",
                                    command=self.screen_flash)
        self.btn_flash.pack(side="left", padx=(0, 4))
        self.btn_view = FlatButton(view_controls, "Fullscreen",
                                   command=self.toggle_fullscreen)
        self.btn_view.pack(side="left")

        self.feed = tk.Frame(self.panel, bg=MonochromeTheme.FEED_BG,
                             highlightthickness=1,
                             highlightbackground=MonochromeTheme.BORDER)
        self.feed.pack(fill="both", expand=True, pady=(8, 0))

        self.feed_label = tk.Label(self.feed, bg=MonochromeTheme.FEED_BG,
                                   compound="center")
        self.feed_label.pack(fill="both", expand=True)

        self.btn_stop = FlatButton(self.feed, "Start Detection",
                                   command=self.toggle_webcam)
        self.btn_stop.configure(state="disabled")
        self.btn_stop.place(relx=1.0, rely=0.0, anchor="ne", x=-12, y=12)

        self.panel.bind("<Configure>", self._on_feed_resize)
        self._show_welcome()

    def toggle_fullscreen(self):
        self.set_view("windowed" if self._fullscreen else "fullscreen")

    def set_view(self, mode):
        if mode == "fullscreen":
            self._fullscreen = True
            self.root.attributes("-fullscreen", True)
            self.btn_view.configure(text="Exit Fullscreen",
                                    state="normal")
            self.btn_view.set_accent(True)
        else:
            self._fullscreen = False
            self.root.attributes("-fullscreen", False)
            self.btn_view.configure(text="Fullscreen", state="normal")
            self.btn_view.set_accent(False)

    def _on_feed_resize(self, event):
        # Frames are re-fitted continuously by _set_feed, so resizing the
        # window / toggling fullscreen is reflected automatically.
        pass

    def _show_welcome(self):
        self._feed_open = False
        self.feed_label.configure(
            image="", text=WELCOME_TEXT,
            fg=MonochromeTheme.MUTED, font=("Segoe UI", 11),
            bg=MonochromeTheme.FEED_BG)

    def _set_feed(self, bgr, flip=False):
        try:
            if flip:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = self._fit(img)
            self._feed_img = ImageTk.PhotoImage(img)
            self.feed_label.configure(image=self._feed_img, text="",
                                      bg=MonochromeTheme.FEED_BG)
        except Exception:
            pass

    def _fit(self, img):
        w, h = img.size
        try:
            fw = max(self.feed.winfo_width(), 40)
            fh = max(self.feed.winfo_height(), 40)
        except Exception:
            fw, fh = FEED_MAX_W, FEED_MAX_H
        ratio = min(fw / w, fh / h)
        new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
        return img.resize((new_w, new_h), Image.BILINEAR)

    # -------------------------------------------------------------- log
    def _build_log(self):
        log_frame = tk.Frame(self.root, bg=MonochromeTheme.BG)
        log_frame.pack(fill="both", expand=True, padx=20)
        tk.Label(log_frame, text="Output Log", font=APP_FONT_BOLD,
                 fg=MonochromeTheme.FG, bg=MonochromeTheme.BG).pack(
                     anchor="w", pady=(8, 6))
        self.log = LogPanel(log_frame, height=7)
        self.log.pack(fill="both", expand=True)

    # -------------------------------------------------------- statusbar
    def _build_statusbar(self):
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=20,
                                                           pady=(8, 0))
        self.status = StatusBar(self.root)
        self.status.pack(fill="x", side="bottom")

    # ----------------------------------------------------------- helpers
    def _glasses_default(self):
        return "auto" if self.force_glasses is None else \
            ("on" if self.force_glasses else "off")

    def _init_detector(self):
        mode = self.glasses_var.get()
        self._set_actions_state("disabled")
        self.status.set("Initialising detection models ...", "busy")
        threading.Thread(target=self._load_detector, args=(mode,),
                         daemon=True).start()

    def _load_detector(self, mode):
        try:
            glasses_mode = GLASSES_MODE[mode]
            self.det = DrowsinessDetector(use_yolo=False, glasses_mode=glasses_mode)
            self.q.put(("ready", self._glasses_label(glasses_mode)))
        except Exception as e:
            self.q.put(("error", str(e)))

    @staticmethod
    def _glasses_label(mode):
        return {0: "Off (no-glasses model)", 1: "On (glasses model)",
                None: "Auto (per-frame)"}[mode]

    def _on_glasses_change(self):
        if self.det is None:
            return
        if self._feed_open:
            self.stop_webcam(silent=True)
        mode = self.glasses_var.get()
        self.status.set("Reloading detector with glasses = %s ..." % mode, "busy")
        self._set_actions_state("disabled")
        self.det = None
        threading.Thread(target=self._load_detector, args=(mode,),
                         daemon=True).start()

    def _set_actions_state(self, state):
        for btn in (self.btn_webcam, self.btn_video, self.btn_image):
            btn.configure(state=state)

    def _log(self, msg, kind="info"):
        self.log.append(msg, kind)

    def _set_status(self, msg, kind="info"):
        self.status.set(msg, kind)

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
                if kind == "ready":
                    self._set_actions_state("normal")
                    self._set_status("Ready. Eye model: %s." % payload, "ok")
                elif kind == "status":
                    self._set_status(payload, "busy")
                elif kind == "log":
                    self._log(payload, kind="info")
                elif kind == "log_ok":
                    self._log(payload, kind="ok")
                elif kind == "log_busy":
                    self._log(payload, kind="busy")
                elif kind == "log_err":
                    self._log(payload, kind="error")
                elif kind == "done":
                    self._set_status(payload, "ok")
                elif kind == "rearm":
                    self._set_actions_state("normal")
                elif kind == "cam_closed":
                    self._on_cam_closed(payload)
                elif kind == "lit":
                    self._toggle_flash_apply()
                elif kind == "error":
                    self._set_status("Error: " + payload, "error")
                    self._log(payload, kind="error")
                    self._set_actions_state("normal")
                    if not self._feed_open:
                        messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _poll_frames(self):
        try:
            while True:
                bgr = self.frame_q.get_nowait()
                if bgr is not None:
                    self._set_feed(bgr)
        except queue.Empty:
            pass
        self.root.after(40, self._poll_frames)

    # ----------------------------------------------------------- webcam
    def start_webcam(self):
        if self.det is None:
            self._warn_not_ready()
            return
        if self._feed_open:
            return
        self.stop_event.clear()
        self._set_status("Starting webcam ...", "busy")
        self._set_actions_state("disabled")
        self._set_stop_button("Stop Detection", "normal")
        self.btn_webcam.set_accent(False)
        self.btn_flash.configure(state="normal")
        self.feed_label.configure(text="Initialising camera ...",
                                  fg=MonochromeTheme.MUTED,
                                  font=("Segoe UI", 11))
        self._cam = cv2.VideoCapture(0)
        if not self._cam.isOpened():
            self._show_welcome()
            self._set_status("Webcam not available.", "error")
            messagebox.showerror("Error", "Webcam not available")
            self._set_actions_state("normal")
            self._set_stop_button("Start Detection", "disabled")
            self.btn_webcam.set_accent(True)
            self.btn_flash.configure(state="disabled")
            return
        self._feed_open = True
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()
        self._boost_camera_sensor()
        mode = self._flash_var.get()
        if mode == "on":
            self._raise_screen_brightness()
        elif mode == "auto":
            self._auto_checked = False
        self._log("live webcam detection started (screen light: %s)" % mode,
                  kind="ok")

    def _cam_loop(self):
        print("live detection started - press the Stop button to quit")
        detector = self.det
        while not self.stop_event.is_set():
            ok, frame = self._cam.read()
            if not ok:
                break
            # Auto screen light: evaluate ambient light on the first frame and
            # decide once whether the panel needs to act as a fill light.
            if self._flash_var.get() == "auto" and not self._auto_checked:
                self._auto_checked = True
                lum = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                if lum < 80:
                    self.q.put(("lit", True))
                else:
                    self.q.put(("log_ok", "lighting adequate - screen fill "
                                          "light not needed"))
            try:
                annotated, _ = detector.process_frame(frame)
            except Exception as e:
                self.q.put(("error", str(e)))
                break
            try:
                self.frame_q.put_nowait(annotated)
            except queue.Full:
                pass
        self._cam.release()
        self.frame_q.put_nowait(None)
        self.q.put(("cam_closed", "Webcam session stopped."))

    def _boost_camera_sensor(self):
        """Industry low-light practice: ask the sensor for more gain/brightness.

        Many UVC webcams ignore these properties, so failures are silent and
        harmless -- the detector already normalises every crop per-image.
        """
        try:
            cam = self._cam
            cam.set(cv2.CAP_PROP_BRIGHTNESS, 130)
            cam.set(cv2.CAP_PROP_GAIN, 130)
        except cv2.error:
            pass

    def _raise_screen_brightness(self):
        """Raise the panel to 100% as a fill light for the webcam user.

        The monitor brightness API only exists on laptops/ACPI panels; on
        desktop monitors it is unsupported and we just skip it."""
        def work():
            current = _screen_brightness_get()
            if current is not None:
                self._brightness_restore = current
                _screen_brightness_set(100)
                self.q.put(("log_ok", "screen brightness raised to 100% "
                                    "(fill light for webcam)"))
            else:
                self.q.put(("log_busy", "screen brightness control not "
                                        "supported on this display"))
        threading.Thread(target=work, daemon=True).start()

    def _toggle_flash_apply(self):
        """Auto-mode handler: low light detected -> use the panel as fill light."""
        self._raise_screen_brightness()
        self.screen_flash()
        self._set_status("Low light detected - screen fill light enabled.", "ok")

    def _restore_screen_brightness(self):
        def work():
            val = self._brightness_restore
            self._brightness_restore = None
            if val is not None:
                _screen_brightness_set(val)
                self.q.put(("log_busy", "screen brightness restored to %d%%" % val))
        if self._brightness_restore is not None:
            threading.Thread(target=work, daemon=True).start()

    def screen_flash(self):
        """Simulate a selfie-camera flash.

        Laptops/desktops do not ship a hardware LED flash for webcams; the
        industry-standard equivalent is a brief, full-white screen flash
        (the phone-style 'flash' behaviour) that fills the face with light.
        Sizing via geometry (not -fullscreen) so it maps reliably on Windows."""
        try:
            flash = tk.Toplevel(self.root)
            flash.overrideredirect(True)
            flash.configure(bg="#FFFFFF")
            w = flash.winfo_screenwidth()
            h = flash.winfo_screenheight()
            flash.geometry("%dx%d+0+0" % (w, h))
            flash.attributes("-topmost", True)
            flash.lift()
            flash.after(160, flash.destroy)
        except tk.TclError:
            pass

    def toggle_webcam(self):
        if self._feed_open:
            self.stop_webcam()
        else:
            self.start_webcam()

    def stop_webcam(self, silent=False):
        if not self._feed_open:
            if silent:
                return
            return
        self._feed_open = False
        self.stop_event.set()

    def _set_stop_button(self, text, state):
        self.btn_stop.configure(text=text, state=state)

    def _on_cam_closed(self, msg):
        self._cam = None
        self._cam_thread = None
        self._show_welcome()
        self._set_stop_button("Start Detection", "normal")
        self.btn_webcam.set_accent(True)
        self.btn_flash.configure(state="disabled")
        self._restore_screen_brightness()
        self._set_actions_state("normal")
        self._set_status(msg, "ok")
        self._log("webcam stopped", kind="busy")

    def on_close(self):
        if self._feed_open:
            self.stop_event.set()
            if self._cam is not None:
                try:
                    self._cam.release()
                except Exception:
                    pass
        self.root.destroy()

    # ----------------------------------------------------------- video
    def start_video(self):
        if self.det is None:
            self._warn_not_ready()
            return
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not path:
            return
        out = os.path.join(OUT_DIR, "annotated_" + os.path.basename(path))
        self._set_status("Processing video ...", "busy")
        self._set_actions_state("disabled")
        self._log("video: %s" % path, kind="busy")
        self._log("output: %s" % out, kind="busy")
        self._worker(lambda: self._run_video(path, out))

    def _run_video(self, path, out):
        stats = detect_video(self.det, path, out)
        msg = ("Video done: %d frames | drowsy=%d | yawn=%d"
               % (stats["frames"], stats["drowsy"], stats["yawn"]))
        self._log(msg, kind="ok")
        self.q.put(("done", "Saved annotated video -> %s" % out))
        self.q.put(("log_busy", "opening result ..."))
        self.q.put(("rearm", ""))
        os.startfile(out)

    # ----------------------------------------------------------- image
    def start_image(self):
        if self.det is None:
            self._warn_not_ready()
            return
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not path:
            return
        out = os.path.join(OUT_DIR, "annotated_" + os.path.basename(path))
        self._set_status("Processing image ...", "busy")
        self._set_actions_state("disabled")
        self._log("image: %s" % path, kind="busy")
        self._worker(lambda: self._run_image(path, out))

    def _run_image(self, path, out):
        _, status = detect_image(self.det, path, out)
        self._log("image result: %s" % status, kind="ok")
        self.q.put(("done", "Saved annotated image -> %s" % out))
        self.q.put(("log_busy", "opening result ..."))
        self.q.put(("rearm", ""))
        os.startfile(out)

    def _warn_not_ready(self):
        messagebox.showinfo("Not Ready", "Models are still loading. Please wait.")


def main(glasses_mode=None):
    root = tk.Tk()
    app = DrowsinessApp(root, glasses_mode=glasses_mode)
    root.bind("<Escape>", lambda e: app.set_view("windowed"))
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
