"""tarina_gui.py — Native Python Desktop GUI for TarinaSR.

Place in the root of the TarinaSR repository (next to tarina.py, sihtisr.py, model.json).
Requires: pip install numpy opencv-python pillow
Run:      python3.13 tarina_gui.py
"""
from __future__ import annotations

import inspect
import json
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import sihtisr
import tarina

LW, LH = 240, 180
HW, HH = LW * 2, LH * 2
MARGIN = 24
DEMO_FRAMES = 200


# ---------- Dynamic module adapters (handles any function naming in sihtisr.py) ----------
def call_bicubic_up(y: np.ndarray) -> np.ndarray:
    for name in ("bicubic_up", "bicubic", "up_bicubic"):
        fn = getattr(sihtisr, name, None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) == 1:
                    return np.asarray(fn(y), dtype=np.float64)
                return np.asarray(fn(y, 2), dtype=np.float64)
            except Exception:
                pass
    return cv2.resize(y, (HW, HH), interpolation=cv2.INTER_CUBIC).astype(np.float64)


def _builtin_sihti_up(y: np.ndarray, model: dict) -> np.ndarray:
    """Direct NumPy implementation of the 216-bucket SihtiSR model.json lookup."""
    h, w = y.shape
    W = np.asarray(model["W"], dtype=np.float64)  # shape (216, 25, 4)
    sthr = model["sthr"]
    coh_thr = model["coh"]
    na, ns, nc = int(model["na"]), int(model["ns"]), int(model["nc"])

    pad = np.pad(y, 2, mode="edge")
    # 5x5 patches for every pixel: shape (h, w, 25)
    patches = np.lib.stride_tricks.sliding_window_view(pad, (5, 5)).reshape(h, w, 25)

    # 3x3 Sobel-like gradients for bucket sorting
    gy = pad[2 : h + 2, 1 : w + 1] - pad[0:h, 1 : w + 1]
    gx = pad[1 : h + 1, 2 : w + 2] - pad[1 : h + 1, 0:w]

    # Structure tensor over 3x3 window
    gxx = cv2.boxFilter(gx * gx, -1, (3, 3), normalize=True)
    gyy = cv2.boxFilter(gy * gy, -1, (3, 3), normalize=True)
    gxy = cv2.boxFilter(gx * gy, -1, (3, 3), normalize=True)

    tr = gxx + gyy
    det = gxx * gyy - gxy * gxy
    disc = np.sqrt(np.maximum(0.0, (tr * 0.5) ** 2 - det))
    l1 = np.maximum(0.0, tr * 0.5 + disc)
    l2 = np.maximum(0.0, tr * 0.5 - disc)

    strength = np.sqrt(l1)
    coherence = (np.sqrt(l1) - np.sqrt(l2)) / (np.sqrt(l1) + np.sqrt(l2) + 1e-12)
    angle = (np.arctan2(gxy, l1 - gyy) + math.pi) % math.pi

    ba = np.clip((angle / math.pi * na).astype(np.int32), 0, na - 1)
    bs = (strength >= sthr[0]).astype(np.int32) + (strength >= sthr[1]).astype(np.int32)
    bc = (coherence >= coh_thr[0]).astype(np.int32) + (coherence >= coh_thr[1]).astype(np.int32)
    bucket = (ba * ns + bs) * nc + bc  # (h, w) in 0..215

    # Apply per-bucket 25x4 filter
    filters = W[bucket]  # (h, w, 25, 4)
    quads = np.einsum("hwk,hwkc->hwc", patches, filters)  # (h, w, 4)

    # Exact Sihti 2x2 downsample conservation lock
    mean_err = y - np.mean(quads, axis=-1)
    quads = quads + mean_err[..., None]

    out = np.empty((h * 2, w * 2), dtype=np.float64)
    out[0::2, 0::2] = quads[:, :, 0]
    out[0::2, 1::2] = quads[:, :, 1]
    out[1::2, 0::2] = quads[:, :, 2]
    out[1::2, 1::2] = quads[:, :, 3]
    return np.clip(out, 0.0, 1.0)


def call_sihti_up(y: np.ndarray, model: dict) -> np.ndarray:
    # 1. Check if tarina.py already exposes the starter upscale helper
    for mod in (sihtisr, tarina):
        for name in (
            "sihti_up",
            "sihtiUp",
            "upscale",
            "up",
            "sr",
            "predict",
            "apply_model",
            "super_resolve",
            "run",
        ):
            fn = getattr(mod, name, None)
            if callable(fn):
                try:
                    res = fn(y, model)
                    if isinstance(res, tuple):
                        res = res[0]
                    arr = np.asarray(res, dtype=np.float64)
                    if arr.shape == (y.shape[0] * 2, y.shape[1] * 2):
                        return arr
                except TypeError:
                    try:
                        res = fn(y)
                        if isinstance(res, tuple):
                            res = res[0]
                        arr = np.asarray(res, dtype=np.float64)
                        if arr.shape == (y.shape[0] * 2, y.shape[1] * 2):
                            return arr
                    except Exception:
                        pass
                except Exception:
                    pass
    # 2. Fallback to built-in 216-bucket lookup
    return _builtin_sihti_up(y, model)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.clip(a[MARGIN : HH - MARGIN, MARGIN : HW - MARGIN], 0.0, 1.0)
    bb = b[MARGIN : HH - MARGIN, MARGIN : HW - MARGIN]
    mse = float(np.mean((aa - bb) ** 2))
    return -10.0 * math.log10(max(mse, 1e-12))


def make_chart_truth() -> np.ndarray:
    """Generate the Siemens star + shrinking text + line-pair test chart (480x360)."""
    im = Image.new("L", (HW, HH), color=214)
    dr = ImageDraw.Draw(im)

    cx, cy, r = 120, 130, 96
    for i in range(36):
        a0 = i * math.pi / 18.0
        a1 = (i + 0.5) * math.pi / 18.0
        pts = [(cx, cy)]
        for s in range(9):
            ang = a0 + (a1 - a0) * s / 8.0
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        dr.polygon(pts, fill=27)

    lines = [
        (22, "TarinaSR makes"),
        (17, "resolution from shake"),
        (13, "Every frame samples new points."),
        (11, "The ring writes what it got wrong"),
        (9, "until the frames stop disagreeing."),
        (8, "Irani & Peleg 1991 · Wronski 2019"),
    ]
    y = 22
    for sz, txt in lines:
        try:
            font = ImageFont.truetype("georgia.ttf", sz)
        except OSError:
            font = ImageFont.load_default()
        dr.text((236, y), txt, fill=21, font=font)
        y += sz + 8

    x = 236.0
    for p in (5.0, 4.0, 3.0, 2.5, 2.0, 1.5):
        for k in range(6):
            x0 = int(round(x + k * 2 * p))
            x1 = int(round(x + k * 2 * p + p))
            dr.rectangle([x0, 200, max(x0, x1 - 1), 270], fill=21)
        x += 12 * p + 8

    arr = np.asarray(im, dtype=np.float64) / 255.0
    for yy in range(284, 344, 3):
        for xx in range(236, 330, 3):
            if ((xx + yy) // 3) % 2 == 0:
                arr[yy : yy + 3, xx : xx + 3] = 21.0 / 255.0
    return arr


def match_exposure(ref: np.ndarray, y: np.ndarray) -> np.ndarray:
    rc = ref[40 : LH - 40, 40 : LW - 40]
    yc = y[40 : LH - 40, 40 : LW - 40]
    mr, my = float(rc.mean()), float(yc.mean())
    vr, vy = float(rc.var()), float(yc.var())
    g = math.sqrt(vr / (vy + 1e-12))
    if abs(g - 1.0) < 0.01 and abs(mr - my) < 0.004:
        return y
    return np.clip((y - my) * g + mr, 0.0, 1.0)


class TarinaGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TarinaSR — Python Native Super-Resolution Ring")
        self.root.configure(bg="#0f1114")

        model_path = Path(__file__).resolve().parent / "model.json"
        self.model = json.loads(model_path.read_text(encoding="utf-8"))

        self.mode = "demo"
        self.frozen = False
        self.lens = 0.0
        self.zoom = 2
        self.cz = [130, 300]
        self.rng = np.random.default_rng(7)

        self.truth = make_chart_truth()
        self.ring = None
        self.x0: np.ndarray | None = None
        self.bic: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.demo_t = 0
        self.written = 0
        self.skipped = 0
        self.misses = 0
        self.cap: cv2.VideoCapture | None = None
        self.fps_t = time.perf_counter()
        self.fps_n = 0

        self._build_ui()
        self.reset_ring()
        self.tick()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        top = tk.Frame(self.root, bg="#0f1114", padx=12, pady=8)
        top.pack(fill=tk.X)

        tk.Label(
            top,
            text="TarinaSR (Python Engine)",
            fg="#f2b544",
            bg="#0f1114",
            font=("Segoe UI", 16, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 16))

        btn_cfg = dict(bg="#171a1f", fg="#e8e6e1", activebackground="#2a2f37", relief=tk.FLAT, padx=10, pady=5)
        tk.Button(top, text="Demo Shake", command=self.use_demo, **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="Live Webcam (Raw)", command=self.use_webcam, **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="Open Video File", command=self.use_video, **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="Demo With Your Image", command=self.use_image, **btn_cfg).pack(side=tk.LEFT, padx=3)

        tk.Label(top, text="Lens σ:", fg="#9a9890", bg="#0f1114").pack(side=tk.LEFT, padx=(14, 4))
        self.lens_var = tk.StringVar(value="0.0")
        lens_box = ttk.Combobox(top, textvariable=self.lens_var, values=["0.0", "0.5", "0.7", "0.9"], width=5, state="readonly")
        lens_box.pack(side=tk.LEFT)
        lens_box.bind("<<ComboboxSelected>>", self.on_lens_change)

        tk.Label(top, text="Zoom:", fg="#9a9890", bg="#0f1114").pack(side=tk.LEFT, padx=(12, 4))
        self.zoom_var = tk.StringVar(value="2")
        zoom_box = ttk.Combobox(top, textvariable=self.zoom_var, values=["1", "2", "4"], width=3, state="readonly")
        zoom_box.pack(side=tk.LEFT)
        zoom_box.bind("<<ComboboxSelected>>", lambda _e: self.render_views())

        self.freeze_btn = tk.Button(top, text="Freeze", command=self.toggle_freeze, **btn_cfg)
        self.freeze_btn.pack(side=tk.RIGHT, padx=3)
        tk.Button(top, text="Reset", command=self.reset_ring, **btn_cfg).pack(side=tk.RIGHT, padx=3)
        tk.Button(top, text="Save PNG", command=self.save_png, **btn_cfg).pack(side=tk.RIGHT, padx=3)

        views_frame = tk.Frame(self.root, bg="#0f1114", padx=12, pady=4)
        views_frame.pack(fill=tk.BOTH, expand=True)

        self.canvases: list[tk.Canvas] = []
        titles = [
            "01 / One Frame (First Frame, SihtiSR ×2)",
            "02 / The Ring (16 Shaky Frames ×2)",
            "03 / What The Ring Wrote (Residue ×6)",
        ]
        for idx, title in enumerate(titles):
            card = tk.Frame(views_frame, bg="#171a1f", bd=1, relief=tk.SOLID)
            card.grid(row=0, column=idx, padx=5, sticky="nsew")
            views_frame.grid_columnconfigure(idx, weight=1)
            tk.Label(card, text=title, fg="#9a9890", bg="#171a1f", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=4)
            cv = tk.Canvas(card, width=HW, height=HH, bg="#000000", highlightthickness=0, cursor="crosshair")
            cv.pack(padx=6, pady=6)
            cv.bind("<Button-1>", self.on_canvas_click)
            self.canvases.append(cv)

        lower = tk.Frame(self.root, bg="#0f1114", padx=12, pady=8)
        lower.pack(fill=tk.X)

        phase_card = tk.Frame(lower, bg="#171a1f", padx=10, pady=8)
        phase_card.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(phase_card, text="Sub-Pixel Phases (2×2 Grid)", fg="#9a9890", bg="#171a1f", font=("Segoe UI", 9, "bold")).pack()
        self.phase_cv = tk.Canvas(phase_card, width=160, height=160, bg="#0b0c0e", highlightthickness=1, highlightbackground="#2a2f37")
        self.phase_cv.pack(pady=4)
        self.phase_lbl = tk.Label(phase_card, text="Phase coverage: 0%", fg="#f2b544", bg="#171a1f", font=("Segoe UI", 9))
        self.phase_lbl.pack()

        stats_card = tk.Frame(lower, bg="#171a1f", padx=14, pady=10)
        stats_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.stat_vars: dict[str, tk.StringVar] = {}
        stat_specs = [
            ("written", "Frames Written"),
            ("skipped", "Frames Skipped"),
            ("shift", "Last Shift (HR px)"),
            ("rot", "Last Rotation"),
            ("res", "Ring Still Writes"),
            ("fps", "Speed"),
            ("p_bic", "Bicubic (1 Frame)"),
            ("p_one", "SihtiSR (1 Frame)"),
            ("p_ring", "TarinaSR Ring PSNR"),
        ]
        for i, (key, label) in enumerate(stat_specs):
            r, c = divmod(i, 3)
            cell = tk.Frame(stats_card, bg="#171a1f")
            cell.grid(row=r, column=c, sticky="w", padx=18, pady=4)
            tk.Label(cell, text=label, fg="#9a9890", bg="#171a1f", font=("Segoe UI", 9)).pack(anchor="w")
            v = tk.StringVar(value="—")
            self.stat_vars[key] = v
            color = "#6fd3c2" if key == "p_ring" else "#e8e6e1"
            tk.Label(cell, textvariable=v, fg=color, bg="#171a1f", font=("Consolas", 14, "bold")).pack(anchor="w")

        self.msg_var = tk.StringVar(value="Click any canvas to pan the zoom window.")
        tk.Label(stats_card, textvariable=self.msg_var, fg="#f2b544", bg="#171a1f", font=("Segoe UI", 10)).grid(
            row=3, column=0, columnspan=3, sticky="w", padx=18, pady=(8, 0)
        )

        self._tk_imgs = [None, None, None]

    def on_lens_change(self, _event=None):
        self.lens = float(self.lens_var.get())
        if self.ring is not None:
            self.ring.lens = self.lens

    def toggle_freeze(self):
        self.frozen = not self.frozen
        self.freeze_btn.config(fg="#f2b544" if self.frozen else "#e8e6e1")

    def use_demo(self):
        self._stop_capture()
        self.mode = "demo"
        self.lens = 0.0
        self.lens_var.set("0.0")
        self.truth = make_chart_truth()
        self.reset_ring()

    def use_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if not path:
            return
        self._stop_capture()
        im = Image.open(path).convert("L").resize((HW, HH), Image.Resampling.BICUBIC)
        self.truth = np.asarray(im, dtype=np.float64) / 255.0
        self.mode = "demo"
        self.lens = 0.0
        self.lens_var.set("0.0")
        self.reset_ring()

    def use_webcam(self):
        self._stop_capture()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.msg_var.set("Could not open webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap = cap
        self.mode = "cam"
        self.truth = None
        self.lens = 0.7
        self.lens_var.set("0.7")
        self.reset_ring()
        self.msg_var.set("Webcam active: center 240×180 native sensor crop. Hold camera or object in hand for micro-shake.")

    def use_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mov;*.avi;*.mkv;*.webm")])
        if not path:
            return
        self._stop_capture()
        self.cap = cv2.VideoCapture(path)
        self.mode = "video"
        self.truth = None
        self.lens = 0.7
        self.lens_var.set("0.7")
        self.reset_ring()

    def _stop_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def reset_ring(self):
        self.ring = None
        self.x0 = None
        self.bic = None
        self.last_frame = None
        self.demo_t = 0
        self.written = 0
        self.skipped = 0
        self.misses = 0
        self.rng = np.random.default_rng(7)
        for k in ("p_bic", "p_one", "p_ring", "shift", "rot", "res"):
            self.stat_vars[k].set("—")

    def next_demo_frame(self) -> np.ndarray:
        t = self.demo_t
        self.demo_t += 1
        if t == 0:
            m = (0.0, 0.0, 0.0)
        else:
            dy = 3.0 * math.sin(0.37 * t + 1.0) + 1.3 * math.sin(1.13 * t) + 0.5 * float(self.rng.standard_normal())
            dx = 3.0 * math.sin(0.29 * t) + 1.3 * math.sin(0.91 * t + 2.0) + 0.5 * float(self.rng.standard_normal())
            dth = 0.004 * math.sin(0.21 * t)
            m = (dy, dx, dth)
        y = tarina.D(tarina.warp(self.truth, m))
        noisy = np.clip(y + 0.005 * self.rng.standard_normal(y.shape), 0.0, 1.0)
        return np.round(noisy * 255.0) / 255.0

    def next_capture_frame(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            if self.mode == "video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None
        h, w = frame.shape[:2]
        if w < LW or h < LH:
            self.msg_var.set("Video smaller than 240×180.")
            return None
        y0 = (h - LH) // 2
        x0 = (w - LW) // 2
        crop = frame[y0 : y0 + LH, x0 : x0 + LW]
        lum = (0.114 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.299 * crop[:, :, 2]) / 255.0
        if float(np.var(lum)) < 1e-4:
            return None
        return lum.astype(np.float64)

    def _create_ring(self, ref: np.ndarray, x0: np.ndarray):
        """Instantiate tarina.Ring regardless of keyword vs dict options signature."""
        try:
            return tarina.Ring(ref, x0, min_peak=0.03, lens=self.lens)
        except TypeError:
            try:
                return tarina.Ring(ref, x0, {"minPeak": 0.03, "lens": self.lens})
            except TypeError:
                return tarina.Ring(ref, x0)

    def feed_frame(self, y: np.ndarray):
        if self.ring is None:
            self.x0 = call_sihti_up(y, self.model)
            self.bic = call_bicubic_up(y) if self.truth is not None else None
            self.ring = self._create_ring(y, self.x0)
            self.written = 0
            self.skipped = 0
            self.misses = 0
            self.last_frame = y
            if self.truth is not None and self.bic is not None:
                self.stat_vars["p_bic"].set(f"{psnr(self.bic, self.truth):.2f} dB")
                self.stat_vars["p_one"].set(f"{psnr(self.x0, self.truth):.2f} dB")
            return

        if self.mode != "demo" and self.last_frame is not None:
            if float(np.mean(np.abs(y - self.last_frame))) < 4e-4:
                return
        self.last_frame = y

        y_in = match_exposure(self.ring.ref, y) if self.mode != "demo" else y
        m = self.ring.step(y_in)
        if m is None:
            self.skipped += 1
            self.misses += 1
            if self.misses > 15 and self.mode != "demo":
                self.msg_var.set("Lost reference lock — restarting ring from current frame.")
                self.ring = None
                self.feed_frame(y)
            return

        self.misses = 0
        self.written += 1
        self.stat_vars["shift"].set(f"{m[0]:.2f}, {m[1]:.2f}")
        self.stat_vars["rot"].set(f"{m[2] * 180.0 / math.pi:.2f}°")
        self.stat_vars["written"].set(str(self.written))
        self.stat_vars["skipped"].set(str(self.skipped))

        res_arr = getattr(self.ring, "last_residue", getattr(self.ring, "lastResidue", None))
        if res_arr is not None:
            res_rms = 255.0 * float(np.sqrt(np.mean(np.asarray(res_arr) ** 2)))
            self.stat_vars["res"].set(f"{res_rms:.2f} grey")
        if self.truth is not None:
            self.stat_vars["p_ring"].set(f"{psnr(self.ring.x, self.truth):.2f} dB")

    def on_canvas_click(self, event):
        z = int(self.zoom_var.get())
        sw, sh = HW / z, HH / z
        sx = min(HW - sw, max(0.0, self.cz[1] - sw / 2.0))
        sy = min(HH - sh, max(0.0, self.cz[0] - sh / 2.0))
        self.cz = [sy + (event.y / HH) * sh, sx + (event.x / HW) * sw]
        if z == 1:
            self.zoom_var.set("2")
        self.render_views()

    def _to_photo(self, arr: np.ndarray) -> ImageTk.PhotoImage:
        z = int(self.zoom_var.get())
        u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if z > 1:
            sw, sh = int(HW // z), int(HH // z)
            sx = int(min(HW - sw, max(0, round(self.cz[1] - sw / 2))))
            sy = int(min(HH - sh, max(0, round(self.cz[0] - sh / 2))))
            crop = u8[sy : sy + sh, sx : sx + sw]
            pil = Image.fromarray(crop).resize((HW, HH), Image.Resampling.NEAREST)
        else:
            pil = Image.fromarray(u8)
        return ImageTk.PhotoImage(pil)

    def render_views(self):
        if self.ring is None or self.x0 is None:
            return
        res_vis = 0.5 + 6.0 * (self.ring.x - self.x0)
        for i, arr in enumerate((self.x0, self.ring.x, res_vis)):
            self._tk_imgs[i] = self._to_photo(arr)
            self.canvases[i].create_image(0, 0, anchor=tk.NW, image=self._tk_imgs[i])

        self.phase_cv.delete("all")
        W = 160
        self.phase_cv.create_line(W / 2, 0, W / 2, W, fill="#2a2f37")
        self.phase_cv.create_line(0, W / 2, W, W / 2, fill="#2a2f37")
        cells = set()
        motions = getattr(self.ring, "motions", [])
        ring_size = getattr(self.ring, "size", 16)
        ring_count = getattr(self.ring, "count", len(motions))
        for k, m in enumerate(motions):
            fy = (m[0] % 2.0 + 2.0) % 2.0
            fx = (m[1] % 2.0 + 2.0) % 2.0
            cells.add(int(fy * 2.0) * 4 + int(fx * 2.0))
            px, py = (fx / 2.0) * W, (fy / 2.0) * W
            is_latest = k == (ring_count - 1) % ring_size
            col = "#f2b544" if is_latest else "#6fd3c2"
            self.phase_cv.create_oval(px - 4, py - 4, px + 4, py + 4, fill=col, outline="")
        cov = int(round(100.0 * len(cells) / 16.0))
        self.phase_lbl.config(text=f"Phase coverage: {cov}% of 16 cells")

    def save_png(self):
        if self.ring is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="tarinasr_python.png")
        if path:
            u8 = np.clip(self.ring.x * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(u8).save(path)
            self.msg_var.set(f"Saved {path}")

    def tick(self):
        if not self.frozen:
            y = None
            if self.mode == "demo" and self.demo_t < DEMO_FRAMES:
                y = self.next_demo_frame()
                if self.demo_t == DEMO_FRAMES:
                    self.msg_var.set(f"Done: {DEMO_FRAMES} shaky frames. Press Reset to run again.")
            elif self.mode in ("cam", "video"):
                y = self.next_capture_frame()

            if y is not None:
                self.feed_frame(y)
                self.render_views()
                self.fps_n += 1

        now = time.perf_counter()
        if now - self.fps_t >= 1.0:
            self.stat_vars["fps"].set(f"{self.fps_n / (now - self.fps_t):.1f} fps")
            self.fps_t = now
            self.fps_n = 0

        self.root.after(15, self.tick)


if __name__ == "__main__":
    root = tk.Tk()
    app = TarinaGUI(root)
    root.mainloop()