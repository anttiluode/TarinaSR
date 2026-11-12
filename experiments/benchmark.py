"""TarinaSR benchmark. 8 standard test images (scikit-image), x2, grayscale.

Each low-res frame = 2x2 mean of the shifted original + camera noise, rounded
to 8 bits. Shifts are estimated by the ring, never given to it.

Controls:
  frames averaged : register and average the 16 frames, then SihtiSR.
                    Same frames, same registration, no sub-pixel use.
                    Separates 'less noise' from 'more resolution'.
  even shifts only: all shifts are whole low-res pixels, so every frame
                    samples the same sub-pixel phase. No new information.
Stress:
  rotation        : each frame also rotated by up to +-0.3 or +-1 degree
  big shake       : shifts up to +-40 high-res px
"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import tarina, sihtisr
from skimage import data, color
from scipy.ndimage import shift as ndshift, rotate
from PIL import Image

model = json.load(open('model.json'))
NAMES = ["camera", "astronaut", "coffee", "chelsea", "rocket", "cat", "brick", "grass"]
N = 16


def load(name):
    im = getattr(data, name)()
    if im.ndim == 3: im = color.rgb2gray(im[..., :3])
    im = im.astype(float); im = im / 255 if im.max() > 1 else im
    return im[:im.shape[0] // 2 * 2, :im.shape[1] // 2 * 2]


def psnr(a, b, m=24):
    a, b = np.clip(a, 0, 1)[m:-m, m:-m], b[m:-m, m:-m]
    return float(-10 * np.log10(np.mean((a - b) ** 2)))


def frames_for(H, rng, amp=6.0, even=False, rot=0.0, noise=0.005):
    out, sh = [], []
    for k in range(N):
        s = np.zeros(2) if k == 0 else rng.uniform(-amp, amp, 2)
        if even: s = 2 * np.round(s / 2)
        img = H if rot == 0 or k == 0 else rotate(H, rng.uniform(-rot, rot), reshape=False, order=3, mode='reflect')
        y = tarina.D(ndshift(img, s, order=3, mode='reflect'))
        y = y + rng.normal(0, noise, y.shape)
        out.append(np.round(np.clip(y, 0, 1) * 255) / 255); sh.append(s)
    return out, sh


def run_ring(frames):
    R = tarina.Ring(frames[0], sihtisr.apply(frames[0], model))
    errs = []
    for f in frames[1:]:
        R.step(f)
    return R


def averaged(frames):
    acc = frames[0].copy()
    for f in frames[1:]:
        d, _ = tarina.register(frames[0], f)
        acc += tarina.warp(f, -d)
    return sihtisr.apply(acc / len(frames), model)


rows = []
for name in NAMES:
    H = load(name)
    rng = np.random.default_rng(NAMES.index(name))
    fr, sh = frames_for(H, rng)
    R = run_ring(fr)
    err = float(np.mean([np.hypot(*(a[:2] - b)) for a, b in zip(R.motions[1:], sh[1:])]))
    row = dict(img=name,
               bicubic=psnr(sihtisr.pil_up(fr[0], Image.BICUBIC), H),
               sihtisr=psnr(sihtisr.apply(fr[0], model), H),
               averaged=psnr(averaged(fr), H),
               tarina=psnr(R.x, H),
               shift_err_hr_px=err)
    for i in (2, 4, 8):
        row[f"tarina_{i}"] = psnr(run_ring(fr[:i]).x, H)
    fe, _ = frames_for(H, np.random.default_rng(100 + NAMES.index(name)), even=True)
    row["even_shifts_only"] = psnr(run_ring(fe).x, H)
    fr2, _ = frames_for(H, np.random.default_rng(200 + NAMES.index(name)), rot=0.3)
    row["rotation_0.3deg"] = psnr(run_ring(fr2).x, H)
    fr2, _ = frames_for(H, np.random.default_rng(250 + NAMES.index(name)), rot=1.0)
    row["rotation_1deg"] = psnr(run_ring(fr2).x, H)
    fr3, _ = frames_for(H, np.random.default_rng(300 + NAMES.index(name)), amp=40)
    row["big_shake_40px"] = psnr(run_ring(fr3).x, H)
    rows.append(row)
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

keys = [k for k in rows[0] if k != "img"]
mean = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in keys}
print("MEAN", mean)
json.dump({"per_image": rows, "mean": mean}, open("results/benchmark.json", "w"), indent=1)
