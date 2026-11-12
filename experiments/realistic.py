"""Stress closer to a real camera: lens blur before the pixels, and JPEG.

Truth is the sharp original. Each frame: shift (+-6 px) and rotation (+-0.5 deg),
Gaussian blur sigma 0.7 high-res px (a lens), 2x2 shrink, noise, JPEG quality 85.
Runs the ring with no lens model, the right one (0.7) and two wrong guesses.
"""
import sys, io, json, numpy as np
sys.path.insert(0, '.')
import tarina, sihtisr
from skimage import data, color
from scipy.ndimage import shift as ndshift, rotate, gaussian_filter
from PIL import Image

model = json.load(open('model.json'))
NAMES = ["camera", "astronaut", "coffee", "chelsea", "rocket", "cat", "brick", "grass"]


def load(name):
    im = getattr(data, name)()
    if im.ndim == 3: im = color.rgb2gray(im[..., :3])
    im = im.astype(float); im = im / 255 if im.max() > 1 else im
    return im[:im.shape[0] // 2 * 2, :im.shape[1] // 2 * 2]


def psnr(a, b, m=24):
    a, b = np.clip(a, 0, 1)[m:-m, m:-m], b[m:-m, m:-m]
    return float(-10 * np.log10(np.mean((a - b) ** 2)))


def jpeg(y, q=85):
    buf = io.BytesIO(); Image.fromarray(np.uint8(np.clip(y, 0, 1) * 255 + .5)).save(buf, 'JPEG', quality=q)
    return np.asarray(Image.open(buf), float) / 255


rows = []
for i, name in enumerate(NAMES):
    H = load(name); rng = np.random.default_rng(400 + i); fr = []
    for k in range(16):
        s = np.zeros(2) if k == 0 else rng.uniform(-6, 6, 2)
        t = 0.0 if k == 0 else rng.uniform(-0.5, 0.5)
        img = rotate(H, t, reshape=False, order=3, mode='reflect') if t else H
        y = tarina.D(gaussian_filter(ndshift(img, s, order=3, mode='reflect'), 0.7))
        fr.append(jpeg(y + rng.normal(0, 0.005, y.shape)))
    row = dict(img=name, bicubic=psnr(sihtisr.pil_up(fr[0], Image.BICUBIC), H),
               sihtisr=psnr(sihtisr.apply(fr[0], model), H))
    for lens in (0.0, 0.5, 0.7, 0.9):          # 0.7 is the true lens; 0.5 and 0.9 are wrong guesses
        R = tarina.Ring(fr[0], sihtisr.apply(fr[0], model), lens=lens)
        for y in fr[1:]: R.step(y)
        row[f"tarina_lens{lens}"] = psnr(R.x, H)
    rows.append(row); print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
mean = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0] if k != 'img'}
print('MEAN', mean)
json.dump({"per_image": rows, "mean": mean}, open('results/realistic.json', 'w'), indent=1)
