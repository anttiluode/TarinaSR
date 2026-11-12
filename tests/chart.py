"""A deterministic test scene (no downloads): rings, bars, edges and texture."""
import numpy as np
from scipy.ndimage import gaussian_filter


def chart(h=288, w=320, seed=0):
    y, x = np.mgrid[0:h, 0:w].astype(float)
    cy, cx = h * 0.45, w * 0.4
    r = np.hypot(y - cy, x - cx)
    a = np.arctan2(y - cy, x - cx)
    img = 0.5 + 0.25 * np.sign(np.sin(24 * a)) * (r < 70)            # Siemens star
    img += 0.2 * np.sin(r ** 2 / 900.0) * (r > 75) * (r < 120)        # zone plate ring
    bars = (np.floor(x / (1 + (y // 40) % 4)) % 2) * ((x > 230) & (x < 300))
    img = np.where((x > 230) & (x < 300), 0.2 + 0.6 * bars, img)
    rng = np.random.default_rng(seed)
    img += 0.08 * gaussian_filter(rng.standard_normal((h, w)), 1.2)
    return np.clip(gaussian_filter(img, 0.6), 0, 1)
