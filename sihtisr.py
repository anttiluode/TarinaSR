"""SihtiSR - put the residue back.

x2 super-resolution with no backprop and no learned sorting:
  1. SORT (hand-written):  each 5x5 low-res patch gets a bucket from its
     structure tensor -> edge angle x edge strength x coherence.
  2. SOLVE (closed form):  each bucket owns one 25->4 linear map, found by
     ridge regression (one matrix solve per bucket).
  3. LOCK (hand-written):  the 2x2 output block is shifted so it averages
     back to the input pixel exactly. Nothing measured is contradicted.

Degradation model: a low-res pixel is the mean of a 2x2 high-res block.

  python sihtisr.py train <BSDS500/data/images>     -> model.json + numbers
  python sihtisr.py up in.png out.png [times]       -> upscale x2 (or x4, ...)
"""
import sys, json, glob, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from PIL import Image

NA, NS, NC = 24, 3, 3            # angle, strength, coherence bins
COH = (0.25, 0.5)                # hand-set coherence thresholds
LAM = 1e-3                       # ridge


def gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def down(H):
    h, w = H.shape[0] // 2 * 2, H.shape[1] // 2 * 2
    H = H[:h, :w]
    return H, (H[0::2, 0::2] + H[0::2, 1::2] + H[1::2, 0::2] + H[1::2, 1::2]) / 4


def sort_patches(L, sthr, na=NA):
    """-> patches (h,w,25) mean-removed, means (h,w), bucket ids (h,w)."""
    P = np.pad(L, 3, mode="edge")
    gx = (P[1:-1, 2:] - P[1:-1, :-2]) / 2          # (h+4, w+4)
    gy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2
    box = lambda a: swv(a, (5, 5)).mean(axis=(2, 3))
    jxx, jyy, jxy = box(gx * gx), box(gy * gy), box(gx * gy)
    d = np.sqrt(((jxx - jyy) / 2) ** 2 + jxy ** 2)
    m = (jxx + jyy) / 2
    s1, s2 = np.sqrt(m + d), np.sqrt(np.maximum(m - d, 0))
    coh = (s1 - s2) / (s1 + s2 + 1e-8)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)     # (-pi/2, pi/2]
    a = np.floor((ang + np.pi / 2) / np.pi * na).astype(int) % na
    s = (s1 > sthr[0]).astype(int) + (s1 > sthr[1])
    c = (coh > COH[0]).astype(int) + (coh > COH[1])
    X = swv(P[1:-1, 1:-1], (5, 5)).reshape(L.shape[0], L.shape[1], 25)
    mu = X.mean(axis=2)
    return X - mu[..., None], mu, (a * NS + s) * NC + c, s1


def apply(L, model, lock=True):
    W = np.asarray(model["W"])                      # (B, 25, 4)
    X, mu, b, _ = sort_patches(L, model["sthr"], model["na"])
    Y = np.einsum("hwk,hwko->hwo", X, W[b]) + mu[..., None]
    if lock:
        Y += (L - Y.mean(axis=2))[..., None]
    h, w = L.shape
    return Y.reshape(h, w, 2, 2).transpose(0, 2, 1, 3).reshape(2 * h, 2 * w)


def fit(files, sthr, na):
    B = na * NS * NC
    AtA, AtB = np.zeros((B, 25, 25)), np.zeros((B, 25, 4))
    for f in files:
        H, L = down(gray(f))
        X, mu, b, _ = sort_patches(L, sthr, na)
        h, w = L.shape
        T = H.reshape(h, 2, w, 2).transpose(0, 2, 1, 3).reshape(h, w, 4) - mu[..., None]
        X, T, b = X.reshape(-1, 25), T.reshape(-1, 4), b.ravel()
        o = np.argsort(b, kind="stable")
        X, T, b = X[o], T[o], b[o]
        cut = np.searchsorted(b, np.arange(B + 1))
        for k in range(B):
            x, t = X[cut[k]:cut[k + 1]], T[cut[k]:cut[k + 1]]
            if len(x):
                AtA[k] += x.T @ x
                AtB[k] += x.T @ t
    G, Gb = AtA.sum(0), AtB.sum(0)                  # global filter = fallback
    Wg = np.linalg.solve(G + LAM * np.eye(25), Gb)
    n = np.trace(AtA, axis1=1, axis2=2)
    W = np.stack([np.linalg.solve(AtA[k] + LAM * (1 + n[k] / 25) * np.eye(25) * 1e-3
                                  + 1e-9 * np.eye(25), AtB[k]) if n[k] > 0 else Wg
                  for k in range(B)])
    return W, Wg


def psnr(a, b, m=4):
    a, b = np.clip(a, 0, 1)[m:-m, m:-m], b[m:-m, m:-m]
    return -10 * np.log10(np.mean((a - b) ** 2))


def pil_up(L, mode):
    return np.asarray(Image.fromarray(L.astype(np.float32), "F").resize(
        (L.shape[1] * 2, L.shape[0] * 2), mode), dtype=np.float64)


def train(root):
    tr = sorted(glob.glob(os.path.join(root, "train", "*.jpg")))
    te = sorted(glob.glob(os.path.join(root, "test", "*.jpg")))
    s1 = np.concatenate([sort_patches(down(gray(f))[1], (0, 0))[3].ravel()[::7] for f in tr[::4]])
    sthr = [float(np.quantile(s1, 1 / 3)), float(np.quantile(s1, 2 / 3))]
    W, Wg = fit(tr, sthr, NA)
    model = {"na": NA, "ns": NS, "nc": NC, "coh": COH, "sthr": sthr, "W": np.round(W, 5).tolist()}
    one = {"na": 1, "sthr": [9, 9], "W": np.tile(Wg, (NS * NC, 1, 1)).tolist()}
    rows = {k: [] for k in ("bicubic", "lanczos", "one filter", "sorted", "sorted+lock")}
    for f in te:
        H, L = down(gray(f))
        rows["bicubic"].append(psnr(pil_up(L, Image.BICUBIC), H))
        rows["lanczos"].append(psnr(pil_up(L, Image.LANCZOS), H))
        rows["one filter"].append(psnr(apply(L, one, lock=False), H))
        rows["sorted"].append(psnr(apply(L, model, lock=False), H))
        rows["sorted+lock"].append(psnr(apply(L, model), H))
    res = {k: round(float(np.mean(v)), 3) for k, v in rows.items()}
    wins = float(np.mean(np.array(rows["sorted+lock"]) > np.array(rows["lanczos"])))
    print(json.dumps(res, indent=1), "\nbeats lanczos on", f"{wins:.0%}", "of", len(te), "test images")
    print("numbers in model:", W.size)
    here0 = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(here0, "model.js"), "w").write("window.SIHTI_MODEL=" + json.dumps(model, separators=(",", ":")) + ";")
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(model, open(os.path.join(here, "model.json"), "w"), separators=(",", ":"))
    json.dump({"test_images": len(te), "train_images": len(tr), "psnr_db": res,
               "beats_lanczos_frac": wins, "numbers": int(W.size)},
              open(os.path.join(here, "results.json"), "w"), indent=1)


def up(src, dst, times=1):
    model = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")))
    im = Image.open(src).convert("YCbCr")
    y = np.asarray(im.getchannel("Y"), dtype=np.float64) / 255
    for _ in range(times):
        y = np.clip(apply(y, model), 0, 1)
    size = (y.shape[1], y.shape[0])
    Y = Image.fromarray((y * 255 + 0.5).astype(np.uint8))
    cb, cr = (im.getchannel(c).resize(size, Image.BICUBIC) for c in ("Cb", "Cr"))
    Image.merge("YCbCr", (Y, cb, cr)).convert("RGB").save(dst)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "train":
        train(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "up":
        up(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    else:
        print(__doc__)
