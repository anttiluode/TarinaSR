"""TarinaSR core: shaky frames -> one sharper image.

Same algorithm as tarina.js (tests/test_js.cjs checks that they agree).

  register : phase correlation gives the shift (integer peak + parabola),
             then Lucas-Kanade refines shift AND a small rotation.
  ring     : the last N frames and their motions circulate. On every new
             frame the comparator visits K ring slots:
                 r = frame - D(G(warp(x, m)))         what this frame says x got wrong
                 x = x + beta * unwarp(G(U(r)), m)    write that residue back
             G is an optional Gaussian lens blur (sigma = 0: none).
  start    : x = SihtiSR(reference frame)          the one-frame prediction

Motion m = (sy, sx, t): a shift plus a small rotation t (radians), applied as
two shears (Paeth / Catmull-Smith two-pass; exact to O(t^2), i.e. < 0.02 px at
1 degree over this image size):
    pass 1, every column i moves down by  a(i) = sy + t (i - cx)
    pass 2, every row y moves right by    b(y) = sx - t (y - sy - cy)
Each pass is 1-D Keys cubic (a = -0.5) with edge clamp, weights constant per
row/column, so a warp costs 8 multiply-adds per pixel.
D = mean of 2x2 blocks, U = 2x2 replicate.
Shifts are in the units of the image they act on (high-res px for x).
"""
import numpy as np

KA = -0.5


def kw(d):
    d = np.abs(d)
    return np.where(d < 1, (KA + 2) * d**3 - (KA + 3) * d**2 + 1,
                    np.where(d < 2, KA * d**3 - 5 * KA * d**2 + 8 * KA * d - 4 * KA, 0.0))


def _shift_rows(x, a):
    """out[y, i] = x[y - a[i], i]: column i moves down by a[i] (cubic, edge clamp)."""
    h, w = x.shape
    f = np.floor(-a); t = -a - f; f = f.astype(int)
    y = np.arange(h)[:, None]
    out = 0.0
    for j in range(4):
        out = out + kw(t - (j - 1))[None, :] * np.take_along_axis(x, np.clip(y + f[None, :] + j - 1, 0, h - 1), 0)
    return out


def _shift_cols(x, b):
    """out[y, i] = x[y, i - b[y]]: row y moves right by b[y]."""
    return _shift_rows(x.T, b).T


def _ab(h, w, m):
    sy, sx, t = m
    cy, cx = (h - 1) / 2, (w - 1) / 2
    a = sy + t * (np.arange(w) - cx)                 # shift of each column (pass 1)
    b = sx - t * (np.arange(h) - sy - cy)            # shift of each row (pass 2)
    return a, b


def warp(x, m):
    """Small-angle rigid motion as two shears:  column shift a(x), then row shift b(y)."""
    a, b = _ab(*x.shape, m)
    return _shift_cols(_shift_rows(x, a), b)


def unwarp(f, m):
    """Exact inverse of the two shears (up to interpolation)."""
    a, b = _ab(*f.shape, m)
    return _shift_rows(_shift_cols(f, -b), -a)


def D(x):
    return (x[0::2, 0::2] + x[0::2, 1::2] + x[1::2, 0::2] + x[1::2, 1::2]) / 4


def U(y):
    return np.repeat(np.repeat(y, 2, 0), 2, 1)


def blur(x, sigma):
    """Separable Gaussian, radius ceil(3 sigma), edge clamp. sigma = 0 is no blur."""
    if sigma <= 0: return x
    r = int(np.ceil(3 * sigma)); k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma * sigma)); k /= k.sum()
    h, w = x.shape
    out = sum(k[j] * x[np.clip(np.arange(h) + j - r, 0, h - 1)] for j in range(2 * r + 1))
    return sum(k[j] * out[:, np.clip(np.arange(w) + j - r, 0, w - 1)] for j in range(2 * r + 1))


def to_hr(m):
    return np.array([2 * m[0], 2 * m[1], m[2]])


# ---------------- registration ----------------
def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)


def phase_corr(ref, img, n=128):
    """d (low-res px) such that img(q) ~ ref(q - d), and the peak height."""
    h, w = ref.shape
    y0, x0 = (h - n) // 2, (w - n) // 2
    a, b = ref[y0:y0 + n, x0:x0 + n], img[y0:y0 + n, x0:x0 + n]
    win = np.outer(_hann(n), _hann(n))
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    c = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(np.argmax(c), c.shape)

    def sub(cm, c0, cp):
        den = cm - 2 * c0 + cp
        return 0.0 if abs(den) < 1e-12 else 0.5 * (cm - cp) / den
    dy = py + sub(c[(py - 1) % n, px], c[py, px], c[(py + 1) % n, px])
    dx = px + sub(c[py, (px - 1) % n], c[py, px], c[py, (px + 1) % n])
    if dy > n / 2: dy -= n
    if dx > n / 2: dx -= n
    return np.array([dy, dx]), float(c[py, px])


def _grad(a):
    return np.gradient(a, axis=0), np.gradient(a, axis=1)


def lk_refine(ref, img, m, iters=5):
    """Gauss-Newton on sum (img - warp(ref, m))^2 over the interior."""
    h, w = ref.shape
    m = np.array(m, float)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    py, px = np.mgrid[0:h, 0:w].astype(float)
    for _ in range(iters):
        bd = int(np.ceil(abs(m[0]) + abs(m[1]) + abs(m[2]) * (h + w) / 2)) + 3
        sl = (slice(bd, h - bd), slice(bd, w - bd))
        base = warp(ref, m)
        gy, gx = _grad(base)
        Jy = -gy + m[2] * gx
        Jx = -gx
        Jt = (py - m[0] - cy) * gx - (px - m[1] - cx) * gy
        J = np.stack([Jy[sl].ravel(), Jx[sl].ravel(), Jt[sl].ravel()], 1)
        r = (img - base)[sl].ravel()
        H = J.T @ J
        if np.linalg.det(H) < 1e-18: break
        step = np.linalg.solve(H, J.T @ r)
        m = m + step
        if max(abs(step[0]), abs(step[1])) < 1e-3 and abs(step[2]) < 1e-5: break
    return m


REG_BLUR = 1.0


def register(ref, img, n=128):
    """Register on blurred copies: fine detail is aliased, and aliasing moves
    differently from the scene, so it would pull the estimate off."""
    a, b = blur(ref, REG_BLUR), blur(img, REG_BLUR)
    d, peak = phase_corr(a, b, n)
    return lk_refine(a, b, [d[0], d[1], 0.0]), peak


# ---------------- the ring ----------------
class Ring:
    def __init__(self, ref, x0, size=16, visits=4, beta=0.4, max_shift=24.0, min_peak=0.0, lens=0.0):
        self.ref, self.x = ref, x0.astype(np.float64).copy()
        self.lens = lens                   # Gaussian lens blur (high-res px) the frames are assumed to carry
        self.size, self.visits, self.beta = size, visits, beta
        self.max_shift, self.min_peak = max_shift, min_peak
        self.frames, self.motions = [], []
        self.head = 0                      # next slot the comparator visits
        self.count = 0                     # frames pushed so far
        self.last_residue = np.zeros_like(self.x)
        self.push(ref, np.zeros(3))

    def push(self, y, m_hr):
        k = self.count % self.size         # overwrite the oldest slot
        if k < len(self.frames):
            self.frames[k] = y; self.motions[k] = np.asarray(m_hr, float)
        else:
            self.frames.append(y); self.motions.append(np.asarray(m_hr, float))
        self.count += 1

    def visit(self, k):
        y, m = self.frames[k], self.motions[k]
        r = y - D(blur(warp(self.x, m), self.lens))
        h, w = y.shape
        b = int(np.ceil((abs(m[0]) + abs(m[1])) / 2 + abs(m[2]) * (h + w) / 2)) + 2
        r[:b] = 0; r[h - b:] = 0; r[:, :b] = 0; r[:, w - b:] = 0
        dx = self.beta * unwarp(blur(U(r), self.lens), m)
        self.x += dx
        return dx

    def step(self, y):
        m, peak = register(self.ref, y)
        self.last_peak = peak
        if max(abs(m[0]), abs(m[1])) > self.max_shift or peak < self.min_peak:
            return None
        self.push(y, to_hr(m))
        acc = 0
        for _ in range(min(self.visits, len(self.frames))):
            acc = acc + self.visit(self.head % len(self.frames))
            self.head = (self.head + 1) % len(self.frames)
        self.last_residue = acc
        return to_hr(m)
