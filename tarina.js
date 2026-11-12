// TarinaSR core in the browser. Mirrors tarina.py (tests/test_js.cjs checks they agree).
// Images are {d: Float64Array, h, w}. Shifts s = [sy, sx]; warp(x, s)(p) = x(p - s).
(function (root) {
  const KA = -0.5;
  function kw(d) {
    d = Math.abs(d);
    if (d < 1) return (KA + 2) * d ** 3 - (KA + 3) * d ** 2 + 1;
    if (d < 2) return KA * d ** 3 - 5 * KA * d ** 2 + 8 * KA * d - 4 * KA;
    return 0;
  }
  const img = (h, w) => ({ d: new Float64Array(h * w), h, w });
  const clamp = (v, n) => v < 0 ? 0 : v >= n ? n - 1 : v;

  // out[y][i] = x[y - a[i]][i]: column i moves down by a[i] (cubic, edge clamp)
  function shiftRows(x, a) {
    const { h, w } = x, o = img(h, w), f = new Int32Array(w), W = [new Float64Array(w), new Float64Array(w), new Float64Array(w), new Float64Array(w)];
    for (let i = 0; i < w; i++) {
      const fl = Math.floor(-a[i]), t = -a[i] - fl; f[i] = fl;
      for (let j = 0; j < 4; j++) W[j][i] = kw(t - (j - 1));
    }
    for (let y = 0; y < h; y++) {
      const ro = y * w;
      for (let i = 0; i < w; i++) {
        const b = y + f[i];
        o.d[ro + i] = W[0][i] * x.d[clamp(b - 1, h) * w + i] + W[1][i] * x.d[clamp(b, h) * w + i]
                    + W[2][i] * x.d[clamp(b + 1, h) * w + i] + W[3][i] * x.d[clamp(b + 2, h) * w + i];
      }
    }
    return o;
  }
  // out[y][i] = x[y][i - b[y]]: row y moves right by b[y]
  function shiftCols(x, b) {
    const { h, w } = x, o = img(h, w);
    for (let y = 0; y < h; y++) {
      const fl = Math.floor(-b[y]), t = -b[y] - fl, w0 = kw(t + 1), w1 = kw(t), w2 = kw(t - 1), w3 = kw(t - 2), r = y * w;
      for (let i = 0; i < w; i++) {
        const c = i + fl;
        o.d[r + i] = w0 * x.d[r + clamp(c - 1, w)] + w1 * x.d[r + clamp(c, w)] + w2 * x.d[r + clamp(c + 1, w)] + w3 * x.d[r + clamp(c + 2, w)];
      }
    }
    return o;
  }
  function ab(h, w, m) {
    const cy = (h - 1) / 2, cx = (w - 1) / 2, a = new Float64Array(w), b = new Float64Array(h);
    for (let i = 0; i < w; i++) a[i] = m[0] + m[2] * (i - cx);
    for (let y = 0; y < h; y++) b[y] = m[1] - m[2] * (y - m[0] - cy);
    return [a, b];
  }
  // small-angle rigid motion m = [sy, sx, t] as two shears (see tarina.py)
  function warp(x, m) { const [a, b] = ab(x.h, x.w, m); return shiftCols(shiftRows(x, a), b); }
  function unwarp(f, m) {
    const [a, b] = ab(f.h, f.w, m);
    return shiftRows(shiftCols(f, b.map(v => -v)), a.map(v => -v));
  }

  function D(x) {
    const h = x.h >> 1, w = x.w >> 1, o = img(h, w), W = x.w;
    for (let y = 0; y < h; y++) for (let i = 0; i < w; i++) {
      const p = 2 * y * W + 2 * i;
      o.d[y * w + i] = (x.d[p] + x.d[p + 1] + x.d[p + W] + x.d[p + W + 1]) / 4;
    }
    return o;
  }
  function U(y) {
    const o = img(2 * y.h, 2 * y.w), W = o.w;
    for (let r = 0; r < y.h; r++) for (let c = 0; c < y.w; c++) {
      const v = y.d[r * y.w + c], p = 2 * r * W + 2 * c;
      o.d[p] = o.d[p + 1] = o.d[p + W] = o.d[p + W + 1] = v;
    }
    return o;
  }

  // separable Gaussian, radius ceil(3 sigma), edge clamp (see tarina.py)
  function blur(x, sigma) {
    if (!(sigma > 0)) return x;
    const r = Math.ceil(3 * sigma), k = [];
    let s = 0; for (let j = -r; j <= r; j++) { const v = Math.exp(-j * j / (2 * sigma * sigma)); k.push(v); s += v; }
    for (let j = 0; j < k.length; j++) k[j] /= s;
    const { h, w } = x, t = img(h, w), o = img(h, w);
    for (let y = 0; y < h; y++) for (let i = 0; i < w; i++) {
      let v = 0; for (let j = 0; j < k.length; j++) v += k[j] * x.d[clamp(y + j - r, h) * w + i]; t.d[y * w + i] = v;
    }
    for (let y = 0; y < h; y++) for (let i = 0; i < w; i++) {
      let v = 0; for (let j = 0; j < k.length; j++) v += k[j] * t.d[y * w + clamp(i + j - r, w)]; o.d[y * w + i] = v;
    }
    return o;
  }

  // ---------- FFT (radix 2, in place) ----------
  function fft(re, im, inv) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const a = (inv ? 2 : -2) * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const p = i + k, q = p + len / 2;
          const tr = re[q] * cr - im[q] * ci, ti = re[q] * ci + im[q] * cr;
          re[q] = re[p] - tr; im[q] = im[p] - ti; re[p] += tr; im[p] += ti;
          const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
        }
      }
    }
    if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  function fft2(re, im, n, inv) {
    const r = new Float64Array(n), i2 = new Float64Array(n);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) { r[x] = re[y * n + x]; i2[x] = im[y * n + x]; }
      fft(r, i2, inv);
      for (let x = 0; x < n; x++) { re[y * n + x] = r[x]; im[y * n + x] = i2[x]; }
    }
    for (let x = 0; x < n; x++) {
      for (let y = 0; y < n; y++) { r[y] = re[y * n + x]; i2[y] = im[y * n + x]; }
      fft(r, i2, inv);
      for (let y = 0; y < n; y++) { re[y * n + x] = r[y]; im[y * n + x] = i2[y]; }
    }
  }

  function phaseCorr(ref, im, n = 128, cache) {
    const y0 = (ref.h - n) >> 1, x0 = (ref.w - n) >> 1;
    const hann = i => 0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 0.5) / n);
    const prep = a => {
      let m = 0;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) m += a.d[(y0 + y) * a.w + x0 + x];
      m /= n * n;
      const re = new Float64Array(n * n), imv = new Float64Array(n * n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
        re[y * n + x] = (a.d[(y0 + y) * a.w + x0 + x] - m) * hann(y) * hann(x);
      fft2(re, imv, n, false);
      return [re, imv];
    };
    let A = cache && cache.ref === ref && cache.n === n ? cache.A : null;
    if (!A) { A = prep(ref); if (cache) { cache.ref = ref; cache.n = n; cache.A = A; } }
    const [ar, ai] = A, [br, bi] = prep(im);
    const re = new Float64Array(n * n), ii = new Float64Array(n * n);
    for (let k = 0; k < n * n; k++) {           // B * conj(A) / |.|
      const r = br[k] * ar[k] + bi[k] * ai[k], q = bi[k] * ar[k] - br[k] * ai[k];
      const m = Math.hypot(r, q) + 1e-9; re[k] = r / m; ii[k] = q / m;
    }
    fft2(re, ii, n, true);
    let best = -Infinity, py = 0, px = 0;
    for (let k = 0; k < n * n; k++) if (re[k] > best) { best = re[k]; py = (k / n) | 0; px = k % n; }
    const c = (y, x) => re[((y + n) % n) * n + ((x + n) % n)];
    const sub = (cm, c0, cp) => { const den = cm - 2 * c0 + cp; return Math.abs(den) < 1e-12 ? 0 : 0.5 * (cm - cp) / den; };
    let dy = py + sub(c(py - 1, px), c(py, px), c(py + 1, px));
    let dx = px + sub(c(py, px - 1), c(py, px), c(py, px + 1));
    if (dy > n / 2) dy -= n; if (dx > n / 2) dx -= n;
    return { d: [dy, dx], peak: best };
  }

  function gradient(a, axis) {   // numpy.gradient, unit spacing
    const { h, w } = a, o = img(h, w);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = y * w + x;
      if (axis === 0) o.d[k] = y === 0 ? a.d[k + w] - a.d[k] : y === h - 1 ? a.d[k] - a.d[k - w] : (a.d[k + w] - a.d[k - w]) / 2;
      else o.d[k] = x === 0 ? a.d[k + 1] - a.d[k] : x === w - 1 ? a.d[k] - a.d[k - 1] : (a.d[k + 1] - a.d[k - 1]) / 2;
    }
    return o;
  }
  function lkRefine(ref, im, m0, iters = 5) {
    const { h, w } = ref, cy = (h - 1) / 2, cx = (w - 1) / 2, m = m0.slice();
    for (let it = 0; it < iters; it++) {
      const bd = Math.ceil(Math.abs(m[0]) + Math.abs(m[1]) + Math.abs(m[2]) * (h + w) / 2) + 3;
      const base = warp(ref, m), gy = gradient(base, 0), gx = gradient(base, 1);
      const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], g = [0, 0, 0], J = [0, 0, 0];
      for (let y = bd; y < h - bd; y++) for (let x = bd; x < w - bd; x++) {
        const i = y * w + x, r = im.d[i] - base.d[i];
        J[0] = -gy.d[i] + m[2] * gx.d[i]; J[1] = -gx.d[i];
        J[2] = (y - m[0] - cy) * gx.d[i] - (x - m[1] - cx) * gy.d[i];
        for (let a = 0; a < 3; a++) { g[a] += J[a] * r; for (let c = 0; c < 3; c++) H[a][c] += J[a] * J[c]; }
      }
      const det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) - H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) + H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);
      if (det < 1e-18) break;
      const st = solve3(H, g);
      m[0] += st[0]; m[1] += st[1]; m[2] += st[2];
      if (Math.max(Math.abs(st[0]), Math.abs(st[1])) < 1e-3 && Math.abs(st[2]) < 1e-5) break;
    }
    return m;
  }
  function solve3(A, b) {   // Gaussian elimination with partial pivoting
    const M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < 3; c++) {
      let p = c; for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < 3; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = [0, 0, 0];
    for (let r = 2; r >= 0; r--) { let v = M[r][3]; for (let k = r + 1; k < 3; k++) v -= M[r][k] * x[k]; x[r] = v / M[r][r]; }
    return x;
  }

  // register on blurred copies: aliased fine detail moves differently from the scene
  const REG_BLUR = 1.0;
  function register(ref, im, n = 128, cache) {
    let a = cache && cache.src === ref ? cache.blurred : null;
    if (!a) { a = blur(ref, REG_BLUR); if (cache) { cache.src = ref; cache.blurred = a; } }
    const b = blur(im, REG_BLUR), pc = phaseCorr(a, b, n, cache);
    return { m: lkRefine(a, b, [pc.d[0], pc.d[1], 0]), peak: pc.peak };
  }
  const toHr = m => [2 * m[0], 2 * m[1], m[2]];

  class Ring {
    constructor(ref, x0, o = {}) {
      this.ref = ref; this.x = { d: Float64Array.from(x0.d), h: x0.h, w: x0.w };
      this.size = o.size ?? 16; this.visits = o.visits ?? 4; this.beta = o.beta ?? 0.4;
      this.maxShift = o.maxShift ?? 24; this.minPeak = o.minPeak ?? 0; this.lens = o.lens ?? 0;
      this.frames = []; this.motions = []; this.head = 0; this.count = 0;
      this.lastResidue = img(x0.h, x0.w); this.lastPeak = 1; this.cache = {};
      this.push(ref, [0, 0, 0]);
    }
    push(y, m) {
      const k = this.count % this.size;
      if (k < this.frames.length) { this.frames[k] = y; this.motions[k] = m; }
      else { this.frames.push(y); this.motions.push(m); }
      this.count++;
    }
    visit(k) {
      const y = this.frames[k], m = this.motions[k], p = D(blur(warp(this.x, m), this.lens));
      const r = img(y.h, y.w);
      const b = Math.ceil((Math.abs(m[0]) + Math.abs(m[1])) / 2 + Math.abs(m[2]) * (y.h + y.w) / 2) + 2;
      for (let a = b; a < y.h - b; a++) for (let c = b; c < y.w - b; c++) {
        const q = a * y.w + c; r.d[q] = y.d[q] - p.d[q];
      }
      const dx = unwarp(blur(U(r), this.lens), m);
      for (let i = 0; i < dx.d.length; i++) { dx.d[i] *= this.beta; this.x.d[i] += dx.d[i]; }
      return dx;
    }
    step(y) {
      const reg = register(this.ref, y, 128, this.cache);
      this.lastPeak = reg.peak;
      if (Math.max(Math.abs(reg.m[0]), Math.abs(reg.m[1])) > this.maxShift || reg.peak < this.minPeak) return null;
      const m = toHr(reg.m);
      this.push(y, m);
      const acc = img(this.x.h, this.x.w);
      for (let v = 0; v < Math.min(this.visits, this.frames.length); v++) {
        const dx = this.visit(this.head % this.frames.length);
        for (let i = 0; i < acc.d.length; i++) acc.d[i] += dx.d[i];
        this.head = (this.head + 1) % this.frames.length;
      }
      this.lastResidue = acc;
      return m;
    }
  }

  root.Tarina = { kw, blur, shiftRows, shiftCols, warp, unwarp, gradient, D, U, fft2, phaseCorr, lkRefine, register, toHr, Ring, img };
})(typeof window !== 'undefined' ? window : globalThis);
