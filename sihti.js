// SihtiSR core: sort each 5x5 patch by its structure tensor, apply that bucket's
// 25->4 linear map, lock the 2x2 block to the input pixel. Mirrors sihtisr.py.
(function (root) {
  function pad3(L, h, w) {
    const H = h + 6, W = w + 6, P = new Float32Array(H * W);
    for (let y = 0; y < H; y++) {
      const sy = Math.min(h - 1, Math.max(0, y - 3));
      for (let x = 0; x < W; x++) P[y * W + x] = L[sy * w + Math.min(w - 1, Math.max(0, x - 3))];
    }
    return P;
  }
  // L: Float32Array h*w in 0..1  ->  Float32Array (2h)*(2w)
  function sihtiUp(L, h, w, M) {
    const P = pad3(L, h, w), PW = w + 6, GW = w + 4, GH = h + 4;
    const gx = new Float32Array(GW * GH), gy = new Float32Array(GW * GH);
    for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
      const p = (y + 1) * PW + x + 1;
      gx[y * GW + x] = (P[p + 1] - P[p - 1]) / 2;
      gy[y * GW + x] = (P[p + PW] - P[p - PW]) / 2;
    }
    const out = new Float32Array(4 * h * w), X = new Float32Array(25);
    const na = M.na, ns = M.ns, nc = M.nc;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let jxx = 0, jyy = 0, jxy = 0, mu = 0, k = 0;
      for (let dy = 0; dy < 5; dy++) for (let dx = 0; dx < 5; dx++) {
        const g = (y + dy) * GW + x + dx, a = gx[g], b = gy[g];
        jxx += a * a; jyy += b * b; jxy += a * b;
        const v = P[(y + dy + 1) * PW + x + dx + 1]; X[k++] = v; mu += v;
      }
      jxx /= 25; jyy /= 25; jxy /= 25; mu /= 25;
      const d = Math.sqrt(((jxx - jyy) / 2) ** 2 + jxy * jxy), m = (jxx + jyy) / 2;
      const s1 = Math.sqrt(m + d), s2 = Math.sqrt(Math.max(m - d, 0));
      const coh = (s1 - s2) / (s1 + s2 + 1e-8);
      const ang = 0.5 * Math.atan2(2 * jxy, jxx - jyy);
      const a = ((Math.floor((ang + Math.PI / 2) / Math.PI * na) % na) + na) % na;
      const s = (s1 > M.sthr[0]) + (s1 > M.sthr[1]);
      const c = (coh > M.coh[0]) + (coh > M.coh[1]);
      const Wb = M.W[(a * ns + s) * nc + c];
      let o0 = 0, o1 = 0, o2 = 0, o3 = 0;
      for (k = 0; k < 25; k++) {
        const v = X[k] - mu, r = Wb[k];
        o0 += v * r[0]; o1 += v * r[1]; o2 += v * r[2]; o3 += v * r[3];
      }
      const lock = L[y * w + x] - (mu + (o0 + o1 + o2 + o3) / 4), base = mu + lock;
      const q = (2 * y) * (2 * w) + 2 * x;
      out[q] = base + o0; out[q + 1] = base + o1;
      out[q + 2 * w] = base + o2; out[q + 2 * w + 1] = base + o3;
    }
    return out;
  }
  // plain bicubic x2 (Keys a=-0.5), the "core" the residue is added to
  const K = [-0.0234375, 0.2265625, 0.8671875, -0.0703125];
  function bicubicUp(L, h, w) {
    const T = new Float32Array(h * 2 * w), O = new Float32Array(4 * h * w);
    const cl = (v, n) => v < 0 ? 0 : v >= n ? n - 1 : v;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let a = 0, b = 0;
      for (let t = 0; t < 4; t++) {
        a += K[t] * L[y * w + cl(x - 2 + t, w)];
        b += K[3 - t] * L[y * w + cl(x - 1 + t, w)];
      }
      T[y * 2 * w + 2 * x] = a; T[y * 2 * w + 2 * x + 1] = b;
    }
    const W2 = 2 * w;
    for (let y = 0; y < h; y++) for (let x = 0; x < W2; x++) {
      let a = 0, b = 0;
      for (let t = 0; t < 4; t++) {
        a += K[t] * T[cl(y - 2 + t, h) * W2 + x];
        b += K[3 - t] * T[cl(y - 1 + t, h) * W2 + x];
      }
      O[(2 * y) * W2 + x] = a; O[(2 * y + 1) * W2 + x] = b;
    }
    return O;
  }
  root.Sihti = { sihtiUp, bicubicUp };
})(typeof window !== 'undefined' ? window : globalThis);
