// TärinäSR page: feeds shaky frames (demo, camera or video file) into Tarina.Ring.
(() => {
  const T = window.Tarina, S = window.Sihti, M = window.SIHTI_MODEL;
  const $ = id => document.getElementById(id);
  const LW = 240, LH = 180, HW = 2 * LW, HH = 2 * LH, MARGIN = 24, DEMO_FRAMES = 200;

  let mode = 'demo', ring = null, x0 = null, truth = null, chroma = null, bic = null;
  let frozen = false, skipped = 0, written = 0, lastFrame = null, demoT = 0, misses = 0;
  let lens = 0, zoom = 2, cz = [HH / 2, HW / 2], busy = false, fpsT = performance.now(), fpsN = 0, stream = null;
  let rng = mulberry(7);

  // ---------- small helpers ----------
  function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const gauss = () => { const u = rng() || 1e-9, v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const toImg = (arr, h, w) => ({ d: Float64Array.from(arr), h, w });
  function psnr(a, b) {
    let s = 0, n = 0;
    for (let y = MARGIN; y < HH - MARGIN; y++) for (let x = MARGIN; x < HW - MARGIN; x++) {
      const k = y * HW + x, v = Math.min(1, Math.max(0, a[k])) - b[k]; s += v * v; n++;
    }
    return -10 * Math.log10(s / n);
  }
  function msg(t) { $('msg').textContent = t || ''; }

  // ---------- demo scene ----------
  function chartTruth() {
    const c = document.createElement('canvas'); c.width = HW; c.height = HH;
    const g = c.getContext('2d');
    g.fillStyle = '#d8d4c8'; g.fillRect(0, 0, HW, HH);
    // Siemens star
    const cx = 120, cy = 130, R = 96;
    for (let i = 0; i < 36; i++) {
      g.beginPath(); g.moveTo(cx, cy);
      g.arc(cx, cy, R, i * Math.PI / 18, (i + 0.5) * Math.PI / 18); g.closePath();
      g.fillStyle = '#1b1b1b'; g.fill();
    }
    // text at shrinking sizes
    g.fillStyle = '#151515'; g.textBaseline = 'top';
    const lines = [[22, 'TÄRINÄ makes'], [17, 'resolution from shake'], [13, 'Every frame samples new points.'],
                   [11, 'The ring writes what it got wrong'], [9, 'until the frames stop disagreeing.'], [8, 'Irani & Peleg 1991 · Wronski 2019']];
    let y = 22;
    for (const [s, t] of lines) { g.font = `${s}px Georgia, serif`; g.fillText(t, 236, y); y += s + 8; }
    // line-pair bars, getting finer
    let x = 236;
    for (const p of [5, 4, 3, 2.5, 2, 1.5]) {
      for (let k = 0; k < 6; k++) { g.fillRect(x + k * 2 * p, 200, p, 70); }
      x += 12 * p + 8;
    }
    // fine checker and a soft photo-like gradient
    for (let yy = 284; yy < 344; yy += 3) for (let xx = 236; xx < 330; xx += 3) if (((xx + yy) / 3) % 2 < 1) g.fillRect(xx, yy, 3, 3);
    const gr = g.createRadialGradient(400, 315, 4, 400, 315, 50); gr.addColorStop(0, '#f6f2e8'); gr.addColorStop(1, '#3a3630');
    g.fillStyle = gr; g.beginPath(); g.arc(400, 315, 40, 0, 7); g.fill();
    g.fillStyle = '#151515'; g.font = '10px Georgia, serif'; g.fillText('16 frames', 66, 250); g.fillText('one ring', 74, 264);
    return lumaOf(g.getImageData(0, 0, HW, HH).data, HW * HH);
  }
  function lumaOf(px, n) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
    return out;
  }
  function imageTruth(im) {
    const c = document.createElement('canvas'); c.width = HW; c.height = HH;
    const g = c.getContext('2d'), s = Math.max(HW / im.width, HH / im.height);
    g.drawImage(im, (HW - im.width * s) / 2, (HH - im.height * s) / 2, im.width * s, im.height * s);
    return lumaOf(g.getImageData(0, 0, HW, HH).data, HW * HH);
  }
  function demoFrame() {
    const t = demoT++;
    const m = [3.0 * Math.sin(0.37 * t + 1) + 1.3 * Math.sin(1.13 * t) + 0.5 * gauss(),
               3.0 * Math.sin(0.29 * t) + 1.3 * Math.sin(0.91 * t + 2) + 0.5 * gauss(),
               0.004 * Math.sin(0.21 * t)];
    if (t === 0) { m[0] = m[1] = m[2] = 0; }
    const y = T.D(T.warp(toImg(truth, HH, HW), m));
    for (let i = 0; i < y.d.length; i++) y.d[i] = Math.round(Math.min(1, Math.max(0, y.d[i] + 0.005 * gauss())) * 255) / 255;
    return y;
  }

  // ---------- camera / video frames ----------
  const grab = document.createElement('canvas'), gctx = grab.getContext('2d', { willReadFrequently: true });
  function videoFrame(v) {
    if (!v.videoWidth || v.readyState < 2) return null;
    const w = Math.min(LW, v.videoWidth), h = Math.min(LH, v.videoHeight);
    if (w < LW || h < LH) { msg('video is smaller than 240×180'); return null; }
    grab.width = LW; grab.height = LH;
    // the camera's own pixels: a centre crop, never rescaled (rescaling would remove the aliasing we need)
    gctx.drawImage(v, (v.videoWidth - LW) >> 1, (v.videoHeight - LH) >> 1, LW, LH, 0, 0, LW, LH);
    const px = gctx.getImageData(0, 0, LW, LH).data;
    const y = T.img(LH, LW), cb = new Float32Array(LW * LH), cr = new Float32Array(LW * LH);
    for (let i = 0; i < LW * LH; i++) {
      const r = px[4 * i] / 255, g = px[4 * i + 1] / 255, b = px[4 * i + 2] / 255;
      y.d[i] = 0.299 * r + 0.587 * g + 0.114 * b; cb[i] = b - y.d[i]; cr[i] = r - y.d[i];
    }
    y.cb = cb; y.cr = cr;
    let m = 0, q = 0; for (let i = 0; i < y.d.length; i += 11) { m += y.d[i]; q += y.d[i] * y.d[i]; }
    const n = Math.ceil(y.d.length / 11); if (q / n - (m / n) ** 2 < 1e-4) return null;   // blank or not yet decoded
    return y;
  }
  function matchTo(ref, y) {           // auto-exposure drift: match mean and contrast of the centre
    let mr = 0, my = 0, vr = 0, vy = 0, n = 0;
    for (let a = 40; a < LH - 40; a++) for (let b = 40; b < LW - 40; b++) { const k = a * LW + b; mr += ref.d[k]; my += y.d[k]; n++; }
    mr /= n; my /= n;
    for (let a = 40; a < LH - 40; a++) for (let b = 40; b < LW - 40; b++) { const k = a * LW + b; vr += (ref.d[k] - mr) ** 2; vy += (y.d[k] - my) ** 2; }
    const g = Math.sqrt(vr / (vy + 1e-12));
    if (Math.abs(g - 1) < 0.01 && Math.abs(mr - my) < 0.004) return y;
    const o = T.img(LH, LW); for (let i = 0; i < o.d.length; i++) o.d[i] = (y.d[i] - my) * g + mr;
    return o;
  }

  // ---------- ring ----------
  function start(ref) {
    x0 = toImg(S.sihtiUp(Float32Array.from(ref.d), LH, LW, M), HH, HW);
    bic = truth ? S.bicubicUp(Float32Array.from(ref.d), LH, LW) : null;
    chroma = ref.cb ? [S.bicubicUp(ref.cb, LH, LW), S.bicubicUp(ref.cr, LH, LW)] : null;
    ring = new T.Ring(ref, x0, { minPeak: 0.03, lens });
    skipped = 0; written = 0; misses = 0; lastFrame = ref;
    if (truth) { $('pBic').textContent = psnr(bic, truth).toFixed(2) + ' dB'; $('pOne').textContent = psnr(x0.d, truth).toFixed(2) + ' dB'; }
  }
  function feed(y) {
    if (!ring) { start(y); return; }
    if (lastFrame && y.cb) {              // the same camera frame twice adds nothing
      let s = 0; for (let i = 0; i < y.d.length; i += 7) s += Math.abs(y.d[i] - lastFrame.d[i]);
      if (s / (y.d.length / 7) < 4e-4) return;
    }
    lastFrame = y;
    const m = ring.step(y.cb ? matchTo(ring.ref, y) : y);
    if (!m) {
      skipped++;
      if (++misses > 15 && mode !== 'demo') { msg('Lost the first frame. Starting again from here.'); start(y); }
      return;
    }
    if (misses) msg(''); misses = 0; written++;
    $('sShift').textContent = `${m[0].toFixed(2)}, ${m[1].toFixed(2)}`;
    $('sRot').textContent = (m[2] * 180 / Math.PI).toFixed(2) + '°';
  }

  // ---------- drawing ----------
  const off = document.createElement('canvas'); off.width = HW; off.height = HH;
  const octx = off.getContext('2d');
  const views = ['cOne', 'cRing', 'cRes'].map(id => { const c = $(id); c.width = HW; c.height = HH; return c; });
  function paint(canvas, lum, kind) {
    const im = octx.createImageData(HW, HH), p = im.data;
    for (let i = 0; i < HW * HH; i++) {
      let r, g, b;
      if (kind === 'res') { r = g = b = 0.5 + 6 * (lum[i] - x0.d[i]); }
      else if (chroma) { const Y = lum[i], cb = chroma[0][i], cr = chroma[1][i]; r = Y + cr; b = Y + cb; g = (Y - 0.299 * r - 0.114 * b) / 0.587; }
      else { r = g = b = lum[i]; }
      p[4 * i] = 255 * r; p[4 * i + 1] = 255 * g; p[4 * i + 2] = 255 * b; p[4 * i + 3] = 255;
    }
    octx.putImageData(im, 0, 0);
    const c = canvas.getContext('2d'), sw = HW / zoom, sh = HH / zoom;
    const sx = Math.min(HW - sw, Math.max(0, cz[1] - sw / 2)), sy = Math.min(HH - sh, Math.max(0, cz[0] - sh / 2));
    c.imageSmoothingEnabled = zoom === 1;
    c.drawImage(off, sx, sy, sw, sh, 0, 0, HW, HH);
    canvas._src = [sx, sy, sw, sh];
  }
  function drawPhase() {
    const c = $('phase'), g = c.getContext('2d'), W = c.width;
    g.fillStyle = '#0b0c0e'; g.fillRect(0, 0, W, W);
    g.strokeStyle = '#2a2f37'; g.lineWidth = 2;
    g.strokeRect(1, 1, W - 2, W - 2); g.beginPath(); g.moveTo(W / 2, 0); g.lineTo(W / 2, W); g.moveTo(0, W / 2); g.lineTo(W, W / 2); g.stroke();
    g.fillStyle = '#5b5f66'; g.font = '20px system-ui'; g.fillText('one low-res pixel = 2×2 high-res', 10, W - 12);
    if (!ring) return;
    const cells = new Set(), n = ring.motions.length;
    ring.motions.forEach((m, k) => {
      const fy = ((m[0] % 2) + 2) % 2, fx = ((m[1] % 2) + 2) % 2;
      cells.add(Math.floor(fy * 2) * 4 + Math.floor(fx * 2));
      const age = ((ring.count - 1 - k) % ring.size + ring.size) % ring.size;
      g.fillStyle = k === (ring.count - 1) % ring.size ? '#f2b544' : `rgba(111,211,194,${1 - 0.8 * age / n})`;
      g.beginPath(); g.arc(fx / 2 * W, fy / 2 * W, 9, 0, 7); g.fill();
    });
    const cov = cells.size / 16;
    $('cover').style.width = (100 * cov).toFixed(0) + '%';
    $('cover').parentElement.nextElementSibling.firstElementChild.textContent = `phase coverage ${(100 * cov).toFixed(0)}% of 16 cells`;
  }
  function render() {
    if (!ring) return;
    paint(views[0], x0.d, 'img'); paint(views[1], ring.x.d, 'img'); paint(views[2], ring.x.d, 'res');
    drawPhase();
    $('sFrames').textContent = written; $('sSkip').textContent = skipped;
    let s = 0; const r = ring.lastResidue.d; for (let i = 0; i < r.length; i++) s += r[i] * r[i];
    $('sRes').textContent = (255 * Math.sqrt(s / r.length)).toFixed(2) + ' grey';
    if (truth) $('pRing').textContent = psnr(ring.x.d, truth).toFixed(2) + ' dB';
  }

  // ---------- loop ----------
  function loop() {
    if (!busy && !frozen) {
      busy = true;
      let y = null;
      if (mode === 'demo' && demoT < DEMO_FRAMES) y = demoFrame();
      else if (mode !== 'demo') y = videoFrame($('vid'));
      if (y) { feed(y); render(); fpsN++; }
      if (mode === 'demo' && demoT === DEMO_FRAMES) msg(`Done: ${DEMO_FRAMES} shaky frames. Press Reset to run it again.`);
      busy = false;
    }
    const now = performance.now();
    if (now - fpsT > 1000) { $('sFps').textContent = (fpsN * 1000 / (now - fpsT)).toFixed(1) + ' frames/s'; fpsT = now; fpsN = 0; }
    requestAnimationFrame(loop);
  }

  // ---------- modes ----------
  function setMode(m) {
    mode = m; ring = null;
    lens = m === 'demo' ? 0 : 0.7; $('lens').value = String(lens); frozen = false; $('bFreeze').classList.remove('on'); msg('');
    $('mDemo').classList.toggle('on', m === 'demo'); $('mCam').classList.toggle('on', m === 'cam');
    document.querySelectorAll('.demo').forEach(e => e.style.display = m === 'demo' ? '' : 'none');
    if (m !== 'demo') { truth = null; }
    if (m !== 'cam' && stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    $('howto').textContent = m === 'demo'
      ? 'Demo: a known picture is shaken by a few pixels and a fraction of a degree, shrunk 2×, given camera noise and rounded to 8 bits. The ring never sees the true shifts. The dB numbers compare each result with the true picture.'
      : m === 'cam'
      ? 'Camera: point it at fine detail (small text works well) and hold it in your hand. Small natural shake is what it needs; a phone resting on a table gives nothing new. The view is the camera’s own pixels from the centre, enlarged 2×.'
      : 'Video file: a handheld clip of a still scene works best. Frames are read from the centre as the video plays.';
  }
  $('mDemo').onclick = () => { truth = chartTruth(); demoT = 0; rng = mulberry(7); setMode('demo'); };
  $('mCam').onclick = async () => {
    setMode('cam');
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      const v = $('vid'); v.srcObject = stream; v.loop = false; await v.play();
    } catch (e) { msg('No camera: ' + e.message); }
  };
  $('fVid').onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    setMode('video');
    const v = $('vid'); v.srcObject = null; v.src = URL.createObjectURL(f); v.loop = true; await v.play();
  };
  $('fImg').onchange = e => {
    const f = e.target.files[0]; if (!f) return;
    const im = new Image(); im.onload = () => { truth = imageTruth(im); demoT = 0; rng = mulberry(7); setMode('demo'); }; im.src = URL.createObjectURL(f);
  };
  $('lens').onchange = e => { lens = +e.target.value; if (ring) ring.lens = lens; };
  $('zoom').onchange = e => { zoom = +e.target.value; render(); };
  $('bFreeze').onclick = () => { frozen = !frozen; $('bFreeze').classList.toggle('on', frozen); };
  $('bReset').onclick = () => {
    if (mode === 'demo') { demoT = 0; rng = mulberry(7); }
    ring = null; msg('');
  };
  $('bSave').onclick = () => {
    if (!ring) return;
    paint(views[1], ring.x.d, 'img');
    off.toBlob(b => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'tarinasr.png'; a.click(); });
  };
  views.forEach(c => c.addEventListener('click', e => {
    const r = c.getBoundingClientRect(), [sx, sy, sw, sh] = c._src || [0, 0, HW, HH];
    cz = [sy + (e.clientY - r.top) / r.height * sh, sx + (e.clientX - r.left) / r.width * sw];
    if (zoom === 1) { zoom = 2; $('zoom').value = '2'; }
    render();
  }));

  window.TarinaApp = { state: () => ({ written, skipped, demoT, psnrRing: ring && truth ? psnr(ring.x.d, truth) : null,
    psnrOne: x0 && truth ? psnr(x0.d, truth) : null, psnrBic: bic && truth ? psnr(bic, truth) : null }) };
  truth = chartTruth(); setMode('demo'); cz = [130, 300];
  requestAnimationFrame(loop);
})();
