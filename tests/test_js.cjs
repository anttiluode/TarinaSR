// JS ring == Python ring on the same frames (fixture from tests/make_fixture.py)
const fs = require('fs'), path = require('path'), vm = require('vm');
const root = path.join(__dirname, '..');
const ctx = { window: {}, globalThis: {} }; ctx.window = ctx; vm.createContext(ctx);
for (const f of ['model.js', 'sihti.js', 'tarina.js']) vm.runInContext(fs.readFileSync(path.join(root, f), 'utf8'), ctx);
const { Tarina, Sihti, SIHTI_MODEL } = ctx;
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixture.json')));
const { h, w } = fx;
const frames = fx.frames.map(a => ({ d: Float64Array.from(a, v => v / 255), h, w }));
const x0 = Tarina.U(frames[0]);   // plain start, so this checks the ring alone
const R = new Tarina.Ring(frames[0], x0, { lens: 0.6 });
let worst = 0;
frames.slice(1).forEach((y, i) => {
  const m = R.step(y);
  for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(m[k] - fx.motions[i][k]));
});
let xs = 0; for (const v of R.x.d) xs += v;
let wx = 0; fx.x_sample.forEach((v, j) => { wx = Math.max(wx, Math.abs(R.x.d[j * 97] - v)); });
console.log('max motion diff', worst.toExponential(2), ' max pixel diff', wx.toExponential(2), ' sum diff', Math.abs(xs - fx.x_sum).toExponential(2));
if (worst > 1e-4 || wx > 1e-4) { console.error('FAIL: JS and Python disagree'); process.exit(1); }
console.log('PASS: JS ring matches Python');
