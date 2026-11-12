import os, sys, json, shutil, subprocess, unittest
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
sys.path[:0] = [here, root]
import tarina, sihtisr
from chart import chart
from scipy.ndimage import shift as ndshift

MODEL = json.load(open(os.path.join(root, 'model.json')))


def psnr(a, b, m=24):
    a, b = np.clip(a, 0, 1)[m:-m, m:-m], b[m:-m, m:-m]
    return -10 * np.log10(np.mean((a - b) ** 2))


def shaky(H, n=16, even=False, seed=0):
    rng = np.random.default_rng(seed); out = []
    for k in range(n):
        s = np.zeros(2) if k == 0 else rng.uniform(-5, 5, 2)
        if even: s = 2 * np.round(s / 2)
        y = tarina.D(ndshift(H, s, order=3, mode='reflect')) + rng.normal(0, 0.004, (H.shape[0] // 2, H.shape[1] // 2))
        out.append(np.round(np.clip(y, 0, 1) * 255) / 255)
    return out


class Core(unittest.TestCase):
    def test_integer_shift_is_a_roll(self):
        x = np.random.default_rng(0).random((40, 50))
        np.testing.assert_allclose(tarina.warp(x, (2, -3, 0))[5:-5, 5:-5], np.roll(np.roll(x, 2, 0), -3, 1)[5:-5, 5:-5], atol=1e-12)

    def test_unwarp_inverts_warp(self):
        H = chart(); m = (2.3, -1.7, 0.006)
        self.assertLess(np.abs(tarina.unwarp(tarina.warp(H, m), m) - H)[24:-24, 24:-24].mean(), 0.02)

    def test_register_finds_shift_and_rotation(self):
        L = tarina.D(chart()); m = (1.7, -2.4, 0.005)
        est, peak = tarina.register(L, tarina.warp(L, m))
        self.assertLess(np.hypot(est[0] - m[0], est[1] - m[1]), 0.05)
        self.assertLess(abs(est[2] - m[2]), 0.001)
        self.assertGreater(peak, 0.3)


class Ring(unittest.TestCase):
    def run_ring(self, frames):
        R = tarina.Ring(frames[0], sihtisr.apply(frames[0], MODEL))
        for y in frames[1:]: R.step(y)
        return R

    def test_shake_adds_resolution(self):
        H = chart(); fr = shaky(H)
        one = psnr(sihtisr.apply(fr[0], MODEL), H)
        ring = psnr(self.run_ring(fr).x, H)
        self.assertGreater(ring - one, 1.5, (one, ring))   # a hard, heavily aliased chart; real photos gain ~5 dB

    def test_no_new_phases_no_gain(self):
        """Control: whole-pixel shifts only, every frame samples the same phase."""
        H = chart(); fr = shaky(H, even=True)
        one = psnr(sihtisr.apply(fr[0], MODEL), H)
        ring = psnr(self.run_ring(fr).x, H)
        self.assertLess(ring - one, 1.0, (one, ring))

    def test_ring_overwrites_oldest(self):
        L = tarina.D(chart()); R = tarina.Ring(L, tarina.U(L), size=4)
        for k in range(6): R.push(L, np.array([k, 0, 0.0]))
        self.assertEqual(len(R.frames), 4)
        self.assertEqual(sorted(m[0] for m in R.motions), [2, 3, 4, 5])


@unittest.skipIf(shutil.which('node') is None, 'node not installed')
class JavaScript(unittest.TestCase):
    def test_js_matches_python(self):
        subprocess.run([sys.executable, os.path.join(here, 'make_fixture.py')], check=True, capture_output=True)
        r = subprocess.run(['node', os.path.join(here, 'test_js.cjs')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == '__main__':
    unittest.main()
