"""Python run of the ring on a fixed shaky sequence; tests/test_js.cjs replays it in JS."""
import sys, json, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [here, os.path.dirname(here)]
import tarina, sihtisr
from chart import chart
from scipy.ndimage import shift as ndshift, rotate


def sequence(n=6, seed=1):
    H = chart()
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for k in range(n):
        s = np.zeros(2) if k == 0 else rng.uniform(-5, 5, 2)
        t = 0.0 if k == 0 else rng.uniform(-0.4, 0.4)
        img = rotate(H, t, reshape=False, order=3, mode='reflect') if t else H
        y = tarina.D(ndshift(img, s, order=3, mode='reflect')) + rng.normal(0, 0.005, (H.shape[0] // 2, H.shape[1] // 2))
        frames.append(np.round(np.clip(y, 0, 1) * 255).astype(int))
        truth.append([*s, t])
    return H, frames, truth


if __name__ == '__main__':
    model = json.load(open(os.path.join(os.path.dirname(here), 'model.json')))
    H, frames, truth = sequence()
    f = [fr / 255.0 for fr in frames]
    x0 = tarina.U(f[0])            # plain start, so the test checks the ring alone
    R = tarina.Ring(f[0], x0, lens=0.6)
    motions = [R.step(y).tolist() for y in f[1:]]
    json.dump({"h": f[0].shape[0], "w": f[0].shape[1], "frames": [fr.ravel().tolist() for fr in frames],
               "motions": motions, "x_sum": float(R.x.sum()), "x_sample": R.x.ravel()[::97].round(6).tolist()},
              open(os.path.join(here, 'fixture.json'), 'w'))
    print('motions', np.round(motions, 4))
