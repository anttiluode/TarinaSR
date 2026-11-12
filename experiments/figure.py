"""results/crops.png: one crop four ways, enlarged 3x with no smoothing."""
import sys, json, numpy as np
sys.path.insert(0, '.')
import tarina, sihtisr
from skimage import data, color
from scipy.ndimage import shift as ndshift
from PIL import Image, ImageDraw

model = json.load(open('model.json'))
H = color.rgb2gray(data.astronaut()[..., :3]).astype(float)
rng = np.random.default_rng(1); fr = []
for k in range(16):
    s = np.zeros(2) if k == 0 else rng.uniform(-6, 6, 2)
    y = tarina.D(ndshift(H, s, order=3, mode='reflect')) + rng.normal(0, 0.005, (256, 256))
    fr.append(np.round(np.clip(y, 0, 1) * 255) / 255)
R = tarina.Ring(fr[0], sihtisr.apply(fr[0], model))
for y in fr[1:]: R.step(y)
panels = [("bicubic, 1 frame", sihtisr.pil_up(fr[0], Image.BICUBIC)), ("SihtiSR, 1 frame", sihtisr.apply(fr[0], model)),
          ("TarinaSR, 16 shaky frames", R.x), ("the true image", H)]
y0, x0, sz, z = 60, 150, 110, 3
tiles = []
for name, im in panels:
    c = np.uint8(np.clip(im[y0:y0 + sz, x0:x0 + sz], 0, 1) * 255 + 0.5)
    t = Image.fromarray(c).resize((sz * z, sz * z), Image.NEAREST).convert('RGB')
    d = ImageDraw.Draw(t); d.rectangle([0, 0, sz * z, 22], fill=(15, 17, 20)); d.text((6, 5), name, fill=(242, 181, 68))
    tiles.append(t)
out = Image.new('RGB', (len(tiles) * sz * z + (len(tiles) - 1) * 6, sz * z), (15, 17, 20))
for i, t in enumerate(tiles): out.paste(t, (i * (sz * z + 6), 0))
out.save('results/crops.png')
print('saved')
