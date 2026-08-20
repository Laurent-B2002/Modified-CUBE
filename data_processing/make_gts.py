from pathlib import Path
from PIL import Image
import numpy as np
import time

STIM_ROOT = Path("pilot_whiten_foveated/stimuli")
GT_ROOT = Path("pilot_whiten_foveated/gts")


new_train = np.load("pilot_whiten_foveated/raw/eeg_train_float16.npy", allow_pickle=True)[()]
new_test = np.load("pilot_whiten_foveated/raw/eeg_test_float16.npy", allow_pickle=True)[()]


missing = []
for s in list(new_train["stimuli"]) + list(new_test["stimuli"]):
    p = GT_ROOT / str(s).lstrip("/")
    if not p.exists():
        missing.append(str(s))

COLOURS = [
    {"name": "Red", "rgb": (255, 0, 0)},
    {"name": "Green", "rgb": (0, 255, 0)},
    {"name": "Blue", "rgb": (0, 0, 255)},
    {"name": "Yellow", "rgb": (255, 255, 0)},
    {"name": "Purple", "rgb": (121, 58, 144)},
    {"name": "Brown", "rgb": (113, 69, 41)},
    {"name": "Pink", "rgb": (225, 118, 178)},
    {"name": "Orange", "rgb": (255, 128, 0)},
    {"name": "Turquoise", "rgb": (63, 185, 177)},
    {"name": "Beige", "rgb": (195, 168, 126)},
    {"name": "White", "rgb": (255, 255, 255)},
    {"name": "Black", "rgb": (0, 0, 0)},
    {"name": "Gray", "rgb": (128, 128, 128)},
]

def rgb_to_palette_indices(image, colours):
    palette = np.array([c["rgb"] for c in colours], dtype=np.float32)

    h, w, _ = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)

    distances = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    indices = distances.argmin(axis=1)

    return indices.reshape(h, w).astype(np.uint8)


#`missing` should be your list of missing GT stimulus paths
#e.g. ['/03_colours/stimuli__0000001.png', ...]

created = 0
skipped = 0

start = time.time()

for stim in sorted(set(missing)):
    stim_path = STIM_ROOT / stim.lstrip("/")
    gt_path = GT_ROOT / stim.lstrip("/")

    if gt_path.exists():
        skipped += 1
        continue

    if not stim_path.exists():
        raise FileNotFoundError(f"Missing stimulus image: {stim_path}")

    gt_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(stim_path).convert("RGB")
    arr = np.array(img)

    label_map = rgb_to_palette_indices(arr, COLOURS)

    #save as RGB index map, matching your existing GT format:
    #index value is repeated across R,G,B channels.
    gt_rgb = np.repeat(label_map[:, :, None], 3, axis=2).astype(np.uint8)

    Image.fromarray(gt_rgb).save(gt_path)

    created += 1
    if created % 100 == 0:
        print(f"Created {created} GTs")

print("Time:", time.time() - start)
print("Created GTs:", created)
print("Skipped existing GTs:", skipped)