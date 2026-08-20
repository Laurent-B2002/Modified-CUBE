from pathlib import Path
import numpy as np
from PIL import Image


#paths
RAW_DIR = Path("pilot_whiten_foveated/raw")
GT_ROOT = Path("pilot_whiten_foveated/gts")
OUT_DIR = Path("pilot_whiten_foveated/colour_vectors")
OUT_DIR.mkdir(parents=True, exist_ok=True)


#colour palette indices
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


#load data
pilot_data = {}

pilot_data["train"] = np.load(RAW_DIR / "eeg_train_float16.npy", allow_pickle=True)[()]

pilot_data["test"] = np.load(RAW_DIR / "eeg_test_float16.npy", allow_pickle=True)[()]


#map path
def stimulus_to_gt_path(stimulus):
    return GT_ROOT / str(stimulus).lstrip("/")

N_COLOURS = 13


def make_gaussian_weight_map(h, w, sigma_frac=0.22):
    y, x = np.mgrid[0:h, 0:w]

    cx = (w - 1) / 2
    cy = (h - 1) / 2

    sigma = sigma_frac * min(h, w)

    weights = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    weights = weights / weights.sum()

    return weights.astype(np.float32)

#gt to 13 dimension histogram
def gt_index_image_to_foveated_colour_vector(gt_path, sigma_frac=0.22):
    gt = np.array(Image.open(gt_path))

    #handle either grayscale GTs: (H, W) or RGB index maps: (H, W, 3)
    if gt.ndim == 3:
        gt = gt[:, :, 0]

    h, w = gt.shape
    weights = make_gaussian_weight_map(h, w, sigma_frac=sigma_frac)

    colour_vector = np.zeros(N_COLOURS, dtype=np.float32)

    for colour_idx in range(N_COLOURS):
        colour_vector[colour_idx] = weights[gt == colour_idx].sum()

    colour_vector = colour_vector / (colour_vector.sum() + 1e-8)

    return colour_vector


def gt_index_image_to_object_equal_bg_vector(gt_path, background_weight=0.3, object_weight=0.7,):
    gt = np.array(Image.open(gt_path))

    if gt.ndim == 3:
        gt = gt[:, :, 0]

    counts = np.bincount(gt.flatten(), minlength=N_COLOURS).astype(np.float32)

    bg_idx = counts.argmax()

    object_indices = np.where(counts > 0)[0]
    object_indices = [idx for idx in object_indices if idx != bg_idx]

    vec = np.zeros(N_COLOURS, dtype=np.float32)

    vec[bg_idx] = background_weight

    if len(object_indices) > 0:
        per_object = object_weight  / len(object_indices)
        for idx in object_indices:
            vec[idx] = per_object
    else:
        #fallback for uniform images
        vec[bg_idx] = 1.0

    return vec

#align vector with eeg stimuli
def build_colour_vectors(stimuli):
    cache = {}
    vectors = []

    for i, stim in enumerate(stimuli):
        if i % 1000 == 0:
            print(f"{i}/{len(stimuli)}")

        stim = str(stim)

        if stim not in cache:
            gt_path = stimulus_to_gt_path(stim)

            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT: {gt_path}")

            if ("/03_colours/" in stim or "/04_colours/" in stim or "/05_colours/" in stim):
                cache[stim] = gt_index_image_to_object_equal_bg_vector(gt_path)
            else:
                cache[stim] = gt_index_image_to_foveated_colour_vector(gt_path,sigma_frac=0.22)

        vectors.append(cache[stim])

    return np.stack(vectors).astype(np.float32)


#train test vectors
colour_gt_train = build_colour_vectors(pilot_data["train"]["stimuli"])

colour_gt_test = build_colour_vectors(pilot_data["test"]["stimuli"])


#validate
print("Train colour GT:", colour_gt_train.shape)
print("Test colour GT:", colour_gt_test.shape)

print("Train row sums:", colour_gt_train.sum(axis=1).min(), colour_gt_train.sum(axis=1).max())
print("Test row sums:", colour_gt_test.sum(axis=1).min(), colour_gt_test.sum(axis=1).max())

print("Train min/max:", colour_gt_train.min(), colour_gt_train.max())
print("Test min/max:", colour_gt_test.min(), colour_gt_test.max())


#save
np.save(OUT_DIR / "colour_gt_train.npy", colour_gt_train)
np.save(OUT_DIR / "colour_gt_test.npy", colour_gt_test)

print("Saved:", OUT_DIR / "colour_gt_train.npy")
print("Saved:", OUT_DIR / "colour_gt_test.npy")


#debug
example_stim = str(pilot_data["train"]["stimuli"][0])
example_gt = stimulus_to_gt_path(example_stim)
example_vec = gt_index_image_to_foveated_colour_vector(example_gt)

print("\nExample stimulus:", example_stim)
print("Example GT:", example_gt)

for idx, value in enumerate(example_vec):
    if value > 0:
        print(idx, COLOURS[idx]["name"], value)