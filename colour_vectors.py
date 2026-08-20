from pathlib import Path
import numpy as np
from PIL import Image


#paths
RAW_DIR = Path("pilot_whiten/raw")
GT_ROOT = Path("pilot_whiten/gts")
OUT_DIR = Path("pilot_whiten/colour_vectors")
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

pilot_data["train"] = np.load(
    RAW_DIR / "eeg_train_float16.npy",
    allow_pickle=True
)[()]

pilot_data["test"] = np.load(
    RAW_DIR / "eeg_test_float16.npy",
    allow_pickle=True
)[()]


#map path
def stimulus_to_gt_path(stimulus):
    return GT_ROOT / str(stimulus).lstrip("/")


#gt to 13 dimension histogram
def gt_index_image_to_colour_vector(gt_path):
    gt = Image.open(gt_path).convert("RGB")
    arr = np.array(gt)

    #gt is an index map stored as RGB where R=G=B=index
    labels = arr[:, :, 0]

    unique_values = np.unique(labels)

    invalid = unique_values[(unique_values < 0) | (unique_values > 12)]
    if len(invalid) > 0:
        raise ValueError(
            f"Invalid palette indices in {gt_path}: {invalid}"
        )

    counts = np.bincount(
        labels.reshape(-1),
        minlength=len(COLOURS)
    ).astype(np.float32)

    total = counts.sum()

    if total == 0:
        raise ValueError(f"No pixels found in {gt_path}")

    hist = counts / total

    return hist


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

            cache[stim] = gt_index_image_to_colour_vector(gt_path)

        vectors.append(cache[stim])

    return np.stack(vectors).astype(np.float32)


#train test vectors
colour_gt_train = build_colour_vectors(
    pilot_data["train"]["stimuli"]
)

colour_gt_test = build_colour_vectors(
    pilot_data["test"]["stimuli"]
)


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
example_vec = gt_index_image_to_colour_vector(example_gt)

print("\nExample stimulus:", example_stim)
print("Example GT:", example_gt)

for idx, value in enumerate(example_vec):
    if value > 0:
        print(idx, COLOURS[idx]["name"], value)