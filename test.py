import torch
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax, expit
from scipy.stats import pearsonr, spearmanr
from PIL import Image


run_dir = "exp_colour_pilot_whiten_foveated/pilot2_final/sub-01_seed0"

pred_logits = np.load(f"{run_dir}/test_outs_colour.npy")
gt = np.load(f"{run_dir}/test_outs_colour_gt.npy")

print("pred:", pred_logits.shape, pred_logits.min(), pred_logits.max())
print("gt:", gt.shape, gt.min(), gt.max())

# BCE
pred = expit(pred_logits)
pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-8, None)

# KLDiv
# pred = softmax(pred_logits, axis=1,)

print("\nPrediction probability range:")
print("minimum:", pred.min())
print("maximum:", pred.max())

print("Mean prediction:")
print(np.round(pred.mean(axis=0), 4))

print("\nStd prediction:")
print(np.round(pred.std(axis=0), 4))

COLOUR_NAMES = [
    "Red", "Green", "Blue", "Yellow", "Purple", "Brown",
    "Pink", "Orange", "Turquoise", "Beige",
    "White", "Black", "Gray"
]

stim_root = Path("pilot2_whiten_foveated/stimuli")

test_data = torch.load(
    "pilot2_whiten_foveated/cube_ready/sub-01/val.pt",
    weights_only=False
)

test_imgs = np.array([str(x) for x in test_data["img"].flatten()])

STIM_TYPES = [
    "03_colours",
    "04_colours",
    "05_colours",
    "random_chars",
    "train_language",
    "uniform_colour",
    "uniform_fg_bg_geometrical",
    "uniform_fg_bg_objects",
]

# keep only types that are actually present in test set
selected_indices = []
selected_types = []

for stim_type in STIM_TYPES:
    matches = [
        i for i, p in enumerate(test_imgs)
        if p.strip("/").startswith(stim_type + "/")
    ]

    if len(matches) == 0:
        print(f"No test samples found for: {stim_type}")
        continue

    selected_indices.append(matches[0])
    selected_types.append(stim_type)

n = len(selected_indices)
x = np.arange(len(COLOUR_NAMES))

fig, axes = plt.subplots(
    nrows=n,
    ncols=2,
    figsize=(15, 3 * n),
    gridspec_kw={"width_ratios": [1, 4]}
)

if n == 1:
    axes = np.array([axes])

for row, idx in enumerate(selected_indices):
    stim_path = test_imgs[idx]
    img_path = stim_root / stim_path.lstrip("/")

    img = Image.open(img_path).convert("RGB")

    axes[row, 0].imshow(img)
    axes[row, 0].axis("off")
    axes[row, 0].set_title(selected_types[row])

    axes[row, 1].bar(x - 0.2, gt[idx], width=0.4, label="GT")
    axes[row, 1].bar(x + 0.2, pred[idx], width=0.4, label="Prediction")

    axes[row, 1].set_ylim(0, 1)
    axes[row, 1].set_ylabel("Probability")
    axes[row, 1].set_xticks(x)
    axes[row, 1].set_xticklabels(COLOUR_NAMES, rotation=45, ha="right")
    # axes[row, 1].set_title(stim_path)

    if row == 0:
        axes[row, 1].legend()

plt.tight_layout()
plt.show()

#histogram for colour frequencies in train set
# train_data = torch.load(
#     "pilot_whiten/cube_ready/sub-01/train.pt",
#     weights_only=False
# )

# stimuli = train_data["img"].flatten()

# gt_root = Path("pilot_whiten/gts")

# colour_counts = np.zeros(13, dtype=np.int64)

# for stim in stimuli:
#     gt_path = gt_root / str(stim).lstrip("/")

#     gt_img = np.array(Image.open(gt_path))

#     # GT saved as RGB index map
#     indices = gt_img

#     gt_img = np.array(Image.open(gt_path))

#     # Handle both grayscale index maps and RGB index maps
#     if gt_img.ndim == 2:
#         indices = gt_img
#     else:
#         indices = gt_img[:, :, 0]

#     counts = np.bincount(indices.flatten(), minlength=13)
#     colour_counts += counts

# colour_freq = colour_counts / colour_counts.sum()

# print("Pixel counts:")
# for name, count in zip(COLOUR_NAMES, colour_counts):
#     print(f"{name:10s}: {count}")

# plt.figure(figsize=(10, 5))
# plt.bar(COLOUR_NAMES, colour_freq)

# plt.ylabel("Fraction of training pixels")
# plt.title("Colour distribution in training GTs")
# plt.xticks(rotation=45, ha="right")

# plt.tight_layout()
# plt.show()


#colour distribution in train set
# colour_dict = np.load(
#     "pilot2/colour_annotations/train.npy",
#     allow_pickle=True
# ).item()

# colour_vectors = np.stack(list(colour_dict.values())).astype(np.float32)

# mean_distribution = colour_vectors.mean(axis=0)

# plt.figure(figsize=(10, 5))
# plt.bar(COLOUR_NAMES, mean_distribution)
# plt.ylabel("Average target probability")
# plt.title("Average colour distribution in training annotations")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()


top1_pred = pred.argmax(axis=1)
top1_gt = gt.argmax(axis=1)

print("Top-1 colour accuracy:", (top1_pred == top1_gt).mean())

# pearsons = []
# spearmans = []

# for p, g in zip(pred, gt):
#     pearsons.append(pearsonr(p, g)[0])
#     spearmans.append(spearmanr(p, g)[0])

# print("Mean Pearson:", np.nanmean(pearsons))
# print("Mean Spearman:", np.nanmean(spearmans))

print("Prediction mean:", pred.mean(axis=0))
print("GT mean:", gt.mean(axis=0))

pred_thresh = 0.10
gt_thresh = 0.001

pred_binary = pred >= pred_thresh
gt_binary = gt >= gt_thresh

print(
    "\nFraction of samples with any predicted "
    f"colour >= {pred_thresh}:",
    pred_binary.any(axis=1).mean(),
)

print(
    "Mean number of predicted colours "
    f">= {pred_thresh}:",
    pred_binary.sum(axis=1).mean(),
)

print(
    "Mean number of GT colours "
    f">= {gt_thresh}:",
    gt_binary.sum(axis=1).mean(),
)