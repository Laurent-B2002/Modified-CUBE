import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr

# for split in ["train", "test"]:
#     f = torch.load(f"pilot/features/{split}.pt", weights_only=False)

#     print(split)
#     print(f.keys())
#     print("image features:", len(f["img_features"]))
#     print("text features:", len(f["text_features"]))

#     first_img = list(f["img_features"].keys())[0]
#     first_text = list(f["text_features"].keys())[0]

#     print("first image key:", first_img)
#     print("image feature shape:", f["img_features"][first_img].shape)
#     print("first text key:", first_text)
#     print("text feature shape:", f["text_features"][first_text].shape)


#for sigmoid
# run_dir = "exp_colour_pilot/pilot_tuning_fix1/sub-01_seed0"

# with open(f"{run_dir}/test_results.json") as f:
#     results = json.load(f)

# print(results)

# pred_colour = np.load(f"{run_dir}/test_outs_colour.npy")
# gt_colour = np.load("pilot/splits/colour_val.npy")

# print("pred:", pred_colour.shape, pred_colour.min(), pred_colour.max())
# print("gt:", gt_colour.shape, gt_colour.min(), gt_colour.max())

# print("First prediction:", pred_colour[0])
# print("First GT:", gt_colour[0])

# names = ["Red","Green","Blue","Yellow","Purple","Brown","Pink",
#          "Orange","Turquoise","Beige","White","Black","Gray"]

# for i in range(5):
#     plt.figure(figsize=(10,3))
#     plt.bar(np.arange(13)-0.2, gt_colour[i], width=0.4, label="GT")
#     plt.bar(np.arange(13)+0.2, 1/(1+np.exp(-pred_colour[i])), width=0.4, label="Prediction")
#     plt.xticks(range(13), names, rotation=45)
#     plt.ylim(0,1)
#     plt.legend()
#     plt.title(f"Validation sample {i}")
#     plt.tight_layout()
#     plt.show()


#for softmax
run_dir = "exp_colour_pilot/pilot_tiny_overfit5/sub-01_seed0"

pred_logits = np.load(f"{run_dir}/test_outs_colour.npy")
gt = np.load(f"{run_dir}/test_outs_colour_gt.npy")

print("pred:", pred_logits.shape, pred_logits.min(), pred_logits.max())
print("gt:", gt.shape, gt.min(), gt.max())

pred = softmax(pred_logits, axis=1)

print("Mean prediction:")
print(np.round(pred.mean(axis=0), 4))

print("\nStd prediction:")
print(np.round(pred.std(axis=0), 4))

COLOUR_NAMES = [
    "Red", "Green", "Blue", "Yellow", "Purple", "Brown",
    "Pink", "Orange", "Turquoise", "Beige",
    "White", "Black", "Gray"
]

for i in range(3):
    pred_prob = softmax(pred_logits[i])

    plt.figure(figsize=(10, 3))
    x = np.arange(13)

    plt.bar(x - 0.2, gt[i], width=0.4, label="GT")
    plt.bar(x + 0.2, pred_prob, width=0.4, label="Prediction")

    plt.xticks(x, COLOUR_NAMES, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title(f"Validation sample {i}")
    plt.legend()
    plt.tight_layout()
    plt.show()


top1_pred = pred.argmax(axis=1)
top1_gt = gt.argmax(axis=1)

print("Top-1 colour accuracy:", (top1_pred == top1_gt).mean())

pearsons = []
spearmans = []

for p, g in zip(pred, gt):
    pearsons.append(pearsonr(p, g)[0])
    spearmans.append(spearmanr(p, g)[0])

print("Mean Pearson:", np.nanmean(pearsons))
print("Mean Spearman:", np.nanmean(spearmans))

print("Prediction mean:", pred.mean(axis=0))
print("GT mean:", gt.mean(axis=0))