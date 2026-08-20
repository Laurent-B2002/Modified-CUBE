import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

COLOUR_NAMES = [
    "Red","Green","Blue","Yellow","Purple","Brown",
    "Pink","Orange","Turquoise","Beige",
    "White","Black","Gray"
]

COLOUR_RGB = [
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0),
    (121,58,144),
    (113,69,41),
    (225,118,178),
    (255,128,0),
    (63,185,177),
    (195,168,126),
    (255,255,255),
    (0,0,0),
    (128,128,128)
]

bar_colours = np.array(COLOUR_RGB) / 255


colour_train = np.load(
    "pilot2/splits/colour_train.npy"
)

stimuli_train = np.load(
    "pilot2/splits/stimuli_train.npy",
    allow_pickle=True
)

eeg_train = np.load(
    "pilot2/splits/eeg_train.npy",
    allow_pickle=True
)

stim_root = Path("pilot2/stimuli")

n_examples = 10

fig, axes = plt.subplots(
    n_examples,
    2,
    figsize=(12, 3*n_examples)
)

for i in range(n_examples):

    stim = str(stimuli_train[i])

    img_path = stim_root / stim.lstrip("/")

    img = Image.open(img_path).convert("RGB")

    vec = colour_train[i]

    axes[i,0].imshow(img)
    axes[i,0].set_title(Path(stim).name)
    axes[i,0].axis("off")

    axes[i,1].bar(
        range(13),
        vec,
        color=bar_colours
    )

    axes[i,1].set_ylim(0,1)
    axes[i,1].set_xticks(range(13))
    axes[i,1].set_xticklabels(
        COLOUR_NAMES,
        rotation=45,
        ha="right"
    )
    axes[i,1].set_ylabel("Probability")

# plt.tight_layout()
# plt.show()


print(colour_train.shape)

print("Mean target:")
print(np.round(colour_train.mean(axis=0), 4))

print()

print("Std target:")
print(np.round(colour_train.std(axis=0), 4))

unique = np.unique(
    np.round(colour_train, 4),
    axis=0
)

print("Unique colour vectors:", len(unique))

for i in range(10):
    plt.figure(figsize=(8,3))
    plt.bar(range(13), colour_train[i])
    plt.title(f"Target {i}")
    plt.show()

print(eeg_train.shape)
print(colour_train.shape)
print(stimuli_train.shape)

print(stimuli_train[:5])
print(colour_train[:5])