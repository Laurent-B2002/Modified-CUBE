from pathlib import Path
import numpy as np
import torch


SPLIT_DIR = Path("pilot/splits")
OUT_DIR = Path("pilot/cube_ready/sub-01")
COLOUR_DIR = Path("pilot/colour_annotations")

OUT_DIR.mkdir(parents=True, exist_ok=True)
COLOUR_DIR.mkdir(parents=True, exist_ok=True)

#make labels for the paths
def stimulus_to_text(stimulus):
    return Path(str(stimulus)).stem

#number the labels
def make_labels(stimuli):
    unique = sorted(set(map(str, stimuli)))
    label_map = {s: i for i, s in enumerate(unique)}
    labels = np.array([label_map[str(s)] for s in stimuli], dtype=np.int64)
    return labels, label_map

#colour vector
def make_colour_dict(stimuli, colour_vectors):
    colour_dict = {}

    for stim, vec in zip(stimuli, colour_vectors):
        stim = str(stim)

        if stim in colour_dict:
            continue

        colour_dict[stim] = vec.astype(np.float32)

    return colour_dict


def save_cube_pt(path, eeg, stimuli, times):
    labels, label_map = make_labels(stimuli)

    eeg = eeg.astype(np.float32)

    if eeg.ndim == 3:
        eeg = eeg[:, None, :, :]          # (N, 1, C, T)
        labels = labels[:, None]          # (N, 1)
        stimuli = np.array(stimuli)[:, None]
        text = np.array([stimulus_to_text(s) for s in stimuli[:, 0]])[:, None]
        session = np.zeros((eeg.shape[0], 1), dtype=np.int64)

    elif eeg.ndim == 4:
        # For true test data like (20, 77, 64, 250)
        n_items, n_repeats = eeg.shape[:2]

        labels = labels[:, None].repeat(n_repeats, axis=1)
        stimuli = np.array(stimuli)[:, None].repeat(n_repeats, axis=1)
        text = np.array([stimulus_to_text(s) for s in stimuli[:, 0]])[:, None].repeat(n_repeats, axis=1)
        session = np.zeros((n_items, n_repeats), dtype=np.int64)

    else:
        raise ValueError(f"Unexpected EEG shape: {eeg.shape}")

    payload = {
        "eeg": eeg,
        "label": labels,
        "img": stimuli,
        "text": text,
        "session": session,
        "times": times,
    }

    torch.save(payload, path)

    return label_map


#load split files
eeg_train = np.load(SPLIT_DIR / "eeg_train.npy")
eeg_val = np.load(SPLIT_DIR / "eeg_val.npy")

colour_train = np.load(SPLIT_DIR / "colour_train.npy")
colour_val = np.load(SPLIT_DIR / "colour_val.npy")

stimuli_train = np.load(SPLIT_DIR / "stimuli_train.npy", allow_pickle=True)
stimuli_val = np.load(SPLIT_DIR / "stimuli_val.npy", allow_pickle=True)

#use original pilot train times
raw_train = np.load("pilot/raw/eeg_train_float16.npy", allow_pickle=True)[()]
times = raw_train["times"]


#cube style eeg files
train_label_map = save_cube_pt(
    OUT_DIR / "train.pt",
    eeg_train,
    stimuli_train,
    times,
)

val_label_map = save_cube_pt(
    OUT_DIR / "test.pt",
    eeg_val,
    stimuli_val,
    times,
)


#cube style colour files
colour_train_dict = make_colour_dict(stimuli_train, colour_train)
colour_val_dict = make_colour_dict(stimuli_val, colour_val)

np.save(COLOUR_DIR / "train.npy", colour_train_dict)
np.save(COLOUR_DIR / "test.npy", colour_val_dict)


#label maps for reference
np.save(OUT_DIR / "train_label_map.npy", train_label_map)
np.save(OUT_DIR / "val_label_map.npy", val_label_map)


#validation
train_pt = torch.load(OUT_DIR / "train.pt", weights_only=False)
test_pt = torch.load(OUT_DIR / "test.pt", weights_only=False)

print("Saved:", OUT_DIR / "train.pt")
print("Saved:", OUT_DIR / "test.pt")
print("Saved:", COLOUR_DIR / "train.npy")
print("Saved:", COLOUR_DIR / "test.npy")

print("\ntrain.pt keys:", train_pt.keys())
print("train eeg:", train_pt["eeg"].shape)
print("train img:", train_pt["img"].shape)

print("\ntest.pt keys:", test_pt.keys())
print("test eeg:", test_pt["eeg"].shape)
print("test img:", test_pt["img"].shape)

print("\ncolour train entries:", len(colour_train_dict))
print("colour val entries:", len(colour_val_dict))

first_img = str(train_pt["img"][0,0])
print("\nFirst train img:", first_img)
print("First colour vector:", colour_train_dict[first_img])
print("Colour vector shape:", colour_train_dict[first_img].shape)

x = torch.load("pilot/cube_ready/sub-01/train.pt", weights_only=False)

for k, v in x.items():
    try:
        print(k, v.shape)
    except:
        print(k, type(v))