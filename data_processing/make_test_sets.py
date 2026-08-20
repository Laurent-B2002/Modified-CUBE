from pathlib import Path
import numpy as np
import torch

ROOT = Path("pilot_whiten_foveated")

SPLIT_DIR = ROOT / "splits"
OUT_DIR = ROOT / "cube_ready" / "sub-01"
COLOUR_DIR = ROOT / "colour_annotations"

OUT_DIR.mkdir(parents=True, exist_ok=True)
COLOUR_DIR.mkdir(parents=True, exist_ok=True)

def stimulus_to_text(stimulus):
    return Path(str(stimulus)).stem


def make_labels(stimuli):
    unique = sorted(set(map(str, stimuli)))

    label_map = {
        stimulus: i
        for i, stimulus in enumerate(unique)}

    labels = np.array(
        [label_map[str(s)] for s in stimuli],
        dtype=np.int64)

    return labels, label_map


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
    stimuli = np.asarray(stimuli)

    if eeg.ndim == 3:

        eeg = eeg[:, None, :, :]

        labels = labels[:, None]

        stimuli = stimuli[:, None]

        text = np.array([
            stimulus_to_text(s)
            for s in stimuli[:, 0]
        ])[:, None]

        session = np.zeros(
            (eeg.shape[0], 1),
            dtype=np.int64
        )

    elif eeg.ndim == 4:

        n_items, n_repeats = eeg.shape[:2]

        labels = labels[:, None].repeat(n_repeats, axis=1)

        stimuli = stimuli[:, None].repeat(n_repeats, axis=1)

        text = np.array([
            stimulus_to_text(s)
            for s in stimuli[:, 0]
        ])[:, None].repeat(n_repeats, axis=1)

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


eeg_test = np.load(SPLIT_DIR / "eeg_test.npy")

colour_test = np.load(SPLIT_DIR / "colour_test.npy")

stimuli_test = np.load(SPLIT_DIR / "stimuli_test.npy", allow_pickle=True)


print("Loaded true test split:")
print("EEG:", eeg_test.shape)
print("Colour:", colour_test.shape)
print("Stimuli:", stimuli_test.shape)


assert len(eeg_test) == len(stimuli_test), (
    f"EEG/stimulus mismatch: "
    f"{len(eeg_test)} vs {len(stimuli_test)}")

assert len(colour_test) == len(stimuli_test), (
    f"Colour/stimulus mismatch: "
    f"{len(colour_test)} vs {len(stimuli_test)}")

assert colour_test.ndim == 2, (
    f"Expected colour_test to be 2-D, got {colour_test.shape}")

assert colour_test.shape[1] == 13, (
    f"Expected 13 colour channels, got {colour_test.shape}")


train_pt_path = OUT_DIR / "train.pt"

if not train_pt_path.exists():
    raise FileNotFoundError(
        f"Existing train.pt not found: {train_pt_path}")

train_pt = torch.load(train_pt_path, weights_only=False)

times = train_pt["times"]

print("\nUsing times from:")
print(train_pt_path)


test_label_map = save_cube_pt(OUT_DIR / "test.pt", eeg_test, stimuli_test, times,)


colour_test_dict = make_colour_dict(stimuli_test, colour_test)

np.save(COLOUR_DIR / "test.npy", colour_test_dict)


#optional reference file
np.save(OUT_DIR / "test_label_map.npy", test_label_map)


test_pt = torch.load(OUT_DIR / "test.pt", weights_only=False)

saved_colour_dict = np.load(COLOUR_DIR / "test.npy", allow_pickle=True)[()]


print("\n========================================")
print("TRUE TEST FILES CREATED")
print("========================================")

print("Saved:", OUT_DIR / "test.pt")
print("Saved:", COLOUR_DIR / "test.npy")
print("Saved:", OUT_DIR / "test_label_map.npy")


print("\ntest.pt keys:")
print(test_pt.keys())

print("\nEEG shape:")
print(test_pt["eeg"].shape)

print("\nImage/stimulus shape:")
print(test_pt["img"].shape)

print("\nText shape:")
print(test_pt["text"].shape)

print("\nLabel shape:")
print(test_pt["label"].shape)

print("\nSession shape:")
print(test_pt["session"].shape)

print("\nUnique colour annotations:")
print(len(saved_colour_dict))

first_img = str(test_pt["img"].reshape(-1)[0])

print("\nFirst test stimulus:")
print(first_img)

print("\nFirst colour vector:")
print(saved_colour_dict[first_img])

print("\nColour vector shape:")
print(saved_colour_dict[first_img].shape)


assert first_img in saved_colour_dict, (f"{first_img} not found in colour annotation dictionary")

print("\nAll test-set checks passed.")