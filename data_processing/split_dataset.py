from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


#paths
RAW_DIR = Path("pilot_whiten_foveated/raw")
COLOUR_VECTOR_DIR = Path("pilot_whiten_foveated/colour_vectors")
SPLIT_DIR = Path("pilot_whiten_foveated/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


#load pilot data
pilot_data = {}

pilot_data["train"] = np.load(
    RAW_DIR / "eeg_train_float16.npy",
    allow_pickle=True
)[()]

pilot_data["test"] = np.load(
    RAW_DIR / "eeg_test_float16.npy",
    allow_pickle=True
)[()]


#load colour vectors
colour_gt_train = np.load(COLOUR_VECTOR_DIR / "colour_gt_train.npy")

colour_gt_test = np.load(COLOUR_VECTOR_DIR / "colour_gt_test.npy")


#validation
train_eeg = pilot_data["train"]["data"]
test_eeg_raw = pilot_data["test"]["data"]

train_stimuli = np.array(pilot_data["train"]["stimuli"])
test_stimuli = np.array(pilot_data["test"]["stimuli"])

assert train_eeg.shape[0] == colour_gt_train.shape[0], (
    train_eeg.shape,
    colour_gt_train.shape
)

assert train_eeg.shape[0] == train_stimuli.shape[0], (
    train_eeg.shape,
    train_stimuli.shape
)

assert colour_gt_train.shape[1] == 13, colour_gt_train.shape
assert colour_gt_test.shape[1] == 13, colour_gt_test.shape

print("Train EEG:", train_eeg.shape)
print("Train colour GT:", colour_gt_train.shape)
print("Train stimuli:", train_stimuli.shape)

print("Raw test EEG:", test_eeg_raw.shape)
print("Test colour GT:", colour_gt_test.shape)
print("Test stimuli:", test_stimuli.shape)


test_eeg = test_eeg_raw

print("unmodified test EEG:", test_eeg.shape)


#train validation split
indices = np.arange(train_eeg.shape[0])

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train indices:", train_idx.shape)
print("Validation indices:", val_idx.shape)


#apply splits
eeg_train = train_eeg[train_idx]
eeg_val = train_eeg[val_idx]

colour_train = colour_gt_train[train_idx]
colour_val = colour_gt_train[val_idx]

stimuli_train = train_stimuli[train_idx]
stimuli_val = train_stimuli[val_idx]


#validation
assert eeg_train.shape[0] == colour_train.shape[0] == stimuli_train.shape[0]
assert eeg_val.shape[0] == colour_val.shape[0] == stimuli_val.shape[0]
assert test_eeg.shape[0] == colour_gt_test.shape[0] == test_stimuli.shape[0]

print("\nFinal split shapes:")
print("eeg_train:", eeg_train.shape)
print("colour_train:", colour_train.shape)
print("stimuli_train:", stimuli_train.shape)

print("eeg_val:", eeg_val.shape)
print("colour_val:", colour_val.shape)
print("stimuli_val:", stimuli_val.shape)

print("eeg_test:", test_eeg.shape)
print("colour_test:", colour_gt_test.shape)
print("stimuli_test:", test_stimuli.shape)


#save split indices
np.save(SPLIT_DIR / "train_idx.npy", train_idx)
np.save(SPLIT_DIR / "val_idx.npy", val_idx)


#save split arrays
np.save(SPLIT_DIR / "eeg_train.npy", eeg_train)
np.save(SPLIT_DIR / "eeg_val.npy", eeg_val)
np.save(SPLIT_DIR / "eeg_test.npy", test_eeg)

np.save(SPLIT_DIR / "colour_train.npy", colour_train)
np.save(SPLIT_DIR / "colour_val.npy", colour_val)
np.save(SPLIT_DIR / "colour_test.npy", colour_gt_test)

np.save(SPLIT_DIR / "stimuli_train.npy", stimuli_train)
np.save(SPLIT_DIR / "stimuli_val.npy", stimuli_val)
np.save(SPLIT_DIR / "stimuli_test.npy", test_stimuli)

print("\nSaved split files to:", SPLIT_DIR)

