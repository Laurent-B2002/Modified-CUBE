from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from PIL import Image


PARTICIPANT = 1

#same scale used for the qualitative heatmaps
SOFTMAX_SCALE = 10.0


if PARTICIPANT == 1:

    RESULT_DIR = Path(
        "exp_colour_pilot_whiten_foveated/pilot_final/sub-01_seed0/final_test/")

    STIMULI_PATH = Path("pilot_whiten_foveated/splits/stimuli_test.npy")

    STIM_ROOT = Path("pilot_whiten_foveated/stimuli")

elif PARTICIPANT == 2:

    RESULT_DIR = Path("exp_colour_pilot_whiten_foveated/pilot2_final/sub-01_seed0/final_test/")

    STIMULI_PATH = Path("pilot2_whiten_foveated/splits/stimuli_test.npy")

    STIM_ROOT = Path("pilot2_whiten_foveated/stimuli")

else:

    raise ValueError("PARTICIPANT must be 1 or 2.")


COLOUR_NAMES = [
    "Red",
    "Green",
    "Blue",
    "Yellow",
    "Purple",
    "Brown",
    "Pink",
    "Orange",
    "Turquoise",
    "Beige",
    "White",
    "Black",
    "Gray",
]


pred_logits = np.load(RESULT_DIR / "test_outs_colour.npy")

stimuli = np.load(STIMULI_PATH, allow_pickle=True).reshape(-1)


print("Raw logits shape:", pred_logits.shape)

print("Raw logits range:", pred_logits.min(),pred_logits.max())

print("Stimulus entries:",len(stimuli))


def extract_word(stimulus):
    stimulus = str(stimulus).replace("\\", "/")
    filename = stimulus.split("/")[-1]
    return Path(filename).stem

#the stimulus file may contain one entry per semantic word, while the prediction array contains one row per EEG trial.
#if so, repeat each stimulus name to match the repeated predictions.
def align_stimuli_with_logits(stimuli,logits):
    stimuli = np.asarray(stimuli).reshape(-1)

    n_stimuli = len(stimuli)
    n_logits = len(logits)

    if n_stimuli == n_logits:

        print("Stimulus identifiers already match prediction rows.")

        return stimuli

    if n_logits % n_stimuli != 0:

        raise ValueError(
            f"Cannot align {n_stimuli} stimuli "
            f"with {n_logits} prediction rows."
        )


    repeats = (n_logits // n_stimuli)

    print("Inferred repetitions per word:", repeats)


    return np.repeat(
        stimuli,
        repeats
    )



#match words to test predictions
trial_stimuli = align_stimuli_with_logits(stimuli,pred_logits)

trial_words = np.array([extract_word(stim) for stim in trial_stimuli])


unique_words = sorted(np.unique(trial_words))


print("\nUnique semantic words:", len(unique_words))



#average raw logits by words
word_predictions = {}
word_mean_logits = {}

for word in unique_words:

    mask = (trial_words == word)


    #average raw logits across repeated EEG trials
    mean_logits = pred_logits[mask].mean(axis=0)


    #scale and softmax for qualitative visualisation
    prediction = softmax(SOFTMAX_SCALE * mean_logits)


    word_mean_logits[word] = (mean_logits)

    word_predictions[word] = (prediction)


#print top 3 colours for each word
print(
    "\n"
    + "=" * 80
)

print(
    f"TOP COLOURS "
    f"(LOGITS x {SOFTMAX_SCALE:g} + SOFTMAX)"
)

print("=" * 80)


for word in unique_words:

    prediction = (word_predictions[word])

    order = np.argsort(prediction)[::-1]


    print(f"\n{word}")


    for j in order[:3]:

        print(
            f"  {COLOUR_NAMES[j]:10s}: "
            f"{prediction[j]:.4f}")


#overall prediction stats

all_word_predictions = np.stack([word_predictions[word] for word in unique_words])


print("\nMean colour score across words:")

for name, value in zip(COLOUR_NAMES, all_word_predictions.mean(axis=0)):

    print(f"{name:10s}: {value:.4f}")


print("\nStd colour score across words:")

for name, value in zip(COLOUR_NAMES, all_word_predictions.std(axis=0)):

    print(f"{name:10s}: {value:.4f}")


#choose words to plot
#change this list depending on which examples you want to show in the prediction distribution

SELECTED_WORDS = [
    "Fire",
    "Love",
    "Heart",
    "Forest",
]


#check that requested words exist
SELECTED_WORDS = [
    word
    for word in SELECTED_WORDS
    if word in word_predictions
]


#find original stimulus for each word
def find_stimulus_path(word):
    for stimulus in stimuli:
        if extract_word(stimulus) == word:
            return Path(str(stimulus).replace("\\", "/"))
    return None


#plot
n = len(SELECTED_WORDS)

x = np.arange(len(COLOUR_NAMES))

fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(15, 3 * n), gridspec_kw={"width_ratios": [1, 4]})


if n == 1:
    axes = np.array([axes])


for row, word in enumerate(SELECTED_WORDS):

    prediction = (word_predictions[word])


    #original textual stimulus
    stim_path = find_stimulus_path(word)


    if stim_path is not None:
        img_path = (STIM_ROOT / str(stim_path).lstrip("/"))

        if img_path.exists():

            img = Image.open(img_path).convert("RGB")

            axes[row, 0].imshow(img)

        else:
            #fall back to showing the word as text
            axes[row, 0].text(
                0.5,
                0.5,
                word,
                ha="center",
                va="center",
                fontsize=24)

    else:

        axes[row, 0].text(
            0.5,
            0.5,
            word,
            ha="center",
            va="center",
            fontsize=24)


    axes[row, 0].axis("off")

    axes[row, 0].set_title(word)


    #predicted colour distribution
    axes[row, 1].bar(x, prediction)


    axes[row, 1].set_ylim(0, max(0.5, all_word_predictions.max() * 1.1))

    axes[row, 1].set_ylabel("Softmax-normalised colour score")

    axes[row, 1].set_xticks(x)

    axes[row, 1].set_xticklabels(COLOUR_NAMES, rotation=45, ha="right")

    axes[row, 1].set_title(f"{word} – Participant {PARTICIPANT}")


plt.tight_layout()

plt.show()