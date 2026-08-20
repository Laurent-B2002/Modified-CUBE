from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

P1_RESULT_DIR = Path(
    "exp_colour_pilot_whiten_foveated/pilot_final/sub-01_seed0/final_test/")

P2_RESULT_DIR = Path(
    "exp_colour_pilot_whiten_foveated/pilot2_final/sub-01_seed0/final_test/")

P1_STIMULI_PATH = Path(
    "pilot_whiten_foveated/splits/stimuli_test.npy"
)

P2_STIMULI_PATH = Path(
    "pilot2_whiten_foveated/splits/stimuli_test.npy"
)

#multiply averaged logits by this value before softmax
SOFTMAX_SCALE = 10.0


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


#softmax
def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    #subtract maximum for numerical stability
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


#extract stimulus word from path

def extract_word(stimulus):
    stimulus = str(stimulus).replace("\\", "/")
    filename = stimulus.split("/")[-1]
    return Path(filename).stem


#make stimulus array match prediction array

def align_stimuli_with_logits(stimuli, logits, participant_name):
    stimuli = np.asarray(stimuli).reshape(-1)

    n_stimuli = len(stimuli)
    n_logits = len(logits)

    print(f"\n{participant_name}")
    print("Number of saved logits:", n_logits)
    print("Number of stimulus entries:", n_stimuli)


    #already one stimulus identifier per prediction
    if n_stimuli == n_logits:
        print("Stimulus identifiers already match logits.")
        return stimuli


    # Check whether predictions are repeated evenly
    if n_logits % n_stimuli != 0:

        raise ValueError(
            f"Cannot align {n_stimuli} stimulus entries "
            f"with {n_logits} prediction rows."
        )


    repeats = (n_logits // n_stimuli)

    print("Repeats inferred per stimulus:", repeats)


    #the test data are grouped by stimulus, so repeat each stimulus identifier for its corresponding trials.
    stimuli = np.repeat(stimuli, repeats)
    return stimuli



#average logits by semantic word, then apply scaled softmax
def average_by_word(result_dir, stimuli_path, participant_name, scale=SOFTMAX_SCALE,):

    logits = np.load(result_dir / "test_outs_colour.npy")

    stimuli = np.load(stimuli_path, allow_pickle=True).reshape(-1)

    #validate logits
    if (logits.ndim != 2 or logits.shape[1] != 13):

        raise ValueError(
            "Expected colour logits with shape "
            f"(N, 13), got {logits.shape}"
        )

    #match stimulus names to prediction rows
    stimuli = align_stimuli_with_logits(stimuli, logits, participant_name)


    words = np.array([extract_word(s) for s in stimuli])


    unique_words = sorted(np.unique(words))


    mean_logits_by_word = []
    colour_scores_by_word = []

    for word in unique_words:

        mask = (words == word)

        #average RAW logits across all EEG trials belonging to this semantic word
        mean_logits = logits[mask].mean(axis=0)

        #Sharpen the averaged logits for visualisation and convert to relative colour scores
        colour_scores = softmax(mean_logits * scale)

        mean_logits_by_word.append(mean_logits)

        colour_scores_by_word.append(colour_scores)


    return (unique_words, np.asarray(mean_logits_by_word), np.asarray(colour_scores_by_word),)



#load participant results
(p1_words, p1_mean_logits, p1_scores) = average_by_word(P1_RESULT_DIR, P1_STIMULI_PATH, "Participant 1")

(p2_words, p2_mean_logits, p2_scores) = average_by_word(P2_RESULT_DIR, P2_STIMULI_PATH, "Participant 2")


#make sure word order is identical
if p1_words != p2_words:
    raise ValueError("Participant word ordering does not match.")

#check softmax
print("\nParticipant 1 row sums:", p1_scores.sum(axis=1))

print("\nParticipant 2 row sums:", p2_scores.sum(axis=1))


#print top colours for each word
print("\n" + "=" * 90)

print(
    f"TOP COLOURS BY WORD "
    f"(LOGITS × {SOFTMAX_SCALE:g} BEFORE SOFTMAX)"
)

print("=" * 90)


for i, word in enumerate(p1_words):

    p1_order = np.argsort(p1_scores[i])[::-1]

    p2_order = np.argsort(p2_scores[i])[::-1]

    print(f"\n{word}")

    print(
        "  P1:",
        ", ".join(
            f"{COLOUR_NAMES[j]} "
            f"({p1_scores[i, j]:.3f})"
            for j in p1_order[:3]))


    print(
        "  P2:",
        ", ".join(
            f"{COLOUR_NAMES[j]} "
            f"({p2_scores[i, j]:.3f})"
            for j in p2_order[:3]))


#heatmap
def make_heatmap(data, words, title, filename, vmax):
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(data, aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(COLOUR_NAMES)))

    ax.set_xticklabels(COLOUR_NAMES, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(words)))

    ax.set_yticklabels(words)


    ax.set_xlabel("Palette colour")

    ax.set_ylabel("Held-out textual stimulus")

    ax.set_title(title)

    colourbar = fig.colorbar(image, ax=ax)

    colourbar.set_label("Softmax-normalised colour score")

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


#common colour scale for both participants
vmax = max(p1_scores.max(), p2_scores.max())

print("\nShared heatmap maximum:", vmax)

#participant 1
make_heatmap(p1_scores, p1_words,
    (
        "Predicted Colour Scores – Participant 1 "
        f"(Logits × {SOFTMAX_SCALE:g})"
    ),
    "ztest_language_softmax_participant1.png", vmax)

#participant 2
make_heatmap(p2_scores, p2_words,
    (
        "Predicted Colour Scores – Participant 2 "
        f"(Logits × {SOFTMAX_SCALE:g})"
    ),
    "ztest_language_softmax_participant2.png", vmax)