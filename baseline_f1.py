from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SPLIT_DIR = Path(
    "pilot_whiten_foveated/splits"
)

TRAIN_COLOUR_PATH = (
    SPLIT_DIR / "colour_train.npy"
)

VAL_COLOUR_PATH = (
    SPLIT_DIR / "colour_val.npy"
)

TRAIN_STIMULI_PATH = (
    SPLIT_DIR / "stimuli_train.npy"
)

VAL_STIMULI_PATH = (
    SPLIT_DIR / "stimuli_val.npy"
)


OUTPUT_DIR = Path(
    "chance_f1_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Use 100 initially for a quick diagnostic run.
# Once everything looks correct, change back to 5000.
N_REPEATS = 5000

RANDOM_SEED = 0

N_COLOURS = 13


# Keep these identical to the current cube.py F1 evaluation.
GT_THRESHOLD = 0.001
PRED_THRESHOLD = 0.10


# Recommended:
# empirical predictions are sampled from training GTs
# belonging to the SAME stimulus category.
CATEGORY_SPECIFIC_EMPIRICAL_SAMPLING = True


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


STIMULUS_TYPES = [
    "03_colours",
    "04_colours",
    "05_colours",
    "random_chars",
    "train_language",
    "uniform_colour",
    "uniform_fg_bg_geometrical",
    "uniform_fg_bg_objects",
]


# ============================================================
# Loading
# ============================================================

def load_array(
    path: Path,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find:\n"
            f"  {path}\n\n"
            "Update the paths at the top "
            "of this script."
        )

    return np.asarray(
        np.load(
            path,
            allow_pickle=True,
        )
    )


def validate_colour_vectors(
    vectors: np.ndarray,
    name: str,
) -> np.ndarray:

    vectors = np.asarray(
        vectors,
        dtype=np.float64,
    )

    if (
        vectors.ndim != 2
        or vectors.shape[1] != N_COLOURS
    ):
        raise ValueError(
            f"{name} must have shape (N, 13), "
            f"but received {vectors.shape}."
        )

    if not np.isfinite(
        vectors
    ).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if np.any(
        vectors < 0
    ):
        raise ValueError(
            f"{name} contains negative values."
        )

    row_sums = vectors.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(
        row_sums <= 0
    ):
        raise ValueError(
            f"{name} contains one or more "
            "zero-sum vectors."
        )

    # GTs should already sum to 1.
    # Normalize defensively only.
    vectors = (
        vectors
        / row_sums
    )

    return vectors


# ============================================================
# Stimulus category extraction
# ============================================================

def stimulus_to_category(
    stimulus: object,
) -> str:

    path = (
        str(stimulus)
        .replace("\\", "/")
        .strip("/")
    )

    if not path:
        return "unknown"

    return path.split("/")[0]


def extract_categories(
    stimuli: np.ndarray,
) -> np.ndarray:

    return np.asarray(
        [
            stimulus_to_category(
                stimulus
            )
            for stimulus in stimuli
        ],
        dtype=object,
    )


# ============================================================
# F1
# ============================================================

def micro_f1_score(
    gt,
    pred_probs,
    gt_thresh=GT_THRESHOLD,
    pred_thresh=PRED_THRESHOLD,
    eps=1e-8,
):

    gt_binary = (
        gt >= gt_thresh
    ).astype(int)

    pred_binary = (
        pred_probs >= pred_thresh
    ).astype(int)

    TP = np.logical_and(
        pred_binary == 1,
        gt_binary == 1,
    ).sum()

    FP = np.logical_and(
        pred_binary == 1,
        gt_binary == 0,
    ).sum()

    FN = np.logical_and(
        pred_binary == 0,
        gt_binary == 1,
    ).sum()

    precision = TP / (
        TP + FP + eps
    )

    recall = TP / (
        TP + FN + eps
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
            + eps
        )
    )

    return f1


def samplewise_f1(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    gt_threshold: float,
    pred_threshold: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Calculate one F1 score per image.

    This matches the CUBE-style evaluation:

        calculate F1 for each 13-D sample
        then average across samples.
    """

    if (
        ground_truth.shape
        != predictions.shape
    ):
        raise ValueError(
            "Ground truth and prediction "
            "shapes differ: "
            f"{ground_truth.shape} versus "
            f"{predictions.shape}."
        )

    gt_binary = (
        ground_truth
        >= gt_threshold
    )

    pred_binary = (
        predictions
        >= pred_threshold
    )

    true_positives = (
        np.logical_and(
            gt_binary,
            pred_binary,
        )
        .sum(axis=1)
    )

    false_positives = (
        np.logical_and(
            ~gt_binary,
            pred_binary,
        )
        .sum(axis=1)
    )

    false_negatives = (
        np.logical_and(
            gt_binary,
            ~pred_binary,
        )
        .sum(axis=1)
    )

    precision = (
        true_positives
        / (
            true_positives
            + false_positives
            + eps
        )
    )

    recall = (
        true_positives
        / (
            true_positives
            + false_negatives
            + eps
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + eps
        )
    )

    return f1.astype(
        np.float64
    )


# ============================================================
# Presence-mask utilities
# ============================================================

def gt_to_presence_mask(
    vectors: np.ndarray,
    gt_threshold: float = GT_THRESHOLD,
) -> np.ndarray:
    """
    Convert GT colour distributions into
    binary colour-presence masks.
    """

    return (
        np.asarray(
            vectors
        )
        >= gt_threshold
    )


def mask_to_probability_distribution(
    masks: np.ndarray,
) -> np.ndarray:
    """
    Convert binary colour masks into normalized
    probability distributions.

    If k colours are active:

        each active colour receives 1/k
        all inactive colours receive 0

    Therefore every prediction sums to 1.
    """

    masks = np.asarray(
        masks,
        dtype=bool,
    )

    counts = masks.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(
        counts == 0
    ):
        raise ValueError(
            "At least one colour mask "
            "contains no active colours."
        )

    max_active = int(
        counts.max()
    )

    minimum_active_probability = (
        1.0 / max_active
    )

    # Because prediction F1 uses >= threshold,
    # exactly 0.10 is still counted as positive.
    if (
        minimum_active_probability
        < PRED_THRESHOLD
    ):
        raise ValueError(
            "At least one baseline sample has "
            f"{max_active} active colours. "
            "Equal weighting would assign "
            f"{minimum_active_probability:.4f} "
            "to each colour, below the "
            f"prediction threshold "
            f"{PRED_THRESHOLD:.4f}."
        )

    predictions = (
        masks.astype(
            np.float64
        )
        / counts
    )

    return predictions


def positive_counts(
    vectors: np.ndarray,
    threshold: float,
) -> np.ndarray:

    return (
        vectors >= threshold
    ).sum(
        axis=1
    )


# ============================================================
# Baseline generator
# ============================================================

def sample_paired_baseline_predictions(
    rng: np.random.Generator,
    empirical_pool: np.ndarray,
    number_of_samples: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate FAIR paired uniform and empirical
    F1-baseline predictions.

    EMPIRICAL BASELINE
    ------------------
    1. Sample a complete GT vector from the
       training set.
    2. Determine which colours are genuinely
       present using GT_THRESHOLD.
    3. Preserve those empirical colour identities.
    4. Give each active colour equal probability.

    UNIFORM BASELINE
    ----------------
    1. Use exactly the SAME number of active
       colours k as the paired empirical sample.
    2. Randomly choose k colours uniformly
       from the 13 palette colours.
    3. Give each selected colour equal probability.

    Consequently:

        empirical sum = 1
        uniform sum = 1

        empirical predicted-positive count
        ==
        uniform predicted-positive count

    Therefore F1 differences cannot be caused
    merely by one baseline predicting more
    colours than the other.
    """

    if len(
        empirical_pool
    ) == 0:
        raise ValueError(
            "Cannot sample from an empty "
            "empirical pool."
        )

    # --------------------------------------------------------
    # Sample complete GT vectors from training
    # --------------------------------------------------------

    sampled_indices = rng.integers(
        low=0,
        high=len(
            empirical_pool
        ),
        size=number_of_samples,
    )

    sampled_training_gt = (
        empirical_pool[
            sampled_indices
        ]
    )

    # --------------------------------------------------------
    # Empirical presence masks
    # --------------------------------------------------------

    empirical_masks = (
        gt_to_presence_mask(
            sampled_training_gt,
            GT_THRESHOLD,
        )
    )

    # --------------------------------------------------------
    # Uniform random masks
    #
    # Same k as empirical, random identities.
    # --------------------------------------------------------

    uniform_masks = np.zeros_like(
        empirical_masks,
        dtype=bool,
    )

    for i, empirical_mask in enumerate(
        empirical_masks
    ):

        k = int(
            empirical_mask.sum()
        )

        if k <= 0:
            raise RuntimeError(
                "Sampled empirical GT contains "
                "zero active colours."
            )

        selected_colours = rng.choice(
            N_COLOURS,
            size=k,
            replace=False,
        )

        uniform_masks[
            i,
            selected_colours,
        ] = True

    # --------------------------------------------------------
    # Convert both into normalized distributions
    # --------------------------------------------------------

    empirical_predictions = (
        mask_to_probability_distribution(
            empirical_masks
        )
    )

    uniform_predictions = (
        mask_to_probability_distribution(
            uniform_masks
        )
    )

    # --------------------------------------------------------
    # Sanity check:
    # thresholded cardinalities MUST match.
    # --------------------------------------------------------

    empirical_counts = (
        positive_counts(
            empirical_predictions,
            PRED_THRESHOLD,
        )
    )

    uniform_counts = (
        positive_counts(
            uniform_predictions,
            PRED_THRESHOLD,
        )
    )

    if not np.array_equal(
        empirical_counts,
        uniform_counts,
    ):
        raise RuntimeError(
            "Uniform and empirical baseline "
            "predicted-positive counts differ. "
            "The paired baseline is invalid."
        )

    return (
        uniform_predictions,
        empirical_predictions,
    )


# ============================================================
# Diagnostics
# ============================================================

def print_active_count_distribution(
    label: str,
    counts: np.ndarray,
) -> None:

    values, frequencies = np.unique(
        counts,
        return_counts=True,
    )

    pieces = []

    for value, frequency in zip(
        values,
        frequencies,
    ):
        pieces.append(
            f"{int(value)} colours: "
            f"{int(frequency)}"
        )

    print(
        f"    {label}: "
        + ", ".join(
            pieces
        )
    )


def print_colour_frequency_diagnostics(
    empirical_pool: np.ndarray,
    validation_gt: np.ndarray,
) -> None:

    train_binary = (
        empirical_pool
        >= GT_THRESHOLD
    )

    val_binary = (
        validation_gt
        >= GT_THRESHOLD
    )

    train_frequency = (
        train_binary.mean(
            axis=0
        )
    )

    val_frequency = (
        val_binary.mean(
            axis=0
        )
    )

    print(
        "\n  Colour-presence frequencies"
    )

    print(
        "  "
        f"{'Colour':<12}"
        f"{'Train':>10}"
        f"{'Validation':>14}"
    )

    for (
        colour,
        train_freq,
        val_freq,
    ) in zip(
        COLOUR_NAMES,
        train_frequency,
        val_frequency,
    ):

        print(
            "  "
            f"{colour:<12}"
            f"{train_freq:>10.3f}"
            f"{val_freq:>14.3f}"
        )

    print(
        "\n  Train frequency SD:",
        f"{train_frequency.std():.6f}",
    )

    print(
        "  Validation frequency SD:",
        f"{val_frequency.std():.6f}",
    )


def print_palette_overlap_diagnostics(
    empirical_pool: np.ndarray,
    validation_gt: np.ndarray,
) -> None:
    """
    Useful especially for uniform_colour,
    where train and validation palette support
    may not overlap.
    """

    train_present = (
        empirical_pool
        >= GT_THRESHOLD
    ).any(
        axis=0
    )

    val_present = (
        validation_gt
        >= GT_THRESHOLD
    ).any(
        axis=0
    )

    shared = (
        train_present
        & val_present
    )

    train_colours = [
        COLOUR_NAMES[i]
        for i in np.where(
            train_present
        )[0]
    ]

    val_colours = [
        COLOUR_NAMES[i]
        for i in np.where(
            val_present
        )[0]
    ]

    shared_colours = [
        COLOUR_NAMES[i]
        for i in np.where(
            shared
        )[0]
    ]

    print(
        "\n  Palette support"
    )

    print(
        "    Train colours:",
        train_colours,
    )

    print(
        "    Validation colours:",
        val_colours,
    )

    print(
        "    Shared colours:",
        shared_colours,
    )


def print_baseline_diagnostics(
    validation_gt: np.ndarray,
    empirical_predictions: np.ndarray,
    uniform_predictions: np.ndarray,
) -> None:

    gt_sums = (
        validation_gt.sum(
            axis=1
        )
    )

    empirical_sums = (
        empirical_predictions.sum(
            axis=1
        )
    )

    uniform_sums = (
        uniform_predictions.sum(
            axis=1
        )
    )

    gt_counts = (
        positive_counts(
            validation_gt,
            GT_THRESHOLD,
        )
    )

    empirical_counts = (
        positive_counts(
            empirical_predictions,
            PRED_THRESHOLD,
        )
    )

    uniform_counts = (
        positive_counts(
            uniform_predictions,
            PRED_THRESHOLD,
        )
    )

    print(
        "\n  Baseline diagnostics "
        "(first Monte Carlo draw)"
    )

    print(
        "    GT row sums:         "
        f"{gt_sums.min():.6f} to "
        f"{gt_sums.max():.6f}"
    )

    print(
        "    Empirical row sums:  "
        f"{empirical_sums.min():.6f} to "
        f"{empirical_sums.max():.6f}"
    )

    print(
        "    Uniform row sums:    "
        f"{uniform_sums.min():.6f} to "
        f"{uniform_sums.max():.6f}"
    )

    print(
        "\n    Mean GT positives:        "
        f"{gt_counts.mean():.3f}"
    )

    print(
        "    Mean empirical positives: "
        f"{empirical_counts.mean():.3f}"
    )

    print(
        "    Mean uniform positives:   "
        f"{uniform_counts.mean():.3f}"
    )

    print(
        "\n    Active-count distributions"
    )

    print_active_count_distribution(
        "GT",
        gt_counts,
    )

    print_active_count_distribution(
        "Empirical",
        empirical_counts,
    )

    print_active_count_distribution(
        "Uniform",
        uniform_counts,
    )

    print(
        "\n    Uniform/empirical "
        "counts identical:",
        np.array_equal(
            empirical_counts,
            uniform_counts,
        ),
    )


# ============================================================
# Monte Carlo
# ============================================================

def summarize_simulations(
    simulation_scores: np.ndarray,
) -> dict[str, float]:

    simulation_scores = np.asarray(
        simulation_scores,
        dtype=np.float64,
    )

    mean = float(
        np.mean(
            simulation_scores
        )
    )

    std = float(
        np.std(
            simulation_scores,
            ddof=1,
        )
    )

    ci_low, ci_high = (
        np.percentile(
            simulation_scores,
            [
                2.5,
                97.5,
            ],
        )
    )

    return {
        "mean_f1": mean,
        "std_f1": std,
        "ci_2.5": float(
            ci_low
        ),
        "ci_97.5": float(
            ci_high
        ),
    }


def run_category_simulation(
    rng: np.random.Generator,
    validation_gt: np.ndarray,
    empirical_pool: np.ndarray,
    n_repeats: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    n_samples = len(
        validation_gt
    )

    uniform_scores = np.empty(
        n_repeats,
        dtype=np.float64,
    )

    empirical_scores = np.empty(
        n_repeats,
        dtype=np.float64,
    )

    for repetition in range(
        n_repeats
    ):

        (
            uniform_predictions,
            empirical_predictions,
        ) = (
            sample_paired_baseline_predictions(
                rng=rng,
                empirical_pool=empirical_pool,
                number_of_samples=n_samples,
            )
        )

        if repetition == 0:

            print_baseline_diagnostics(
                validation_gt=(
                    validation_gt
                ),
                empirical_predictions=(
                    empirical_predictions
                ),
                uniform_predictions=(
                    uniform_predictions
                ),
            )

        uniform_f1 = (
            samplewise_f1(
                ground_truth=(
                    validation_gt
                ),
                predictions=(
                    uniform_predictions
                ),
                gt_threshold=(
                    GT_THRESHOLD
                ),
                pred_threshold=(
                    PRED_THRESHOLD
                ),
            )
        )

        empirical_f1 = (
            samplewise_f1(
                ground_truth=(
                    validation_gt
                ),
                predictions=(
                    empirical_predictions
                ),
                gt_threshold=(
                    GT_THRESHOLD
                ),
                pred_threshold=(
                    PRED_THRESHOLD
                ),
            )
        )

        uniform_scores[
            repetition
        ] = (
            uniform_f1.mean()
        )

        empirical_scores[
            repetition
        ] = (
            empirical_f1.mean()
        )

        if (
            repetition == 0
            or (
                repetition + 1
            ) % 500 == 0
            or repetition + 1
            == n_repeats
        ):

            print(
                "    repetition "
                f"{repetition + 1}/"
                f"{n_repeats}"
            )

    return (
        uniform_scores,
        empirical_scores,
    )


# ============================================================
# Plotting
# ============================================================

def plot_results(
    results: pd.DataFrame,
) -> None:

    plot_data = results[
        results[
            "stimulus_type"
        ] != "Overall"
    ].copy()

    overall_row = results[
        results[
            "stimulus_type"
        ] == "Overall"
    ].iloc[0]

    uniform_overall = float(
        overall_row[
            "uniform_mean_f1"
        ]
    )

    empirical_overall = float(
        overall_row[
            "empirical_mean_f1"
        ]
    )

    x = np.arange(
        len(
            plot_data
        )
    )

    width = 0.38

    uniform_error_low = (
        plot_data[
            "uniform_mean_f1"
        ]
        - plot_data[
            "uniform_ci_2.5"
        ]
    )

    uniform_error_high = (
        plot_data[
            "uniform_ci_97.5"
        ]
        - plot_data[
            "uniform_mean_f1"
        ]
    )

    empirical_error_low = (
        plot_data[
            "empirical_mean_f1"
        ]
        - plot_data[
            "empirical_ci_2.5"
        ]
    )

    empirical_error_high = (
        plot_data[
            "empirical_ci_97.5"
        ]
        - plot_data[
            "empirical_mean_f1"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.bar(
        x - width / 2,
        plot_data[
            "uniform_mean_f1"
        ],
        width=width,
        yerr=np.vstack(
            [
                uniform_error_low,
                uniform_error_high,
            ]
        ),
        capsize=4,
        label=(
            "Uniform random baseline"
        ),
    )

    ax.bar(
        x + width / 2,
        plot_data[
            "empirical_mean_f1"
        ],
        width=width,
        yerr=np.vstack(
            [
                empirical_error_low,
                empirical_error_high,
            ]
        ),
        capsize=4,
        label=(
            "Empirical distribution baseline"
        ),
    )

    # --------------------------------------------------------
    # Overall average F1 horizontal lines
    # --------------------------------------------------------

    ax.axhline(
        y=uniform_overall,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Uniform overall F1 "
            f"({uniform_overall:.3f})"
        ),
    )

    ax.axhline(
        y=empirical_overall,
        linestyle=":",
        linewidth=1.5,
        label=(
            "Empirical overall F1 "
            f"({empirical_overall:.3f})"
        ),
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        plot_data[
            "stimulus_type"
        ],
        rotation=40,
        ha="right",
    )

    ax.set_xlabel(
        "Stimulus type"
    )

    ax.set_ylabel(
        "Chance F1"
    )

    ax.set_title(
        "Monte Carlo chance F1 "
        "by stimulus type"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "chance_f1_by_stimulus_type.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"\nSaved plot: "
        f"{output_path}"
    )

    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Load arrays
    # --------------------------------------------------------

    train_colour = (
        validate_colour_vectors(
            load_array(
                TRAIN_COLOUR_PATH
            ),
            "Training colour vectors",
        )
    )

    val_colour = (
        validate_colour_vectors(
            load_array(
                VAL_COLOUR_PATH
            ),
            "Validation colour vectors",
        )
    )

    train_stimuli = (
        load_array(
            TRAIN_STIMULI_PATH
        )
        .reshape(-1)
    )

    val_stimuli = (
        load_array(
            VAL_STIMULI_PATH
        )
        .reshape(-1)
    )

    if (
        len(train_colour)
        != len(train_stimuli)
    ):
        raise ValueError(
            "Training colour-vector and "
            "stimulus counts differ: "
            f"{len(train_colour)} versus "
            f"{len(train_stimuli)}."
        )

    if (
        len(val_colour)
        != len(val_stimuli)
    ):
        raise ValueError(
            "Validation colour-vector and "
            "stimulus counts differ: "
            f"{len(val_colour)} versus "
            f"{len(val_stimuli)}."
        )

    train_categories = (
        extract_categories(
            train_stimuli
        )
    )

    val_categories = (
        extract_categories(
            val_stimuli
        )
    )

    print(
        "Training colour shape:",
        train_colour.shape,
    )

    print(
        "Validation colour shape:",
        val_colour.shape,
    )

    # --------------------------------------------------------
    # Category counts
    # --------------------------------------------------------

    print(
        "\nTraining categories:"
    )

    unique_train, train_counts = (
        np.unique(
            train_categories,
            return_counts=True,
        )
    )

    for category, count in zip(
        unique_train,
        train_counts,
    ):
        print(
            f"  {category}: "
            f"{count}"
        )

    print(
        "\nValidation categories:"
    )

    unique_val, val_counts = (
        np.unique(
            val_categories,
            return_counts=True,
        )
    )

    for category, count in zip(
        unique_val,
        val_counts,
    ):
        print(
            f"  {category}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    result_rows = []

    all_uniform_scores = {}

    all_empirical_scores = {}

    # --------------------------------------------------------
    # Categories to evaluate
    # --------------------------------------------------------

    categories_to_evaluate = [
        category
        for category in STIMULUS_TYPES
        if np.any(
            val_categories
            == category
        )
    ]

    # --------------------------------------------------------
    # Category-wise Monte Carlo
    # --------------------------------------------------------

    for category in (
        categories_to_evaluate
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Stimulus type: "
            f"{category}"
        )

        print(
            "=" * 70
        )

        val_mask = (
            val_categories
            == category
        )

        category_validation_gt = (
            val_colour[
                val_mask
            ]
        )

        if (
            CATEGORY_SPECIFIC_EMPIRICAL_SAMPLING
        ):

            train_mask = (
                train_categories
                == category
            )

            empirical_pool = (
                train_colour[
                    train_mask
                ]
            )

        else:

            empirical_pool = (
                train_colour
            )

        print(
            "Validation samples:",
            len(
                category_validation_gt
            ),
        )

        print(
            "Empirical sampling pool:",
            len(
                empirical_pool
            ),
        )

        if len(
            empirical_pool
        ) == 0:
            raise RuntimeError(
                "No empirical training "
                f"samples found for "
                f"'{category}'."
            )

        # ----------------------------------------------------
        # GT active-colour diagnostics
        # ----------------------------------------------------

        gt_active_counts = (
            positive_counts(
                category_validation_gt,
                GT_THRESHOLD,
            )
        )

        print(
            "\n  Validation GT "
            "active-colour distribution"
        )

        print_active_count_distribution(
            "GT",
            gt_active_counts,
        )

        # For your constructed categories:
        #
        # 03_colours:
        # 3 foreground + 1 background = 4
        #
        # 04_colours:
        # 4 foreground + 1 background = 5
        #
        # 05_colours:
        # 5 foreground + 1 background = 6

        # ----------------------------------------------------
        # Colour frequency diagnostics
        # ----------------------------------------------------

        print_colour_frequency_diagnostics(
            empirical_pool=(
                empirical_pool
            ),
            validation_gt=(
                category_validation_gt
            ),
        )

        print_palette_overlap_diagnostics(
            empirical_pool=(
                empirical_pool
            ),
            validation_gt=(
                category_validation_gt
            ),
        )

        # ----------------------------------------------------
        # Monte Carlo
        # ----------------------------------------------------

        (
            uniform_scores,
            empirical_scores,
        ) = (
            run_category_simulation(
                rng=rng,
                validation_gt=(
                    category_validation_gt
                ),
                empirical_pool=(
                    empirical_pool
                ),
                n_repeats=(
                    N_REPEATS
                ),
            )
        )

        all_uniform_scores[
            category
        ] = (
            uniform_scores
        )

        all_empirical_scores[
            category
        ] = (
            empirical_scores
        )

        uniform_summary = (
            summarize_simulations(
                uniform_scores
            )
        )

        empirical_summary = (
            summarize_simulations(
                empirical_scores
            )
        )

        empirical_minus_uniform = (
            empirical_summary[
                "mean_f1"
            ]
            - uniform_summary[
                "mean_f1"
            ]
        )

        result_rows.append(
            {
                "stimulus_type": (
                    category
                ),
                "validation_samples": (
                    len(
                        category_validation_gt
                    )
                ),
                "empirical_pool_samples": (
                    len(
                        empirical_pool
                    )
                ),

                "uniform_mean_f1": (
                    uniform_summary[
                        "mean_f1"
                    ]
                ),

                "uniform_std_f1": (
                    uniform_summary[
                        "std_f1"
                    ]
                ),

                "uniform_ci_2.5": (
                    uniform_summary[
                        "ci_2.5"
                    ]
                ),

                "uniform_ci_97.5": (
                    uniform_summary[
                        "ci_97.5"
                    ]
                ),

                "empirical_mean_f1": (
                    empirical_summary[
                        "mean_f1"
                    ]
                ),

                "empirical_std_f1": (
                    empirical_summary[
                        "std_f1"
                    ]
                ),

                "empirical_ci_2.5": (
                    empirical_summary[
                        "ci_2.5"
                    ]
                ),

                "empirical_ci_97.5": (
                    empirical_summary[
                        "ci_97.5"
                    ]
                ),

                "empirical_minus_uniform": (
                    empirical_minus_uniform
                ),
            }
        )

        print(
            "\nUniform random baseline"
        )

        print(
            "  Mean F1: "
            f"{uniform_summary['mean_f1']:.6f}"
        )

        print(
            "  SD:      "
            f"{uniform_summary['std_f1']:.6f}"
        )

        print(
            "  95% interval: "
            f"["
            f"{uniform_summary['ci_2.5']:.6f}, "
            f"{uniform_summary['ci_97.5']:.6f}"
            f"]"
        )

        print(
            "\nEmpirical distribution baseline"
        )

        print(
            "  Mean F1: "
            f"{empirical_summary['mean_f1']:.6f}"
        )

        print(
            "  SD:      "
            f"{empirical_summary['std_f1']:.6f}"
        )

        print(
            "  95% interval: "
            f"["
            f"{empirical_summary['ci_2.5']:.6f}, "
            f"{empirical_summary['ci_97.5']:.6f}"
            f"]"
        )

        print(
            "\nEmpirical - Uniform:"
            f" {empirical_minus_uniform:.6f}"
        )

    # ========================================================
    # Overall Monte Carlo
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Overall"
    )

    print(
        "=" * 70
    )

    overall_uniform_scores = (
        np.empty(
            N_REPEATS,
            dtype=np.float64,
        )
    )

    overall_empirical_scores = (
        np.empty(
            N_REPEATS,
            dtype=np.float64,
        )
    )

    for repetition in range(
        N_REPEATS
    ):

        uniform_predictions = (
            np.empty_like(
                val_colour,
                dtype=np.float64,
            )
        )

        empirical_predictions = (
            np.empty_like(
                val_colour,
                dtype=np.float64,
            )
        )

        # ----------------------------------------------------
        # Generate paired predictions category by category
        # ----------------------------------------------------

        for category in np.unique(
            val_categories
        ):

            val_mask = (
                val_categories
                == category
            )

            n_category_samples = int(
                val_mask.sum()
            )

            if (
                CATEGORY_SPECIFIC_EMPIRICAL_SAMPLING
            ):

                empirical_pool = (
                    train_colour[
                        train_categories
                        == category
                    ]
                )

            else:

                empirical_pool = (
                    train_colour
                )

            if len(
                empirical_pool
            ) == 0:

                raise RuntimeError(
                    "No training empirical "
                    "pool available for "
                    f"category '{category}'."
                )

            (
                category_uniform,
                category_empirical,
            ) = (
                sample_paired_baseline_predictions(
                    rng=rng,
                    empirical_pool=(
                        empirical_pool
                    ),
                    number_of_samples=(
                        n_category_samples
                    ),
                )
            )

            uniform_predictions[
                val_mask
            ] = (
                category_uniform
            )

            empirical_predictions[
                val_mask
            ] = (
                category_empirical
            )

        # ----------------------------------------------------
        # Overall mean per-image F1
        # ----------------------------------------------------

        overall_uniform_scores[
            repetition
        ] = (
            samplewise_f1(
                ground_truth=(
                    val_colour
                ),
                predictions=(
                    uniform_predictions
                ),
                gt_threshold=(
                    GT_THRESHOLD
                ),
                pred_threshold=(
                    PRED_THRESHOLD
                ),
            )
            .mean()
        )

        overall_empirical_scores[
            repetition
        ] = (
            samplewise_f1(
                ground_truth=(
                    val_colour
                ),
                predictions=(
                    empirical_predictions
                ),
                gt_threshold=(
                    GT_THRESHOLD
                ),
                pred_threshold=(
                    PRED_THRESHOLD
                ),
            )
            .mean()
        )

        if (
            repetition == 0
            or (
                repetition + 1
            ) % 500 == 0
            or repetition + 1
            == N_REPEATS
        ):

            print(
                "    repetition "
                f"{repetition + 1}/"
                f"{N_REPEATS}"
            )

    overall_uniform_summary = (
        summarize_simulations(
            overall_uniform_scores
        )
    )

    overall_empirical_summary = (
        summarize_simulations(
            overall_empirical_scores
        )
    )

    overall_difference = (
        overall_empirical_summary[
            "mean_f1"
        ]
        - overall_uniform_summary[
            "mean_f1"
        ]
    )

    result_rows.append(
        {
            "stimulus_type": (
                "Overall"
            ),

            "validation_samples": (
                len(
                    val_colour
                )
            ),

            "empirical_pool_samples": (
                len(
                    train_colour
                )
            ),

            "uniform_mean_f1": (
                overall_uniform_summary[
                    "mean_f1"
                ]
            ),

            "uniform_std_f1": (
                overall_uniform_summary[
                    "std_f1"
                ]
            ),

            "uniform_ci_2.5": (
                overall_uniform_summary[
                    "ci_2.5"
                ]
            ),

            "uniform_ci_97.5": (
                overall_uniform_summary[
                    "ci_97.5"
                ]
            ),

            "empirical_mean_f1": (
                overall_empirical_summary[
                    "mean_f1"
                ]
            ),

            "empirical_std_f1": (
                overall_empirical_summary[
                    "std_f1"
                ]
            ),

            "empirical_ci_2.5": (
                overall_empirical_summary[
                    "ci_2.5"
                ]
            ),

            "empirical_ci_97.5": (
                overall_empirical_summary[
                    "ci_97.5"
                ]
            ),

            "empirical_minus_uniform": (
                overall_difference
            ),
        }
    )

    print(
        "\nOverall uniform random baseline"
    )

    print(
        "  Mean F1: "
        f"{overall_uniform_summary['mean_f1']:.6f}"
    )

    print(
        "  95% interval: "
        f"["
        f"{overall_uniform_summary['ci_2.5']:.6f}, "
        f"{overall_uniform_summary['ci_97.5']:.6f}"
        f"]"
    )

    print(
        "\nOverall empirical distribution baseline"
    )

    print(
        "  Mean F1: "
        f"{overall_empirical_summary['mean_f1']:.6f}"
    )

    print(
        "  95% interval: "
        f"["
        f"{overall_empirical_summary['ci_2.5']:.6f}, "
        f"{overall_empirical_summary['ci_97.5']:.6f}"
        f"]"
    )

    print(
        "\nOverall empirical - uniform:"
        f" {overall_difference:.6f}"
    )

    # ========================================================
    # Results table
    # ========================================================

    results = pd.DataFrame(
        result_rows
    )

    print(
        "\nFinal results"
    )

    print(
        results.to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.6f}"
            ),
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_DIR
        / "chance_f1_by_stimulus_type.csv"
    )

    results.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Save raw Monte Carlo samples
    # --------------------------------------------------------

    np.savez(
        OUTPUT_DIR
        / "chance_f1_simulations.npz",

        overall_uniform=(
            overall_uniform_scores
        ),

        overall_empirical=(
            overall_empirical_scores
        ),

        **{
            f"uniform_{category}":
            scores

            for category, scores
            in all_uniform_scores.items()
        },

        **{
            f"empirical_{category}":
            scores

            for category, scores
            in all_empirical_scores.items()
        },
    )

    print(
        f"\nSaved table: "
        f"{output_csv}"
    )

    print(
        "Saved raw simulations:",
        OUTPUT_DIR
        / "chance_f1_simulations.npz",
    )

    # ========================================================
    # Plot
    # ========================================================

    plot_results(
        results
    )


if __name__ == "__main__":
    main()