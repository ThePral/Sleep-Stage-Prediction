from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "W",
    "N1",
    "N2",
    "N3",
    "REM",
]

FEATURE_COLUMNS = [
    "delta_absolute",
    "theta_absolute",
    "alpha_absolute",
    "beta_absolute",
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "beta_relative",
    "spectral_entropy",
    "dominant_frequency",
]

EXPECTED_RECORDINGS = 197
EXPECTED_PARTICIPANTS = 100

EXPECTED_TRAIN_PARTICIPANTS = 70
EXPECTED_VAL_PARTICIPANTS = 15
EXPECTED_TEST_PARTICIPANTS = 15

EXPECTED_SEQUENCE_LENGTH = 10
EXPECTED_N_FEATURES = 10
EXPECTED_N_CLASSES = 5


# ============================================================
# MAIN VALIDATION
# ============================================================

def main() -> None:
    """
    Validate the complete corrected final-model package.

    This script checks:

    1. Required files exist.
    2. Final configuration is readable.
    3. Saved model is loadable.
    4. Saved scaler is loadable.
    5. Dataset is readable.
    6. Dataset contains the expected recordings/participants.
    7. No NaN/infinite feature values exist.
    8. Participant-level train/validation/test splits are valid.
    9. No participant appears in multiple splits.
    10. Model input/output dimensions match the configuration.
    11. The scaler produces finite values on training data.
    12. The saved scaler is fitted and usable.

    The script does NOT evaluate the test set and therefore does
    not use test performance for validation.
    """

    print("=" * 70)
    print("FINAL MODEL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    print(
        "Project root:"
    )

    print(
        project_root
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    model_path = (
        project_root
        / "final_model"
        / "stacked_lstm_32_32_10_epochs.keras"
    )

    scaler_path = (
        project_root
        / "final_model"
        / "scaler.pkl"
    )

    config_path = (
        project_root
        / "final_model"
        / "config.json"
    )

    data_path = (
        project_root
        / "data"
        / "features"
        / "sleep_edf_features.csv"
    )

    split_path = (
        project_root
        / "data"
        / "features"
        / "participant_split.csv"
    )

    required_files = {
        "model": model_path,
        "scaler": scaler_path,
        "configuration": config_path,
        "features": data_path,
        "participant split": split_path,
    }

    # --------------------------------------------------------
    # File existence
    # --------------------------------------------------------

    print(
        "\nChecking required files..."
    )

    for name, path in required_files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name.capitalize()} file not found:\n"
                f"{path}"
            )

        print(
            f"[OK] {path}"
        )

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    try:

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise RuntimeError(
            "Failed to load final configuration."
        ) from exc

    print(
        "\n[OK] Configuration loaded."
    )

    # --------------------------------------------------------
    # Validate configuration values
    # --------------------------------------------------------

    required_config_keys = [
        "dataset",
        "recordings",
        "participants",
        "split_unit",
        "train_participants",
        "validation_participants",
        "test_participants",
        "eeg_channel",
        "sampling_frequency_hz",
        "features",
        "sequence_length_epochs",
        "epoch_duration_seconds",
        "temporal_context_minutes",
        "model",
        "lstm_units",
        "dropout",
        "loss",
        "optimizer",
        "initial_learning_rate",
        "batch_size",
        "classes",
    ]

    missing_config_keys = [
        key
        for key in required_config_keys
        if key not in config
    ]

    if missing_config_keys:

        raise ValueError(
            "Configuration is missing required keys: "
            f"{missing_config_keys}"
        )

    if config["split_unit"] != "participant":

        raise ValueError(
            "Final configuration must use "
            "participant-level splitting."
        )

    if config["recordings"] != EXPECTED_RECORDINGS:

        raise ValueError(
            "Unexpected recording count in configuration: "
            f"{config['recordings']} "
            f"(expected {EXPECTED_RECORDINGS})."
        )

    if config["participants"] != EXPECTED_PARTICIPANTS:

        raise ValueError(
            "Unexpected participant count in configuration: "
            f"{config['participants']} "
            f"(expected {EXPECTED_PARTICIPANTS})."
        )

    if config[
        "train_participants"
    ] != EXPECTED_TRAIN_PARTICIPANTS:

        raise ValueError(
            "Unexpected number of training participants: "
            f"{config['train_participants']} "
            f"(expected {EXPECTED_TRAIN_PARTICIPANTS})."
        )

    if config[
        "validation_participants"
    ] != EXPECTED_VAL_PARTICIPANTS:

        raise ValueError(
            "Unexpected number of validation participants: "
            f"{config['validation_participants']} "
            f"(expected {EXPECTED_VAL_PARTICIPANTS})."
        )

    if config[
        "test_participants"
    ] != EXPECTED_TEST_PARTICIPANTS:

        raise ValueError(
            "Unexpected number of test participants: "
            f"{config['test_participants']} "
            f"(expected {EXPECTED_TEST_PARTICIPANTS})."
        )

    if config[
        "sequence_length_epochs"
    ] != EXPECTED_SEQUENCE_LENGTH:

        raise ValueError(
            "Unexpected sequence length: "
            f"{config['sequence_length_epochs']} "
            f"(expected {EXPECTED_SEQUENCE_LENGTH})."
        )

    configured_features = list(
        config["features"]
    )

    if configured_features != FEATURE_COLUMNS:

        raise ValueError(
            "Feature ordering in config.json does not "
            "match the expected final feature ordering.\n"
            f"Expected: {FEATURE_COLUMNS}\n"
            f"Found: {configured_features}"
        )

    configured_classes = list(
        config["classes"]
    )

    if configured_classes != CLASS_NAMES:

        raise ValueError(
            "Class ordering in config.json does not "
            "match the expected class ordering.\n"
            f"Expected: {CLASS_NAMES}\n"
            f"Found: {configured_classes}"
        )

    print(
        "[OK] Configuration values are consistent."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            model_path
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load the saved Keras model."
        ) from exc

    print(
        "[OK] Model loaded."
    )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    try:

        scaler = joblib.load(
            scaler_path
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load the saved scaler."
        ) from exc

    print(
        "[OK] Scaler loaded."
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            data_path
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load feature dataset."
        ) from exc

    try:

        participant_split_df = pd.read_csv(
            split_path
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load participant split."
        ) from exc

    print(
        "\n[OK] Feature dataset loaded."
    )

    print(
        "[OK] Participant split loaded."
    )

    # --------------------------------------------------------
    # Validate required dataset columns
    # --------------------------------------------------------

    required_data_columns = [
        "subject_id",
        "epoch_index",
        "start_time",
        "end_time",
        "raw_stage",
        "target_stage",
        *FEATURE_COLUMNS,
    ]

    missing_data_columns = [
        column
        for column in required_data_columns
        if column not in df.columns
    ]

    if missing_data_columns:

        raise ValueError(
            "Feature dataset is missing required columns: "
            f"{missing_data_columns}"
        )

    required_split_columns = [
        "participant_id",
        "split",
    ]

    missing_split_columns = [
        column
        for column in required_split_columns
        if column not in participant_split_df.columns
    ]

    if missing_split_columns:

        raise ValueError(
            "Participant split is missing required columns: "
            f"{missing_split_columns}"
        )

    print(
        "[OK] Dataset columns are present."
    )

    # --------------------------------------------------------
    # Derive participant ID
    # --------------------------------------------------------

    df["participant_id"] = (
        df["subject_id"]
        .astype(str)
        .str[:5]
    )

    # --------------------------------------------------------
    # Dataset size checks
    # --------------------------------------------------------

    recordings = (
        df["subject_id"]
        .nunique()
    )

    participants = (
        df["participant_id"]
        .nunique()
    )

    print(
        "\nDataset summary:"
    )

    print(
        f"  Rows: {len(df)}"
    )

    print(
        f"  Recordings: {recordings}"
    )

    print(
        f"  Participants: {participants}"
    )

    if recordings != EXPECTED_RECORDINGS:

        raise ValueError(
            "Unexpected number of recordings: "
            f"{recordings} "
            f"(expected {EXPECTED_RECORDINGS})."
        )

    if participants != EXPECTED_PARTICIPANTS:

        raise ValueError(
            "Unexpected number of participants: "
            f"{participants} "
            f"(expected {EXPECTED_PARTICIPANTS})."
        )

    print(
        "[OK] Recording and participant counts are correct."
    )

    # --------------------------------------------------------
    # Validate participant mapping
    # --------------------------------------------------------

    mapping = (
        df[
            [
                "subject_id",
                "participant_id",
            ]
        ]
        .drop_duplicates()
    )

    recordings_per_participant = (
        mapping
        .groupby(
            "participant_id"
        )
        .size()
    )

    if not (
        recordings_per_participant
        .isin([1, 2])
        .all()
    ):

        invalid = (
            recordings_per_participant[
                ~recordings_per_participant
                .isin([1, 2])
            ]
        )

        raise ValueError(
            "Unexpected number of recordings for "
            f"participant(s):\n{invalid}"
        )

    print(
        "[OK] Recording-to-participant mapping is valid."
    )

    # --------------------------------------------------------
    # Validate participant split file
    # --------------------------------------------------------

    participant_split_unique = (
        participant_split_df[
            [
                "participant_id",
                "split",
            ]
        ]
        .drop_duplicates()
    )

    split_participants = (
        participant_split_unique[
            "participant_id"
        ]
        .nunique()
    )

    if (
        split_participants
        != EXPECTED_PARTICIPANTS
    ):

        raise ValueError(
            "Participant split does not contain all "
            f"{EXPECTED_PARTICIPANTS} participants. "
            f"Found {split_participants}."
        )

    duplicate_participant_assignments = (
        participant_split_unique
        .groupby(
            "participant_id"
        )["split"]
        .nunique()
    )

    duplicated_split_assignments = (
        duplicate_participant_assignments[
            duplicate_participant_assignments > 1
        ]
    )

    if len(
        duplicated_split_assignments
    ):

        raise ValueError(
            "Some participants have multiple split assignments:\n"
            f"{duplicated_split_assignments}"
        )

    print(
        "[OK] Every participant has exactly one split assignment."
    )

    # --------------------------------------------------------
    # Merge split assignments
    # --------------------------------------------------------

    split_lookup = (
        participant_split_unique
        .set_index(
            "participant_id"
        )["split"]
    )

    df["split"] = (
        df["participant_id"]
        .map(split_lookup)
    )

    missing_split_assignments = (
        df["split"]
        .isna()
        .sum()
    )

    if missing_split_assignments:

        raise ValueError(
            "Found "
            f"{missing_split_assignments} rows without "
            "a participant-level split assignment."
        )

    print(
        "[OK] Every dataset row has a split assignment."
    )

    # --------------------------------------------------------
    # Participant-level split counts
    # --------------------------------------------------------

    participants_by_split = (
        df[
            [
                "participant_id",
                "split",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "split"
        )["participant_id"]
        .nunique()
    )

    train_count = int(
        participants_by_split.get(
            "train",
            0,
        )
    )

    val_count = int(
        participants_by_split.get(
            "val",
            0,
        )
    )

    test_count = int(
        participants_by_split.get(
            "test",
            0,
        )
    )

    print(
        "\nParticipants by split:"
    )

    print(
        f"  Train: {train_count}"
    )

    print(
        f"  Validation: {val_count}"
    )

    print(
        f"  Test: {test_count}"
    )

    if train_count != EXPECTED_TRAIN_PARTICIPANTS:

        raise ValueError(
            "Unexpected training participant count: "
            f"{train_count}"
        )

    if val_count != EXPECTED_VAL_PARTICIPANTS:

        raise ValueError(
            "Unexpected validation participant count: "
            f"{val_count}"
        )

    if test_count != EXPECTED_TEST_PARTICIPANTS:

        raise ValueError(
            "Unexpected test participant count: "
            f"{test_count}"
        )

    print(
        "[OK] Participant split counts are correct."
    )

    # --------------------------------------------------------
    # Explicit participant leakage check
    # --------------------------------------------------------

    split_sets = {}

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        split_sets[split_name] = set(
            df.loc[
                df["split"] == split_name,
                "participant_id",
            ].unique()
        )

    train_val_overlap = (
        split_sets["train"]
        & split_sets["val"]
    )

    train_test_overlap = (
        split_sets["train"]
        & split_sets["test"]
    )

    val_test_overlap = (
        split_sets["val"]
        & split_sets["test"]
    )

    print(
        "\nParticipant overlap:"
    )

    print(
        "  Train ∩ Validation:",
        train_val_overlap,
    )

    print(
        "  Train ∩ Test:",
        train_test_overlap,
    )

    print(
        "  Validation ∩ Test:",
        val_test_overlap,
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):

        raise ValueError(
            "Participant leakage detected."
        )

    print(
        "[OK] No participant appears in multiple splits."
    )

    # --------------------------------------------------------
    # Feature quality
    # --------------------------------------------------------

    feature_values = df[
        FEATURE_COLUMNS
    ]

    nan_counts = (
        feature_values
        .isna()
        .sum()
    )

    total_nan = int(
        nan_counts.sum()
    )

    if total_nan != 0:

        raise ValueError(
            "Feature dataset contains "
            f"{total_nan} NaN values."
        )

    feature_array = (
        feature_values
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        feature_array
    ).all():

        raise ValueError(
            "Feature dataset contains "
            "infinite values."
        )

    print(
        "[OK] No NaN or infinite feature values."
    )

    # --------------------------------------------------------
    # Epoch duration validation
    # --------------------------------------------------------

    epoch_durations = (
        df["end_time"]
        - df["start_time"]
    )

    unique_durations = (
        epoch_durations
        .dropna()
        .unique()
    )

    if not np.allclose(
        unique_durations,
        30.0,
    ):

        raise ValueError(
            "Not all epochs have a 30-second duration.\n"
            f"Unique durations: {unique_durations}"
        )

    print(
        "[OK] All epochs are exactly 30 seconds."
    )

    # --------------------------------------------------------
    # Target-stage validation
    # --------------------------------------------------------

    observed_classes = sorted(
        df["target_stage"]
        .dropna()
        .unique()
        .tolist()
    )

    expected_classes_sorted = sorted(
        CLASS_NAMES
    )

    if observed_classes != expected_classes_sorted:

        raise ValueError(
            "Unexpected target-stage labels.\n"
            f"Expected: {expected_classes_sorted}\n"
            f"Found: {observed_classes}"
        )

    print(
        "[OK] Target stages are valid."
    )

    # --------------------------------------------------------
    # Model input/output validation
    # --------------------------------------------------------

    model_input_shape = (
        model.input_shape
    )

    model_output_shape = (
        model.output_shape
    )

    print(
        "\nModel input shape:",
        model_input_shape,
    )

    print(
        "Model output shape:",
        model_output_shape,
    )

    expected_input_shape = (
        EXPECTED_SEQUENCE_LENGTH,
        EXPECTED_N_FEATURES,
    )

    if model_input_shape[1:] != (
        expected_input_shape
    ):

        raise ValueError(
            "Model input shape does not match "
            f"the expected {expected_input_shape}."
        )

    if (
        model_output_shape[-1]
        != EXPECTED_N_CLASSES
    ):

        raise ValueError(
            "Model output does not contain "
            f"{EXPECTED_N_CLASSES} classes."
        )

    print(
        "[OK] Model input/output dimensions are correct."
    )

    # --------------------------------------------------------
    # Model configuration consistency
    # --------------------------------------------------------

    if config["model"] != "stacked_lstm":

        raise ValueError(
            "Unexpected final model type: "
            f"{config['model']}"
        )

    configured_lstm_units = list(
        config["lstm_units"]
    )

    if configured_lstm_units != [
        32,
        32,
    ]:

        raise ValueError(
            "Unexpected LSTM configuration: "
            f"{configured_lstm_units}"
        )

    if not np.isclose(
        float(config["dropout"]),
        0.30,
    ):

        raise ValueError(
            "Unexpected dropout value: "
            f"{config['dropout']}"
        )

    if config["loss"] != (
        "unweighted_sparse_categorical_crossentropy"
    ):

        raise ValueError(
            "Unexpected final loss configuration: "
            f"{config['loss']}"
        )

    print(
        "[OK] Final model configuration is consistent."
    )

    # --------------------------------------------------------
    # Verify scaler functionality on TRAINING data only
    # --------------------------------------------------------

    train_df = (
        df[
            df["split"] == "train"
        ]
        .sort_values(
            [
                "subject_id",
                "epoch_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    try:

        train_scaled = scaler.transform(
            train_df
        )

    except Exception as exc:

        raise RuntimeError(
            "Saved scaler could not transform "
            "the training data."
        ) from exc

    if not isinstance(
        train_scaled,
        pd.DataFrame,
    ):

        raise TypeError(
            "Expected SleepFeatureScaler.transform() "
            "to return a pandas DataFrame, but received "
            f"{type(train_scaled).__name__}."
        )

    scaled_values = (
        train_scaled[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        scaled_values
    ).all():

        raise ValueError(
            "Saved scaler produced NaN or "
            "infinite values."
        )

    if list(
        train_scaled.columns
    ) != FEATURE_COLUMNS:

        raise ValueError(
            "Scaler output columns do not match "
            "the expected feature order."
        )

    print(
        "[OK] Saved scaler successfully transforms "
        "training data."
    )

    # --------------------------------------------------------
    # Verify training normalization
    #
    # SleepFeatureScaler performs log10 transformation on
    # absolute powers followed by standardization.
    # Therefore the final transformed training features
    # should have approximately zero mean and unit variance.
    # --------------------------------------------------------

    train_means = (
        scaled_values
        .mean(
            axis=0
        )
    )

    train_stds = (
        scaled_values
        .std(
            axis=0,
            ddof=1,
        )
    )

    if not np.allclose(
        train_means,
        0.0,
        atol=1e-2,
    ):

        raise ValueError(
            "Scaled training means are not approximately zero."
        )

    if not np.allclose(
        train_stds,
        1.0,
        atol=1e-2,
    ):

        raise ValueError(
            "Scaled training standard deviations are "
            "not approximately one."
        )

    print(
        "[OK] Training normalization statistics are valid."
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION PASSED"
    )

    print(
        "=" * 70
    )

    print(
        f"Recordings: {recordings}"
    )

    print(
        f"Participants: {participants}"
    )

    print(
        f"Train participants: {train_count}"
    )

    print(
        f"Validation participants: {val_count}"
    )

    print(
        f"Test participants: {test_count}"
    )

    print(
        f"Sequence length: "
        f"{config['sequence_length_epochs']}"
    )

    print(
        f"Temporal context: "
        f"{config['temporal_context_minutes']} minutes"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Classes: {len(CLASS_NAMES)}"
    )

    print(
        f"Model: {model.name}"
    )

    print(
        "Split unit: participant"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()