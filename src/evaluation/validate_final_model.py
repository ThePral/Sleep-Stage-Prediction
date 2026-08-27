from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


CLASS_NAMES = [
    "W",
    "N1",
    "N2",
    "N3",
    "REM",
]


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    model_path = (
        project_root
        / "final_model"
        / "stacked_lstm_32_32.keras"
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
        / "subject_split.csv"
    )

    print("=" * 70)
    print("FINAL MODEL VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # File validation
    # ---------------------------------------------------------

    required_files = [
        model_path,
        scaler_path,
        config_path,
        data_path,
        split_path,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        print(
            f"[OK] {path}"
        )

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )

    print(
        "\n[OK] Configuration loaded."
    )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = tf.keras.models.load_model(
        model_path
    )

    print(
        "\n[OK] Model loaded."
    )

    # ---------------------------------------------------------
    # Load scaler
    # ---------------------------------------------------------

    scaler = joblib.load(
        scaler_path
    )

    print(
        "[OK] Scaler loaded."
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        data_path
    )

    split_df = pd.read_csv(
        split_path
    )

    df = df.merge(
        split_df,
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )

    print(
        "\nDataset rows:",
        len(df),
    )

    print(
        "Subjects:",
        df["subject_id"]
        .nunique(),
    )

    # ---------------------------------------------------------
    # Data-quality checks
    # ---------------------------------------------------------

    feature_columns = list(
        config["features"]
    )

    feature_values = df[
        feature_columns
    ]

    if feature_values.isna().any().any():

        raise ValueError(
            "Dataset contains NaN values."
        )

    if np.isinf(
        feature_values.to_numpy()
    ).any():

        raise ValueError(
            "Dataset contains infinite values."
        )

    print(
        "[OK] No NaN or infinite feature values."
    )

    # ---------------------------------------------------------
    # Split integrity
    # ---------------------------------------------------------

    subjects_by_split = (
        df[
            [
                "subject_id",
                "split",
            ]
        ]
        .drop_duplicates()
        .groupby("split")
        ["subject_id"]
        .apply(set)
    )

    train_subjects = subjects_by_split.get(
        "train",
        set(),
    )

    val_subjects = subjects_by_split.get(
        "val",
        set(),
    )

    test_subjects = subjects_by_split.get(
        "test",
        set(),
    )

    if (
        train_subjects
        & val_subjects
    ):

        raise ValueError(
            "Train and validation subjects overlap."
        )

    if (
        train_subjects
        & test_subjects
    ):

        raise ValueError(
            "Train and test subjects overlap."
        )

    if (
        val_subjects
        & test_subjects
    ):

        raise ValueError(
            "Validation and test subjects overlap."
        )

    print(
        "[OK] Subject splits are mutually exclusive."
    )

    # ---------------------------------------------------------
    # Model input/output validation
    # ---------------------------------------------------------

    expected_sequence_length = int(
        config[
            "sequence_length_epochs"
        ]
    )

    expected_features = len(
        feature_columns
    )

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

    if model_input_shape[1:] != (
        expected_sequence_length,
        expected_features,
    ):

        raise ValueError(
            "Model input shape does not "
            "match final configuration."
        )

    if model_output_shape[-1] != len(
        CLASS_NAMES
    ):

        raise ValueError(
            "Model output does not contain "
            "five classes."
        )

    print(
        "[OK] Model architecture matches configuration."
    )

    # ---------------------------------------------------------
    # Scaler validation
    # ---------------------------------------------------------

    train_df = (
        df[
            df["split"] == "train"
        ]
        .sort_values(
            [
                "subject_id",
                "start_time",
            ]
        )
        .reset_index(drop=True)
    )

    train_scaled = scaler.transform(
        train_df[
            feature_columns
        ]
    )

    train_scaled_values = (
        train_scaled.to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        train_scaled_values
    ).all():

        raise ValueError(
            "Scaler produced NaN or infinite values."
        )

    print(
        "[OK] Scaler produces finite training values."
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION PASSED")
    print("=" * 70)

    print(
        f"Subjects: {df['subject_id'].nunique()}"
    )

    print(
        f"Features: {expected_features}"
    )

    print(
        f"Sequence length: "
        f"{expected_sequence_length}"
    )

    print(
        f"Classes: {len(CLASS_NAMES)}"
    )

    print(
        "Model:",
        model.name,
    )


if __name__ == "__main__":
    main()