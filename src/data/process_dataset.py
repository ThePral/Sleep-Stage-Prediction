from pathlib import Path
import sys
import traceback
import pandas as pd


# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from src.data.pairing import (
    build_recording_pairs,
    validate_recording_pairs,
)

from src.preprocessing.pipeline import (
    process_sleep_edf_subject,
)


# ============================================================
# Paths
# ============================================================

DATA_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

FEATURES_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# Configuration
# ============================================================

EEG_CHANNEL = "EEG Fpz-Cz"

LOW_FREQ = 0.3
HIGH_FREQ = 35.0

EPOCH_DURATION = 30.0

NPERSEG = 1024


# ============================================================
# Directory preparation
# ============================================================

def prepare_directories():
    """
    Create output directories if they don't already exist.
    """

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Process one recording
# ============================================================

def process_one_recording(
    recording_id,
    psg_path,
    hypnogram_path,
):
    """
    Process one Sleep-EDF recording.

    Returns
    -------
    pandas.DataFrame
        Feature DataFrame for the recording.
    """

    print()
    print("=" * 70)
    print(f"Processing: {recording_id}")
    print("=" * 70)

    print(f"PSG:       {psg_path}")
    print(f"Hypnogram: {hypnogram_path}")

    dataframe = process_sleep_edf_subject(
        psg_path=psg_path,
        hypnogram_path=hypnogram_path,
        recording_id=recording_id,
        eeg_channel=EEG_CHANNEL,
        low_freq=LOW_FREQ,
        high_freq=HIGH_FREQ,
        epoch_duration=EPOCH_DURATION,
        nperseg=NPERSEG,
    )

    print(
        f"Valid epochs: {len(dataframe)}"
    )

    print("\nClass distribution:")

    print(
        dataframe[
            "target_stage"
        ].value_counts()
    )

    return dataframe


# ============================================================
# Save one recording
# ============================================================

def save_recording_features(
    dataframe,
    recording_id,
):
    """
    Save one recording's features as CSV.
    """

    output_path = (
        FEATURES_DIR
        / f"{recording_id}_features.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved: {output_path}"
    )

    return output_path


# ============================================================
# Main dataset processing
# ============================================================

def main():
    """
    Process all Sleep-EDF recordings.
    """

    prepare_directories()

    print(
        f"Raw dataset directory:\n"
        f"{DATA_RAW_DIR}"
    )

    # --------------------------------------------------------
    # Find and pair recordings
    # --------------------------------------------------------

    pairings = build_recording_pairs(
        DATA_RAW_DIR
    )

    print()
    print(
        f"Total recording IDs: "
        f"{len(pairings)}"
    )

    validation = (
        validate_recording_pairs(
            pairings
        )
    )

    if not validation["is_valid"]:

        print(
            "\nERROR: Recording pairing "
            "validation failed."
        )

        if len(
            validation["missing_psg"]
        ) > 0:

            print(
                "\nMissing PSG files:"
            )

            print(
                validation[
                    "missing_psg"
                ]
            )

        if len(
            validation["missing_hypnogram"]
        ) > 0:

            print(
                "\nMissing hypnogram files:"
            )

            print(
                validation[
                    "missing_hypnogram"
                ]
            )

        raise RuntimeError(
            "Every recording must have "
            "both PSG and hypnogram files."
        )

    print(
        "Pairing validation: PASSED"
    )

    # --------------------------------------------------------
    # Process recordings
    # --------------------------------------------------------

    successful = []
    failed = []

    all_dataframes = []

    for _, row in pairings.iterrows():

        recording_id = row[
            "recording_id"
        ]

        psg_path = Path(
            row["psg"]
        )

        hypnogram_path = Path(
            row["hypnogram"]
        )

        try:

            dataframe = process_one_recording(
                recording_id=recording_id,
                psg_path=psg_path,
                hypnogram_path=hypnogram_path,
            )

            save_recording_features(
                dataframe,
                recording_id,
            )

            all_dataframes.append(
                dataframe
            )

            successful.append(
                recording_id
            )

        except Exception as error:

            print()
            print(
                "ERROR while processing "
                f"{recording_id}"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            traceback.print_exc()

            failed.append(
                recording_id
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"Total recordings: "
        f"{len(pairings)}"
    )

    print(
        f"Successfully processed: "
        f"{len(successful)}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
    )

    if failed:

        print("\nFailed recordings:")

        for recording_id in failed:
            print(
                f"  - {recording_id}"
            )

    # --------------------------------------------------------
    # Combine successful recordings
    # --------------------------------------------------------

    if not all_dataframes:

        print(
            "\nNo recordings were "
            "successfully processed."
        )

        return

    combined_df = pd.concat(
        all_dataframes,
        ignore_index=True,
    )

    combined_path = (
        FEATURES_DIR
        / "sleep_edf_features.csv"
    )

    combined_df.to_csv(
        combined_path,
        index=False,
    )

    print()
    print(
        f"Combined dataset saved to:"
    )
    print(combined_path)

    print()
    print(
        "Combined dataset shape:"
    )
    print(combined_df.shape)

    print()
    print(
        "Combined class distribution:"
    )
    print(
        combined_df[
            "target_stage"
        ].value_counts()
    )

    print()
    print(
        "Subjects in combined dataset:"
    )
    print(
        combined_df[
            "subject_id"
        ].nunique()
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()