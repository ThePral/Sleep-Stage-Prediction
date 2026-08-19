from pathlib import Path
import traceback

import pandas as pd

from src.data.pairing import (
    build_recording_pairs,
    validate_recording_pairs,
)

from src.data.process_dataset import (
    process_one_recording,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

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

INDIVIDUAL_DIR = (
    FEATURES_DIR
    / "individual"
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
# Helpers
# ============================================================

def prepare_directories():
    INDIVIDUAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def is_valid_feature_file(
    path: Path,
):
    """
    Check whether an existing CSV looks like a valid
    completed feature file.

    This prevents an interrupted/empty file from being
    treated as successfully processed.
    """

    if not path.exists():
        return False

    if path.stat().st_size == 0:
        return False

    required_columns = {
        "subject_id",
        "epoch_index",
        "target_stage",
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
    }

    try:
        header = pd.read_csv(
            path,
            nrows=0,
        )

        return required_columns.issubset(
            set(header.columns)
        )

    except Exception:
        return False


def save_recording_features(
    dataframe,
    recording_id,
):
    output_path = (
        INDIVIDUAL_DIR
        / f"{recording_id}_features.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# Main
# ============================================================

def main():
    prepare_directories()

    print("=" * 70)
    print("Sleep-EDF Expanded Feature Processing")
    print("=" * 70)

    pairings = build_recording_pairs(
        DATA_RAW_DIR
    )

    print(
        f"Total recording pairs: "
        f"{len(pairings)}"
    )

    validation = validate_recording_pairs(
        pairings
    )

    if not validation["is_valid"]:
        raise RuntimeError(
            "Recording pairing validation failed."
        )

    print(
        "Pairing validation: PASSED"
    )

    all_dataframes = []

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    failed_recordings = []

    total = len(pairings)

    for position, row in enumerate(
        pairings.itertuples(index=False),
        start=1,
    ):

        recording_id = row.recording_id

        output_path = (
            INDIVIDUAL_DIR
            / f"{recording_id}_features.csv"
        )

        print()
        print("=" * 70)
        print(
            f"[{position}/{total}] "
            f"{recording_id}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if is_valid_feature_file(
            output_path
        ):

            print(
                "Already processed."
                " Skipping."
            )

            skipped_count += 1

            try:
                existing = pd.read_csv(
                    output_path
                )

                all_dataframes.append(
                    existing
                )

            except Exception as exc:

                print(
                    "Existing file could not "
                    "be read. Reprocessing."
                )

                print(
                    repr(exc)
                )

                try:
                    output_path.unlink()
                except FileNotFoundError:
                    pass

            else:
                continue

        # ----------------------------------------------------
        # Process recording
        # ----------------------------------------------------

        try:

            dataframe = process_one_recording(
                recording_id=recording_id,
                psg_path=row.psg,
                hypnogram_path=row.hypnogram,
            )

            save_recording_features(
                dataframe,
                recording_id,
            )

            all_dataframes.append(
                dataframe
            )

            processed_count += 1

            print(
                f"Valid epochs: "
                f"{len(dataframe)}"
            )

            print(
                "Saved:"
            )

            print(
                output_path
            )

        except Exception as exc:

            failed_count += 1

            failed_recordings.append(
                recording_id
            )

            print()
            print(
                f"ERROR processing "
                f"{recording_id}"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            traceback.print_exc()

            print(
                "\nProcessing will stop."
            )

            raise

    # ========================================================
    # Combine individual files
    # ========================================================

    if not all_dataframes:
        raise RuntimeError(
            "No feature files were available "
            "for combination."
        )

    combined = pd.concat(
        all_dataframes,
        ignore_index=True,
    )

    combined_path = (
        FEATURES_DIR
        / "sleep_edf_features.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Total recordings: {total}"
    )

    print(
        f"Newly processed: {processed_count}"
    )

    print(
        f"Skipped/resumed: {skipped_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    print(
        f"Combined rows: {len(combined)}"
    )

    print(
        "Subjects:",
        combined["subject_id"].nunique(),
    )

    print(
        "\nCombined class distribution:"
    )

    print(
        combined[
            "target_stage"
        ].value_counts()
    )

    print(
        "\nCombined dataset:"
    )

    print(
        combined_path
    )


if __name__ == "__main__":
    main()