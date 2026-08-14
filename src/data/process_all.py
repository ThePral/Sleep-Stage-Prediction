from pathlib import Path

import pandas as pd

from src.data.pairing import (
    build_recording_pairs,
    validate_recording_pairs,
)
from src.data.process_dataset import (
    process_one_recording,
)


def process_all_recordings(
    data_directory: Path,
    output_directory: Path,
) -> pd.DataFrame:
    """
    Process every paired Sleep-EDF recording.

    Each recording is processed independently and saved
    as an individual CSV file.

    A final combined CSV containing all recordings is
    also created.

    Parameters
    ----------
    data_directory:
        Root directory containing sleep-cassette and
        sleep-telemetry.

    output_directory:
        Directory where feature files will be saved.

    Returns
    -------
    pd.DataFrame
        Combined feature dataset.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    individual_directory = (
        output_directory / "individual"
    )

    individual_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Building Sleep-EDF recording pairs")
    print("=" * 70)

    pairings = build_recording_pairs(
        data_directory
    )

    print(
        f"Total recording pairs: {len(pairings)}"
    )

    validation = validate_recording_pairs(
        pairings
    )

    if not validation["is_valid"]:
        raise RuntimeError(
            "Recording-pair validation failed."
        )

    print("Pairing validation: PASSED")

    all_data = []

    total = len(pairings)

    for index, row in pairings.iterrows():

        recording_id = row["recording_id"]

        print()
        print("=" * 70)
        print(
            f"[{index + 1}/{total}] "
            f"Processing {recording_id}"
        )
        print("=" * 70)

        try:
            features = process_one_recording(
                recording_id=recording_id,
                psg_path=row["psg"],
                hypnogram_path=row[
                    "hypnogram"
                ],
            )

            output_path = (
                individual_directory
                / f"{recording_id}_features.csv"
            )

            features.to_csv(
                output_path,
                index=False,
            )

            all_data.append(
                features
            )

            print(
                f"Valid epochs: "
                f"{len(features)}"
            )

            print(
                "Class distribution:"
            )

            print(
                features[
                    "target_stage"
                ].value_counts()
            )

            print(
                f"Saved: {output_path}"
            )

        except Exception as exc:

            print(
                f"ERROR processing "
                f"{recording_id}:"
            )

            print(
                repr(exc)
            )

            raise

    if not all_data:
        raise RuntimeError(
            "No recordings were processed."
        )

    combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    combined_output = (
        output_directory
        / "sleep_edf_features.csv"
    )

    combined.to_csv(
        combined_output,
        index=False,
    )

    print()
    print("=" * 70)
    print("FULL DATASET PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Number of recordings: "
        f"{combined['subject_id'].nunique()}"
    )

    print(
        f"Total valid epochs: "
        f"{len(combined)}"
    )

    print()
    print("Overall class distribution:")

    print(
        combined[
            "target_stage"
        ].value_counts()
    )

    print()
    print(
        f"Combined dataset saved to:"
    )

    print(
        combined_output
    )

    return combined


if __name__ == "__main__":

    project_root = Path(
        __file__
    ).resolve().parents[2]

    data_directory = (
        project_root
        / "data"
        / "raw"
    )

    output_directory = (
        project_root
        / "data"
        / "features"
    )

    process_all_recordings(
        data_directory=data_directory,
        output_directory=output_directory,
    )