from pathlib import Path

import pandas as pd


def find_edf_files(data_directory):
    """
    Find all PSG and hypnogram EDF files in the Sleep-EDF dataset.

    Parameters
    ----------
    data_directory : str or Path
        Path to the directory containing sleep-cassette
        and/or sleep-telemetry.

    Returns
    -------
    psg_files : list[Path]
        All PSG EDF files.
    hypnogram_files : list[Path]
        All hypnogram EDF files.
    """

    data_directory = Path(data_directory)

    psg_files = sorted(
        data_directory.rglob("*-PSG.edf")
    )

    hypnogram_files = sorted(
        data_directory.rglob("*-Hypnogram.edf")
    )

    return psg_files, hypnogram_files


def get_recording_id(psg_path):
    """
    Extract the recording ID from a PSG filename.

    Example
    -------
    SC4001E0-PSG.edf -> SC4001
    """

    psg_path = Path(psg_path)

    filename = psg_path.name

    return filename.split("E0-PSG.edf")[0]


def build_recording_pairs(data_directory):
    """
    Pair PSG files with their corresponding hypnogram files.

    Sleep-EDF uses slightly different suffixes for hypnogram files,
    so pairing is performed using the recording ID.

    Returns
    -------
    pandas.DataFrame
        Columns:
        recording_id
        psg
        hypnogram
    """

    data_directory = Path(data_directory)

    psg_files, hypnogram_files = find_edf_files(
        data_directory
    )

    psg_mapping = {}
    hypnogram_mapping = {}

    for psg in psg_files:
        recording_id = get_recording_id(psg)
        psg_mapping[recording_id] = psg

    for hypnogram in hypnogram_files:
        filename = hypnogram.name

        # Example:
        # SC4001EC-Hypnogram.edf -> SC4001
        recording_id = filename.split("E")[0]

        hypnogram_mapping[recording_id] = hypnogram

    recording_ids = sorted(
        set(psg_mapping.keys())
        | set(hypnogram_mapping.keys())
    )

    rows = []

    for recording_id in recording_ids:
        rows.append(
            {
                "recording_id": recording_id,
                "psg": (
                    str(psg_mapping[recording_id])
                    if recording_id in psg_mapping
                    else None
                ),
                "hypnogram": (
                    str(hypnogram_mapping[recording_id])
                    if recording_id in hypnogram_mapping
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_recording_pairs(pairings_df):
    """
    Validate that every recording has both a PSG
    and a hypnogram file.

    Returns
    -------
    dict
        Validation information.
    """

    missing_psg = pairings_df[
        pairings_df["psg"].isna()
    ]

    missing_hypnogram = pairings_df[
        pairings_df["hypnogram"].isna()
    ]

    return {
        "total_recordings": len(pairings_df),
        "missing_psg": missing_psg,
        "missing_hypnogram": missing_hypnogram,
        "is_valid": (
            len(missing_psg) == 0
            and len(missing_hypnogram) == 0
        ),
    }