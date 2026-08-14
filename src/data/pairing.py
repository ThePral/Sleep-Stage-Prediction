from pathlib import Path

import pandas as pd


def find_edf_files(data_directory):
    """
    Find all PSG and hypnogram EDF files.
    """

    data_directory = Path(
        data_directory
    )

    psg_files = sorted(
        data_directory.rglob(
            "*-PSG.edf"
        )
    )

    hypnogram_files = sorted(
        data_directory.rglob(
            "*-Hypnogram.edf"
        )
    )

    return (
        psg_files,
        hypnogram_files,
    )


def get_psg_recording_id(
    psg_path,
):
    """
    Extract the recording ID from a Sleep-EDF
    PSG filename.

    Examples:
        SC4001E0-PSG.edf -> SC4001
        SC4261F0-PSG.edf -> SC4261

    The final two characters before "-PSG.edf"
    are part of the EDF recording filename and
    are not part of the recording ID.
    """

    filename = Path(
        psg_path
    ).name

    suffix = "-PSG.edf"

    if not filename.endswith(
        suffix
    ):
        raise ValueError(
            f"Unexpected PSG filename: "
            f"{filename}"
        )

    base = filename[
        :-len(suffix)
    ]

    if len(base) < 3:
        raise ValueError(
            f"Unexpected PSG filename: "
            f"{filename}"
        )

    # Remove the final two characters.
    #
    # SC4001E0 -> SC4001
    # SC4261F0 -> SC4261
    return base[:-2]


def get_hypnogram_recording_id(
    hypnogram_path,
):
    """
    Extract recording ID from a hypnogram filename.

    Sleep-EDF hypnogram files can have
    different suffixes such as:

        SC4001EC-Hypnogram.edf
        SC4011EH-Hypnogram.edf
        SC4022EJ-Hypnogram.edf

    The recording ID is the part before
    the final two-character EDF-specific
    suffix.
    """

    filename = Path(
        hypnogram_path
    ).name

    suffix = "-Hypnogram.edf"

    if not filename.endswith(
        suffix
    ):
        raise ValueError(
            f"Unexpected hypnogram filename: "
            f"{filename}"
        )

    base = filename[
        :-len(suffix)
    ]

    # Remove the final two characters.
    #
    # Examples:
    # SC4001EC -> SC4001
    # SC4011EH -> SC4011
    # SC4022EJ -> SC4022
    return base[:-2]


def build_recording_pairs(
    data_directory,
):
    """
    Pair every PSG with its corresponding
    hypnogram.
    """

    data_directory = Path(
        data_directory
    )

    psg_files, hypnogram_files = (
        find_edf_files(
            data_directory
        )
    )

    psg_mapping = {}

    for psg in psg_files:

        recording_id = (
            get_psg_recording_id(
                psg
            )
        )

        psg_mapping[
            recording_id
        ] = psg

    hypnogram_mapping = {}

    for hypnogram in hypnogram_files:

        recording_id = (
            get_hypnogram_recording_id(
                hypnogram
            )
        )

        hypnogram_mapping[
            recording_id
        ] = hypnogram

    recording_ids = sorted(
        set(psg_mapping)
        | set(hypnogram_mapping)
    )

    rows = []

    for recording_id in recording_ids:

        rows.append(
            {
                "recording_id": recording_id,
                "psg": (
                    str(
                        psg_mapping[
                            recording_id
                        ]
                    )
                    if recording_id
                    in psg_mapping
                    else None
                ),
                "hypnogram": (
                    str(
                        hypnogram_mapping[
                            recording_id
                        ]
                    )
                    if recording_id
                    in hypnogram_mapping
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def validate_recording_pairs(
    pairings_df,
):
    """
    Check whether every recording has
    both PSG and hypnogram files.
    """

    missing_psg = pairings_df[
        pairings_df["psg"].isna()
    ]

    missing_hypnogram = pairings_df[
        pairings_df[
            "hypnogram"
        ].isna()
    ]

    return {
        "total_recordings": len(
            pairings_df
        ),
        "missing_psg": missing_psg,
        "missing_hypnogram": (
            missing_hypnogram
        ),
        "is_valid": (
            len(missing_psg) == 0
            and len(missing_hypnogram) == 0
        ),
    }