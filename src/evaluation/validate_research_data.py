"""Validate prepared collections and optionally reproduce the frozen source baseline.

This command never evaluates predictive performance on ISRUC.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.epoch_store import EpochStore
from src.data.pairing import build_recording_pairs
from src.data.prepare_research_data import ROOT, participant_assignments
from src.preprocessing.normalization import ALL_FEATURES


def validate_collection(root, *, require_complete=False, check_baseline=False):
    summary = {"datasets": {}, "baseline": None}
    assignments = participant_assignments(ROOT / "final_model_corrected/participant_split.csv")
    test_sequences, test_labels = [], []
    scaler = None
    if check_baseline:
        import joblib
        scaler = joblib.load(ROOT / "final_model/scaler.pkl")
    for dataset in ["sleepedf", "isruc"]:
        paths = sorted((Path(root) / dataset).glob("*/manifest.json"))
        counts = {"recordings": 0, "epochs": 0, "quality_epochs": 0, "supervised_epochs": 0,
                  "participants": set(), "participants_by_split": {}, "windows_10_by_split": {}}
        seen = set()
        for path in paths:
            with EpochStore(path.parent) as store:
                frame = store.epochs
                actual_dataset, participant, recording, split = frame.iloc[0][["dataset", "participant_id", "recording_id", "split"]]
                if actual_dataset != dataset or recording != path.parent.name or recording in seen:
                    raise ValueError("Store identity disagrees with its collection path.")
                if dataset == "sleepedf" and assignments.get(participant) != split:
                    raise ValueError("Store disagrees with the frozen participant manifest.")
                seen.add(recording)
                counts["recordings"] += 1
                counts["epochs"] += len(frame)
                counts["quality_epochs"] += int(frame.quality_valid.sum())
                counts["supervised_epochs"] += int(frame.supervised_valid.sum())
                counts["participants"].add(participant)
                counts["participants_by_split"].setdefault(split, set()).add(participant)
                windows = store.sequence_positions(10)
                counts["windows_10_by_split"][split] = counts["windows_10_by_split"].get(split, 0) + len(windows)
                if check_baseline and dataset == "sleepedf" and split == "test":
                    if store.config.epoch_duration != 30 or store.config.sampling_frequency != 100:
                        raise ValueError("Baseline validation requires its original epoch/sample policy.")
                    valid = frame.supervised_valid.to_numpy()
                    values = np.zeros((len(frame), len(ALL_FEATURES)), dtype=np.float32)
                    values[valid] = scaler.transform(frame.loc[valid, ALL_FEATURES]).to_numpy(dtype=np.float32)
                    test_sequences.append(values[windows])
                    test_labels.append(frame.target_stage.to_numpy()[windows[:, -1]])
        if require_complete:
            if dataset == "sleepedf":
                expected = set(build_recording_pairs(ROOT / "data/raw").recording_id)
            else:
                expected = {p.name for p in (ROOT / "data/external/ISRUC-Sleep-I").iterdir() if p.is_dir() and p.name.isdigit()}
            if not expected or seen != expected:
                raise ValueError(f"Incomplete {dataset} collection; missing={sorted(expected - seen)}, extra={sorted(seen - expected)}")
        counts["participants"] = len(counts["participants"])
        counts["participants_by_split"] = {key: len(value) for key, value in counts["participants_by_split"].items()}
        summary["datasets"][dataset] = counts
    if check_baseline:
        import pandas as pd
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from src.inference.predict import CLASS_NAMES, SleepStagePredictor
        if not test_sequences:
            raise ValueError("No source test sequences were prepared.")
        predictor = SleepStagePredictor.from_directory()
        probabilities = predictor.model.predict(np.concatenate(test_sequences), batch_size=256, verbose=0)
        predicted = np.asarray(CLASS_NAMES)[probabilities.argmax(axis=1)]
        true = np.concatenate(test_labels)
        matrix = confusion_matrix(true, predicted, labels=CLASS_NAMES)
        expected = pd.read_csv(ROOT / "results/final_evaluation/final_confusion_matrix.csv", index_col=0).to_numpy()
        if not np.array_equal(matrix, expected):
            raise ValueError("Prepared-data predictions differ from the frozen source confusion matrix.")
        summary["baseline"] = {
            "test_windows": len(true), "accuracy": accuracy_score(true, predicted),
            "macro_f1": f1_score(true, predicted, average="macro"),
            "confusion_matrix_matches": True,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/processed/research_v1")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--check-baseline", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    summary = validate_collection(args.root, require_complete=args.require_complete, check_baseline=args.check_baseline)
    output = json.dumps(summary, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
