"""Streaming five-class metrics with explicit handling of absent stages."""

import numpy as np

from src.preprocessing.sequences import CLASS_LABELS


class ClassificationAccumulator:
    def __init__(self):
        self.matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
        self.negative_log_likelihood = 0.0

    def update(self, targets, probabilities):
        targets = np.asarray(targets)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Targets must be a one-dimensional integer array.")
        if probabilities.shape != (len(targets), 5) or not np.isfinite(probabilities).all():
            raise ValueError("Expected finite five-class probabilities aligned with targets.")
        if np.any(targets < 0) or np.any(targets >= 5):
            raise ValueError("Invalid target class.")
        if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Class probabilities must be nonnegative and sum to one.")
        np.add.at(self.matrix, (targets, probabilities.argmax(axis=1)), 1)
        self.negative_log_likelihood += float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-7, 1)).sum())

    def result(self):
        count = int(self.matrix.sum())
        if not count:
            raise ValueError("No examples were evaluated.")
        true = self.matrix.sum(axis=1)
        predicted = self.matrix.sum(axis=0)
        tp = self.matrix.diagonal()
        precision = np.divide(tp, predicted, out=np.zeros(5), where=predicted > 0)
        recall = np.divide(tp, true, out=np.zeros(5), where=true > 0)
        f1 = np.divide(2 * tp, true + predicted, out=np.zeros(5), where=true + predicted > 0)
        accuracy = float(tp.sum() / count)
        expected = float(np.dot(true / count, predicted / count))
        return {
            "examples": count, "loss": self.negative_log_likelihood / count,
            "accuracy": accuracy, "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[true > 0].mean()),
            "weighted_f1": float(np.dot(f1, true) / count),
            "cohen_kappa": float((accuracy - expected) / (1 - expected)) if expected < 1 else None,
            "macro_f1_policy": "fixed_five_classes_zero_division_zero",
            "balanced_accuracy_policy": "mean_recall_over_truth_classes_present",
            "per_stage": {label: {"support": int(true[i]), "precision": float(precision[i]),
                                  "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, label in enumerate(CLASS_LABELS)},
            "confusion_matrix": self.matrix.tolist(),
        }
