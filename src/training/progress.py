"""Periodic console progress; no model, optimizer or random-state operations."""

import time


class BatchProgress:
    def __init__(self, phase, epoch, epochs, total_batches, interval=0):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
            raise ValueError("Progress interval must be a nonnegative integer.")
        if total_batches < 1:
            raise ValueError("Progress requires at least one batch.")
        self.phase, self.epoch, self.epochs = phase, epoch, epochs
        self.total, self.interval = total_batches, interval
        self.started = time.perf_counter()
        if interval:
            print(f"{phase} epoch {epoch}/{epochs}: starting {total_batches} batches", flush=True)

    @staticmethod
    def duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update(self, completed, logs=None):
        if not self.interval:
            return
        if not 1 <= completed <= self.total:
            raise ValueError("Progress batch falls outside its epoch.")
        if completed != 1 and completed != self.total and completed % self.interval:
            return
        elapsed = time.perf_counter() - self.started
        eta = elapsed * (self.total - completed) / completed
        metrics = " ".join(f"{name}={float(logs[name]):.5f}" for name in
                           ["loss", "accuracy", "retrieval_accuracy"] if logs and name in logs)
        print(f"{self.phase} epoch {self.epoch}/{self.epochs} batch {completed}/{self.total} "
              f"({100 * completed / self.total:.1f}%) {metrics} "
              f"elapsed={self.duration(elapsed)} ETA~{self.duration(eta)}", flush=True)
