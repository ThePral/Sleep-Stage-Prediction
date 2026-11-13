"""Opt-in supervised epoch recovery; training equivalence needs the user-run gate."""
import csv
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.training.recovery_state import load_recovery, publish_recovery


def variable_layout(variables):
    return [{'path': variable.path, 'shape': list(variable.shape), 'dtype': str(variable.dtype)}
            for variable in variables]


def keras_global_rng():
    # Version-pinned Keras state; reject a changed package via the resume contract.
    from keras.src.random.seed_generator import global_seed_generator
    return global_seed_generator().state


class AtomicBest(keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        score = float(logs['val_macro_f1'])
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation macro-F1.')
        if self.best is None or score > self.best:
            staging = self.path.with_name('model.pending.keras')
            self.model.save(staging)
            with staging.open('rb+') as stream:
                os.fsync(stream.fileno())
            os.replace(staging, self.path)
            self.best = score


class EpochEvents(keras.callbacks.Callback):
    """Forward epoch/batch events while preserving callbacks across one-epoch fits."""
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def on_epoch_begin(self, epoch, logs=None):
        self.callbacks.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        # Preserve the same logs dict for the outer fit's progress/history too.
        for callback in self.callbacks.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        self.callbacks.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self.callbacks.on_train_batch_end(batch, logs)


class SupervisedRecovery(keras.callbacks.Callback):
    """Last callback: serialize post-LR/early-stop state and an authoritative history."""
    def __init__(self, run, contract, best, early, reduce, tf_rng):
        super().__init__()
        self.run, self.contract = Path(run), contract
        self.best, self.early, self.reduce, self.tf_rng = best, early, reduce, tf_rng
        self.history = []
        self.started = time.perf_counter()
        self.previous_elapsed = 0.

    def _arrays(self):
        arrays = {}
        for prefix, variables in [('model', self.model.variables), ('optimizer', self.model.optimizer.variables)]:
            arrays.update({f'{prefix}_{i}': variable.numpy() for i, variable in enumerate(variables)})
        arrays.update({f'best_{i}': value for i, value in enumerate(self.early.best_weights or [])})
        arrays['keras_global_rng'] = keras_global_rng().numpy()
        arrays['tf_global_rng'] = self.tf_rng.state.numpy()
        return arrays

    def save(self):
        arrays = self._arrays()  # Instantiate global Keras RNG before capturing Python RNG.
        numpy_state = np.random.get_state()
        state = {'history': self.history, 'model_layout': variable_layout(self.model.variables),
                 'optimizer_layout': variable_layout(self.model.optimizer.variables),
                 'best_weights_count': len(self.early.best_weights or []),
                 'early': {name: getattr(self.early, name) for name in ('best', 'wait', 'stopped_epoch', 'best_epoch')},
                 'reduce': {name: getattr(self.reduce, name) for name in ('best', 'wait', 'cooldown_counter')},
                 'best_score': self.best.best, 'stop_training': bool(self.model.stop_training),
                 'python_rng': random.getstate(),
                 'numpy_rng': [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
                 'tf_rng_algorithm': int(self.tf_rng.algorithm),
                 'elapsed_seconds': self.previous_elapsed + time.perf_counter() - self.started}
        # Keras callback metrics may be NumPy scalar values.
        for group in ('early', 'reduce'):
            state[group] = {key: (value.item() if isinstance(value, np.generic) else value)
                            for key, value in state[group].items()}
        generation = publish_recovery(self.run / 'recovery', self.contract, len(self.history), state,
                                      arrays, self.best.path if self.best.path.exists() else None)
        self.write_history()
        return generation

    def write_history(self):
        if not self.history:
            return
        destination = self.run / 'training_history.csv'
        staging = destination.with_suffix('.pending.csv')
        keys = ['epoch', *sorted(set().union(*(set(row) for row in self.history)) - {'epoch'})]
        with staging.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, destination)

    def on_epoch_end(self, epoch, logs=None):
        if epoch != len(self.history):
            raise ValueError('Recovery epoch ordering mismatch.')
        row = {'epoch': epoch, **{key: float(value) for key, value in logs.items()}}
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError('Nonfinite epoch metrics; recovery was not advanced.')
        self.history.append(row)
        self.save()

    def restore(self, source):
        generation, state, arrays = load_recovery(source, self.contract)
        for prefix, variables in [('model', self.model.variables), ('optimizer', self.model.optimizer.variables)]:
            if state[f'{prefix}_layout'] != variable_layout(variables):
                raise ValueError(f'Recovery {prefix} variable layout changed.')
            for i, variable in enumerate(variables):
                values = arrays[f'{prefix}_{i}']
                if list(values.shape) != list(variable.shape) or str(values.dtype) != str(variable.dtype):
                    raise ValueError('Recovery array shape or dtype mismatch.')
                variable.assign(values)
        for name, value in state['early'].items():
            setattr(self.early, name, value)
        for name, value in state['reduce'].items():
            setattr(self.reduce, name, value)
        self.early.best_weights = ([arrays[f'best_{i}'] for i in range(state['best_weights_count'])]
                                   if state['best_weights_count'] else None)
        self.best.best = state['best_score']
        self.history = state['history']
        if (generation / 'best.keras').exists():
            shutil.copyfile(generation / 'best.keras', self.best.path)
        self.previous_elapsed = state['elapsed_seconds']
        self.model.stop_training = state['stop_training']
        if int(self.tf_rng.algorithm) != state['tf_rng_algorithm']:
            raise ValueError('TensorFlow RNG algorithm changed.')
        self.tf_rng.state.assign(arrays['tf_global_rng'])
        keras_global_rng().assign(arrays['keras_global_rng'])
        random.setstate((state['python_rng'][0], tuple(state['python_rng'][1]), state['python_rng'][2]))
        rng = state['numpy_rng']
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
        self.write_history()
        return len(self.history)


def fit_recoverable(model, batches, run, training, validation_callback, contract,
                    progress=None, resume_from=None, verbose=2, elapsed_start=None):
    """Fresh one-epoch datasets avoid prefetch advancing an implicit sampler cursor."""
    model.optimizer.build(model.trainable_variables)
    best = AtomicBest(Path(run) / 'model.keras')
    early = keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max', patience=training.patience,
                                          restore_best_weights=True)
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=.5, patience=2, min_lr=1e-6)
    tf_rng = tf.random.get_global_generator()
    recovery = SupervisedRecovery(run, contract, best, early, reduce, tf_rng)
    if elapsed_start is not None:
        recovery.started = elapsed_start
    ordered = [validation_callback, best, early, reduce]
    if progress is not None:
        ordered.append(progress)
    ordered.append(recovery)
    callbacks = keras.callbacks.CallbackList(ordered, model=model, epochs=training.epochs, steps=len(batches))
    model.stop_training = False
    callbacks.on_train_begin()
    initial_epoch = recovery.restore(resume_from) if resume_from else 0
    if initial_epoch > training.epochs:
        raise ValueError('Recovery epoch exceeds the declared schedule.')
    recovery.save()  # Epoch zero or a verified restored boundary belongs to this new attempt too.
    if not model.stop_training:
        for epoch in range(initial_epoch, training.epochs):
            model.fit(batches.as_dataset(epoch=epoch), initial_epoch=epoch, epochs=epoch + 1,
                      callbacks=[EpochEvents(callbacks)], shuffle=False, verbose=verbose)
            if model.stop_training:
                break
    callbacks.on_train_end()  # Restore best weights exactly once, after all training.
    return initial_epoch, recovery
