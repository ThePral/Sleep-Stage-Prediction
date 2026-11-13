"""User-run synthetic optimizer gate worker. Never invoke during read-only checks."""
import argparse
from pathlib import Path
from unittest import mock

from src.models.multiview import ModelConfig
from src.training.supervised import SourceValidation, TrainingConfig, train_supervised
from src.training.supervised_recovery import SupervisedRecovery


class IntentionalInterruption(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--interrupt', action='store_true')
    args = parser.parse_args()
    # Controlled callback inputs deliberately exercise patience and LR state.
    # These are synthetic test metrics, not experimental performance results.
    def controlled_validation(self, epoch, logs=None):
        logs.update(val_macro_f1=[.5, .6, .55, .54, .53, .52][epoch],
                    val_loss=[1., .9, 1., 1.1, 1.2, 1.3][epoch],
                    val_accuracy=.5, val_balanced_accuracy=.5)
    original = SupervisedRecovery.save
    def save_then_interrupt(self):
        result = original(self)
        if args.interrupt and len(self.history) == 3:
            raise IntentionalInterruption('Power-loss simulation after sealed epoch three.')
        return result
    config = ModelConfig(variant='gated', sequence_length=2, filters=(4, 8), kernels=(7, 5), strides=(10, 10),
                         token_dim=8, attention_heads=2, temporal_head='bilstm', temporal_units=4, dropout=.3)
    training = TrainingConfig(batch_size=2, epochs=6, patience=2, recovery=True, compiled_validation=True)
    with mock.patch.object(SourceValidation, 'on_epoch_end', controlled_validation), \
         mock.patch.object(SupervisedRecovery, 'save', save_then_interrupt):
        try:
            train_supervised(args.data_root, args.run_dir, config, training,
                train_recordings=['SC4001', 'SC4031'], validation_recordings=['SC4011'],
                split_file=args.data_root / 'split.csv', max_train_batches=3, max_validation_batches=2,
                resume_from=args.resume_from, verbose=2)
        except IntentionalInterruption:
            if not args.interrupt:
                raise
            print('Expected interruption; checkpoint remains sealed.', flush=True)


if __name__ == '__main__':
    main()
