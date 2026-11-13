"""Review already completed histories; no model execution or optimizer updates."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run import load_study, verify_job_result

out = Path(__file__).resolve().parent
study_dir = ROOT / 'runs/bilstm_terminal_2026-11-02'
study, _ = load_study(study_dir)
ledger = json.loads((study_dir / 'ledger.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
colors = ['#777777', '#1764ab', '#168579', '#c16b10', '#8051a6', '#458b21', '#bf3644']
rows = []
for job in study['jobs']:
    if job['kind'] != 'supervised':
        continue
    entry = ledger['jobs'][job['job_id']]
    assert entry['status'] == 'completed'
    attempt = study_dir / entry['attempt']
    verify_job_result(attempt, study, job['job_id'])
    history = pd.read_csv(attempt / 'artifacts/training_history.csv')
    assert np.isfinite(history.to_numpy()).all()
    summary = json.loads((attempt / 'artifacts/summary.json').read_text())
    best_index = int(history.val_macro_f1.to_numpy().argmax())
    assert best_index + 1 == summary['best_epoch']
    rows.append({'budget_id': job['budget_id'], 'condition': job['condition'],
                 'epochs_completed': len(history), 'best_epoch': best_index + 1,
                 'first_val_macro_f1': float(history.val_macro_f1.iloc[0]),
                 'best_val_macro_f1': float(history.val_macro_f1.iloc[best_index]),
                 'last_val_macro_f1': float(history.val_macro_f1.iloc[-1]),
                 'first_training_loss': float(history.loss.iloc[0]),
                 'last_training_loss': float(history.loss.iloc[-1]),
                 'last_validation_loss': float(history.val_loss.iloc[-1]),
                 'last_training_accuracy': float(history.accuracy.iloc[-1]),
                 'last_validation_accuracy': float(history.val_accuracy.iloc[-1])})
    row_index = 0 if job['budget_id'] == 'draw42_n18' else 1
    color = colors[int(job['condition'][1:])]
    for column, field in [(0, 'loss'), (1, 'val_macro_f1')]:
        axes[row_index, column].plot(history.epoch + 1, history[field], marker='.',
                                     color=color, label=job['condition'], linewidth=1.4)
for row_index, budget in enumerate([18, 70]):
    for column, name in enumerate(['Training loss', 'Pooled source-validation macro-F1']):
        ax = axes[row_index, column]
        ax.set_title(f'{budget} labeled participants: {name}', fontsize=11)
        ax.set_xlabel('Completed epoch')
        ax.set_ylabel('Cross-entropy loss' if column == 0 else 'Macro-F1')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
axes[0, 0].legend(ncol=4, fontsize=8, frameon=False)
fig.suptitle('Completed Sleep-EDF study: learning curves (seed 42)', fontsize=14)
for suffix in ('png', 'svg'):
    fig.savefig(out / f'learning_curves.{suffix}', dpi=170)
plt.close(fig)
pd.DataFrame(rows).to_csv(out / 'learning_summary.csv', index=False)
assert 'tensorflow' not in sys.modules
print(pd.DataFrame(rows)[['budget_id', 'condition', 'epochs_completed', 'best_epoch',
                          'first_val_macro_f1', 'best_val_macro_f1', 'last_val_macro_f1']].to_string(index=False))
print('No TensorFlow import, optimizer updates, or held-out scoring.')
