"""Plot audited pooled validation scores, without retraining or scoring models."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
frame = pd.read_csv(root / 'validation_results.csv')
positions = np.arange(7)
fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
for offset, budget, label, color in [(-.19, 'draw42_n18', '18 labeled participants', '#659ac4'),
                                    (.19, 'draw42_n70', '70 labeled participants', '#193e63')]:
    values = frame.loc[frame.budget_id == budget].set_index('condition').loc[[f'M{i}' for i in range(7)], 'pooled_macro_f1']
    bars = ax.bar(positions + offset, values, width=.35, color=color, label=label)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
ax.set_xticks(positions, ['M0\nSpectral', 'M1\nRaw', 'M2\nConcat', 'M3\nAttention',
                        'M4\nRaw + SSL', 'M5\nConcat + SSL', 'M6\nAttention + SSL'])
ax.set_ylim(0, 1)
ax.set_ylabel('Pooled five-class macro-F1')
ax.set_title('Sleep-EDF source-validation results\nOne training seed (42); checkpoint-selection cohort', pad=14)
ax.legend(loc='upper left', frameon=False, ncol=2)
ax.set_axisbelow(True)
ax.yaxis.grid(True, alpha=.2)
ax.spines[['top', 'right']].set_visible(False)
for suffix in ('png', 'svg'):
    fig.savefig(root / f'validation_macro_f1.{suffix}', dpi=180)
plt.close(fig)
print('Saved validation_macro_f1.png and validation_macro_f1.svg')
