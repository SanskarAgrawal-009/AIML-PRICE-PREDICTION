import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASELINE_METRICS = Path('models') / 'metrics.csv'
ADVANCED_METRICS = Path('models_advanced') / 'advanced_metrics.json'
OUT_DIR = Path('notebooks') / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load baseline
if BASELINE_METRICS.exists():
    base = pd.read_csv(BASELINE_METRICS)
    # baseline expected columns: ticker, mean_accuracy
else:
    base = pd.DataFrame()

# load advanced
adv = []
if ADVANCED_METRICS.exists():
    with open(ADVANCED_METRICS, 'r') as f:
        adv = json.load(f)
adv_df_rows = []
for entry in adv:
    t = entry.get('ticker')
    rf = entry.get('rf', {})
    rf_folds = rf.get('folds', [])
    if rf_folds:
        accs = [f.get('accuracy') for f in rf_folds]
        adv_df_rows.append({'ticker': t, 'advanced_rf_mean_accuracy': sum(accs)/len(accs)})

adv_df = pd.DataFrame(adv_df_rows)

# merge
if not base.empty and not adv_df.empty:
    # baseline may have 'mean_accuracy' or similar; try common names
    baseline_col = None
    for col in base.columns:
        if 'mean' in col and 'acc' in col:
            baseline_col = col
            break
    if baseline_col is None and 'mean_accuracy' in base.columns:
        baseline_col = 'mean_accuracy'
    if baseline_col is None:
        # try accuracy
        if 'accuracy' in base.columns:
            baseline_col = 'accuracy'
        else:
            baseline_col = base.columns[1] if len(base.columns)>1 else base.columns[0]
    merged = base.rename(columns={baseline_col: 'baseline_mean_accuracy'}).merge(adv_df, on='ticker', how='outer')
    merged.to_csv('notebooks/compare_summary.csv', index=False)

    # plot
    merged = merged.dropna(subset=['baseline_mean_accuracy','advanced_rf_mean_accuracy'])
    if not merged.empty:
        merged = merged.sort_values('baseline_mean_accuracy')
        plt.figure(figsize=(10,6))
        plt.plot(merged['ticker'], merged['baseline_mean_accuracy'], label='baseline')
        plt.plot(merged['ticker'], merged['advanced_rf_mean_accuracy'], label='advanced RF')
        plt.xticks(rotation=90)
        plt.ylabel('Mean Accuracy')
        plt.title('Baseline vs Advanced (RF) Mean Accuracy per Ticker')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR/'accuracy_comparison.png')
        print('Saved', OUT_DIR/'accuracy_comparison.png')
else:
    print('Baseline or advanced metrics missing; baseline exists:', BASELINE_METRICS.exists(), 'advanced exists:', ADVANCED_METRICS.exists())
