"""Quick EDA script for a single ticker or all tickers in data/raw.

Run from project root:
    python notebooks/eda.py --sample RELIANCE.csv
"""
import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the project root is on sys.path so `src` package can be imported when
# running this script directly (python notebooks/eda.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import prepare_for_classification, add_technical_indicators


def plot_sample(csv_path, out_dir='notebooks/figs'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    # add tech indicators
    df2 = add_technical_indicators(df)
    # price plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(df2.index, df2['Close'], label='Close')
    ax[0].plot(df2.index, df2['sma_20'], label='SMA20')
    ax[0].legend()
    ax[1].plot(df2.index, df2['rsi_14'], label='RSI14')
    ax[1].axhline(70, color='red', linestyle='--')
    ax[1].axhline(30, color='green', linestyle='--')
    ax[1].legend()
    ax[2].plot(df2.index, df2['bb_width'], label='BB Width')
    ax[2].legend()
    plt.tight_layout()
    fig_path = out_dir / f"{Path(csv_path).stem}_eda.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print('Saved', fig_path)


def summary_metrics(data_dir='data/raw', out_file='notebooks/eda_summary.csv'):
    data_dir = Path(data_dir)
    rows = []
    for csv in sorted(data_dir.glob('*.csv')):
        try:
            df = pd.read_csv(csv, parse_dates=True, index_col=0)
        except Exception as e:
            print(f"Skipping {csv.name}: could not read CSV ({e})")
            continue
        try:
            X, y = prepare_for_classification(df)
        except KeyError as e:
            print(f"Skipping {csv.name}: no price column found ({e})")
            continue
        n = len(y)
        if n == 0:
            print(f"Skipping {csv.name}: no usable rows after feature prep")
            continue
        balance = y.mean()
        rows.append({'ticker': csv.stem, 'n_rows': n, 'up_fraction': balance})
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print('Wrote', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', help='CSV filename in data/raw to plot (e.g., RELIANCE.csv)')
    parser.add_argument('--data_dir', default='data/raw')
    args = parser.parse_args()
    if args.sample:
        sample_path = Path(args.data_dir)/args.sample
        plot_sample(sample_path)
    else:
        summary_metrics(data_dir=args.data_dir)
