"""Train baseline models for all CSVs in data/raw/ and save metrics.

Usage: python -m src.train_all --data_dir data/raw --out_dir models
"""
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from src.features import prepare_for_classification


def train_one(csv_path, out_dir='models'):
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    X, y = prepare_for_classification(df)
    if X.empty:
        return None
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
    # final model on all data
    final = LogisticRegression(max_iter=1000)
    final.fit(X, y)
    ticker = Path(csv_path).stem
    os.makedirs(out_dir, exist_ok=True)
    model_path = Path(out_dir) / f"{ticker}.pkl"
    joblib.dump(final, model_path)
    return {'ticker': ticker, 'mean_accuracy': float(pd.Series(accuracies).mean()), 'fold_accuracies': accuracies}


def main(data_dir='data/raw', out_dir='models'):
    data_dir = Path(data_dir)
    results = []
    for csv in sorted(data_dir.glob('*.csv')):
        print('Training', csv.name)
        try:
            res = train_one(csv, out_dir=out_dir)
            if res:
                results.append(res)
        except Exception as e:
            print('Failed', csv.name, e)
    if results:
        df = pd.DataFrame([{'ticker': r['ticker'], 'mean_accuracy': r['mean_accuracy'], 'fold_accuracies': r['fold_accuracies']} for r in results])
        df.to_csv(Path(out_dir)/'metrics.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw')
    parser.add_argument('--out_dir', default='models')
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir)
