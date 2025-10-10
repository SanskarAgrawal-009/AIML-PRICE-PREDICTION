 """Train a simple baseline classifier (logistic regression) for next-day direction."""
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from src.features import prepare_for_classification


def train_baseline(csv_path, model_out='models/baseline_lr.pkl'):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    X, y = prepare_for_classification(df)
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
    # train final on all data
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X, y)
    joblib.dump(final_model, model_out)
    return accuracies



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to ticker CSV')
    parser.add_argument('--out', default='models/baseline_lr.pkl')
    args = parser.parse_args()
    accs = train_baseline(args.csv, model_out=args.out)
    print('Fold accuracies:', accs)
