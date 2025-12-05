
import os
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.features import prepare_for_classification
from scipy.stats import randint, uniform

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def eval_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }


def train_models_for_csv(csv_path, out_dir='models_advanced', n_splits=5, tune=False, n_iter=10, save_importances=True, top_k=None):
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    X, y = prepare_for_classification(df)
    if X.empty:
        return None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rf_fold_metrics = []
    xgb_fold_metrics = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1] if hasattr(rf, 'predict_proba') else None
        rm = eval_metrics(y_test, rf_pred)
        if rf_prob is not None:
            try:
                rm['roc_auc'] = float(roc_auc_score(y_test, rf_prob))
            except Exception:
                rm['roc_auc'] = None
        rf_fold_metrics.append(rm)
        if HAS_XGB:
            xg = xgb.XGBClassifier(eval_metric='logloss', n_estimators=100, verbosity=0)
            xg.fit(X_train, y_train)
            xgb_pred = xg.predict(X_test)
            xgb_prob = xg.predict_proba(X_test)[:, 1] if hasattr(xg, 'predict_proba') else None
            xm = eval_metrics(y_test, xgb_pred)
            if xgb_prob is not None:
                try:
                    xm['roc_auc'] = float(roc_auc_score(y_test, xgb_prob))
                except Exception:
                    xm['roc_auc'] = None
            xgb_fold_metrics.append(xm)

    ticker = Path(csv_path).stem
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = {'ticker': ticker, 'rf': {'folds': rf_fold_metrics},}


    if tune:
        print(f'Tuning RandomForest for {ticker} (n_iter={n_iter})')
        param_dist_rf = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 20),
            'min_samples_split': randint(2, 20),
            'max_features': ['sqrt', 'log2', None]
        }
        rsearch = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_distributions=param_dist_rf, n_iter=n_iter, cv=tscv, n_jobs=-1, random_state=42)
        rsearch.fit(X, y)
        rf_final = rsearch.best_estimator_
        results['rf']['tuned_params'] = rsearch.best_params_
    else:
        rf_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_final.fit(X, y)

    rf_path = Path(out_dir) / f"{ticker}_rf.pkl"
    joblib.dump(rf_final, rf_path)
    results['rf']['model_path'] = str(rf_path)


    if save_importances and hasattr(rf_final, 'feature_importances_'):
        fi = pd.Series(rf_final.feature_importances_, index=X.columns).sort_values(ascending=False)
        if top_k:
            fi = fi.head(top_k)
        fi.to_csv(Path(out_dir)/f"{ticker}_rf_feature_importances.csv")

    if HAS_XGB:
        results['xgb'] = {'folds': xgb_fold_metrics}
        if tune:
            print(f'Tuning XGBoost for {ticker} (n_iter={n_iter})')
            param_dist_xgb = {
                'n_estimators': randint(50, 300),
                'max_depth': randint(2, 12),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.5, 0.5),
                'colsample_bytree': uniform(0.5, 0.5)
            }
            xgb_clf = xgb.XGBClassifier(eval_metric='logloss', verbosity=0, use_label_encoder=False) if hasattr(xgb, 'XGBClassifier') else xgb.XGBClassifier(eval_metric='logloss', verbosity=0)
            xsearch = RandomizedSearchCV(xgb_clf, param_distributions=param_dist_xgb, n_iter=n_iter, cv=tscv, n_jobs=-1, random_state=42)
            xsearch.fit(X, y)
            xg_final = xsearch.best_estimator_
            results['xgb']['tuned_params'] = xsearch.best_params_
        else:
            xg_final = xgb.XGBClassifier(eval_metric='logloss', n_estimators=200, verbosity=0)
            xg_final.fit(X, y)
        xg_path = Path(out_dir)/f"{ticker}_xgb.pkl"
        joblib.dump(xg_final, xg_path)
        results['xgb']['model_path'] = str(xg_path)

    return results


def main(data_dir='data/raw', out_dir='models_advanced', tune=False, n_iter=10, n_splits=5, save_importances=True, top_k=None):
    data_dir = Path(data_dir)
    all_results = []
    for csv in sorted(data_dir.glob('*.csv')):
        print('Training', csv.name)
        try:
            res = train_models_for_csv(csv, out_dir=out_dir, n_splits=n_splits, tune=tune, n_iter=n_iter, save_importances=save_importances, top_k=top_k)
            if res:
                all_results.append(res)
        except Exception as e:
            print('Failed', csv.name, e)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir)/'advanced_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw')
    parser.add_argument('--out_dir', default='models_advanced')
    parser.add_argument('--tune', action='store_true', help='Enable RandomizedSearchCV tuning (can be slow)')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of parameter samples for RandomizedSearchCV')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of TimeSeriesSplit folds')
    parser.add_argument('--no_importances', action='store_true', help='Do not save feature importances')
    parser.add_argument('--top_k', type=int, default=None, help='Save only top_k feature importances')
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir, tune=args.tune, n_iter=args.n_iter, n_splits=args.n_splits, save_importances=(not args.no_importances), top_k=args.top_k)
