import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys, os
# Make project root importable so `from src.features import ...` works when running Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.features import prepare_for_classification
import json

st.set_page_config(layout='wide')
st.title('NIFTY Next-day Direction Predictor')

DATA_DIR = Path('data/raw')
MODELS_DIR = Path('models_advanced')

col1, col2 = st.columns([1,3])

with col1:
    st.header('Controls')
    tickers = sorted([p.stem for p in DATA_DIR.glob('*.csv')])
    ticker = st.selectbox('Choose ticker CSV', options=tickers)
    model_files = sorted([p.name for p in MODELS_DIR.glob('*_rf.pkl')])
    model_choice = st.selectbox('Choose RF model', options=['']+model_files)
    run_predict = st.button('Predict latest')

with col2:
    st.header('Model & Data Info')
    if model_choice:
        st.subheader('Selected model')
        st.write(model_choice)
        metric_file = MODELS_DIR / 'advanced_metrics.json'
        if metric_file.exists():
            with open(metric_file,'r') as f:
                adv = json.load(f)
            # find ticker entry
            t = ticker
            res = next((e for e in adv if e.get('ticker')==t), None)
            if res:
                st.write('Advanced metrics (summary):')
                rf_folds = res.get('rf',{}).get('folds', [])
                if rf_folds:
                    accs = [f['accuracy'] for f in rf_folds]
                    st.write('Mean accuracy:', round(sum(accs)/len(accs),4))
                    st.write('Fold accuracies:', [round(a,4) for a in accs])
                # show feature importances if present
                fi_path = MODELS_DIR / f"{t}_rf_feature_importances.csv"
                if fi_path.exists():
                    fi = pd.read_csv(fi_path, index_col=0, header=None).squeeze()
                    st.write('Top features:')
                    st.table(fi.head(10))
    else:
        st.info('Select a model to show metrics')

st.markdown('---')

if run_predict:
    csv_path = DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        st.error('Ticker CSV not found: ' + str(csv_path))
    else:
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)

        # helper to find price column
        def _find_price_col(df):
            candidates = ['Close', 'close', 'Adj Close', 'adj close', 'adjusted close', 'Last', 'last']
            cols = {c.lower().strip(): c for c in df.columns}
            for cand in candidates:
                if cand.lower() in cols:
                    return cols[cand.lower()]
            # fallback to first numeric column
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
            raise KeyError('No numeric column found for price')

        try:
            price_col = _find_price_col(df)
        except KeyError:
            st.error('Could not find price column in CSV')
            price_col = None

        X, y = prepare_for_classification(df)
        if X.empty or price_col is None:
            st.error('No usable features found for this CSV')
        else:
            latest = X.tail(1)
            st.write('Latest features (last row):')
            st.table(latest.T)
            # build price plot with historical predictions if model selected
            # Try to use Plotly for interactive plotting; fall back to Matplotlib if unavailable
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                use_plotly = True
            except Exception:
                import matplotlib.pyplot as plt
                use_plotly = False

            if use_plotly:
                fig = px.line(df, x=df.index, y=price_col, title=f'{ticker} price with predicted signals')

                # UI controls for historical vs latest-only predictions
                show_historical = st.sidebar.checkbox('Show historical predictions', value=True)
                show_latest_only = st.sidebar.checkbox('Show only latest prediction', value=False)
                prob_threshold = st.sidebar.slider('Probability threshold for up signal', 0.0, 1.0, 0.5)

                @st.cache_resource
                def _load_model(path: str):
                    return joblib.load(path)

                if model_choice:
                    model_path = MODELS_DIR / model_choice
                    model = _load_model(str(model_path))
                    # historical predictions aligned with X index
                    try:
                        preds_all = model.predict(X)
                        probs_all = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                    except Exception:
                        preds_all = None
                        probs_all = None

                    if show_historical and preds_all is not None and len(preds_all) == len(X) and not show_latest_only:
                        pred_dates = X.index
                        # apply probability threshold if probabilities available
                        if probs_all is not None:
                            up_idx = [d for d,p,pr in zip(pred_dates, preds_all, probs_all) if p==1 and pr[1] >= prob_threshold]
                            down_idx = [d for d,p,pr in zip(pred_dates, preds_all, probs_all) if p==0 and (probs_all is None or pr[1] < prob_threshold)]
                        else:
                            up_idx = [d for d,p in zip(pred_dates, preds_all) if p==1]
                            down_idx = [d for d,p in zip(pred_dates, preds_all) if p==0]

                        # overlay small markers for predicted up/down days
                        fig.add_trace(go.Scatter(x=up_idx, y=df.loc[up_idx, price_col], mode='markers', marker=dict(color='green', size=6), name='predicted_up'))
                        fig.add_trace(go.Scatter(x=down_idx, y=df.loc[down_idx, price_col], mode='markers', marker=dict(color='red', size=6), name='predicted_down'))

                    # latest prediction and probability
                    pred = model.predict(latest)[0]
                    prob = model.predict_proba(latest)[0] if hasattr(model, 'predict_proba') else None
                    last_date = latest.index[-1]
                    last_price = df.loc[last_date, price_col]
                    color = 'green' if pred==1 else 'red'

                    if (show_latest_only) or (not show_latest_only):
                        # always show latest marker/annotation
                        fig.add_trace(go.Scatter(x=[last_date], y=[last_price], mode='markers', marker=dict(color=color, size=12, symbol='star'), name='latest_pred'))
                        ann_text = f'Pred: {int(pred)}'
                        if prob is not None:
                            ann_text += f' | P(up)={prob[1]:.2f}'
                        fig.add_annotation(x=last_date, y=last_price, text=ann_text, showarrow=True, arrowhead=1)

                st.plotly_chart(fig, use_container_width=True)
            else:
                # Matplotlib fallback
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index, df[price_col], label=price_col)

                if model_choice:
                    model_path = MODELS_DIR / model_choice
                    model = joblib.load(model_path)
                    try:
                        preds_all = model.predict(X)
                    except Exception:
                        preds_all = None
                    if preds_all is not None and len(preds_all) == len(X):
                        pred_dates = X.index
                        up_idx = [d for d,p in zip(pred_dates, preds_all) if p==1]
                        down_idx = [d for d,p in zip(pred_dates, preds_all) if p==0]
                        ax.scatter(up_idx, df.loc[up_idx, price_col], color='green', s=10, label='predicted_up')
                        ax.scatter(down_idx, df.loc[down_idx, price_col], color='red', s=10, label='predicted_down')

                    pred = model.predict(latest)[0]
                    prob = model.predict_proba(latest)[0] if hasattr(model, 'predict_proba') else None
                    last_date = latest.index[-1]
                    last_price = df.loc[last_date, price_col]
                    color = 'green' if pred==1 else 'red'
                    ax.scatter([last_date], [last_price], color=color, s=60, marker='*', label='latest_pred')
                    ann_text = f'Pred: {int(pred)}'
                    if prob is not None:
                        ann_text += f' | P(up)={prob[1]:.2f}'
                    ax.annotate(ann_text, xy=(last_date, last_price), xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

                ax.legend()
                ax.set_title(f'{ticker} price with predicted signals')
                st.pyplot(fig)
