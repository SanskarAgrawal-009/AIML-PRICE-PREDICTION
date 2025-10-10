# Stock Direction Prediction (NIFTY 50)

Project scaffold for predicting next-day stock direction (up/down) for NIFTY 50 constituents. This repo contains data ingestion, preprocessing, feature engineering, models, and a starter Streamlit app for visualization.

Structure:

- data/               # raw and processed CSVs
- notebooks/          # exploratory notebooks
- src/                # python modules
- models/             # trained model artifacts
- app/                # Streamlit/Dash app

Getting started:

1. Create a virtual environment and install requirements:

   python -m venv venv
   venv\Scripts\Activate.ps1; pip install -r requirements.txt

2. Run the data downloader:

   python -m src.data.download --tickers NIFTY50.csv --start 2004-01-01 --end 2025-01-01

3. Open notebooks for EDA and feature engineering in `notebooks/`.

Notes:
- This scaffold assumes you have 21 years of historical data available for NIFTY 50 constituents.
- Next steps: implement feature engineering, baseline models, and model training/evaluation.
