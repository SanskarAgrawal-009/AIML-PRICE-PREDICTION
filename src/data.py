"""Simple data download utilities using yfinance."""
import os
from datetime import datetime
import yfinance as yf
import pandas as pd


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def download_ticker(ticker, start, end, out_dir="data/raw"):
    ensure_dir(out_dir)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    file_path = os.path.join(out_dir, f"{ticker.replace('/', '_')}.csv")
    df.to_csv(file_path)
    return file_path


def download_tickers_from_list(ticker_list, start, end, out_dir="data/raw"):
    paths = {}
    for t in ticker_list:
        try:
            p = download_ticker(t, start, end, out_dir=out_dir)
            paths[t] = p
        except Exception as e:
            print(f"Skipping {t}: {e}")
    return paths


def load_csv(path):
    return pd.read_csv(path, parse_dates=True, index_col=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download historical data using yfinance")
    parser.add_argument("--tickers", type=str, required=True,
                        help="Comma-separated tickers or path to a .txt/.csv file containing tickers")
    parser.add_argument("--start", type=str, default="2004-01-01")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--out", type=str, default="data/raw")

    args = parser.parse_args()
    if args.tickers.endswith('.csv') or args.tickers.endswith('.txt'):
        # read file
        df = pd.read_csv(args.tickers, header=None)
        tickers = df.iloc[:,0].astype(str).tolist()
    else:
        tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]

    print(f"Downloading {len(tickers)} tickers to {args.out}")
    download_tickers_from_list(tickers, args.start, args.end, out_dir=args.out)
