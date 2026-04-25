import yfinance as yf
import pandas as pd
from datetime import date


def get_stock_data(ticker):
    df = yf.download(ticker, start="2010-01-01", end=date.today(), progress=False)

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- VIX ---
    vix = yf.download("^VIX", start="2010-01-01", end=date.today(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    df['VIX'] = vix['Close']

    # --- SPY ---
    spy = yf.download("SPY", start="2010-01-01", end=date.today(), progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    df['SPY_Close'] = spy['Close']

    # --- Fundamentals ---
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        df['PE_Ratio']       = info.get('trailingPE', 0)
        df['Profit_Margin']  = info.get('profitMargins', 0)
        df['Revenue_Growth'] = info.get('revenueGrowth', 0)
        df['Debt_Equity']    = info.get('debtToEquity', 0)
        for col in ['PE_Ratio', 'Profit_Margin', 'Revenue_Growth', 'Debt_Equity']:
            df[col] = df[col].ffill()
    except Exception:
        for col in ['PE_Ratio', 'Profit_Margin', 'Revenue_Growth', 'Debt_Equity']:
            df[col] = 0

    df.dropna(subset=['VIX', 'SPY_Close'], inplace=True)
    return df