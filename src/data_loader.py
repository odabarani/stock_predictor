import yfinance as yf

def get_stock_data(ticker):
    df = yf.download(ticker, start="2019-01-01", end="2024-01-01")
    return df
