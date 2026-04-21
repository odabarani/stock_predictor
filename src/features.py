def add_features(df):
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Momentum'] = df['Close'].pct_change(5)
    df['Volatility'] = df['Close'].rolling(10).std()
    df['Volume_Change'] = df['Volume'].pct_change()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    return df
