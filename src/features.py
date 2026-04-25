def add_features(df):
    # Moving averages
    df['MA_5']         = df['Close'].rolling(5).mean()
    df['MA_20']        = df['Close'].rolling(20).mean()

    # Momentum
    df['Momentum']     = df['Close'].pct_change(5)

    # Volatility
    df['Volatility']   = df['Close'].rolling(10).std()

    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change()

    # RSI
    delta              = df['Close'].diff()
    gain               = delta.clip(lower=0).rolling(14).mean()
    loss               = -delta.clip(upper=0).rolling(14).mean()
    df['RSI']          = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12              = df['Close'].ewm(span=12).mean()
    ema26              = df['Close'].ewm(span=26).mean()
    df['MACD']         = ema12 - ema26
    df['MACD_Signal']  = df['MACD'].ewm(span=9).mean()

    # Bollinger Band position
    rolling_mean       = df['Close'].rolling(20).mean()
    rolling_std        = df['Close'].rolling(20).std()
    df['BB_Position']  = (df['Close'].squeeze() - rolling_mean.squeeze()) / (2 * rolling_std.squeeze())

    # Target — 5 day direction
    df['Target']       = (df['Close'].shift(-5) > df['Close']).astype(int)

    df.dropna(inplace=True)
    return df