def add_features(df):

    # --- Moving Averages ---
    df['MA_5']          = df['Close'].rolling(5).mean()
    df['MA_20']         = df['Close'].rolling(20).mean()

    # --- Momentum ---
    df['Momentum']      = df['Close'].pct_change(5)

    # --- Volatility ---
    df['Volatility']    = df['Close'].rolling(10).std()

    # --- Volume Features ---
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA20']   = df['Volume'].rolling(20).mean()
    df['Volume_Ratio']  = df['Volume'].squeeze() / df['Volume_MA20'].squeeze()

    # --- RSI ---
    delta               = df['Close'].diff()
    gain                = delta.clip(lower=0).rolling(14).mean()
    loss                = -delta.clip(upper=0).rolling(14).mean()
    df['RSI']           = 100 - (100 / (1 + gain / loss))

    # --- MACD ---
    ema12               = df['Close'].ewm(span=12).mean()
    ema26               = df['Close'].ewm(span=26).mean()
    df['MACD']          = ema12 - ema26
    df['MACD_Signal']   = df['MACD'].ewm(span=9).mean()

    # --- Bollinger Band Position ---
    rolling_mean        = df['Close'].rolling(20).mean()
    rolling_std         = df['Close'].rolling(20).std()
    df['BB_Position']   = (df['Close'].squeeze() - rolling_mean.squeeze()) / (2 * rolling_std.squeeze())

    # --- VIX ---
    df['VIX']           = df['VIX']

    # --- Relative Strength vs SPY ---
    df['RS_SPY']        = df['Close'].squeeze() / df['SPY_Close'].squeeze()

    # --- Candlestick / Price Action ---
    df['Body_Size']     = abs(df['Close'].squeeze() - df['Open'].squeeze())
    df['Upper_Wick']    = df['High'].squeeze() - df[['Close', 'Open']].max(axis=1).squeeze()
    df['Lower_Wick']    = df[['Close', 'Open']].min(axis=1).squeeze() - df['Low'].squeeze()
    df['Gap']           = df['Open'].squeeze() - df['Close'].shift(1).squeeze()

    # --- Target: 5-day direction ---
    df['Target']        = (df['Close'].shift(-5) > df['Close']).astype(int)

    df.dropna(inplace=True)
    return df