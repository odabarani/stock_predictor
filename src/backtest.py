def backtest(df, model, features):

    cash = 10000  # starting capital
    position = 0  # number of shares
    portfolio_values = []

    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        next_price = df['Close'].iloc[i + 1]

        X = row[features].values.reshape(1, -1)

        prediction = model.predict(X)[0]
        confidence = max(model.predict_proba(X)[0])

        # simple strategy
        if prediction == 1 and confidence > 0.65 and position == 0:
            # BUY
            position = cash / row['Close']
            cash = 0

        elif prediction == 0 and confidence > 0.65 and position > 0:
            # SELL
            cash = position * row['Close']
            position = 0

        # portfolio value
        value = cash + position * row['Close']
        portfolio_values.append(value)

    return portfolio_values