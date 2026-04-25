def backtest(df, model, features):

    cash = 10000
    position = 0
    portfolio_values = []

    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        next_price = float(df['Close'].iloc[i + 1].item())

        X = row[features].values.reshape(1, -1)

        prediction = model.predict(X)[0]
        confidence = max(model.predict_proba(X)[0])

        current_price = float(row['Close'].item())

        # BUY
        if prediction == 1 and confidence > 0.65 and position == 0:
            position = cash / current_price
            cash = 0

        # SELL
        elif prediction == 0 and confidence > 0.65 and position > 0:
            cash = position * current_price
            position = 0

        value = cash + position * current_price
        portfolio_values.append(value)

    return portfolio_values