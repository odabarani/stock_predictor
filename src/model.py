from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def train_model(df, features):
    X = df[features]
    y = df['Target']

    # Walk-forward split — train on first 70%, test on last 30%
    # This prevents data leakage in the backtest
    split        = int(len(X) * 0.70)
    X_train      = X.iloc[:split]
    y_train      = y.iloc[:split]

    tscv  = TimeSeriesSplit(n_splits=5)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )

    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')

    # Final fit on training portion only
    model.fit(X_train, y_train)

    return model, scores.mean()