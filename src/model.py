from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def train_model(df, features):
    X = df[features]
    y = df['Target']

    # 🚨 split LAST row out
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    tscv = TimeSeriesSplit(n_splits=5)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )

    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')

    model.fit(X_train, y_train)

    return model, scores.mean()