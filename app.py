import streamlit as st
import plotly.graph_objects as go
from src.data_loader import get_stock_data
from src.features import add_features
from src.model import train_model
from src.signals import generate_signal
from src.backtest import backtest



FEATURES = [
    'MA_5',
    'MA_20',
    'Momentum',
    'Volatility',
    'Volume_Change',
    'RSI',
    'MACD',
    'MACD_Signal',
    'BB_Position'
]

# --- Page Config ---
st.set_page_config(page_title="Stock ML Predictor", page_icon="📈", layout="wide")

# --- Header ---
st.title("📈 Stock ML Predictor")
st.caption("ML-powered 5-day price direction prediction using technical indicators.")
st.divider()

# --- Input ---
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
with col2:
    st.write("")
    st.write("")
    run = st.button("Run Prediction", use_container_width=True)

# --- Main Logic ---
if run:
    with st.spinner(f"Fetching data for {ticker}..."):
        df = get_stock_data(ticker)
        df = add_features(df)

    with st.spinner("Training model..."):
        model, cv_accuracy = train_model(df, FEATURES)

    portfolio = backtest(df, model, FEATURES)

    latest     = df[FEATURES].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest)[0]
    confidence = max(model.predict_proba(latest)[0])
    direction  = "📈 UP" if prediction == 1 else "📉 DOWN"
    signal     = generate_signal(prediction, confidence)

    st.divider()

    # --- Metrics Row ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction (5-Day)", direction)
    m2.metric("Confidence", f"{round(confidence * 100, 2)}%")
    m3.metric("CV Accuracy", f"{round(cv_accuracy * 100, 2)}%")

    # --- Signal Badge ---
    st.divider()
    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(signal, "gray")
    st.markdown(f"### Signal: :{signal_color}[{signal}]")

    # --- Chart ---
    st.divider()
    st.subheader(f"{ticker} — Price & Moving Averages")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'].squeeze(),
        name='Close Price', line=dict(color='#2196F3', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA_5'].squeeze(),
        name='5-Day MA', line=dict(color='#FF9800', width=1.5, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA_20'].squeeze(),
        name='20-Day MA', line=dict(color='#4CAF50', width=1.5, dash='dash')
    ))
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Backtest performance ---
    st.divider()
    st.subheader("Backtest Performance")

    st.line_chart(portfolio)

    if len(portfolio) > 0:
        total_return = (portfolio[-1] / portfolio[0] - 1) * 100
        st.metric("Strategy Return", f"{round(total_return, 2)}%")

    # --- Feature Importance ---
    st.divider()
    st.subheader("Feature Importance")
    importance_df = dict(zip(FEATURES, model.feature_importances_))
    st.bar_chart(importance_df)