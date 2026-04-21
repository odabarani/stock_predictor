{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPKb+G5Vh0dzWBEKD1aRr3Y",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/odabarani/stock_predictor/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "orXbRAa7hovQ"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "\n",
        "from src.data_loader import get_stock_data\n",
        "from src.features import add_features\n",
        "from src.model import train_model\n",
        "from src.signals import generate_signal\n",
        "\n",
        "\n",
        "FEATURES = [\n",
        "    'MA_5',\n",
        "    'MA_20',\n",
        "    'Momentum',\n",
        "    'Volatility',\n",
        "    'Volume_Change'\n",
        "]\n",
        "\n",
        "\n",
        "st.title(\"📈 Stock ML Predictor\")\n",
        "\n",
        "\n",
        "ticker = st.text_input(\"Enter Stock Ticker\", \"AAPL\")\n",
        "\n",
        "\n",
        "if st.button(\"Run Prediction\"):\n",
        "\n",
        "    st.write(\"Fetching data...\")\n",
        "    df = get_stock_data(ticker)\n",
        "\n",
        "    st.write(\"Building features...\")\n",
        "    df = add_features(df)\n",
        "\n",
        "    st.write(\"Training model...\")\n",
        "    model, accuracy = train_model(df, FEATURES)\n",
        "\n",
        "    st.write(f\"Accuracy: {round(accuracy * 100, 2)}%\")\n",
        "\n",
        "\n",
        "    latest = df[FEATURES].iloc[-1].values.reshape(1, -1)\n",
        "\n",
        "    prediction = model.predict(latest)[0]\n",
        "    confidence = max(model.predict_proba(latest)[0])\n",
        "\n",
        "    direction = \"UP\" if prediction == 1 else \"DOWN\"\n",
        "    signal = generate_signal(prediction, confidence)\n",
        "\n",
        "\n",
        "    st.subheader(\"Result\")\n",
        "    st.write(\"Prediction:\", direction)\n",
        "    st.write(\"Confidence:\", round(confidence * 100, 2), \"%\")\n",
        "    st.write(\"Signal:\", signal)"
      ]
    }
  ]
}