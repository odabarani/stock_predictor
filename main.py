{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO36wJwSjjgONyEj4o7M7nh",
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
        "<a href=\"https://colab.research.google.com/github/odabarani/stock_predictor/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
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
        "# =========================\n",
        "# MAIN APPLICATION PIPELINE\n",
        "# =========================\n",
        "\n",
        "from src.data_loader import get_stock_data\n",
        "from src.features import add_features\n",
        "from src.model import train_model\n",
        "from src.signals import generate_signal\n",
        "\n",
        "\n",
        "# -------------------------\n",
        "# CONFIG\n",
        "# -------------------------\n",
        "TICKER = input(\"Enter stock ticker (e.g. AAPL, TSLA): \").upper()\n",
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
        "# -------------------------\n",
        "# STEP 1: DATA LOADING\n",
        "# -------------------------\n",
        "print(\"\\n[1/5] Fetching data...\")\n",
        "df = get_stock_data(TICKER)\n",
        "\n",
        "\n",
        "# -------------------------\n",
        "# STEP 2: FEATURE ENGINEERING\n",
        "# -------------------------\n",
        "print(\"[2/5] Engineering features...\")\n",
        "df = add_features(df)\n",
        "\n",
        "\n",
        "# -------------------------\n",
        "# STEP 3: TRAIN MODEL\n",
        "# -------------------------\n",
        "print(\"[3/5] Training model...\")\n",
        "model, accuracy = train_model(df, FEATURES)\n",
        "\n",
        "print(f\"Model Accuracy: {round(accuracy * 100, 2)}%\")\n",
        "\n",
        "\n",
        "# -------------------------\n",
        "# STEP 4: PREDICTION\n",
        "# -------------------------\n",
        "print(\"[4/5] Generating prediction...\")\n",
        "\n",
        "latest = df[FEATURES].iloc[-1].values.reshape(1, -1)\n",
        "\n",
        "prediction = model.predict(latest)[0]\n",
        "confidence = max(model.predict_proba(latest)[0])\n",
        "\n",
        "direction = \"UP\" if prediction == 1 else \"DOWN\"\n",
        "\n",
        "\n",
        "# -------------------------\n",
        "# STEP 5: SIGNAL GENERATION\n",
        "# -------------------------\n",
        "print(\"[5/5] Generating trading signal...\")\n",
        "\n",
        "signal = generate_signal(prediction, confidence)\n",
        "\n",
        "\n",
        "# -------------------------\n",
        "# OUTPUT\n",
        "# -------------------------\n",
        "print(\"\\n=========================\")\n",
        "print(f\"TICKER: {TICKER}\")\n",
        "print(f\"PREDICTION: {direction}\")\n",
        "print(f\"CONFIDENCE: {round(confidence * 100, 2)}%\")\n",
        "print(f\"SIGNAL: {signal}\")\n",
        "print(\"=========================\\n\")"
      ]
    }
  ]
}