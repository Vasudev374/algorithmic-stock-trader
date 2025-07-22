# Nvidia Trader 

An AI-powered trading system for Nvidia stock using FastAPI, LSTM prediction, and n8n automation.

## Features
- `/predict`: Returns price prediction and trading signal
- `/retrain`: Dynamically retrains LSTM model using latest data
- **n8n Integration**: Automates decision-making logic
- FastAPI + TensorFlow + Alpaca + yFinance

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # (Windows)
pip install -r requirements.txt
