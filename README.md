# algorithmic-stock-trader
An AI-based automated stock trading system that uses **LSTM architecture**, **FastAPI**, and **n8n workflow automation** to simulate real-time Nvidia (NVDA) stock trading by providing real-time predictions for Buy/Sell/Hold orders based on dynamic prices pulled in real-time that are carried out in Alpaca. 

---

## Features

-  Real-time Nvidia stock prediction using LSTM architecture 
-  Dynamic model retraining with NVDA prices 
-  FastAPI-powered REST endpoints for prediction and retraining
-  n8n workflow integration for automated trading decisions on Alpaca
-  Trained using historical NVDA data via Alpaca API
-  Scikit-learn MinMaxScaler for consistent scaling
-  Organized project structure and environment control

-  ---

##  How It Works

1. **Data** is fetched from Alpaca (or preloaded) and scaled.
2. **LSTM model** makes a prediction on future price movement.
3. **FastAPI** serves:
   - `/predict` → returns next price + action (buy/sell/hold)
   - `/retrain` → updates model on fresh data
4. **n8n** workflows can hit these endpoints to automate paper-trading logic

---
_
##  API Usage

### `POST /predict`
Returns the model's prediction and a trading signal.

```json
{
  "predicted_price": 132.87,
  "signal": "buy"
}



