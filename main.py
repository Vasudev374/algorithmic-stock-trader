from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import subprocess
import numpy as np
import sys 
import pickle 
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import os 
from tensorflow.keras.models import load_model
import joblib

app = FastAPI()

# Set your Alpaca keys
API_KEY = os.getenv("APCA_API_KEY_ID", "PKP2V53YA48D3ER3CJM1")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "OTBfdUiFrlhPJLaXxgKz8jv7M7fHNDxxBvMxbSbW")
BASE_URL = "https://paper-api.alpaca.markets"

# Alpaca API setup
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# Load model and scaler once at startup
try:
    model = load_model("nvda_lstm_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print("Error loading model or scaler at startup:", e)
    model = None
    scaler = None

# Get latest 50 NVDA prices
def get_latest_nvda_prices(limit=50):
    end = datetime.utcnow()
    start = end - timedelta(days=100)  # buffer to ensure enough trading days
    bars = api.get_bars("NVDA", "1D",
                      start=start.strftime("%Y-%m-%dT%H:%M:%SZ"), 
                      end=end.strftime("%Y-%m-%dT%H:%M:%SZ"), feed= 'iex').df

    if len(bars) < limit:
        raise ValueError(f"Only got {len(bars)} bars, need {limit}.")
    
    bars = bars.tail(limit)
    return bars["close"].tolist()

# Prediction endpoint (no input required)
@app.post ("/predict")
def predict():
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail= "Model or scaler not loaded. Please retrain or check files")
        # Get latest 50 prices
        prices = get_latest_nvda_prices()

        if not prices or len(prices) < 50:
            raise ValueError(f"Not enough price data to make prediction. Got {len(prices)} prices.")

        # Scale and reshape input
        scaled_prices = scaler.transform(np.array(prices).reshape(-1, 1))
        X = np.array(scaled_prices).reshape(1, 50, 1)

        # Predict
        predicted_scaled = model.predict(X)[0][0]
        predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

        # Decision logic
        last_price = prices[-1]
        threshold = 0.01 

        if predicted_price > last_price * (1+threshold):
            signal = "buy"
        elif predicted_price < last_price * (1- threshold):
            signal = "sell"
        else:
            signal  = "hold"

        print(f"Last: {last_price:.2f}, Predicted: {predicted_price:.2f}, Signal: {signal}")
        
        return {
            "last_price": round(last_price, 2),
            "predicted_price": round(predicted_price, 2),
            "signal": signal
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
       
    
# Retrain endpoint
@app.post("/retrain")
def retrain():
    try:
        result = subprocess.run([sys.executable, "retrain_model.py"],
        check=True,
        capture_output=True,
        text=True
        
    )

        return {
            "status": "Retraining completed successfully",
            "output": result.stdout # this will now show logs
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr or str(e))





    
    
    
    
        

    

 