import numpy as np
import pandas as pd
import os 
import requests
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

API_KEY = 'PKP2V53YA48D3ER3CJM1'
SECRET_KEY = 'OTBfdUiFrlhPJLaXxgKz8jv7M7fHNDxxBvMxbSbW'
BASE_URL = 'https://data.alpaca.markets/v2'

def get_nvda_data(days=100):
    # Use timezone-aware UTC datetime
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    url= f"{BASE_URL}/stocks/NVDA/bars"
    params = {
        "start": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timeframe": "1Day",
        "limit": days
    
    }
    headers = {
        "APCA-API-KEY-ID": 'PKP2V53YA48D3ER3CJM1',
        "APCA-API-SECRET-KEY": 'OTBfdUiFrlhPJLaXxgKz8jv7M7fHNDxxBvMxbSbW'
    }
     
    print(f"Fetching data from {params['start']} to {params['end']}")   
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    bars = response.json().get("bars", [])

    if not bars:
        raise ValueError ("No bars returned. Check your API key, symbol, or time window.")

    close_prices = [bar["c"] for bar in bars]
    print(f"Fetched{len(close_prices)} prices.")
    return np.array(close_prices).reshape(-1, 1)

def preprocess_data(prices):
     scaler = MinMaxScaler()
     scaled = scaler.fit_transform(prices)

     X, y = [], []
     for i in range(50, len(scaled)):
         X.append(scaled[i-50:i])
         y.append(scaled[i])

     X = np.array(X)
     y = np.array(y)
     print(f"Preprocessed data: X.shape = {X.shape}, y.shape = {y.shape}")
     return X, y, scaler
     
def train_model(X, y):
    print("Training model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    return model
     
def save_model(model, scaler):
    model_path = "nvda_lstm_model.h5"
    try:
        model.save(model_path) # Use the variable, NOT the string literal
        print("Saved retrained model to:", os.path.abspath(model_path))
    except Exception as e:
        print("Error saving model:", e)
            
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
print("Scaler saved to scaler.pkl")
    
def main():
    print("Starting retraining process...")
    prices = get_nvda_data(200)
    X, y, scaler = preprocess_data(prices)
    model = train_model(X, y)
    save_model(model, scaler)
    print("Retraining complete and saved.")
    print(f"Model retrained at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")

if __name__ == "__main__":
    main()

     
     
    

       


    
              
