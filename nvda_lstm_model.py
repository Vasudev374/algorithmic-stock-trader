
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame

API_KEY = 'PKP2V53YA48D3ER3CJM1'
API_SECRET = ' OTBfdUiFrlhPJLaXxgKz8jv7M7fHNDxxBvMxbSbW'
BASE_URL = 'https://paper-api.alpaca.markets'

api = REST(API_KEY, API_SECRET, base_url=BASE_URL)

# Fetch daily NVDA prices for 2 years
df = api.get_bars("NVDA", TimeFrame.Day, limit=1000).df
df = df[['close']]
df = df.rename(columns={'close': 'Price'})
df = df.reset_index(drop=True)
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Price']])

X = []
y = []
sequence_length = 50  # 50 previous days

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, time steps, features)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)
model.save("nvda_lstm_model.h5")
print("Model saved as nvda_lstm_model.h5")
