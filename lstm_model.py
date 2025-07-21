import yfinance as yf
import numpy as np  
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.callbacks import EarlyStopping 
import matplotlib.pyplot as plt

ticker = "NVDA"
data = yf.download(ticker, start="2018-01-01", end= "2024-12-31")
data = data[['Close']] 

scaler = MinMaxScaler()
scaled_data =  scaler.fit_transform(data)

def create_sequences(data, sequence_length=60):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

x, y = create_sequences(scaled_data)

x = x.reshape((x.shape[0], x.shape[1], 1))

train_size = int(len(x)* 0.8)
x_train, x_val = x[:train_size], x[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
from tensorflow.keras.losses import MeanSquaredError
model.compile(optimizer='adam', loss=MeanSquaredError())
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=32, callbacks=[early_stop])

model.save("model.h5")

import joblib
joblib.dump(scaler, "scaler.pkl")

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("LSTM Training Loss")
plt.show()
print("LSTM model script ran successfully!")