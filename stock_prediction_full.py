
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping               


folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]

if not csv_files:
    raise FileNotFoundError("No CSV file found in the script folder!")
elif len(csv_files) > 1:
    print("Multiple CSV files found, using the first one:", csv_files[0])

csv_file = os.path.join(folder, csv_files[0])
print("Using CSV file:", csv_file)


data = pd.read_csv(csv_file)

data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

print("Dataset loaded successfully. Shape:", data.shape)
print(data.head())




use_recent_years = True
recent_years = 10  

if use_recent_years:
    start_date = pd.Timestamp.today() - pd.DateOffset(years=recent_years)
    data = data[data.index >= start_date]
    print(f"Using last {recent_years} years for training. Shape:", data.shape)



scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')



es = EarlyStopping(monitor='loss', patience=5)
print("Training LSTM model...")
model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[es])


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
print("Test RMSE:", rmse)

plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(test_predict, label='Predicted Price')
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title("Apple Stock Price Prediction")
plt.legend()
plt.show()

last_60_days = scaled_data[-60:]
X_input = last_60_days.reshape(1, -1)
X_input = X_input.reshape((1, 60, 1))

pred_price = model.predict(X_input)
pred_price = scaler.inverse_transform(pred_price)
print("Next day predicted price:", pred_price[0][0])
