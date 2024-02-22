import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yfin


#Getting stock data

df = yfin.download('F', start='2017-01-01', end='2023-05-03')

#show data
print(df)

#get columns and rows of data
df.shape

#visualize closing price history
#plt.figure(figsize=(16, 8))
#plt.title('Close Price History')
#plt.plot(df['Close'])
#plt.xlabel('Date', fontsize=18)
#plt.ylabel('Close Price USD ($)', fontsize=18)
#plt.show()

#create a new dataframe with only the close column
data = df.filter(['Close'])
#convert dataframe to a numpy array
dataset = data.values
#get number of rows to train model on
training_data_len = math.ceil( len(dataset) * 0.8)

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#create the scaled training dataset
train_data = scaled_data[0:training_data_len ,:]
#split the data into x_train and y_train datasets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()


#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences= True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer= 'adam', loss= 'mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size= 1, epochs= 1)

#Create the testing dataset
#Create a new array containing scaled values from index 1543
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len: :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60: i, 0])

#convert data to a numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date' , fontsize= 18)
plt.ylabel('Close Price USD ($)' , fontsize= 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc= 'lower right')
plt.show()

#show the real and predicted prices
#Can't figure out how to print, look into it later


#Predicting price for the next day (This is what we want)
#Get stock quote
stock_quote = yfin.download('F', start='2017-01-01', end='2023-05-03')
#Create a new dataframe
new_df = stock_quote.filter(['Close'])
#Get the last 60 day closing prices and convert dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 1 and 0
last_60_days_scaled = scaler.transform(last_60_days)
#Create and empty list
X_test = []
#append the last 60 days
X_test.append(last_60_days_scaled)
#convert the X_test dataset to a numpy array
X_test = np.array(X_test)
#reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print("Predicted price: ")
print(pred_price)
