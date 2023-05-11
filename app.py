import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model

beginning = '2015-01-01'
ending = '2022-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# using AAPL to start/train model, then generalizing
# it later
data = yf.download(user_input, start=beginning, end=ending)

dataframe = pd.DataFrame(data)

dataframe.head()

# explain data

st.subheader('Data From 2015-2022')
st.write(dataframe.describe())

#visualizations
st.subheader('Closing Price vs. Time')
fig = plt.figure(figsize = (12,6))
plt.plot(dataframe.Close)
st.pyplot(fig)

st.subheader('Closing Price vs. Time — 100 Moving Average')
movav100 = dataframe.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(movav100, 'r')
plt.plot(dataframe.Close)
st.pyplot(fig)

st.subheader('Closing Price vs. Time — 100/200 Moving Average')
movav100 = dataframe.Close.rolling(100).mean()
movav200 = dataframe.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(movav100, 'r')
plt.plot(movav200, 'g')
plt.plot(dataframe.Close, 'b')
st.pyplot(fig)

df = dataframe

# split data to train/test

train_split = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test_split = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

train_array = scaler.fit_transform(train_split)


# load model

model = load_model('keras_model.h5')

# testing


last100 = train_split.tail(100)
df_updated = pd.concat([last100, test_split], ignore_index=True)
input = scaler.fit_transform(df_updated)

xtest = []
ytest = []

for i in range (100, input.shape[0]):
    xtest.append(input[i-100:i])
    ytest.append(input[i,0])

xtest, ytest = np.array(xtest), np.array(ytest)
y_predict = model.predict(xtest)

scalefactor = 1/scaler.scale_
y_predict = y_predict * scalefactor
ytest = ytest*scalefactor


# final graph

st.subheader('Predictions vs. Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(ytest,'b',label = 'original price')
plt.plot(y_predict,'r', label = 'predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()
st.pyplot(fig2)


