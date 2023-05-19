import numpy as np
import pandas as pd
import pandas_datareader as data
#import yfinance as yf
#import cryptography
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


st.title('Stock Trend Prediction')

user_input = st.text_input('Stock Symbool')
end=st.text_input('TODAY DATE')

start='2010-01-01'

#tickerData = yf.Ticker(user_input)
tickerDf = data.DataReader(user_input,'yahoo',start,end)
tickerDf=tickerDf.reset_index()
tickerDf=tickerDf.drop(['Date'],axis=1)

st.subheader('Data from 2010 to May 2023')
st.write(tickerDf.describe())

#visualising the closing price
st.subheader('Closing price')
fig=plt.figure(figsize=(12,6))
plt.plot(tickerDf.Close)    
st.pyplot(fig)

mavg100=tickerDf.Close.rolling(100).mean()
mavg200=tickerDf.Close.rolling(200).mean()


#visualising the moving averages

st.subheader('Moving average of 100 Days and closing price')
fig=plt.figure(figsize=(12,6))
plt.plot(mavg100)  
plt.plot(tickerDf.Close)
st.pyplot(fig)

st.subheader('Moving average of 100 Days Vs  200 Days')
fig=plt.figure(figsize=(12,6))
plt.plot(mavg100)  
plt.plot(mavg200)
st.pyplot(fig)

st.subheader('Closing Vs Moving average of 100 Days and 200 Days')
fig=plt.figure(figsize=(12,6))
plt.plot(mavg100)
plt.plot(mavg200)    
plt.plot(tickerDf.Close,'r')
st.pyplot(fig)


data_train =pd.DataFrame(tickerDf['Close'][0:int(len(tickerDf)*0.7)])
data_test= pd.DataFrame(tickerDf['Close'][int(len(tickerDf)*0.7):int(len(tickerDf))])

#for lstm model we need to scale the data ,at present data is not supported for futher processing 
scaler=MinMaxScaler(feature_range=(0,1))

#scaling data as per lstm format
data_train_array= scaler.fit_transform(data_train)

#divide train data array into x train and y train
x_train=[]
y_train=[]
for i in range(100,data_train_array.shape[0]):
    x_train.append(data_train_array[i-100: i])
    y_train.append(data_train_array[i, 0])

#converting into numpy arrays
x_train,y_train=np.array(x_train),np.array(y_train)
#commented above trainig part because we have already trained the model

#loading trained model
model=load_model('keras_model.h5')

#testing part
#data_test=data_test.drop(['level_0'],axis=1)

past_100_days=data_train.tail(100)
final_df=pd.concat([past_100_days,data_test])

input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scalerfactor=1/(scaler.scale_[0])

y_predicted=np.array(y_predicted)

y_test=(y_test*scalerfactor)
y_predicted=(y_predicted*scalerfactor)

#final graph

st.subheader('Predictions Vs Orignal')
fig2=plt.figure(figsize=(12,6))
plt.plot((y_test),'b',label='Original Price')
plt.plot((y_predicted),'r',label='predicted Price')
plt.legend()
st.pyplot(fig2)

