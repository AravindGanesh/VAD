# import tensorflow as tf
# import os
# import glob
# import csv
import numpy as np
# from numpy import genfromtxt
import pandas 
# import librosa
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout
# from keras.layers import Embedding
# from keras.layers import LSTM


df = pandas.read_csv('train13.csv', sep=',', header=None)
train_data = np.array(df)
labels = pandas.read_csv('labels13.csv',sep=',', header=None)
labels = np.array(labels)

print(train_data.shape)
print(labels.shape)


model = Sequential()

model.add(Dense(3, input_dim=13))
model.add(Dropout(0.3))
model.add(Dense(3, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='tanh'))



model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(train_data, labels, batch_size=32, epochs=5, shuffle=True, validation_split=0.3, verbose=1)

model.save('vad13.h5')
model.summary()

"""
model.add(Dense(3, input_dim=13, use_bias=True, bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(LSTM(3, activation='tanh', use_bias=True, bias_initializer="zeros" ))
model.add(LSTM(3, activation='tanh', use_bias=True, bias_initializer="zeros" ))
model.add(LSTM(1, activation='tanh', use_bias=True, bias_initializer="zeros" ))
"""