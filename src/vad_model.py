# import tensorflow as tf
# import os
# import glob
# import csv
import numpy as np
import librosa
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM


# filenames = glob.glob('/home/aravind/IDP/VAD/data/audioset_train/*.tfrecord')
train_data = np.genfromtxt('train13.csv')
labels = np.genfromtxt('lables.csv')

model = Sequential()

model.add(Dense(3, input_dim=13, use_bias=True, bias_initializer='zeros'))

model.Dropout(0.2)

model.add(LSTM(3, activation='tanh', use_bias=True, bias_initializer="zeros" ))

model.add(LSTM(3, activation='tanh', use_bias=True, bias_initializer="zeros" ))

model.add(LSTM(1, activation='tanh', use_bias=True, bias_initializer="zeros" ))




model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

model.fit(train_data, labels, batch_size=32, epochs=10, shuffle=True, validation_split=0.25, verbose=2)

model.save('my_vad.h5')
