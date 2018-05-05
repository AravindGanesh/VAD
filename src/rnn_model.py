import numpy as np
import os
import pandas
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense 
from keras.layers import Input
from keras.layers import SimpleRNN

# from keras.layers import Dropout
# from keras.layers import Embedding
# from keras.layers import LSTM
# from keras.layers import RNN
# from keras.utilis import np_utilis 


df = pandas.read_csv('train13.csv', sep=',', header=None)
train_data = np.array(df)
labels = pandas.read_csv('labels13.csv',sep=',', header=None)
labels = np.array(labels)


# data = keras.preprocessing.sequence.TimeseriesGenerator(train_data, labels, length=10, sampling_rate=1, batch_size=32)
train_data = train_data[0:int(train_data.shape[0]/10)*10]
labels = labels[0:int(labels.shape[0]/10)*10]
 
data = np.reshape(train_data, ((int(train_data.shape[0]/10)), 10,train_data.shape[1]))

targets = np.reshape(labels, ((int(labels.shape[0]/10)), 10,labels.shape[1]))




inputs = Input(shape=(10,13))


# x = Embedding(output_dim=3, input_dim=10000, input_length=10)(layer_in)

layer_H0 = SimpleRNN(
	3,
	activation=None,
	use_bias=True,
	bias_initializer='zeros',
	return_sequences=True,
	return_state=False,
	go_backwards=False
	)

layer_H1 = SimpleRNN(
	3,
	activation=None,
	use_bias=True, 
	bias_initializer='zeros', 
	return_sequences=True, 
	return_state=False, 
	go_backwards=False
	)


layer_out = SimpleRNN(
	1,
	activation='tanh', 
	use_bias=True, 
	bias_initializer='zeros', 
	return_sequences=True, 
	return_state=True, 
	go_backwards=False
	)


# layer_out = Dense(
# 	1, 
# 	activation='tanh', 
# 	use_bias=True, 
# 	bias_initializer='zeros'
# 	)

x = layer_H0(inputs)
x = layer_H1(x)
predictions = layer_out(x) 

model = Model(inputs=inputs, outputs=predictions)
# model.add(LSTM(units=3, input_shape=(40, 13), activation=None, bias_initializer='zeros', return_sequences=True))
# model.add(LSTM(1, return_sequences=True))

model.compile(
	loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'],
	sample_weight_mode='temporal',

	)

model.fit(
	x=data,
	y=labels, 
	batch_size=None, 
	epochs=5, 
	shuffle=False, 
	validation_split=0.3, 
	verbose=1)



model.save('rnn_model.h5')

model.summary()


# keras.preprocessing.sequence.TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)








# https://stackoverflow.com/questions/48140989/keras-lstm-input-dimension-setting




''' 
keras.layers.RNN(
	cell, 
	return_sequences=False, 
	return_state=False, 
	go_backwards=False, 
	stateful=False, 
	unroll=False
	)


keras.layers.SimpleRNNCell(
	units,
	activation='tanh', 
	use_bias=True, 
	kernel_initializer='glorot_uniform', 
	recurrent_initializer='orthogonal', 
	bias_initializer='zeros', 
	kernel_regularizer=None, 
	recurrent_regularizer=None, 
	bias_regularizer=None, 
	kernel_constraint=None, 
	recurrent_constraint=None, 
	bias_constraint=None, 
	dropout=0.0, 
	recurrent_dropout=0.0
	)

'''
'''
keras.layers.LSTM(
	units, 
	activation='tanh', 
	recurrent_activation='hard_sigmoid', 
	use_bias=True, 
	kernel_initializer='glorot_uniform', 
	recurrent_initializer='orthogonal', 
	bias_initializer='zeros', 
	unit_forget_bias=True, 
	kernel_regularizer=None, 
	recurrent_regularizer=None, 
	bias_regularizer=None, 
	activity_regularizer=None, 
	kernel_constraint=None, 
	recurrent_constraint=None, 
	bias_constraint=None, 
	dropout=0.0, 
	recurrent_dropout=0.0, 
	implementation=1, 
	return_sequences=False, 
	return_state=False, 
	go_backwards=False, 
	stateful=False, 
	unroll=False
	)

'''
'''
class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]
'''