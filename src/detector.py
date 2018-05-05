# import os
# import glob
import librosa
import numpy as np
import pandas
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model


FEATURE_LENGTH = 13

def normalize(_y):
	_mean = np.mean(_y,axis=0)
	_std = np.std(_y, axis=0)
	x = (_y - _mean)/_std
	return x


def normalize_feature(_feature):

	for i in range(0,len(_feature)):
		# _mean = np.mean(_feature[i] , axis=0)
		# _std = np.std(_feature[i] , axis=0)
		_feature[i] = normalize(_feature[i])
	return _feature


model = load_model('vad13.h5')

filename = '/home/aravind/IDP/VAD/src/try.wav'

y, sr = librosa.core.load(filename, sr=16000, mono=True)
y = normalize(y)

feature13 = librosa.feature.mfcc(y, sr=sr, S=None, n_mfcc=int(len(y)/100), hop_length=int(len(y)/(FEATURE_LENGTH-1)))

feature13 = normalize_feature(feature13)

print(feature13.shape)

results = model.predict(x=feature13, batch_size=20, verbose=1)

print(np.mean(results))
print(np.std(results))

for result in results: print(result)
# for result in results: print(result)

plt.plot(y)
plt.show()