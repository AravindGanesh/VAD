"""  Feature extraction from speech data """

import glob
import numpy as np
import librosa
import pandas

SAMPLE_RATE = 16000
FRAME_DURATION = 1.0 # sec
SAMPLE_WINDOW = int(SAMPLE_RATE * FRAME_DURATION) 
OUTPUT_LENGTH = 13 # length of feature column

filenames = glob.glob('/home/aravind/IDP/VAD/data/train/*.flac')
# filenames = glob.glob('/home/aravind/IDP/VAD/data/train/652-130726-0017.flac')

def read_audio(_filename):
	_y, _sr = librosa.core.load(_filename, sr=SAMPLE_RATE, mono=True)
	return _y, _sr

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



data_seq = []
# noise_data_seq = []
labels = np.array([])
count = 0


for filename in filenames:

	# if count == 2000: break;
	y, sr = read_audio(filename)
	intervals = librosa.effects.split(y, top_db=25, frame_length=2000, hop_length=100)
	y = normalize(y)
	noise = np.random.normal(0, 0.75, len(y))
	data = y + noise
	# print(intervals.shape)
	# print(intervals)
	#output length = (seconds) * (sample rate) / (hop_length)
	# n_mfcc per 1000 samples = 10 
	for interval in intervals:
		frame_length = librosa.core.get_duration(data[interval[0]:interval[1]])
		speech_feat13 = librosa.feature.mfcc(
			data[interval[0]:interval[1]],
			sr = SAMPLE_RATE,
			S=None,
			n_mfcc = int((interval[1]-interval[0])/100.0),
			n_fft = 2048,
			hop_length = int((interval[1]-interval[0]) / (OUTPUT_LENGTH-1))
			)
		speech_feat13 = normalize_feature(speech_feat13)
		# print(speech_feat13.shape[1])
		if speech_feat13.shape[1]==OUTPUT_LENGTH:
			data_seq.extend(speech_feat13)
			labels = np.append(labels, np.ones(speech_feat13.shape[0]))

	for i in range(intervals.shape[0]-1):
		frame_length = librosa.core.get_duration(data[intervals[i][1]:intervals[i+1][0]])
		noise_feat13 = librosa.feature.mfcc(
			data[intervals[i][1]:intervals[i+1][0]],
			sr = SAMPLE_RATE,
			n_mfcc = int((intervals[i+1][0]-intervals[i][1]) / 100.0),
			n_fft = 2048,
			S = None,
			hop_length = int((intervals[i+1][0]-intervals[i][1]) / (OUTPUT_LENGTH-1))
			)
		noise_feat13 = normalize_feature(noise_feat13)
		# print(noise_feat13.shape[1])
		if noise_feat13.shape[1]==OUTPUT_LENGTH: #13 is the output length
			data_seq.extend(noise_feat13) 
			labels = np.append(labels, np.zeros(noise_feat13.shape[0]))			

	print('features extracted from' , filename)
	print('count = ', count)
	count+=1


print(len(data_seq))
# print(len(noise_data_seq))


df = pandas.DataFrame(data_seq)
df.to_csv("train13.csv", index=False, header=False, sep=',')

label = pandas.DataFrame(labels)
label.to_csv('labels13.csv', index=False, header=False, sep=',')
