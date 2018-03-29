"""  Feature extraction from speech data """
import glob
import numpy as np
import librosa
import pandas

SAMPLE_RATE = 16000
FRAME_DURATION = 1.0 # sec
SAMPLE_WINDOW = int(SAMPLE_RATE*FRAME_DURATION) 
OUTPUT_LENGTH = 13 # length of feature column

filenames = glob.glob('/home/aravind/IDP/VAD/data/train_data/*.flac')
# filenames = glob.glob('/home/aravind/IDP/VAD/data/train_data/1272-141231-0030.flac')

def read_audio(filename):
	y, sr = librosa.core.load(filename, sr=SAMPLE_RATE, mono=True)
	return y, sr

speech_data_seq = []
noise_data_seq = []
labels = np.array([])

for filename in filenames:

	y, sr = read_audio(filename)
	noise = np.random.normal(0, 1, len(y))
	data = y + noise
	intervals = librosa.effects.split(y, top_db=10, frame_length=2000, hop_length=100)
	print(intervals.shape)
	# print(intervals)
	#output length = (seconds) * (sample rate) / (hop_length)
	# n_mfcc per 1000 samples = 10 
	for interval in intervals:
		frame_length = librosa.core.get_duration(data[interval[0]:interval[1]])
		speech_feat13 = librosa.feature.mfcc(
			data[interval[0]:interval[1]],
			sr = SAMPLE_RATE,
			S=None,
			n_mfcc = int((interval[1]-interval[0])/100),
			n_fft = 2048,
			hop_length = int((interval[1]-interval[0]) / (OUTPUT_LENGTH-1))
			)
		if speech_feat13.shape[1]==13:
			speech_data_seq.extend(speech_feat13)
			labels = np.append(labels, np.ones(speech_feat13.shape[0]))

	for i in range(intervals.shape[0]-1):
		frame_length = librosa.core.get_duration(data[intervals[i][1]:intervals[i+1][0]])
		noise_feat13 = librosa.feature.mfcc(
			data[intervals[i][1]:intervals[i+1][0]],
			sr = SAMPLE_RATE,
			n_mfcc = int((intervals[i+1][0]-intervals[i][1]) / 100),
			n_fft = 2048,
			S = None,
			hop_length = int((intervals[i+1][0]-intervals[i][1]) / (OUTPUT_LENGTH-1))
			)
		if noise_feat13.shape[1]==13: 
			noise_data_seq.extend(noise_feat13) 
			labels = np.append(labels, np.zeros(noise_feat13.shape[0]))			

	print('features extracted from' , filename)
	print(len(speech_data_seq))
	print(len(noise_data_seq))



df = pandas.DataFrame(speech_data_seq + noise_data_seq)
df.to_csv("train13.csv", index=False, header=False, sep=',')

label = pandas.DataFrame(labels)
label.to_csv('lables.csv', index=False, header=False, sep=',')


