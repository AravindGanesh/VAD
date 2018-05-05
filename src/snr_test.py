import numpy as np
import random
import glob
import librosa
import matplotlib.pyplot as plt


def normalize(_y):
	_mean = np.mean(_y,axis=0)
	_std = np.std(_y, axis=0)
	x = (_y - _mean)/_std
	return x

filenames = glob.glob('/home/aravind/IDP/VAD/data/train/*.flac')
audio_file = random.choice(filenames)
data, sr = librosa.core.load(audio_file, sr=16000, mono=True)

intervals = librosa.effects.split(data, top_db=25, frame_length=2000, hop_length=100)
data = normalize(data)

data_amp = np.linalg.norm(data) # l2 norm
amp_db = 20*np.log10(data_amp)


speech = np.asarray([])

for interval in intervals:
	speech = np.append(speech, data[interval[0]:interval[1]])
# speech_th = 20.0


speech_amp = np.linalg.norm(speech)
speech_db = 20*np.log10(speech_amp)
top = 10.0**(20.0/20.0)

snr = speech_amp/top
SNR = data_amp/top
N0 = 0.5/snr
# noise_sd = np.sqrt(N0)
noise_sd = np.sqrt(0.5)/speech_amp

noise = np.random.normal(0.0, 0.75, len(data))
noise_amp = np.linalg.norm(noise)
noise_db = 20*np.log10(noise_amp)

print audio_file
print 'mean = ', np.mean(data)
print 'sd = ',   np.std(data)
print'data amp = ', data_amp
print 'data db =  ',  amp_db
print 'speech amp = ', speech_amp
print 'speech db = ' , speech_db
print 'noise amp = ', noise_amp
print 'noise db = ', noise_db
print 'SNR = ', snr
print 'Global SNR = ', SNR

plt.figure(1)
plt.plot(data)
plt.figure(2)
plt.plot(speech)
plt.figure(3)
plt.plot(data)
plt.plot(noise)
# plt.figure(4)
# plt.plot(data+noise)
plt.show()