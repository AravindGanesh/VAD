""" Code to remove silence intervals in an audio file """
# TODO: Use suitable  top_db for FAR-FIELD voice activity detection

import numpy as np
import librosa


filename = 'samples/brian.wav'

data, rate = librosa.load(filename)

intervals = librosa.effects.split(data, top_db=20, frame_length=2048, hop_length=2)

no_silence = []

for interval in intervals:
	no_silence.append(data[interval[0]:interval[1]])

no_silence = np.asarray(no_silence)
print no_silence[0]

blah = np.concatenate(no_silence)

print intervals.shape
print no_silence.shape
print blah.shape

librosa.output.write_wav("bark.wav", blah, rate)


# y => audio  signal - ndarray
# top_db => The threshold (in decibels) below reference to consider as silence
# ref : number or callable,  The reference power. By default, it uses np.max and compares to the peak power in the signal.
# frame_length : int > 0, The number of samples per analysis frame
# hop_length : int > 0,  The number of samples between analysis frames
# Returns: intervals : np.ndarray, shape=(m, 2) intervals[i] == (start_i, end_i) are the start and end time (in samples) of non-silent interval i.

