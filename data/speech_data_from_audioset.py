#This script copies features labeled 'Speech' from audioset to the repository. 
#Make sure to set the destination correctly before running this script. 
import os

# Speech class 
#have a look at https://github.com/audioset/ontology/blob/master/ontology.json for ontology
speech_class = '/m/09x0r'

# Train speech data from audioset  
# data found here - https://research.google.com/audioset//download.html
train_csv = 'balanced_train_segments.csv'
train_csv_file = open(train_csv)
train_data = train_csv_file.read()
train_examples = [example for example in train_data.split('\n') if speech_class in example]
train_tfrecord_prefixes = set([i[:2] for i in train_examples])
train_tfrecord_filenames = ["bal_train/" + i + ".tfrecord" for i in train_tfrecord_prefixes]
train_destination = '/home/aravind/IDP/VAD/data/audioset_train'

for tfrecord in train_tfrecord_filenames:
	os.system('cp '+ tfrecord + ' ' + train_destination)

# Eval speech data from audioset
eval_csv = 'eval_segments.csv'
eval_csv_file = open(eval_csv)
eval_data = eval_csv_file.read()
eval_examples = [example for example in eval_data.split('\n') if speech_class in example]
eval_tfrecord_prefixes = set([i[:2] for i in eval_examples])
eval_tfrecord_filenames = ["eval/" + i + ".tfrecord" for i in eval_tfrecord_prefixes]
eval_destination = '/home/aravind/IDP/VAD/data/audioset_eval'

for tfrecord in eval_tfrecord_filenames:
	os.system('cp '+ tfrecord + ' ' + eval_destination)
