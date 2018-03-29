# Data from librespeech 

import os
import shutil
import random

root_dir = '/home/aravind/IDP/datasets/TrainClean/'
output_dir = '/home/aravind/IDP/VAD/data/train_data/'

ref = 5

for root, dirs, files in os.walk(root_dir):
    number_of_files = len(os.listdir(root)) 
    if number_of_files > ref:
        ref_copy = int(round(0.2 * number_of_files))
        for i in xrange(ref_copy):
            chosen_one = random.choice(os.listdir(root))
            file_in_track = root
            file_to_copy = file_in_track + '/' + chosen_one
            if os.path.isfile(file_to_copy) == True:
                shutil.copy(file_to_copy,output_dir)
                #print file_to_copy
    else:
        for i in xrange(len(files)):
            track_list = root
            file_in_track = files[i]
            file_to_copy = track_list + '/' + file_in_track
            if os.path.isfile(file_to_copy) == True:
                shutil.copy(file_to_copy,output_dir)
                # print file_to_copy

os.system(' find . -name \*.TXT -type f -delete')
os.system(' find . -name \*.txt -type f -delete')

print 'Finished !' 

