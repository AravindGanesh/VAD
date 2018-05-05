# Data from LibreSpeech

import os
import shutil
import random



root_dir = '/home/aravind/IDP/VAD/data/LibriSpeech'
output_dir = '/home/aravind/IDP/VAD/data/train'

os.system('sudo rm -rf  ' + output_dir)
os.system('mkdir ' + output_dir)

for root, dirs, files in os.walk(root_dir):
    number_of_files = len(os.listdir(root))
    for the_file in os.listdir(root):
        file_in_track = root
        file_to_copy = file_in_track + '/' + the_file
        if os.path.isfile(file_to_copy) == True:
            shutil.copy(file_to_copy,output_dir)


os.system(' find ' + output_dir + ' -name \*.txt -type f -delete')
os.system(' find ' + output_dir + ' -name \*.TXT -type f -delete')
os.system('sudo rm -rf ' + root_dir)



print 'Finished !' 

