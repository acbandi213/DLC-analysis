import numpy as np
from numpy import save
import pandas as pd
import glob
import os
import cv2  
import sys
import yaml
import random
from ruamel.yaml import YAML

#mp4 filepath
file_path = r"Y:\DLC\ACC-DMS_nphr_final_skel-acb-2020-10-05"
data_path = file_path + r"\videos\DMS_Opto_videos\test"
save_path = file_path + r"\videos\DMS_Opto_videos\frame"

how_many_frames_to_extract = 10

def Getlistofdata(datatype): #function for making a list of all files in a directory 
    os.chdir(data_path)
    datalist = [
    os.path.join(data_path, fn)
    for fn in os.listdir(os.curdir)
    if os.path.isfile(fn)
    and fn.endswith(datatype)
    and "_labeled." not in fn
    and "_full." not in fn
            ] 
    
    return datalist

movie_list = Getlistofdata('.mp4') #get a list of all mp4 files in directory 

for movie in list(range(len(movie_list))): #for loop for all videos in directory 
    full_name = movie_list[movie]
    a = full_name.split("\\test\\")
    b = a[1].split(".mp4")
    movie_name = b[0]
    vidpath = data_path + "/" + movie_name + ".mp4"
     
    vid = cv2.VideoCapture(vidpath)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) #gets video frame count 
    
    for num in range(how_many_frames_to_extract): #for loop for getting the required number of frames 
        start_frame = random.randint(0, length)
        vid.set(1, start_frame)
        ret, still = vid.read()
    
        os.chdir(save_path)
        cv2.imwrite(movie_name + f'_img{start_frame}.png', still)