import numpy as np
from numpy import save
import pandas as pd
import glob
import os
import cv2  
import sys
import yaml
from ruamel.yaml import YAML

#.h5 filepath
file_path = "/tigress/acbandi/DLC/ACC_DMS_imaging_skel-acb-2020-10-01"
data_path = file_path + "/videos"
save_path = file_path + "/labeled-data/Outliers"

##import config file info 
config = file_path + "/config.yaml"
ruamelFile = YAML()
cfg = ruamelFile.load(config)

#bodyparts & relevant config data 
bodyparts = ['body', 'left ear', 'right ear', 'Scope 1', 'Scope 2', 'base of tail', 'nosepoke']
p_bound = 0.01
epsilon = 30 #number of pixel distance
how_many_frames_to_extract = 5

def calc_jump(h5_file):
    df = pd.read_hdf(h5_file) ##import data file 
        
    nframes = len(df)
    startindex = max([int(np.floor(nframes * 0)), 0])
    stopindex = min([int(np.ceil(nframes * 1)), nframes])
    Index = np.arange(stopindex - startindex) + startindex
        
    df = df.iloc[Index]
    mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
    df_temp = df.loc[:, mask] #temp df set up 
    
    temp_dt = df_temp.diff(axis=0) ** 2
    temp_dt.drop("likelihood", axis=1, level=-1, inplace=True)
    temp_dt.drop("nosepoke", axis=1, level=1, inplace=True)
    sum_ = temp_dt.sum(axis=1, level=1)
    sum_df = pd.DataFrame(df_temp.index[(sum_ > epsilon ** 2).any(axis=1)])
    ind2 = sum_df.values
    
    return ind2

def Getlistofdata(datatype):
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

data_list = Getlistofdata('.h5')

for i in data_list:
    outlier_df = pd.DataFrame(calc_jump(i), columns = ['frame'])
    jump_outliers = pd.DataFrame(outlier_df['frame'].sample(n=how_many_frames_to_extract))
    num_list = list(range(how_many_frames_to_extract))
    
    a = i.split("/videos/")
    b = a[1].split("DLC")
    c = b[0] + ".h5"
    
    vidpath = data_path + "/" + b[0] + ".mp4"
    vid = cv2.VideoCapture(vidpath)
    for num in num_list:
        start_frame = jump_outliers['frame'].iloc[num]
        end_frame = start_frame + 1 
        for ran in range(start_frame, end_frame):
            vid.set(1, ran)
            ret, still = vid.read()
            os.chdir(save_path)
            cv2.imwrite(b[0] + f'_img{start_frame}.png', still)
