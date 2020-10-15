import numpy as np
from numpy import save
import pandas as pd
import glob
import os
import cv2  
import sys
import yaml
from ruamel.yaml import YAML

#filepaths
file_path = "/tigress/acbandi/DLC/ACC-DMS_nphr_final_skel-acb-2020-10-05
data_path = file_path + "/videos"
save_path = file_path + "/labeled-data/Outliers"

##import config file info 
config = file_path + "/config.yaml"
ruamelFile = YAML()
cfg = ruamelFile.load(config)

#bodyparts & relevant config data 
bodyparts = ['Implant', 'Body', 'Left Ear', 'Right Ear', 'Base of Tail', 'Nosepoke'] 
epsilon = 40 #number of pixel distance
how_many_frames_to_extract = 100

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

all_outlier = []
for i in data_list:
    a = i.split("/videos/")
    b = a[1].split("DLC")
    c = b[0] + ".h5"
    outlier_df = pd.DataFrame(calc_jump(i), columns = ['frame'])
    outlier_df['movie'] = b[0]
    all_outlier.append(outlier_df)
all_outlier = pd.concat(all_outlier)

rand_outliers = pd.DataFrame(all_outlier.sample(n=how_many_frames_to_extract))
os.chdir(file_path)
list_tracked_frames = np.load("list_track.npy") #list of frames used to train the network
rand_label = rand_outliers['movie'].values
rand_count = list(range(0,len(rand_label)))

rand_arr = []
for x in rand_count:
    arr = rand_label[x] + "/" + str(rand_outliers['frame'].iloc[x])
    rand_arr.append(arr)
    
while len(rand_arr) == how_many_frames_to_extract:

    if len(np.intersect1d(list_tracked_frames, rand_arr)) == 0:
        num_list = list(range(how_many_frames_to_extract))
        for num in num_list:
            movie_name = rand_outliers['movie'].iloc[num]
            vidpath = data_path + "/" + movie_name + ".mp4"
            vid = cv2.VideoCapture(vidpath)
            trackpath = data_path + "/" + movie_name + "DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000" + ".h5"
            h5_df = pd.read_hdf(trackpath)
            
            start_frame = rand_outliers['frame'].iloc[num]
            
            vid.set(1, start_frame)
            ret, still = vid.read()

            body_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Implant']['x'].iloc[start_frame])
            body_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Implant']['y'].iloc[start_frame])
            left_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Body']['x'].iloc[start_frame])
            left_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Body']['y'].iloc[start_frame])
            right_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Left Ear']['x'].iloc[start_frame])
            right_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Left Ear']['y'].iloc[start_frame])
            scope1_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Right Ear']['x'].iloc[start_frame])
            scope1_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Right Ear']['y'].iloc[start_frame])
            base_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Base of Tail']['x'].iloc[start_frame])
            base_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Base of Tail']['y'].iloc[start_frame])
            nose_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Nosepoke']['x'].iloc[start_frame])
            nose_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Nosepoke']['y'].iloc[start_frame])
                
            image = cv2.circle(still, (body_x,body_y), radius=5, color=(255, 0, 0), thickness = 2)
            image = cv2.circle(still, (left_x,left_y), radius=5, color=(255, 0, 0), thickness = 2)
            image = cv2.circle(still, (right_x,right_y), radius=5, color=(255, 0, 0), thickness = 2)
            image = cv2.circle(still, (scope1_x,scope1_y), radius=5, color=(255, 0, 0), thickness = 2)
            image = cv2.circle(still, (scope2_x,scope2_y), radius=5, color=(255, 0, 0), thickness = 2)
            image = cv2.circle(still, (base_x,base_y), radius=5, color=(255, 0, 0), thickness = 2)
            image = cv2.circle(still, (nose_x,nose_y), radius=5, color=(255, 0, 0), thickness = 2)
                
            os.chdir(save_path)
            cv2.imwrite(movie_name + f'_img{start_frame}.png', still)
        break

    elif len(np.intersect1d(list_tracked_frames, rand_arr)) != 0:
        print('tracked frame is in the random outlier extract!')
