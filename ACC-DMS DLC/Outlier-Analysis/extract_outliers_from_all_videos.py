#acbandi 10/15/20 
#This script is what was used to identify and extract outliers that went into re-training the ACC-DMS DLC networks 

#Imports 
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
file_path = "/tigress/acbandi/DLC/ACC-DMS_nphr_final_skel-acb-2020-10-05" #must be in this format to run on tiger cluster. can be replaces with sys.argv[1]
data_path = file_path + "/videos" #location of h5 files 
save_path = file_path + "/labeled-data/Outliers" #location to save the extracted frames- in this case I made a folder called 'Outliers' in the dlc labeled-data folder 

##import config file info 
config = file_path + "/config.yaml" #location of config file - make sure the project path is correct in config file 
ruamelFile = YAML()
cfg = ruamelFile.load(config) #loads the config file 

#bodyparts & relevant config data 
bodyparts = ['Implant', 'Body', 'Left Ear', 'Right Ear', 'Base of Tail', 'Nosepoke'] #list the bodyparts in order from the config file 
epsilon = 40 #number of pixel distance in joint position 
how_many_frames_to_extract = 400 #how many outlier frames to extract - 400 used for both Imaging and Opto networks 

#this function loads the h5 data file for each tracked video, calculates the euclidean distance 
#between all joint's position from frame to frame and finds the proportion of frames that exceed the 'jump' (epsilon) threshold  

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
    temp_dt.drop("likelihood", axis=1, level=-1, inplace=True) #removes liklihood column 
    temp_dt.drop("nosepoke", axis=1, level=1, inplace=True) #removes nosepoke positional data - fixed point that moves when mouse enters the nosepoke area (prone to jumps) 
    sum_ = temp_dt.sum(axis=1, level=1)
    sum_df = pd.DataFrame(df_temp.index[(sum_ > epsilon ** 2).any(axis=1)])
    ind2 = sum_df.values
    
    return ind2

#this function creates a list of all h5 files in the folder 
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

#this takes each file name that was stored in data_list splits the strings to make only the video title visible 
#and then runs the calc_jump funtion on all files stored in data_list 

all_outlier = [] #datframe with all outlier frames from all videos (quite large) 
for i in data_list: 
    a = i.split("/videos/")
    b = a[1].split("DLC")
    c = b[0] + ".h5"
    outlier_df = pd.DataFrame(calc_jump(i), columns = ['frame'])
    outlier_df['movie'] = b[0]
    all_outlier.append(outlier_df)
all_outlier = pd.concat(all_outlier)

rand_outliers = pd.DataFrame(all_outlier.sample(n=how_many_frames_to_extract)) #randomly selects the number of frames you want from a list of all outlier frames 
os.chdir(file_path)
list_tracked_frames = np.load("list_track.npy") #list of frames used to train the original network 
rand_label = rand_outliers['movie'].values
rand_count = list(range(0,len(rand_label)))

rand_arr = []
for x in rand_count:
    arr = rand_label[x] + "/" + str(rand_outliers['frame'].iloc[x])
    rand_arr.append(arr)
    
while len(rand_arr) == how_many_frames_to_extract: #this loop only comletes when there is no frame that was used to train the network present in the new outlier frame selection

    if len(np.intersect1d(list_tracked_frames, rand_arr)) == 0:
        num_list = list(range(how_many_frames_to_extract)) #goes thru each of the randomly selected outlier frames and extracts that frame as a .png
        for num in num_list:
            movie_name = rand_outliers['movie'].iloc[num]
            vidpath = data_path + "/" + movie_name + ".mp4"
            vid = cv2.VideoCapture(vidpath) #opens the video  
            trackpath = data_path + "/" + movie_name + "DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000" + ".h5" #opens corresponding h5 data file 
            h5_df = pd.read_hdf(trackpath)
            
            start_frame = rand_outliers['frame'].iloc[num]
            
            vid.set(1, start_frame) #opens the video from the start to the selected frame 
            ret, still = vid.read() #extracts the selected frame 

            ## uncomment below if you want to plot the dlc tracked joints on the extracted frame to see how the tracking looks!
            #body_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Implant']['x'].iloc[start_frame])
            #body_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Implant']['y'].iloc[start_frame])
            #left_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Body']['x'].iloc[start_frame])
            #left_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Body']['y'].iloc[start_frame])
            #right_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Left Ear']['x'].iloc[start_frame])
            #right_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Left Ear']['y'].iloc[start_frame])
            #scope1_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Right Ear']['x'].iloc[start_frame])
            #scope1_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Right Ear']['y'].iloc[start_frame])
            #base_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Base of Tail']['x'].iloc[start_frame])
            #base_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Base of Tail']['y'].iloc[start_frame])
            #nose_x = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Nosepoke']['x'].iloc[start_frame])
            #nose_y = int(h5_df['DLC_resnet50_ACC-DMS nphr final skelOct05shuffle1_450000']['Nosepoke']['y'].iloc[start_frame])
                
            #image = cv2.circle(still, (body_x,body_y), radius=5, color=(255, 0, 0), thickness = 2)
            #image = cv2.circle(still, (left_x,left_y), radius=5, color=(255, 0, 0), thickness = 2)
            #image = cv2.circle(still, (right_x,right_y), radius=5, color=(255, 0, 0), thickness = 2)
            #image = cv2.circle(still, (scope1_x,scope1_y), radius=5, color=(255, 0, 0), thickness = 2)
            #image = cv2.circle(still, (scope2_x,scope2_y), radius=5, color=(255, 0, 0), thickness = 2)
            #image = cv2.circle(still, (base_x,base_y), radius=5, color=(255, 0, 0), thickness = 2)
            #image = cv2.circle(still, (nose_x,nose_y), radius=5, color=(255, 0, 0), thickness = 2)
                
            os.chdir(save_path)
            cv2.imwrite(movie_name + f'_img{start_frame}.png', still) #saves the extracted outlier frame 
        break

    elif len(np.intersect1d(list_tracked_frames, rand_arr)) != 0:
        print('tracked frame is in the random outlier extract!')
