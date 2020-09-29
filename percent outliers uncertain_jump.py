## acbandi 9/28 - adpated from DeepLabCut
## This script finds what percent of overall frames are considered outlier frames using the DLC 'uncertain' and 'jump' outlier extraction method 

##IMPORTS
import numpy as np
from numpy import save
import pandas as pd
import glob
import os
import yaml
from ruamel.yaml import YAML

#.h5 filepath
path = r"Y:\DLC\outlier_test\*.h5"

##import config file info 
config = r"Y:\DLC\ACC_DMS_imaging-acb-2020-09-01\config.yaml"
ruamelFile = YAML()
cfg = ruamelFile.load(config)

#bodyparts & relevant config data 
dataname = 'DLC_resnet50_ACC_DMS_imagingSep1shuffle1_300000'
bodyparts = ['left ear', 'right ear', 'Scope 1', 'Scope 2', 'base of tail', 'nosepoke']
p_bound = 0.01
epsilon = 20 #number of pixel distance
ARdegree=3
MAdegree=1
alpha=0.01

def calc_uncertain(fname):
    df = pd.read_hdf(fname) ##import data file 
        
    nframes = len(df)
    startindex = max([int(np.floor(nframes * 0)), 0])
    stopindex = min([int(np.ceil(nframes * 1)), nframes])
    Index = np.arange(stopindex - startindex) + startindex
        
    df = df.iloc[Index]
    mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
    df_temp = df.loc[:, mask] #temp df set up 
    
    p = df_temp.xs("likelihood", level=-1, axis=1)
    ind1 = df_temp.index[(p < p_bound).any(axis=1)].size
    uncertain = ind1/nframes

    return uncertain

def calc_jump(fname):
    df = pd.read_hdf(fname) ##import data file 
        
    nframes = len(df)
    startindex = max([int(np.floor(nframes * 0)), 0])
    stopindex = min([int(np.ceil(nframes * 1)), nframes])
    Index = np.arange(stopindex - startindex) + startindex
        
    df = df.iloc[Index]
    mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
    df_temp = df.loc[:, mask] #temp df set up 
    
    temp_dt = df_temp.diff(axis=0) ** 2
    temp_dt.drop("likelihood", axis=1, level=-1, inplace=True)
    sum_ = temp_dt.sum(axis=1, level=1)
    ind2 = df_temp.index[(sum_ > epsilon ** 2).any(axis=1)].size
    
    jump = ind2/nframes

    return jump

def calculate_percent_outlier_frames(path):
    uncertain_val = []
    jump_val = []
    for fname in glob.glob(path):
        a = calc_uncertain(fname)
        uncertain_val.append(a)
        b = calc_jump(fname)
        jump_val.append(b)
    save('uncertain.npy', uncertain_val)
    save('jump.npy', jump_val)

calculate_percent_outlier_frames(path)

# find and save file names 
list = []
for fname in glob.glob(path):
    list.append(fname[20:28])
save('file_names.npy', list)

# dataframe of data from all outlier extraction methods 
#df = pd.DataFrame(np.load('file_names.npy'), columns = ['video'])
#df['uncertain'] = np.load('uncertain.npy')
#df['jump'] = np.load('jump.npy')
