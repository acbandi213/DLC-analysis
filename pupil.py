##calculates mean diameter H & V of pupil for each epoch in each trial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
%matplotlib inline
import os

df = pd.DataFrame(np.load('_ibl_trials.intervals.npy')[:,0], columns = ['Trial Start'])
df['GC trigger'] = np.load('_ibl_trials.goCueTrigger_times.npy') #Go cue trigger time
df['Stim On'] = np.load('_ibl_trials.stimOn_times.npy') #time of stim onset
df['Go Cue'] = np.load('_ibl_trials.goCue_times.npy') #Go cue delivery time
df['Response'] = np.load('_ibl_trials.response_times.npy') #time that response was registered
df['Feedback times'] = np.load('_ibl_trials.feedback_times.npy') #Time of feedback delivery (reward or noise) in seconds relative to session start
df['Trial End'] = np.load('_ibl_trials.intervals.npy')[:,1]

df['Trial Choice'] = np.load('_ibl_trials.choice.npy') # -1 (turn CCW), +1 (turn CW), or 0 (nogo)
df['Trial Contrast L'] = np.load('_ibl_trials.contrastLeft.npy') #contrast of L-stim 0-1, NaN if stim is on other side
df['Trial Contrast R'] = np.load('_ibl_trials.contrastRight.npy') #contrast of R-stim 0-1, NaN if stim is on other side
df['Feedback type'] = np.load('_ibl_trials.feedbackType.npy') #-1 for negative, 1 for positive, 0 for no feedback
df['Opto'] = np.load('_ibl_trials.opto.npy') #Whether trial was Opto (1) or No (0)
df['Opto Prob'] = np.load('_ibl_trials.opto_probability_left.npy') #Probability of Opto

#times converted to approx frame based on frame rate of 150Hz
df['fTS'] = df['Trial Start']*150
df['fSO'] = df['Stim On']*150
df['fRes'] = df['Response']*150
df['fTE'] = df['Trial End']*150

#time length of each epoch
df['tS1'] = df['Stim On'] - df['Trial Start']
df['tS2'] = df['Response'] - df['Stim On']
df['tS3'] = df['Trial End'] - df['Response']

ITI_time = []

for sesh in df[(df.index < 737)].index:

    start = df['Trial End'][sesh]
    stop = df['Trial Start'][sesh+1]

    x = stop - start
    ITI_time.append(x)

ITI_time.append(0)
df['tITI'] = np.array(ITI_time)

extension = 'csv'
result = glob.glob('*.{}'.format(extension))

df2 = pd.read_csv(result[0], skiprows=2)

df2 = df2.iloc[2:len(df2.index), np.r_[0,1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,
                                       43,44,46,47,49,50,52,53,55,56,58,59]]  # subselect dataframe

df2.columns = ['frame','earx','eary','eyetopx','eyetopy','eyerx','eyery','eyebotx','eyeboty','eyelx','eyely','nosetipx','nosetipy','tubetopx','tubetopy','tubebottomx','tubebottomy',
             'chinx','chiny','whiskerx','whiskery','tonguelx','tonguely','tonguerx','tonguery',
             'pinkyrx','pinkyry','ringrx','ringry','middlerx','middlery','pointerrx','pointerry',
             'pinkylx','pinkyly','ringlx','ringly','middlelx','middlely','pointerlx','pointerly']

def euclid(x1,y1,x2,y2):
    return (((((x2-x1)**2 + (y2-y1)**2)**.5)/9.6)/10)

df2['diamH'] = euclid(df2['eyelx'],df2['eyely'],df2['eyerx'],df2['eyery']) # horizontal diameter
df2['diamV'] = euclid(df2['eyetopx'],df2['eyetopy'],df2['eyebotx'],df2['eyeboty']) # vertical diameter

def calc_diam(x,y,z,w):

    Diam_mean_H = []
    Diam_mean_V = []

    for val in df.index:

        start = int((df[x][val]))
        stop = int((df[y][val]))

        global eucH
        global eucV

        for val2 in df2[(df2.index >= start) & (df2.index < stop)].index:

            eucH = np.array(euclid(df2['eyelx'].iloc[val2],df2['eyely'].iloc[val2],
            df2['eyerx'].iloc[val2],df2['eyery'].iloc[val2])).mean()

            eucV = np.array(euclid(df2['eyetopx'].iloc[val2],df2['eyetopy'].iloc[val2],
            df2['eyebotx'].iloc[val2],df2['eyeboty'].iloc[val2])).mean()

        Diam_mean_H.append(eucH)
        Diam_mean_V.append(eucV)

    df[z] = np.array(Diam_mean_H)
    df[w] = np.array(Diam_mean_V)

calc_diam('fTS','fSO','S1H','S1V') #Mean diam from trial start to stim on time
calc_diam('fSO','fRes','S2H','S2V') #Mean diam from stim onset to response registered time
calc_diam('fRes','fTE','S3H','S3V') #Mean diam from Response registered time to trial end

#Mean diam during ITI

Diam_mean_H = []
Diam_mean_V = []

for val in df[(df.index < 737)].index:

    start = int((df['fTE'][val]))
    stop = int((df['fTS'][val+1]))

    global eucH
    global eucV

    for val2 in df2[(df2.index >= start) & (df2.index < stop)].index:

        eucH = np.array(euclid(df2['eyelx'].iloc[val2],df2['eyely'].iloc[val2],
        df2['eyerx'].iloc[val2],df2['eyery'].iloc[val2])).mean()

        eucV = np.array(euclid(df2['eyetopx'].iloc[val2],df2['eyetopy'].iloc[val2],
        df2['eyebotx'].iloc[val2],df2['eyeboty'].iloc[val2])).mean()

    Diam_mean_H.append(eucH)
    Diam_mean_V.append(eucV)

Diam_mean_H.append(0)
df['S-ITI-H'] = np.array(Diam_mean_H)

Diam_mean_V.append(0)
df['S-ITI-V'] = np.array(Diam_mean_V)

filea = result[0].split('.csv')
addpaw = 'pupil'
ext = '.csv'
input = filea[0] + addpaw + ext

df.to_csv(input,index=False)

print('analyzed - pupil')
