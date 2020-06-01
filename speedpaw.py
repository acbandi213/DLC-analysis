import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
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
df['fGo'] = df['Go Cue']*150
df['fRes'] = df['Response']*150
df['fFeed'] = df['Feedback times']*150
df['fTE'] = df['Trial End']*150

extension = 'csv'
result = glob.glob('*.{}'.format(extension))

df2 = pd.read_csv(result[0], skiprows=2)

df2 = df2.iloc[2:len(df2.index), np.r_[0,1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,
                                       43,44,46,47,49,50,52,53,55,56,58,59]]  # subselect dataframe

df2.columns = ['frame','earx','eary','eyetopx','eyetopy','eyerx','eyery','eyebotx','eyeboty','eyelx','eyely','nosetipx','nosetipy','tubetopx','tubetopy','tubebottomx','tubebottomy',
             'chinx','chiny','whiskerx','whiskery','tonguelx','tonguely','tonguerx','tonguery',
             'pinkyrx','pinkyry','ringrx','ringry','middlerx','middlery','pointerrx','pointerry',
             'pinkylx','pinkyly','ringlx','ringly','middlelx','middlely','pointerlx','pointerly']

jointx = 'pointerrx'
jointy = 'pointerry'

def euclid(x1,y1,x2,y2):
    return (((((x2-x1)**2 + (y2-y1)**2)**.5)/9.18)/10)

def speed(x1,y1,x2,y2):
    return euclid(x1,y1,x2,y2)/(1/150)

def calc_mean(x,y,z):

    speed_mean = []

    for val in df.index:

        start = int((df[x][val]))
        stop = int((df[y][val]))

        speed_calc = []

        global sp

        for val2 in df2[(df2.index >= start) & (df2.index < stop)].index:

            sp = np.array(speed(df2[jointx].iloc[val2],df2[jointy].iloc[val2],
            df2[jointx].iloc[val2+1],df2[jointy].iloc[val2+1])).mean()

        speed_calc.append(sp)

        speed_mean.append(speed_calc)

    df[z] = np.array(speed_mean)

calc_mean('fTS','fSO','S1') #Mean speed from trial start to stim on time

calc_mean('fSO','fRes','S2') #Mean speed from stim onset to Go Cue

calc_mean('fRes','fTE','S3') #Mean speed from stim onset to Response registered time

#Mean speed of right pointer finger joint during ITI

speed_mean = []

for val in df[(df.index < 737)].index:

    start = int((df['fTE'][val]))
    stop = int((df['fTS'][val+1]))

    speed_calc = []

    for val2 in df2[(df2.index >= start) & (df2.index < stop)].index:

        sp = np.array(speed(df2[jointx].iloc[val2],df2[jointy].iloc[val2],
            df2[jointx].iloc[val2+1],df2[jointy].iloc[val2+1])).mean()

    speed_calc.append(sp)

    speed_mean.append(speed_calc)

speed_mean.append([0])
df['S-ITI'] = np.array(speed_mean)


filea = result[0].split('.csv')
addpaw = 'paw'
ext = '.csv'
input = filea[0] + addpaw + ext

df.to_csv(input,index=False)

print('analyzed')
