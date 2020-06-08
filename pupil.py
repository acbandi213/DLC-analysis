##calculates mean speed of joint for each epoch in each trial
def dlc(path):
    import numpy as np
    import pandas as pd
    import math
    import glob
    import os

    os.chdir(path)

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

    frame = pd.DataFrame(np.load('_ibl_rightCamera.times.npy'), columns = ['Camera times'])

    a = []
    b = []

    for val in df[df.index >= 1].index:
        x = frame.loc[(frame['Camera times'] > df['Trial Start'].iloc[val]) &
           (frame['Camera times'] < df['Stim On'].iloc[val])]
        y = x.index[0]
        z = x.index.max()
        a.append(y)
        b.append(z)

    A = np.insert(a, 0, 0, axis=0) #fTS
    B = np.insert(b, 0, 0, axis=0) #fSO

    c = []

    for val in df[df.index >= 1].index:
        x = frame.loc[(frame['Camera times'] > df['Stim On'].iloc[val]) &
           (frame['Camera times'] < df['Response'].iloc[val])]
        y = x.index.max()
        c.append(y)

    C = np.insert(c, 0, 0, axis=0) #fRes

    d = []

    for val in df[df.index >= 1].index:
        x = frame.loc[(frame['Camera times'] > df['Response'].iloc[val]) &
           (frame['Camera times'] < df['Trial End'].iloc[val])]
        y = x.index.max()
        d.append(y)

    D = np.insert(d, 0, 0, axis=0) #fTE

    #times converted to approx frame based on frame rate of 150Hz
    df['fTS'] = A
    df['fSO'] = B
    df['fRes'] = C
    df['fTE'] = D

    result = glob.glob('dop*.csv')
    df2 = pd.read_csv(result[0], skiprows=2)
    df2 = df2.iloc[0:len(df2.index), np.r_[0,1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,
                                           43,44,46,47,49,50,52,53,55,56,58,59]]  # subselect dataframe
    df2.columns = ['frame','earx','eary','eyetopx','eyetopy','eyerx','eyery','eyebotx','eyeboty','eyelx','eyely','nosetipx','nosetipy','tubetopx','tubetopy','tubebottomx','tubebottomy',
                 'chinx','chiny','whiskerx','whiskery','tonguelx','tonguely','tonguerx','tonguery',
                 'pinkyrx','pinkyry','ringrx','ringry','middlerx','middlery','pointerrx','pointerry',
                 'pinkylx','pinkyly','ringlx','ringly','middlelx','middlely','pointerlx','pointerly']

    def euclid(x1,y1,x2,y2):
        return (((((x2-x1)**2 + (y2-y1)**2)**.5)/9.6)/10)

    df2['diamH'] = euclid(df2['eyelx'],df2['eyely'],df2['eyerx'],df2['eyery']) # horizontal diameter
    df2['diamV'] = euclid(df2['eyetopx'],df2['eyetopy'],df2['eyebotx'],df2['eyeboty']) # vertical diameter
    df2['Circ'] = (math.pi)*(df2['diamH'])

    def calc_diam(x,y,z,w,j):
        Diam_mean_H = []
        Diam_mean_V = []
        Circ = []
        for val in df.index:
            start = int((df[x][val]))
            stop = int((df[y][val]))
            global eucH
            global eucV
            eucH = df2['diamH'].iloc[start:stop].mean()
            eucV = df2['diamV'].iloc[start:stop].mean()
            circ = df2['Circ'].iloc[start:stop].mean()
            Diam_mean_H.append(eucH)
            Diam_mean_V.append(eucV)
            Circ.append(circ)
        df[z] = np.array(Diam_mean_H)
        df[w] = np.array(Diam_mean_V)
        df[j] = np.array(Circ)

    calc_diam('fTS','fSO','S1H','S1V','S1C') #Mean diam from trial start to stim on time
    calc_diam('fSO','fRes','S2H','S2V','S2C') #Mean diam from stim onset to response registered time
    calc_diam('fRes','fTE','S3H','S3V','S3C') #Mean diam from Response registered time to trial end

    #Mean diam during ITI
    Diam_mean_H = []
    Diam_mean_V = []
    Circ = []
    for val in df[(df.index < df.index.max())].index:
        start = int((df['fTE'][val]))
        stop = int((df['fTS'][val+1]))
        global eucH
        global eucV
        eucH = df2['diamH'].iloc[start:stop].mean()
        eucV = df2['diamV'].iloc[start:stop].mean()
        circ = df2['Circ'].iloc[start:stop].mean()
        Diam_mean_H.append(eucH)
        Diam_mean_V.append(eucV)
        Circ.append(circ)
    Diam_mean_H.append(0)
    df['S-ITI-H'] = np.array(Diam_mean_H)
    Diam_mean_V.append(0)
    df['S-ITI-V'] = np.array(Diam_mean_V)
    Circ.append(0)
    df['S-ITI-C'] = np.array(Circ)

    filea = result[0].split('.csv')
    addpaw = 'pupil'
    ext = '.csv'
    input = addpaw + ext

    df.to_csv(input,index=False)

    print('analyzed - pupil')
