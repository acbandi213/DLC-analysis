import numpy as np
import pandas as pd
import glob
import os

# goes thru each csv file and finds the frames in which each joint is past the 400pixel y zone
path = r"Y:\DLC\ACC_DMS_imaging-acb-2020-09-01\videos\*.csv"
d = []
for fname in glob.glob(path):
    df = pd.read_csv(fname, skiprows=2)
    df2 = df[(df['y'] > 400) & (df['y.1'] > 400) & (df['y.2'] > 400) & (df['y.3'] > 400) & (df['y.4'] > 400)]
    dis = df2['y'].count()
    tot = df['y'].count()
    val = (dis/tot)*100
    d.append(val)
    print(val)

# does the same for imaging videos 
path = r"Y:\DLC\ACC_DMS_imaging-acb-2020-09-01\videos\*.csv"
d2 = []
for fname in glob.glob(path):
    df = pd.read_csv(fname, skiprows=2)
    df2 = df[(df['y'] > 400) & (df['y.1'] > 400) & (df['y.2'] > 400) & (df['y.3'] > 400) & (df['y.4'] > 400)]
    dis = df2['y'].count()
    tot = df['y'].count()
    val = (dis/tot)*100
    d2.append(val)
    print(val)

# make boxplots 
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

plt.subplot(2,2,1)
ax = sns.boxplot(x=d2)
ax.set_title('imaging')

plt.subplot(2,2,2)
ax = sns.boxplot(x=d)
ax.set_title('Opto')

# save numpy arrays of disenaged percentages 
from numpy import save
save('imaging.npy', d2)
save('opto.npy', d)

# find and save file names 
path = r"Y:\DLC\ACC-DMS_nphr-acb-2020-07-30\videos\pre-track\*.csv"
frames = []
for fname in glob.glob(path):
    frames.append(fname)
    print(fname)
save('opto_names.npy', frames)

# dataframe of both file names + percent disengaged 
df = pd.DataFrame(np.load('imaging_names.npy'), columns = ['names'])
df['disengaged'] = np.load('imaging.npy')
df2 = df[df['disengaged'] > 15] #new data frame with outlier percentage points 
df2
# df2['names'].str.slice(45, 72) #file names are annoyingly long so use this to get the important file name info


