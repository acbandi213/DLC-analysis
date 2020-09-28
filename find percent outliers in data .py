## acbandi 9/28 - adpated from DeepLabCut

##IMPORTS
import numpy as np
from numpy import save
import pandas as pd
import glob
import os
import yaml
from ruamel.yaml import YAML
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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

def convertparms2start(pn):
    """ Creating a start value for sarimax in case of an value error
    See: https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk """
    if "ar." in pn:
        return 0
    elif "ma." in pn:
        return 0
    elif "sigma" in pn:
        return 1
    else:
        return 0

def FitSARIMAXModel(x, p, pcutoff, alpha, ARdegree, MAdegree, nforecast=0, disp=False):
    # Seasonal Autoregressive Integrated Moving-Average with eXogenous regressors (SARIMAX)
    # see http://www.statsmodels.org/stable/statespace.html#seasonal-autoregressive-integrated-moving-average-with-exogenous-regressors-sarimax
    Y = x.copy()
    Y[p < pcutoff] = np.nan  # Set uncertain estimates to nan (modeled as missing data)
    if np.sum(np.isfinite(Y)) > 10:

        # SARIMAX implemetnation has better prediction models than simple ARIMAX (however we do not use the seasonal etc. parameters!)
        mod = sm.tsa.statespace.SARIMAX(
            Y.flatten(),
            order=(ARdegree, 0, MAdegree),
            seasonal_order=(0, 0, 0, 0),
            simple_differencing=True,
        )
        # Autoregressive Moving Average ARMA(p,q) Model
        # mod = sm.tsa.ARIMA(Y, order=(ARdegree,0,MAdegree)) #order=(ARdegree,0,MAdegree)
        try:
            res = mod.fit(disp=disp)
        except ValueError:  # https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk (let's update to statsmodels 0.10.0 soon...)
            startvalues = np.array([convertparms2start(pn) for pn in mod.param_names])
            res = mod.fit(start_params=startvalues, disp=disp)
        except np.linalg.LinAlgError:
            # The process is not stationary, but the default SARIMAX model tries to solve for such a distribution...
            # Relaxing those constraints should do the job.
            mod = sm.tsa.statespace.SARIMAX(
                Y.flatten(),
                order=(ARdegree, 0, MAdegree),
                seasonal_order=(0, 0, 0, 0),
                simple_differencing=True,
                enforce_stationarity=False,
                enforce_invertibility=False,
                use_exact_diffuse=False,
            )
            res = mod.fit(disp=disp)

        predict = res.get_prediction(end=mod.nobs + nforecast - 1)
        return predict.predicted_mean, predict.conf_int(alpha=alpha)
    else:
        return np.nan * np.zeros(len(Y)), np.nan * np.zeros((len(Y), 2))

def compute_deviations(Dataframe, dataname, p_bound, alpha, ARdegree, MAdegree, storeoutput=None):
    """ Fits Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model to data and computes confidence interval
    as well as mean fit. """

    print("Fitting state-space models with parameters:", ARdegree, MAdegree)
    df_x, df_y, df_likelihood = Dataframe.values.reshape((Dataframe.shape[0], -1, 3)).T
    preds = []
    for row in range(len(df_x)):
        x = df_x[row]
        y = df_y[row]
        p = df_likelihood[row]
        meanx, CIx = FitSARIMAXModel(x, p, p_bound, alpha, ARdegree, MAdegree)
        meany, CIy = FitSARIMAXModel(y, p, p_bound, alpha, ARdegree, MAdegree)
        distance = np.sqrt((x - meanx) ** 2 + (y - meany) ** 2)
        significant = (
            (x < CIx[:, 0]) + (x > CIx[:, 1]) + (x < CIy[:, 0]) + (y > CIy[:, 1])
        )
        preds.append(np.c_[distance, significant, meanx, meany, CIx, CIy])

    columns = Dataframe.columns
    prod = []
    for i in range(columns.nlevels - 1):
        prod.append(columns.get_level_values(i).unique())
    prod.append(
        [
            "distance",
            "sig",
            "meanx",
            "meany",
            "lowerCIx",
            "higherCIx",
            "lowerCIy",
            "higherCIy",
        ]
    )
    pdindex = pd.MultiIndex.from_product(prod, names=columns.names)
    data = pd.DataFrame(np.concatenate(preds, axis=1), columns=pdindex)
    # average distance and average # significant differences avg. over comparisonbodyparts
    d = data.xs("distance", axis=1, level=-1).mean(axis=1).values
    o = data.xs("sig", axis=1, level=-1).mean(axis=1).values

    if storeoutput == "full":
        data.to_hdf(
            dataname.split(".h5")[0] + "filtered.h5",
            "df_with_missing",
            format="table",
            mode="w",
        )
        return d, o, data
    else:
        return d, o

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

def calc_fitting(fname):
    df = pd.read_hdf(fname) ##import data file 
        
    nframes = len(df)
    startindex = max([int(np.floor(nframes * 0)), 0])
    stopindex = min([int(np.ceil(nframes * 1)), nframes])
    Index = np.arange(stopindex - startindex) + startindex
        
    df = df.iloc[Index]
    mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
    df_temp = df.loc[:, mask] #temp df set up 
   
    warnings.simplefilter('ignore', ConvergenceWarning)
    d, o = compute_deviations(df_temp, dataname, p_bound, alpha, ARdegree, MAdegree)
    ind = np.flatnonzero(d > epsilon)  # time points with at least average difference of epsilon
    ind3 = len(ind)
    fitting = ind3/nframes

    return fitting

def calculate_percent_outlier_frames(path):
    uncertain_val = []
    jump_val = []
    #fitting_val =[]
    for fname in glob.glob(path):
        a = calc_uncertain(fname)
        uncertain_val.append(a)
        b = calc_jump(fname)
        jump_val.append(b)
        #c = calc_fitting(fname)
        #fitting_val.append(c)

    return uncertain_val jump_val #fitting_val 

calculate_percent_outlier_frames(path)

# find and save file names 
# find and save file names 
list = []
for fname in glob.glob(path):
    list.append(fname[20:28])
save('file_names.npy', list)

# dataframe of all outlier percentages 
df = pd.DataFrame(np.load('file_names.npy'), columns = ['videos'])
df['uncertain'] = np.load('uncertain.npy')
df['jump'] = np.load('jump.npy')