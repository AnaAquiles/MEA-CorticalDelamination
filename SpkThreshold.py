#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:21:52 2023

@author: aaquiles
"""

"""
Spike sorting threshold


# 0 - Reorganize every channel 
# 1 - Filter  from 200 Hz to 5kHz
# 2- Find a threshold between median {signal/Factor}
 (standard deciation of background) noise
# 3- Save in a DATAFRAME 
 
"""

from scipy.signal import hilbert
import matplotlib.pyplot as plt 
import math
import pandas as pd
import numpy as np
from scipy import signal
import scipy.io
from scipy import stats
from shapely.geometry import LineString 


# Prueba con un canal primero 16 enero 
# filename="AP011-CTRL-900"
# data = np.loadtxt(filename + ".csv",delimiter=',')                         # DataFilt to 200 Hz

####     CHANNELS REORGANIZATION


Array = [23,25,28,31,34,36,20,21,24,29,30,35,38,39,18,19,22,
          27,32,37,40,41,15,16,17,26,33,42,43,44,
          14,13,12,3,56,47,46,45,11,10,7,2,57,52,
          49,48,9,8,5,0,59,54,51,50,6,4,1,58,55,53]

idx = np.empty_like(Array)
idx[Array] = np.arange(len(Array))

data[:] = data[idx,:]
data = np.delete(data, 30,0)
#%%
#######

u_T = int(len(data[0,:])/25000)
down = int(u_T * 22000)

# UN CANAL
channel = data[1,:].astype(float)                                              # select the channel that you wanna try
# Dchannel = signal.resample(channel, down)                                       # how many samples are necesary to acquiere 900 seconds  at 1.25 kHz
Time = np.linspace(0,u_T,len(channel))

# DownSample = []
# for i in range(0,60):
#     DownSample.append(signal.resample(data[i,:],down))

# DownData = np.array(DownSample)

                                   
## BAND PASS

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


fs = 25000
lowcut = 200                                                                 #Elegir la banda a estudiar
highcut = 5000

# UN CANAL 
DFilt = butter_bandpass_filter(channel, lowcut, highcut, fs, order = 3)
#%%

fs = 25000
lowcut = 200                                                                 #Elegir la banda a estudiar
highcut = 5000

# fs = 1000
filtered = []
for i in range(0,60):
    filtered.append(butter_bandpass_filter(data[i], lowcut, highcut, fs, order = 3))
DataFiltBP = np.array(filtered)

# Array = [23,25,28,31,34,36,20,21,24,29,30,35,38,39,18,19,22,
#           27,32,37,40,41,15,16,17,26,33,42,43,44,
#           14,13,12,3,56,47,46,45,11,10,7,2,57,52,
#           49,48,9,8,5,0,59,54,51,50,6,4,1,58,55,53]


# idx = np.empty_like(Array)
# idx[Array] = np.arange(len(Array))

# DataFiltBP[:] = DataFiltBP[idx,:]
# DataFiltBP = np.delete(DataFiltBP, 30,0) ### REFERENCE

#%%
##########        THRESHOLD 

# Time = np.linspace(0,u_T,len(Dchannel))


def threshold (Factor, n_points):
    Emp = np.ones(len(n_points))
    x = n_points/Factor
    sigma = np.median(x)
    Th = Emp*(5*sigma) 
    return Th
def line_threshold(Th, data):
    idx = np.argwhere(np.diff(np.sign(Th - data)
                              ) != 0).reshape(-1)+0
    return idx

Intercepts = threshold(0.0059, DFilt)

#exploracion
idx = np.argwhere(np.diff(np.sign(Intercepts - DFilt)
                          )!= 0).reshape(-1)+0

Spiga1 = DFilt[idx[0]-8:idx[1]+10] # exploracion
Spiga2 = DFilt[idx[2]-8:idx[3]+10]
Spiga3 = DFilt[idx[4]-8:idx[5]+10]


plt.figure(0)
plt.plot(Time,Intercepts, 'r-')
plt.plot(Time, DFilt, 'k-', alpha =0.3)
plt.plot(Time, DFilt, 'ro', alpha =0.05)
plt.box(False)

# ZOOM first-1000 pts
plt.figure(1)
plt.plot(Time[:1000],Intercepts[:1000], 'r-')
plt.plot(Time[:1000], DFilt[:1000], 'k-', alpha =0.3)
plt.box(False)
#%%

### SAVE EVERY CHUNK OF THE SIGNAL 

# Voltage 
Pair_ID = []
for i in range(len(idx)): 
    Pair_ID.append(DFilt[idx[i]-8:idx[i]+1+10])
    
Pair_IDarr = np.array(Pair_ID)

# Time mean
TimeID = []
for i in range(len(idx)): 
    TimeID.append(np.mean(Time[idx[i]-8:idx[i]+1+10]))
    
Time_ID = np.array(TimeID)
# Time_IDav = np.average(Time_ID,axis=1)
X_Label = np.arange(0,len(Time_ID),1)

## MAKE THE DATA FRAME

d = {'x' : [], 'Time(s)' : [], 'Values' : []}

Spikes_Th = pd.DataFrame(d)
Spikes_Th['Values'] = Pair_ID
Spikes_Th['Time(s)'] = Time_ID
Spikes_Th['x'] = X_Label
Spikes_Th['Channel'] = '1'

# SAVE
compression_opts = dict(method='zip',
                        archive_name='Chan1_011CTRL0Mg02.csv')

Spikes_Th.to_csv('Chan1_011CTRL0Mg02.zip', index =False, 
                 compression=compression_opts)

