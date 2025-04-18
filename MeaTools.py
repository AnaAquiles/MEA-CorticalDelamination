#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:35:33 2022

@author: aaquiles
"""

import sys, importlib, os
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from scipy.signal import butter, lfilter, freqz
from scipy import signal

import math
import numpy as np


# Data IMPORTATION AND EXPLORATION

test_data_folder = r'/media/2TB_HDD/AnaAquiles'          
channel_raw_data = McsPy.McsData.RawData(os.path.join(
    test_data_folder, 'AP005_BCNU_p35_60_200_cero0002.h5'))


analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
analog_stream_0_data = analog_stream_0.channel_data

np_analog_stream_0_data = np.transpose(analog_stream_0_data)

channel_ids = channel_raw_data.recordings[0].analog_streams[0].channel_infos.keys()
print(channel_ids)

channel_id = list(channel_raw_data.recordings[0].analog_streams[0].channel_infos.keys())[0]

stream = channel_raw_data.recordings[0].analog_streams[0]
time = stream.get_channel_sample_timestamps(channel_id, 0, )

scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
time_in_sec = time[0] * scale_factor_for_second

signal = stream.get_channel_in_range(channel_id, 0, 22500000)

sampling_frequency = stream.channel_infos[channel_id].sampling_frequency.magnitude 

data = channel_raw_data.recordings[0].analog_streams[0].channel_data[:, 0:]
aspect_ratio = 1000

#%%

#### Downsampling from 25kHz to 1.25 kHz len 900 s

# for one channel ...
from scipy import signal
from scipy.signal import butter,filtfilt

u_T = len(data[0,:])/25000

down = int(u_T * 1000)

channel = data[35,:].astype(float)                                              # select the channel that you wanna try
Dchannel = signal.resample(channel, down)                                       # how many samples are necesary to acquiere 900 seconds  at 1.25 kHz

DownSample = []

for i in range(0,60):
    DownSample.append(signal.resample(data[i,:],down))

DownData = np.array(DownSample)

# Low pass filer by butter method

order = 5
fs = 1000  
cutoff = 200
nyq = 0.5 * fs

Time = np.linspace(0,900,len(Dchannel))

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
    

y = butter_lowpass_filter(Dchannel, cutoff, fs, order)

plt.plot(Time, Dchannel,'go-', Time, y, '.-', 900, Dchannel[0],'ro')



#%%
# for all channels ...

from scipy import signal
from scipy.signal import butter,filtfilt,lfilter

u_T = int(len(data[0,:])/25000)

down = int(u_T * 1000)

DownSample = []

for i in range(0,60):
    DownSample.append(signal.resample(data[i,:],down))

DownData = np.array(DownSample)

## save you downsampled data 
np.savetxt('AP011-CTRL-basal-DWSAMPLE.csv', DownData, delimiter = ',')


#### BAND PASS EXPLORATION

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

fs = 1000
lowcut = 0.5
highcut = 3

filtered = []
for i in range(0,60):
    filtered.append(butter_bandpass_filter(DownData[i], lowcut, highcut, fs, order = 3))
    
DataFiltBP = np.array(filtered)

Array = [23,25,28,31,34,36,20,21,24,29,30,35,38,39,18,19,22,
          27,32,37,40,41,15,16,17,26,33,42,43,44,
          14,13,12,3,56,47,46,45,11,10,7,2,57,52,
          49,48,9,8,5,0,59,54,51,50,6,4,1,58,55,53]


idx = np.empty_like(Array)
idx[Array] = np.arange(len(Array))

DataFiltBP[:] = DataFiltBP[idx,:]

    
#%%
####### EXPLORE YOUR ORIGINAL, DOWNSAMPLED AND BAND FILTERED DATA 

plt.figure(2)
plt.subplot(411)
plt.plot(time_in_sec[:], data[27,:], label = 'Raw Data')
plt.legend()
plt.box(False)

plt.subplot(412)
plt.plot(Time[:60000], DownData[27,120000:180000], label = 'Downsample Data')
plt.legend()
plt.box(False)

plt.subplot(413)
plt.plot(Time[:60000], DataFilt[27,120000:180000], label = 'Low pass 200Hz Data')
# plt.xlim(0,4)
plt.legend()
plt.box(False)

plt.subplot(414)
powerSpectrum, frec, y = plt.magnitude_spectrum(DataFilt[27,120000:180000],Fs =1000,
                                                color = 'coral', alpha = 1, 
                       label = 'Magnitude Spectrum')
plt.ylim(0,300)
plt.xlim(0,100)
plt.legend()
plt.box(False)



