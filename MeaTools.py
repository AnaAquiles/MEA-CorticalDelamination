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

# import bokeh.io
# import bokeh.plottinging 
# # from bokeh.palettes import Spectral11

# Data exploration

test_data_folder = r'/media/2TB_HDD/AnaAquiles'          # /AnaAquiles/AP062021adjust this!
channel_raw_data = McsPy.McsData.RawData(os.path.join(
    test_data_folder, 'AP005_BCNU_p35_60_200_cero0002.h5'))


analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
analog_stream_0_data = analog_stream_0.channel_data

np_analog_stream_0_data = np.transpose(analog_stream_0_data)

# print("Old shape:", analog_stream_0_data.shape)
# print("New shape:", np_analog_stream_0_data.shape)
# print()
# print(np_analog_stream_0_data)

channel_ids = channel_raw_data.recordings[0].analog_streams[0].channel_infos.keys()
# print(channel_ids)

channel_id = list(channel_raw_data.recordings[0].analog_streams[0].channel_infos.keys())[0]

stream = channel_raw_data.recordings[0].analog_streams[0]
time = stream.get_channel_sample_timestamps(channel_id, 0, )

scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
time_in_sec = time[0] * scale_factor_for_second

signal = stream.get_channel_in_range(channel_id, 0, 22500000)

sampling_frequency = stream.channel_infos[channel_id].sampling_frequency.magnitude 

data = channel_raw_data.recordings[0].analog_streams[0].channel_data[:, 0:]
aspect_ratio = 1000


# plt.figure(figsize=(20,12))

# plt.set_cmap("jet")
# plt.imshow(data,)# interpolation='nearest', )
# plt.colorbar()
# plt.xlabel('Sample Index')
# plt.ylabel('Channel Number')
# plt.title('BCNU Basal 6')
# plt.show()


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

# def butter_lowpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a

# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = lfilter(b, a, data)
#     return y
    
# order = 2
# fs = 1250  
# cutoff = 100

# b, a = butter_lowpass(cutoff, fs, order)

# y = butter_lowpass_filter(Dchannel, cutoff, fs, order)
# y = butter_lowpass_filter(Dchannel, cutoff, fs, order)


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


np.savetxt('AP011-CTRL-basal-DWSAMPLE.csv', DownData, delimiter = ',')


order = 5
fs = 1000 
cutoff = 200
nyq = 0.5 * fs

channel = data[35,:].astype(float)                                              # select the channel that you wanna try
Dchannel = signal.resample(channel, down)                                       # how many samples are necesary to acquiere 900 seconds  at 1.25 kHz
Time = np.linspace(0,u_T,len(Dchannel))

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
    
filtered = []
for i in range(0,60):
    filtered.append(butter_lowpass_filter(DownData[i,:], cutoff, fs, order))
    
DataFilt = np.array(filtered)

np.savetxt('AP011-Basal-CTRL-DWSAMPLE.csv', DownData, delimiter = ',')

#%%

### 

filename="AP016-Basal-BCNU-DWSAMPLE"
DataFilt = np.loadtxt(filename + ".csv",delimiter=',')

order = 5
fs = 1000 
cutoff = 12
nyq = 0.5 * fs

Time = np.linspace(0,u_T,len(Dchannel))

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
    
filtered = []
for i in range(0,60):
    filtered.append(butter_lowpass_filter(DataFilt[i], cutoff, fs, order))
    
DataFilt2 = np.array(filtered)


### Reorder the electrodes according to the MEA 60  200/

Array = [23,25,28,31,34,36,20,21,24,29,30,35,38,39,18,19,22,
          27,32,37,40,41,15,16,17,26,33,42,43,44,
          14,13,12,3,56,47,46,45,11,10,7,2,57,52,
          49,48,9,8,5,0,59,54,51,50,6,4,1,58,55,53]

idx = np.empty_like(Array)
idx[Array] = np.arange(len(Array))

DataFilt2[:] = DataFilt2[idx,:]

plt.matshow(DataFilt[:,:60000], interpolation='nearest', aspect ='auto', 
            cmap='RdYlBu', 
           vmin =-1000,vmax=1000) 
plt.title('CTRL Basal activity 1 minute')
plt.box(False)
plt.ylabel('Channel')
plt.xlabel('Time (s)')
plt.colorbar()


# f, t, Sxx = signal.spectrogram(DataFilt[43,:], fs, noverlap = 100, nfft = 1000 )

# plt.figure(2)
# plt.pcolormesh(t, f, Sxx, vmin= 500, vmax= 2500)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim(0,10)
# plt.colorbar()
# plt.show()
#%%

#### BAND PASS COHERENCE

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

a = 10
b = 1
c = 1


fig = plt.figure(2,figsize =(36,8))

for i in range(10,20,1):
    plt.subplot(a,b,c)
    plt.title('Channel {}, subplot: {},{},{}'.format(i,a,b,c))
    plt.plot(DataFiltBP[i, :])
    plt.box(False)
    c = c + 1


#%%

fig, axs = plt.subplots(nrows=30, ncols=1, figsize = (15,12))  
plt.subplots_adjust(hspace=0.)

for n in enumerate(len(DownData,axis=0)):
    ax = plt.subplot(DownData[n,:400000])
    
    
    
#%%

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


PowerT, frec, y = plt.magnitude_spectrum(DataFilt[28,:5000],Fs =1250,
                                                color = 'coral', alpha = 1, 
                       label = 'Magnitude Spectrum')

Baseline = np.min(powerSpectrum)                                              #4 seconds window
PowerN = np.log10(powerSpectrum/Baseline)


#%%


#### Signal decomposition

from scipy.signal import hilbert
from matplotlib import pyplot 
import math

fourier = np.fft.fft(DataFilt2)
hilbert = hilbert(DataFilt2)
amplitude_envelope = np.abs(hilbert) # np.abs(fourier)

# Amplitude_Valuemeantime = np.mean(amplitude_envelope,axis=1)*1e2
AmpNorm = np.log10(amplitude_envelope/ (np.min(amplitude_envelope)))


plt.figure(1)
plt.clf()
pyplot.imshow(AmpNorm[:,:], 
               aspect ='auto', cmap='jet', vmax=6.5)
plt.xlabel('Time')
plt.ylabel('Channels')
plt.title('BCNU 009 Basal, 15 min')
plt.colorbar()
plt.grid(False)
plt.box(False)

#%%

for i in range(0,59):
    f, t,Sxx = signal.spectrogram(DataFiltBP[i,:], 1000, noverlap = 250, nfft = 600,)

plt.pcolormesh(t, f,(Sxx), cmap = "jet",)# vmax = 5000) #10*np.log10 #Blues para CTRL
plt.colorbar()
plt.xlim(0,301)
plt.ylim(1,20)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


#%%

## 05/10/22 

"""
      ANALISIS DE REGISTROS DESPUÉS DEL DOWNSAMPLING 1KHz    
      
      - ISPC, con descomposición de Hilbert 
"""
from scipy.signal import hilbert
from matplotlib import pyplot 
import math

# https://towardsdatascience.com/instantaneous-phase-and-magnitude-with-the-hilbert-transform-40a73985be07
fs = 1000
### 

filename="AP016-Basal-BCNU-DWSAMPLE"
DataFilt = np.loadtxt(filename + ".csv",delimiter=',')

z= hilbert(DataFilt)                                                            #form the analytical signal
Amp = np.abs(z)                                                                 #envelope extraction
Phase = np.angle(z)                                                             #inst phase
inst_freq = np.diff(Phase)/(2*np.pi)*fs                                         #inst frequency


# Phase of 2 minute 


PhSh = Phase[:,:60000]                                                          # - 1 min
Challenge = PhSh[7,:] - PhSh[10,:]

Freq = inst_freq[10,:600000]
Time = np.linspace(0,600,len(Freq))

# f, t,Sxx = signal.spectrogram(DataFilt[7,:600000], 1000, noverlap = 250, nfft = 600,)

size = (301,600000)
M = np.zeros(size)

for i in range(0, len(M)):
    x[i] = f
    y[i] = Time
    Mat[x][y] = Probe[i]
    

# X,Y = np.meshgrid(f,Probe)
# plt.contour(f, Time, Y)

#%%

### 1 way to obtain phase coherence 


#https://github.com/emma-holmes/Phase-Coherence-for-Python/blob/master/PhaseCoherence.py

def PhaseCoherence(freq, timeSeries, FS):
    
    # Get parameters of input data
    nMeasures	 = np.shape(timeSeries)[0]
    nSamples 	= np.shape(timeSeries)[1]
    nSecs = nSamples / FS
    print('Number of measurements =', nMeasures)
    print('Number of time samples =', nSamples, '=', nSecs, 'seconds')
    
    # Calculate FFT for each measurement (spect is freq x measurements)
    spect = np.fft.fft(timeSeries, axis=1)
    
    # Normalise by amplitude
    spect = spect / abs(spect)
    
    # Find spectrum values for frequency bin of interest
    freqRes = 1 / nSecs;
    foibin = round(freq / freqRes + 1) - 1
    spectFoi = spect[:,foibin]
    
    # Find individual phase angles per measurement at frequency of interest
    anglesFoi = np.arctan2(spectFoi.imag, spectFoi.real)
    
    # PC is root mean square of the sums of the cosines and sines of the angles
    PC = np.sqrt((np.sum(np.cos(anglesFoi)))**2 + (np.sum(np.sin(anglesFoi)))**2) / np.shape(anglesFoi)[0]
    
    # Print the value
    print('----------------------------------');
    print('Phase coherence value = ' + str("{0:.3f}".format(PC)));
        
    return PC

PhSh = Phase[:,:60000]                                                          # - 1 min
Probe = PhSh[7,:] - PhSh[10,:]

Freq = inst_freq[10,:600000]
Time = np.linspace(0,600,len(Freq))

PhCoherence = []

for i in range(1,10,1):
    PhCoherence.append(PhaseCoherence(i, Signals, FS = 1000))
    
