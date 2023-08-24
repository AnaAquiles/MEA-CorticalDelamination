#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:41:36 2022

@author:  Phase coherence aaquiles
"""


from scipy.signal import hilbert
import matplotlib.pyplot as plt 
import math
import pandas as pd
import numpy as np
from scipy import signal
import scipy.io
from scipy import stats 


filename="AP004-Basal-CTRL-DWSAMPLE"
DownData = np.loadtxt(filename + ".csv",delimiter=',')                         # DataFilt to 200 Hz

fs = 1000

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

fs = 1000
lowcut = 0.5                                                                   #Elegir la banda a estudiar
highcut = 40
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
DataFiltBP = np.delete(DataFiltBP, 30,0)

# make a matrix of every sample decomposition

z= hilbert(DataFiltBP[:,:300000])                                                  #form the analytical signal
# Z_noREF = np.delete(z, 30,0)
Amp = np.abs(z)
# AmpNorm = Amp/ np.min(Amp)                                                                #envelope extraction

AmpNorm = np.log10(Amp/ (np.max(Amp)))
Phase = np.angle(z)                                                             #inst phase
inst_freq = np.diff(Phase)/(2*np.pi)*fs                                         #inst frequency

plt.figure(5)
plt.clf()
plt.imshow(AmpNorm[:,:], 
               aspect ='auto', cmap='jet', )
plt.xlabel('Time')
plt.ylabel('Channels')
plt.title('Control 5min BASAL- GammaF')
plt.colorbar()
plt.grid(False)
plt.box(False)


def z_score(values) : 
    for i in range(0,len(values)):
        Mean = np.mean(values, axis =1)
        std = np.std(values,axis =1)
        
    Power = []
    for i in range(0,len(values)):
        Power.append((values[i,:] - Mean[i])/ std[i])
    Power = np.array(Power)
    return Power
Zscore = z_score(Amp)



plt.figure(2)
plt.subplot(411)
plt.plot(DataFiltBP[5,:100000], 'k-', alpha = 0.8)
plt.box(False)

plt.subplot(412)
plt.plot(Phase[40,:100000], 'r-', alpha = 0.4, label = 'Delta')
plt.legend()
plt.box(False)

plt.subplot(413)
plt.plot(Phase1[40,:100000], 'b-', alpha = 0.4, label = 'Theta')
plt.legend()
plt.box(False)

plt.subplot(414)
plt.plot(Phase2[40,:100000], 'g-', alpha = 0.4, label = 'Beta')
plt.legend()
plt.box(False)




#%%
##### guardar info en formato H5 

import h5py 

h5f = h5py.File('Electrodo1.h5', 'w')

elec = DataFiltBP[0,:]

h5f.create_dataset('Electrodo1', data=elec)
# Out[17]: <HDF5 dataset "Electrodo1": shape (2413600,), type "<f8">

with h5py.File('Electrodo1.h5', 'r') as hf:
    data = hf['Electrodo1'][:]

#%%

### coordinates and Magnitude value

#https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import stats
# import pandas as pd

# Coordenadas de cada electrodo
Coord  =pd.read_csv('CoordMEA.csv')

# Coord = Coord.drop(30)

x = Coord['X'].values
y = Coord['Y'].values

x = np.delete(x,0,0)
y = np.delete(y,0,0)


CorrMat_shape = np.delete(CorrMat_shape,0,1)

filenames = []

for i in range(1000):
    plt.figure(figsize = (8,4))
    plt.subplot(121)
    plt.scatter(x,y, s = 200, c = Zscore[:,i], cmap = 'jet',
                vmin=-1, vmax =1, alpha = 0.8)
    plt.box(False)
    plt.subplot(122)
    plt.scatter(x,y, s = 200, c = ZscoreB[:,i], cmap = 'jet',
                vmin=-1, vmax =1, alpha = 0.8)
    plt.title(f'LOW MAGNESIUM {i}')
    plt.colorbar()
    plt.box(False)
    filename = f'{i}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()
    
with imageio.get_writer('BothampZscore-0-DTB.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
for filename in set(filenames):
    os.remove(filename)        

# plt.scatter(x,y,s = 100, cmap = 'jet')
# plt.colorbar
# plt.box(False)

#%%
### FOR 2 CHANNELS, PHASE COHERENCE


PhSh = Phase                                                                    # - 1 min
Challenge = PhSh[21,:600] - PhSh[45,:600]                                             # np.exp(1j*(PhSh[21,:] - PhSh[45,:]))
FreqDiff = np.mean((inst_freq[21,:600],inst_freq[45,:600]), axis = 0)


d = {'Frequency' : [], 'PhaseDiff' : [], 'Time' : []}

DataFrame = pd.DataFrame(d)

time = np.linspace(0,60,600)

DataFrame['PhaseDiff'] = Challenge[:]
DataFrame['Frequency'] = FreqDiff
DataFrame['Time'] = time[:]
#DataFrame['Channel'] = '21-45'

# DataFrame.to_csv('FreqBCNU20Hz21-45.csv', index = False)

DataVal = DataFrame.values
DatavalShort = DataVal[:100,:]

rows, rows_pos = np.unique(DataVal[:,0],return_inverse = True)
cols, col_pos = np.unique(DataVal[:,2], return_inverse = True)

# pivot_table = np.zeros((len(rows), len(cols)), dtype=data.dtype)
# pivot_table[rows_pos, col_pos] = Dataval[:, 1]

import scipy.sparse as sps
pivot_table = sps.coo_matrix((DataVal[:, 1], (rows_pos, col_pos)),
                             shape=(len(rows), len(cols))).A 



#%%

##### SLIDING WINDOW


### Phase coherence 
# lst = [1,5,8,2,7,9,11,2,4,5,9,10,12,3]                                           # lista ejemplo

def sliding_window(elements, window_size):
    if len(elements) <= window_size:
        return elements
    
    # for i in range(0,len(elements) - window_size + 1, window_size//2):        # indica hasta donde va a terminar de construir una ventana
    #     for j in range(0,len(elements) - window_size + 1, window_size//2)
    sw_gen = []
    for i in range(0,len(elements) - window_size + 1):        # indica hasta donde va a terminar de construir una ventana
        for j in range(0,len(elements) - window_size + 1):
            sw_gen.append(np.subtract(elements[i:i+window_size], elements[j:j+window_size]))                               # overlap de 50% de la ventana
    sw_gen = np.array(sw_gen)
    return sw_gen

# Window1 = sliding_window(lst, 4)                                              # prueba ejemplo

PhaseDiff = []
for i in range(0,60):
    for j in range (0,60):
        PhaseDiff.append(Phase[i,i:i+100] - Phase[j,j:j+100])
        
PhaseDiff = np.array(PhaseDiff)
PhaseDiff_mean = np.mean(PhaseDiff, axis =1)
mat_reshape = np.reshape(PhaseDiff_mean, (60,60))                              # mat reshape
# datosPhase = np.array([PhaseDiff_mean[i:i+60] for i in range(0,60,1)])            # phase coherene without sliding window
plt.figure(8)
plt.imshow(mat_reshape, cmap = 'seismic', vmin = -3, vmax = 3)
plt.title('Window: 8')
plt.xlabel('Channels')
plt.ylabel('Channels')
plt.colorbar()


Windows = []
for i in range(0,60):
    Windows.append(sliding_window(Phase[i,:],1000))                    # Window for every phase

Windows = np.array(Windows)

### calculo de todas las diferencias posibles 

# https://www.geeksforgeeks.org/moviepy-creating-animation-using-matplotlib/

#%%

### phase correlation 

# def sliding_window(elements, window_size):
#     if len(elements) <= window_size:
#         return elements
        
#     CorrMat= []
#     for i in range(0,len(elements[0,:]) - window_size + 1):       
#         spcorr.append(stats.spearmanr(elements[:,i:i+window_size], axis=1))                              # overlap de 50% de la ventana
         
#     CorrMat=np.array(spcorr)
#     return CorrMat
   
# Windows = []
# for i in range(0,60):
#     Windows.append(sliding_window(Phase,1000))                    # Window for every phase

# Windows = np.array(Windows)

"######## correlación contra todos los electrodos + SLIDING WINDOW"

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import stats

CorrMat = []
spcorr = []

for i in range(0,len(Phase[0,:]) - 1000 + 1, 500):       
    spcorr.append(stats.spearmanr(Phase[:,i:i+1000], axis=1))                              # overlap de 50% de la ventana

CorrMat = np.array(spcorr)


filenames = []
for i in range(len(CorrMat[:,0,:,0])):
    plt.imshow(CorrMat[i,0,:,:], cmap = 'seismic', vmin=-1, vmax =1)
    plt.colorbar()
    plt.title('Control cero Mg activity 1 min')
    plt.box(False)
    filename = f'{i}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()
    
with imageio.get_writer('ControlmatCero.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
for filename in set(filenames):
    os.remove(filename)        



"###### CORRELACIÓN PONIENDO UNA SEMILLA + SLIDING WINDOW"

CorrMat = []
spcorr = []

for i in range(0,len(Phase[0,:]) - 1000 + 1, 500):
    for j in range(0,len(Phase[:,0]),1):
       spcorr.append(stats.spearmanr(Phase[0,i:i+1000], Phase[j,i:i+1000], axis=1))                              # overlap de 50% de la ventana

CorrMat = np.array(spcorr)
Spcorr = CorrMat[:,0]
CorrMat_shape = np.array([Spcorr[i:i+59] for i in range(0,len(Spcorr),59)])


#%%

### para 5 min corrleación de fase
CorrMat = []
spcorr = []
for j in range(0,len(Phase[:,0]),1):
    spcorr.append(stats.spearmanr(Phase[0,:300000], Phase[j,:300000], axis=1))   
    CorrMat = np.array(spcorr)
Spcorr = CorrMat[:,0]
Spcorr = np.array(Spcorr)

x = Coord['X'].values
y = Coord['Y'].values
x = np.delete(x,0,0)
y = np.delete(y,0,0)
CorrMat_shape = np.delete(Spcorr,0)
plt.scatter(x,y, s = 200, c = CorrMat_shape, cmap = 'seismic',
                vmin=-1, vmax =1, alpha = 1)
plt.colorbar()
plt.title('BCNU Basal Activity Beta E0')
plt.box(False)
#%% 

# z= hilbert(DataFilt)                                                         #form the analytical signal
# Amp = np.abs(z)                                                              #envelope extraction
# Phase = np.angle(z)                                                          #inst phase
# inst_freq = np.diff(Phase)/(2*np.pi)*fs                                      #inst frequency
# PhSh = Phase[:,:60000]                                                       # modify the len of time to evalute
###### WITH WAVELETS!!!

Channel1 = DataFilt[7,:60000]
Channel2 = DataFilt[10,:60000]

freqs2use = np.logspace(np.log10(1),np.log10(15),15)
time2save = np.arange(0,70,10).astype(float)
timewindow = np.linspace(1,3,len(freqs2use))
baselinetm = np.array([0,5])

# Wavelets y FFT params

time = np.arange(-1,1.001,1/1000)
half_wavelet = int((len(time)-1)/2)
num_cycles = np.logspace(np.log10(4), np.log10(8), len(freqs2use))
n_wavelet = len(time)
n_data = 2 * 60000
n_convolution = n_wavelet + n_data-1

# Time in indices
Times = np.linspace(0,60,len(Channel1))


def dsearchn(x, v):
    z=np.atleast_2d(x)-np.atleast_2d(v).T
    return np.where(np.abs(z).T==np.abs(z).min(axis=1))[0]

time2saveidx = dsearchn(Times, time2save).astype(float)
baselineidx = dsearchn(time2save, baselinetm).astype(float)

#initialize
size = len(freqs2use), len(time2save)
ispc = np.zeros(size)
ps = np.zeros(size)

#data FFTs

data_fft1 = np.fft.fft(Channel1,n_convolution)
data_fft2 = np.fft.fft(Channel2, n_convolution)



for i in range(0, len(freqs2use)):
    # wavelet and FFT
    s = num_cycles[i]/ (2 * np.pi * freqs2use[i])
    wavelet_fft = np.fft.fft(((np.exp(2 * 1j * np.pi * freqs2use[i])) * time) * np.exp(-time)**2 / (2*(s**2)),
                             n_convolution)
    # phase angles from channel 1 via convolution
    convolution_result_fft = np.fft.ifft((wavelet_fft * data_fft1), n_convolution)
    convolution_result_fft = convolution_result_fft[half_wavelet:-half_wavelet]
    phase_sig1 = np.angle(convolution_result_fft)
    
    # phase angles from channel 2 via convolution
    convolution_result_fft = np.fft.ifft((wavelet_fft * data_fft2), n_convolution)
    convolution_result_fft = convolution_result_fft[half_wavelet:-half_wavelet]
    phase_sig2 = np.angle(convolution_result_fft)
    
    # phase angle difference
    phase_diffs = phase_sig1 - phase_sig2
    
    # compute ICPS over trials 
    ps[i,:] = np.abs(np.mean(np.exp(1j*phase_diffs[time2saveidx,:]), 2))
    
    # compute time window in indices for this frequency
    time_window_idx = np.round((1000/freqs2use[i]) * timewindow[i]/(1000/1000))
    
    for t in range(0,len(time2save)):
        # compute phase synchronization 
        phasesynch = np.abs(np.mean(
            np.exp(1j*phase_diffs(time2saveidx[t] - time_window_idx:time2saveidx[i] + time_window_idx,:)),1))
        #average over trials
        
        ispc[i,t] = mean(phasesynch)
        
#%%
        
### TRANSFORMADA DE HILBERT 
        

### LOAD DATA

filename="AP016-Basal-BCNU-DWSAMPLE"
DataFilt = np.loadtxt(filename + ".csv",delimiter=',')

fs = 1000                                                                      #Frequency sample (DOWNSAMPLING)

### DataFilter 20 Hz

from scipy import signal
from scipy.signal import butter,filtfilt,lfilter

order = 5
fs = 1000 
cutoff = 20
nyq = 0.5 * fs


channel = DataFilt[35,:].astype(float)                                         # select the channel that you wanna try
Time = np.linspace(0,60,60000)

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
    
filtered = []
for i in range(0,60):
    filtered.append(butter_lowpass_filter(DataFilt[i,:], cutoff, fs, order))
    
DataFilt2 = np.array(filtered)

### 
#####Datos filtrados a 
#%%

#  Signal decomposition by Hilbert transform


z= hilbert(DataFiltBP[:,:60000])                                                            #form the analytical signal
Amp = np.abs(z)                                                                 #envelope extraction
Phase = np.angle(z)                                                             #inst phase
inst_freq = np.diff(Phase)/(2*np.pi)*fs                                         #inst frequency

PhSh = Phase                                                                    # - 1 min
Challenge = PhSh[21,:] - PhSh[45,:]                                             # np.exp(1j*(PhSh[21,:] - PhSh[45,:]))
FreqDiff = np.mean((inst_freq[21,:],inst_freq[45,:]), axis = 0)


d = {'Frequency' : [], 'PhaseDiff' : [], 'Time' : []}

DataFrame = pd.DataFrame(d)

time = np.linspace(0,60,60000)

DataFrame['PhaseDiff'] = Challenge[:-1]
DataFrame['Frequency'] = FreqDiff
DataFrame['Time'] = time[:-1]
#DataFrame['Channel'] = '21-45'

# DataFrame.to_csv('FreqBCNU20Hz21-45.csv', index = False)

DataVal = DataFrame.values
DatavalShort = DataVal[:100,:]

rows, rows_pos = np.unique(DataVal[:,0],return_inverse = True)
cols, col_pos = np.unique(DataVal[:,2], return_inverse = True)

# pivot_table = np.zeros((len(rows), len(cols)), dtype=data.dtype)
# pivot_table[rows_pos, col_pos] = Dataval[:, 1]

import scipy.sparse as sps
pivot_table = sps.coo_matrix((DataVal[:, 2], (rows_pos, col_pos)),
                             shape=(len(rows), len(cols))).A 
#%%
#### All the differences between channels

PhaseDiff = []

for i in range(0,60):
    for j in range (0,59):
        PhaseDiff.append(Phase[i,:] - Phase[j+1,:])
        
PhaseDiff = np.array(PhaseDiff)

###

# Signal decomposition by Fourier Transform 
N = 60000
T = 1.0 / 50
yf = np.fft.fft(DataFilt[2,:60000])
freq = np.fft.fftfreq(yf.size, d=T)
magnitude = np.abs(yf)
Phase = np.angle(yf)

####

# Phase of 1 minute channel vs channel

# PhSh = Phase[:,:60000] 
# Challenge1 = np.exp(1j*(PhSh[2,:] - PhSh[5,:]))
# PhSh = Phase[:,:60000] 
# Challenge2 = np.exp(1j*(PhSh[2,:] - PhSh[5,:]))
# PhSh = Phase[:,:60000] 
# Challenge3 = np.exp(1j*(PhSh[2,:] - PhSh[5,:]))
# PhSh = Phase[:,:60000] 
# Challenge4 = np.exp(1j*(PhSh[2,:] - PhSh[5,:]))

Change = np.array([Challenge, Challenge1, Challenge2, Challenge3, Challenge4])

Freq = inst_freq[10,:60000]
Ind_Freq = np.array(np.where((Freq < 13) & (Freq > 0))).reshape(1600)
Freq_1 = Freq[Ind_Freq]

PhCoh1 = Challenge[Ind_Freq]
PhCoh2 = Challenge1[Ind_Freq]

Time = np.linspace(0,60,len(Freq))    
    
size = len(Ind_Freq), len(Time)
ispc = np.zeros(size)
ps = np.zeros(size)

for i in range(0, len(Change)):
    ps[:,i] = np.mean(np.exp(1j * Change))
    
    
plt.figure(1)

plt.subplot(511)
plt.plot(Challenge)
plt.box(False)

plt.subplot(512)
plt.plot(Challenge1)
plt.box(False)    
    
plt.subplot(513)
plt.plot(Challenge2)
plt.box(False)    
        
plt.subplot(514)
plt.plot(Challenge3)
plt.box(False)     
    
plt.subplot(515)
plt.plot(Challenge4)
plt.box(False)     
        
#%%

Fs = 8000
f = 10
sample = 8000
x = np.arange(sample)
y = np.sin(2 * np.pi * f * x / Fs)

###

Fs = 8000
f = 10
sample = 8000
z = np.arange(sample)
w = np.sin(2 * np.pi * f * z / Fs)

Fs = 8000
f = 6
sample = 8000
u = np.arange(sample)
v = np.sin(2 * np.pi * f * u / Fs)

Fs = 8000
f = 10
sample = 8000
r = np.arange(sample)
s = np.sin(2 * np.pi * f * r / Fs)


time = np.linspace(0,1,8000)
Data = np.array((y,w,v,s))

z= hilbert(Data)                                                                #form the analytical signal
Amp = np.abs(z)                                                                 #envelope extraction
Phase = np.angle(z)                                                             #inst phase
inst_freq = np.diff(Phase)/(2*np.pi)*fs                                         #inst frequency
freqMean = np.mean(inst_freq, axis =0)

PhSh = Phase                                                                    # - 1 min

ChallengeProb = PhSh[0] - PhSh[1]
####

d = {'Frequency' : [], 'PhaseDiff' : []}

DataFrame = pd.DataFrame(d)

DataFrame['PhaseDiff'] = ChallengeProb[:-1]
DataFrame['Frequency'] = freqMean
DataFrame['Time'] = time[:-1]
DataFrame['P'] = 1
# DataFrame['Channel'] = '21-45'

DataFrame.to_csv('FreqProb5.csv', index = False)

#%%


#### MOVIEpy

from moviepy.editor import *
img = [ '1.png', '2.png', '3.png', '4.png', '5.png', '6.png',
       '7.png', '8.png']

clips = [ImageClip(m).set_duration(2)
      for m in img]

duration = 4

# animation = VideoClip(clips, duration = duration)
# animation.ipython_display(fps = 20, loop = True, autoplay = True)


concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.set_duration(5)
concat_clip.write_videofile("testctrl.mp4", fps=8)
