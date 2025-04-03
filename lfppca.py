

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:57:33 2023

@author: aaquiles
"""

from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
import matplotlib.pyplot as plt 
import math
import pandas as pd
import numpy as np
from scipy import signal
import scipy.io
from scipy import stats 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.signal.windows import dpss 

filename="AP016-Mg1-BCNU-DWSAMPLE"
DownData = np.loadtxt(filename + ".csv",delimiter=',')                         # DataFilt to 200 Hz

fs = 1000
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

###             Notch Filter
a_N, b_N = signal.iirnotch(2*60/fs, 90)
SignF = signal.filtfilt(a_N, b_N, DownData)
                        
lowcut = 0.5                                                         #Elegir la banda a estudiar
highcut = 100

filtered = []
for i in range(0,60):
    filtered.append(butter_bandpass_filter(SignF[i], lowcut, highcut, fs, order = 3))
    
DataFiltBP = np.array(filtered)

Array = [23,25,28,31,34,36,20,21,24,29,30,35,38,39,18,19,22,
          27,32,37,40,41,15,16,17,26,33,42,43,44,
          14,13,12,3,56,47,46,45,11,10,7,2,57,52,
          49,48,9,8,5,0,59,54,51,50,6,4,1,58,55,53]


idx = np.empty_like(Array)
idx[Array] = np.arange(len(Array))
# Time = np.linspace(0,900,len(Dchannel))

DataFiltBP[:] = DataFiltBP[idx,:]
DataFiltBP = np.delete(DataFiltBP, 30,0) #electrode reference position


####

#%%

#              POWER SPECTRUM OF EVERY CHANNEL  ADJUST 
import powerlaw as pwl
from scipy import optimize
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
plt.style.use('fivethirtyeight')


'''
###     Power spectrum only from the frequencies 0.5-100 Hz
         #Fitting of aperiodic exponent using the same formula
             of the paper 2022 Thomas Donoghue Technical Report NATURE

'''
###  USING LINEAR REGRESSION AND #### Lorentzian Function 

def powerSpec(datos,fs, th):   
    freqs = np.fft.rfftfreq(len(datos), d=1/fs)
    powerspec = np.abs(np.fft.rfft(datos))**2
    threshold = th
    indices = freqs > threshold
    # log_freq = np.log(freqs[indices])
    # log_power = np.log(powerspec[indices])
    Freq = freqs[indices]
    Pow = powerspec[indices]
    totalPow = np.sum(powerspec)
    normPow = powerspec/totalPow
    
    PowN = normPow[indices]
      
    # using linear regression
    # slope,intercept, r, p , stderr = linregress(log_freq,log_power)
    # aperiodicExp = -slope
    return Freq,Pow,PowN

# Alpha, Freqs, PowerS, Ind = powerSpec(DataFiltBP[5,:100000],1000,100)
    
Freqs = np.zeros((59,49950)) #150001 - 5 min  # 50001 -100 s  #5001 - 10s
PowerN = np.zeros((59, 49950))
Power = np.zeros((59, 49950))

for i in range(59):
    Freqs[i,:], Power[i,:],PowerN[i,:] = powerSpec(DataFiltBP[i,:100000],1000,0.5) 
    
    #last param indicates since which freq it will begin
    
    
##### #### Lorentzian Function 


def combinedMod(f,alpha,k):
    return -(np.log(k + f**(alpha))) #### Lorentzian Function 

def fit (freqs,Power):
    popt, pcov = curve_fit(combinedMod, freqs, Power)
    alpha, k= popt
    return alpha

AperiodicExp = np.zeros(59)

for i in range(59):
    AperiodicExp[i] = np.abs(fit(Freqs[i,:], Power[i,:]))
    
    
smoothed_power = []

for i in range(59):
    smoothed_power.append(gaussian_filter1d(PowerN[i,:], sigma = 5))
    
smoothed_power = np.array(smoothed_power)

##### plot the normalized power spec  plotted with a gaussian filter 
plt.figure(0)
for j in range(59):
    plt.plot(Freqs[j,:], smoothed_power[j,:], c= 'pink', alpha = 0.2, linewidth = 0.8)
    
    plt.xlim(0,30)
    # plt.ylim(0.001,1)
    plt.title('Eulam baseline  (004)')
    plt.xlabel('Log Frequency')
    plt.ylabel('Log Power')
    plt.xscale('log')
    plt.yscale('log')

# plt.savefig('GroupCompMgFilt.svg')

#### MEAN VARIANCE 

powerMEAN = np.mean(smoothed_power, axis = 0)  
variancePower = np.var(smoothed_power, axis = 0)
STDpower = np.std(smoothed_power, axis = 0)

log_powerSpec = np.log10(smoothed_power + 1e-10)
logpowerMEAN = np.mean(log_powerSpec, axis = 0)  
logvariancePower = np.var(log_powerSpec, axis = 0)


plt.figure(2)
plt.plot(Freqs[0,:], logpowerMEAN, color ='grey')
plt.fill_between(Freqs[0,:], logpowerMEAN - logvariancePower, 
                 logpowerMEAN + logvariancePower,
                 color = 'grey', alpha = 0.3)

plt.xscale('log')
# plt.yscale('log')
plt.savefig('BothCero2VARMEAN30Hz.svg')


#%%


'''
         POWER SPECTRUM DENSITY
         
      Hack : in matplotlib exits the function   
      plt.fill_between()
'''

confidence = 95
Interval = (100 - confidence)/ 2

MeanPow = np.mean(PowerN, axis = 0)
PercentileMin = np.percentile(PowerN, Interval, axis = 0)
PercentileMax = np.percentile(PowerN, 100 -Interval, axis = 0)
MinPow = np.min


plt.figure(4)
# plt.plot(MeanPow , 'k-')
plt.fill_between(Freqs[0,:], PercentileMin, PercentileMax, color = "pink", 
                 alpha = 0.5)
plt.xscale('log')
plt.yscale('log')

plt.savefig('005Mg.svg')

#%%

    
'''
           APERIODIC OR ALPHA VALUES PLOTS, HISTOGRAM
               KERNEL DENSITY  AND ROC CURVE

'''

df = pd.read_csv('AlphaValuesCondition.csv')
plt.style.use('fivethirtyeight')

sns.catplot(data=df, x="x", y="AperiodicExp", hue="Group",
            kind="boxen", showfliers = False)

df = df.replace("Eulaminated",1)  #'Control'
df = df.replace("Dislaminated",0) #'BCNU trated'

BinVal = df['Group'].values
AperVal = df['AperiodicExp'].values

from sklearn.metrics import roc_curve, auc

###### ROC Curve to compare the aperiodic exponent between layering

fpr, tpr, thr = roc_curve(BinVal, AperVal)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw = 2, label = f'ROC curve(area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.legend(loc='lower right')   

### histogram Normalized 
from scipy.stats import gaussian_kde
df = pd.read_csv('AlphaValuesCondition.csv')

df = df.replace("Eulaminated",1)  #'Control'
df = df.replace("Dislaminated",0) #'BCNU trated'

data1 = df[df['Group']==1]
data1 = data1[data1['Activity']=='Recovery 2']
data1 = data1['Aperiodic Value'].values

data2 = df[df['Group']==0]
data2 = data2[data2['Activity']=='Recovery 2']
data2 = data2['Aperiodic Value'].values

Kde1 = gaussian_kde(data1)
Kde2 = gaussian_kde(data2)
x = np.linspace(0, 1.5, 100)

kde1_values = Kde1(x)
kde2_values = Kde2(x)

kde1Norm = kde1_values / np.sum(kde1_values)
kde2Norm = kde2_values / np.sum(kde2_values)

plt.figure()
plt.plot(x, kde1Norm, label = 'Eulaminated', color = 'k', alpha = 0.5)
plt.plot(x, kde2Norm, label = 'Dislaminated', color = 'pink', alpha = 0.5)
plt.title('Aperiodic values with powerlaw Funct')
plt.xlabel('Aperiodic Values')
plt.ylabel('Density')
plt.grid('False')
plt.legend()
plt.savefig('AperiodicPowerLawRecovery2.svg')




####    Binomial distribution validation 

D1 = df[df['x'] == 5]
D1 = D1['AperiodicExp'].values

# Compute Bimodality Coefficient
def bimodality_coefficient(data):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    skewness = np.mean(((data - mean) / std_dev) ** 3)
    kurtosis = np.mean(((data - mean) / std_dev) ** 4) - 3  # Excess kurtosis

    bc = (skewness**2 + 1) / (kurtosis + 3 * ((n-1)**2) / ((n-2)*(n-3)))
    return bc, skewness, kurtosis

bc, skew, kurt = bimodality_coefficient(D1)


### dip stat < 0.05 - evidence of bimodality
### dip stat > 0.05 - likely unimodality

# Print results
print(f"Bimodality Coefficient: {bc}")
print(f"Skewness: {skew}")
print(f"Kurtosis (Excess): {kurt}")


### Hartigan-s Dip Test

### dip stat < 0.05 - strong evidence of bimodality
### dip stat > 0.05 - cannot reject unimodality



def dip_test(data, nbins = 10):
    hist, bin_edges = np.histogram(data, bins = nbins, density = True)
    
    #find local maximal modes
    modes = 0
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            modes += 1
    dip_stat = modes / nbins
    return dip_stat, modes

Dipstat, modes = dip_test(D1)


#%%

########   Correlation between epsilon values and bimodality Coeficient 


#### correlation matrix with seaborn 

df = pd.read_csv('EpsilonValues.csv')
data = df[["ID","Condition","Epsilon2","CB", 'Proportion']]


data1 = data.sort_values('Epsilon2', ascending = False)
data1 = data1[['Epsilon2','CB']].values

# sns.clustermap(data1, metric = 'euclidean', method='ward',)
from sklearn.linear_model import LinearRegression

# Example Data (for 8 subjects)
X = data1[:,0].reshape(-1, 1)  # Independent Variable
Y =  data1[:,1] # Dependent Variable

# Model
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

# Plot
plt.scatter(X, Y, color='blue', label='Original Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel(r"$\epsilon$")
plt.ylabel("Binomial Coefficient")
plt.legend()
plt.show()
# Regression Coefficients
print(f"Slope: {model.coef_[0]}, Intercept: {model.intercept_}")

### If slope is negative, Y(Binomial coeff) increase as X(epsilon decrease)


from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Example Data
X = data1[:,0].reshape(-1, 1)  # Independent Variable
Y =  data1[:,1] # Dependent Variable

# Model
model = LinearRegression()
model.fit(X, Y)

# Get values
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, Y)  # R^2 value
r, pval_ = pearsonr(X.flatten(), Y)  # Pearson Correlation Coefficient

# Print rvesults
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"Equation: Y = {slope}X + {intercept}")
print(f"R (Correlation Coefficient): {r}")
print(f"R² (Coefficient of Determination): {r_squared}")

# Plot
Y_pred = model.predict(X)
plt.figure()
plt.scatter(X, Y, color='blue', label='Original Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel("X Variable")
plt.ylabel("Y Variable")
plt.legend()
plt.show()




#%%
#               SLEPIAN MULTITAPERS 



def slepian_multitapers(d,fs, window, NW, tapers,overlap, electrodes):

    fs = fs # 1000
    timeWin =  window #1000
    timeWinID = int(np.round(timeWin/(window/fs)))
    tapers = dpss(timeWinID, NW, tapers, overlap) # 3, 6
    
    ### window fixed 
    LenWind = window #1000
    OverLap = int(LenWind/overlap) #75% #3
    electrodes = len(d)
    tap = len(tapers)
    # Channel = channel[:5000000]
    
    data = []
    for n in range(0,len(d[0,:]) - LenWind + 1, OverLap):
        for i in range(0,len(d)):
            data.append(d[i,n:n+LenWind])
            
    data = np.array(data)  #data 
    data = np.array([data[i:i+electrodes] for i in range(0,len(data),electrodes)])
    data = signal.detrend(data, axis =-1, type='constant')                          #remove the mean value of the signal
    
    
    taperD= [] #np.zeros([int(len(data)*2),LenWind])
    for i in range(0,len(data)):
        for j in range(0, len(tapers)):
            for n in range(0,len(data[0,:,:])):
                taperD.append(data[i,n,:] * tapers[j,:])       
    taperD = np.array(taperD)
    
    taperDd = np.array([taperD[i:i+tap] for i in range(0,len(taperD),tap)])    #len tapers 
    taperDd_ele = np.array([taperDd[i:i+electrodes] for i in range(0,len(taperDd),electrodes)])  #len electrodes
       
    Fft_Tap = np.fft.fft(taperDd_ele, axis = 3)
    # Fft_Tap = np.fft.fft(np.mean(taperDd,axis=1), axis = 1)
    
    ###   MEAN ALONG THE TAPERS  
    Fft_PowerT = np.mean(np.absolute(Fft_Tap), axis=2)
    
    # TAKE THE FREQUENCIES OF INTEREST 100 < Y EN ESPACIO LOGARITMICO
    f = np.linspace(0.5,window/2,(timeWinID//2))  #(0.5,1000/2,(timeWinID//2)
    F_t = np.argmin(np.abs(f-99))
    # Sxx = Sxx[:,0:F_t,:]
    f = f[0:F_t]
    #### log selection of frequencies 
    N = len(f)
    F_log = np.log10(np.logspace(f[0],f[-1], 60))                ###select only 60 F partitions
    
    Fid = []
    for i in range(0,len(F_log)):
      Fid.append(np.argmin(np.abs(f-F_log[i])))
      
    # f_Norm = np.mean(Fft_PoweT, axis=0)
    Fid = np.array(Fid).astype(int)
    f = np.take(f,Fid)
    Fft_PowerN = np.take(Fft_PowerT,Fid,axis = 2)
    f_Norm = np.mean(Fft_PowerN, axis=0)
    
    Fft_PowerNorm = []
    for i in range(0,electrodes):
        Fft_PowerNorm.append(np.log10(Fft_PowerN[:,i,:]/f_Norm[i,:]))
    
    Fft_PowerNorm = np.array(Fft_PowerNorm)
    # ff_F = np.mean(np.real(Fft_Tap), axis =1)
    
    time = np.linspace(1,300,len(taperDd_ele))
    Spectro = np.concatenate(Fft_PowerNorm, axis=1)                               ### What to covariate   axis = 0, time  axis=1, frequency
    return Fft_PowerNorm,time, f


# fs = 1000
# timeWin = 1000
# timeWinID = int(np.round(timeWin/(1000/fs)))

channel = DataFiltBP[56,:]               
u_T = int(len(DataFiltBP[0,:])/25000)                                                                        # how many samples are necesary to acquiere 900 seconds  at 1.25 kHz
Time = np.linspace(0,u_T,len(channel))


datos = signal.detrend(np.squeeze(DataFiltBP[:,:900000]))                          # channel 56

Fft_Pow, TimeTap, F_f = slepian_multitapers(datos, 1000,1000, 3, 6, 3, 59)
# plt.figure(1)
# plt.contourf(time, f, Fft_PowerNorm[2,:,:].T, 50, cmap ='seismic')
# # plt.contourf(time, f, Fft_PowerNorm[2,:int(LenWind//2)].T, 50, cmap ='jet')
# plt.ylim(0.5,100)
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time(s)')
# plt.colorbar()

#%%

#####   PLOTS OF SOME spectrogram  ELECTRODES 

# SpecZero = np.zeros((59,898,60))

Fft_NormZero = np.insert(Fft_Pow,0,0, axis=0)
Fft_NormZero = np.insert(Fft_NormZero,7,0, axis=0)
Fft_NormZero = np.insert(Fft_NormZero,24,0, axis=0)
Fft_NormZero = np.insert(Fft_NormZero,56,0, axis=0)
Fft_NormZero = np.insert(Fft_NormZero,63,0, axis=0)

fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(16, 12))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("ALL electrodes 003", fontsize=12, y=0.95)

# fig.colorbar(orientation='vertical')

axs = axs.ravel()

for d in range(64):
    # filter df for ticker and plot on specified axes
    axs[d].contourf(TimeTap, F_f, Fft_NormZero[d,:,:].T, 50, cmap ='seismic',
                    vmin= -1, vmax=1)
    axs[d].set_ylim(0,100)
    # chart formatting
    # axs[d].get_legend().remove()
    axs[d].set_title(str(d))
    
    

#%%
'''
######    covar from EVERY ELECTRODE FREQUENCY
####            CovE = F*T
'''

electrodes = len(DataFiltBP)

CovElec = []
for i in range(electrodes):
    CovElec.append(np.cov(Fft_Pow[i,:,:].T))
    
CovElec = np.array(CovElec)

#PCA per electrode
eigval = np.zeros((electrodes,60))
eigvec = np.zeros((electrodes,60,60))

for i in range(electrodes):
    eigval[i,:], eigvec[i,:,:] = np.linalg.eigh(CovElec[i,:,:])

indexes = np.flip(np.argsort(eigval[0,:]))
eigval = eigval[:,indexes]
eigvec = eigvec[:,:,indexes]

maximum = np.max(np.abs(eigvec))
minimum = np.min(np.abs(eigvec))
Maximum = np.max(np.abs(CovElec))

#plt.plot(np.cumsum(eigval[0,:])/np.sum(eigval[0,:]))AP016-Basal-BCNU-DWSAMPLE
#  pca pesado a las frecuencias
eigvecE = eigvec[:,:,0].T ## PC1
eigvecE2 = eigvec[:,:,1].T ## PC2
MatPC1 = eigvecE * F_f [:,None]    #eigval, frequency 
MatPC2 = eigvecE2 * F_f[:,None]     #eigval, frequency 

CovMat = []
for i in range(electrodes):
  for j in range(electrodes):
    CovMat.append(MatPC1[:,i]* MatPC1[:,j])
    
CovMat = np.array(CovMat)
CovMat = np.array([CovMat[i:i+electrodes] for i in range(0,len(CovMat),electrodes)]) 
CovMat = np.mean(CovMat, axis=2)
# CovMat = np.cov(CovMat)

eigval1, eigvec1 = np.linalg.eigh(CovMat)
indexes = np.flip(np.argsort(eigval1))
eigval1 = eigval1[indexes]
eigvec1 = eigvec1[:, indexes]
maximum = np.max(np.abs(eigvec1))
#%%
### plot variance explained
tot = sum(eigval[0,:])
var_exp = [(i / tot) for i in sorted(eigval[0,:], reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plot explained variances
plt.figure(5)
plt.subplot(121)
plt.bar(range(60), var_exp, color= 'k',alpha=0.3,
        align='center', label='Varianza individual')
plt.ylim(0,0.2)
plt.ylabel('Proportion of variance')
plt.xlabel('Modes')
plt.legend(loc='best')
plt.box(False)
plt.grid(False)


plt.subplot(122)
plt.step(range(60), cum_var_exp, where='mid',
         label='Varianza acumulada',color= 'k',alpha = 0.3)
plt.ylabel('Proportion of variance')
plt.xlabel('Modes')
plt.legend(loc='best')
plt.box(False)
plt.grid(False)
plt.show()
plt.savefig('VarBaseDESLAM.svg')

#%%  


#### SCATTER BY POSITION 

d = {'Electrodes': [], 'Layer' :[]}   #
dataF = pd.DataFrame(d)               #
dataF['PC1'] = np.abs(eigvec1[:,0].T) #              # Principal components 1 and 2, choice
dataF['PC2'] = np.abs(eigvec1[:,1].T) # 

plt.figure(figsize = (10,10))
plt.scatter(dataF['PC1'],dataF['PC2'],
            c = np.arange(0,electrodes,1), s =300,      # c must be a float not int number, that because we multiply 0.5
            cmap = 'Set2', alpha = 0.9)
plt.colorbar()
plt.box(False)

plt.style.use('fivethirtyeight')
plt.figure()
plt.plot(dataF['PC1'].values, label = 'Mode 1')
plt.plot(dataF['PC2'].values, label = 'Mode 2')
plt.legend()
# plt.savefig('005Mode1and2.svg')

#%%

## Mean shift clustering

from sklearn.cluster import MeanShift, estimate_bandwidth

Coord  =pd.read_csv('CoordMEA.csv')

X = np.array((dataF['PC1'].values, dataF['PC2'].values))
bandwidth = estimate_bandwidth(X.T, quantile = 0.45, n_samples=80) 
clustering = MeanShift(bandwidth=bandwidth).fit(X.T)

labels = clustering.labels_
center = clustering.cluster_centers_

####   SCATTER BY MEAN SHIFT CLUSTERING 
plt.figure(figsize = (6,5))
plt.clf()
plt.scatter(dataF['PC1'],dataF['PC2'], c = labels.astype(float)*0.5,  s =300,
            alpha =0.9)
plt.scatter(center[:,0], center[:,1], c= 'r', marker = 'x', alpha = 0.9)
plt.title("011-basal ")
plt.colorbar()
plt.box(False)
# plt.savefig("011-basalPCA.svg")
### Scatter by real coord
plt.figure(figsize = (6,5))
plt.clf()
plt.scatter(Coord['X'],Coord['Y'], c = labels.astype(float)*0.5,  s =300,
            alpha =0.9)
#plt.scatter(center[:,0], center[:,1], c='r', marker = 'x', alpha = 0.9)
plt.title("010-basal ")
plt.colorbar()
plt.box(False)
# plt.savefig("010-basalPCAclus.svg")


#%%

####   Extract the signals of the groups 

Label_list = list(labels)

def countX(lst,x):
    count = 0
    for ele in lst:
        if (ele==x):
            count = count+1
    return count

Counts = []

for i in range(len(center)):
    Counts.append(countX(Label_list, i))
    
Electrodes = 59 

# Comment or uncomment
Cluster1 = np.array(np.where(labels == 0))
Cluster2 = np.array(np.where(labels == 1))
Cluster3 = np.array(np.where(labels == 2))


### FILTER AGAIN THE SIGNAL BUT THIS TIME USING THE COMMON BAND
filename="AP004-cero-CTRL-DWSAMPLE"
DownData = np.loadtxt(filename + ".csv",delimiter=',')                         # DataFilt to 200 Hz

fs = 1000
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
# fs = 1000
### common band 
lowcut = 0.5                                                           #Elegir la banda a estudiar
highcut = 30

filtered = []
for i in range(0,60):
    filtered.append(butter_bandpass_filter(DownData[i], lowcut, highcut, fs, order = 3))
    
DataFiltBP = np.array(filtered)

# filteredB = []
# for i in range(0,60):
#     filteredB.append(butter_bandpass_filter(DownDataBC[i], lowcut, highcut, fs, order = 3))
    
# DataFiltBP_BC = np.array(filteredB)

Array = [23,25,28,31,34,36,20,21,24,29,30,35,38,39,18,19,22,
          27,32,37,40,41,15,16,17,26,33,42,43,44,
          14,13,12,3,56,47,46,45,11,10,7,2,57,52,
          49,48,9,8,5,0,59,54,51,50,6,4,1,58,55,53]


idx = np.empty_like(Array)
idx[Array] = np.arange(len(Array))
# Time = np.linspace(0,900,len(Dchannel))

DataFiltBP[:] = DataFiltBP[idx,:]
DataFiltBP = np.delete(DataFiltBP, 30,0)


### extract electric signal / RUN BAND PASS FROM MUTUALINFO SCRIPT

Act1 = DataFiltBP[Cluster1].reshape(len(Cluster1[0,:]),len(DataFiltBP[0,:]))
Act2 = DataFiltBP[Cluster2].reshape(len(Cluster2[0,:]),len(DataFiltBP[0,:]))
Act3 = DataFiltBP[Cluster3].reshape(len(Cluster3[0,:]),len(DataFiltBP[0,:]))




#%%
"""
####   covar from electrodes, under the assumption of every electrode are 
 #######       TIME FIXED  CovT = F * E
"""  
    
TimeW = int(len(Fft_Pow[0,:,:]))

CovSpec = [] 
for i in range(0,len(Fft_Pow[0,:,:])):
    CovSpec.append(np.cov(Fft_Pow[:,i,:].T))
    
CovSpec = np.array(CovSpec)
eigval = np.zeros((TimeW,60))
eigvec = np.zeros((TimeW,60,60))

for i in range(TimeW):
    eigval[i,:],eigvec[i,:,:] = np.linalg.eigh(CovSpec[i,:,:])
    
indexes = np.flip(np.argsort(eigval[0,:]))
eigval = eigval[:,indexes]
eigvec = eigvec[:,:,indexes]


####### plot spectrogram per PCA ELECTRODE CovTime result 

fig,axs = plt.subplots(nrows = 1, ncols = 4, figsize=(8,6))

fig.suptitle("First 4 components - Control" , fontsize=12, y=0.95)
axs = axs.ravel()
for d in range(4):
    axs[d].pcolormesh(TimeTap,F_f[:],np.abs(eigvec[:,:,d].T),
                      vmin = 0, vmax=0.8, cmap = 'rainbow')
    axs[d].set_ylim(0,25)
    axs[d].set_title(str(d))

###### variance cumulative distr
# plt.figure(2)
# plt.plot(np.cumsum(eigval[0,:])/ np.sum(eigval[0,:]), 'ro')
# plt.box(False)

tot = sum(eigval[0,:])
var_exp = [(i / tot) for i in sorted(eigval[0,:], reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plot explained variances
plt.figure(7)
plt.bar(range(60), var_exp, color ='r',alpha=0.5,
        align='center', label='Varianza individual')
plt.step(range(60), cum_var_exp, where='mid',
         label='Varianza acumulada',color = 'r',alpha = 0.5)
plt.ylabel('Proportion of variance')
plt.xlabel('Modes')
plt.legend(loc='best')
plt.box(False)
plt.show()

#####

eigvecMean = np.mean(eigvec, axis=0)
plt.figure()
plt.plot(np.abs(eigvecMean[:,0].T), label = 'Mode 1')
plt.plot(np.abs(eigvecMean[:,1].T), label = 'Mode 2')
plt.legend()
plt.savefig('004Mode1and2BAND.svg')
#%%

# #### covar Cov = E * F

# CovEF = [] 
# for i in range(0,len(Fft_PowerNorm[:,0,:])):
#     CovEF.append(np.cov(Fft_PowerNorm[i,:,:]))
# CovEF = np.array(CovEF)

# # CovSpecW = CovSpec[:,:-1,:-1] * CovEF

# eigval = np.zeros((electrodes,TimeW))
# eigvec = np.zeros((electrodes,TimeW,TimeW))

# for i in range(electrodes):
#     eigval[i,:],eigvec[i,:,:] = np.linalg.eigh(CovEF[i,:,:])
    
# indexes = np.flip(np.argsort(eigval[0,:]))
# eigval = eigval[:,indexes]
# eigvec = eigvec[:,:,indexes]


#%%
####   covar from electrodes, under the assumption of every electrode are 
 #######         Cov = E * E, sumado en las frecuencias, por cada tiempo

Fft_PowerB0M = np.concatenate((Fft_PowerNorm, Fft_PowerNorm2, Fft_PowerNorm3), axis= 1)

time = np.linspace(1,2100,len(Fft_PowerB0M[0,:,:]))    
    
TimeW = int(len(Fft_PowerB0M[0,:,:]))

CovE = [] 
for i in range(0,TimeW):
    CovE.append(np.cov(Fft_PowerB0M[:,i,:]))
    
CovE = np.array(CovE)
eigval = np.zeros((TimeW, electrodes))
eigvec = np.zeros((TimeW,electrodes,electrodes))

for i in range(TimeW):
    eigval[i,:],eigvec[i,:,:] = np.linalg.eigh(CovE[i,:,:])
    
indexes = np.flip(np.argsort(eigval[0,:]))
eigval = eigval[:,indexes]
eigvec = eigvec[:,:,indexes]


fig,axs = plt.subplots(nrows = 1, ncols = 3, figsize=(12,6))

fig.suptitle("PC 1-3, Cov E,T CTRL 004" , fontsize=12, y=0.95)
axs = axs.ravel()
for d in range(3):
    axs[d].pcolormesh(time,np.arange(0,electrodes,1),np.abs(eigvec[:,:,d].T),
                      vmin = 0, vmax = 0.8,
                       cmap = 'jet')
    axs[d].set_ylabel("electrodes")
    axs[d].set_xlabel("Time s")
    axs[d].axvline(x = 300, ymin = 0, ymax = 59, c ='red', linewidth = 2, alpha = 0.6)
    axs[d].axvline(x = 1200, ymin = 0, ymax = 59, c ='yellow', linewidth = 2, alpha = 0.6)
    # axs[d].set_ylim(0,59)
    axs[d].set_title(str(d))


####### plot spectrogram per PCA ELECTRODE CovTime result 

fig,axs = plt.subplots(nrows = 2, ncols = 4, figsize=(8,6))

fig.suptitle("First 6 components - Control" , fontsize=12, y=0.95)
axs = axs.ravel()
for d in range(8):
    axs[d].pcolormesh(time,f[:],np.abs(eigvec[:,:,d].T),
                      vmin = 0, vmax=0.8, cmap = 'jet')
    axs[d].set_ylim(0,50)
    axs[d].set_title(str(d))





