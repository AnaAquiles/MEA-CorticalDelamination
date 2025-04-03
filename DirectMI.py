# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:01:27 2024

@author: aaquiles
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

start = time.time()


def calc_MI(X,Y):
    
    binsY = np.histogram_bin_edges(Y,bins = 'sturges', range =(0,5))
    binsX = np.histogram_bin_edges(X,bins = 'sturges', range =(0,5))
    
    c_XY = np.histogram2d(X,Y,binsX)[0]
    c_X = np.histogram(X,binsX)[0]
    c_Y = np.histogram(Y,binsY)[0]
 
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
 
    MI = H_X + H_Y - H_XY
    return MI
 
def shan_entropy(c):
    c_normalized = np.nan_to_num(c / float(np.sum(c)))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H



def compute_directionality(set_a, set_b, lags=np.arange(-50, 51)):

    """
    Compute directionality between two signal sets (Set A -> Set B and Set B -> Set A).
    Returns the directionality flow and best lags for each signal pair.  using the full band -   
    """
    results = {
        "A_to_B": np.zeros((len(set_a), len(set_b), len(lags))),
        "B_to_A": np.zeros((len(set_b), len(set_a), len(lags))),
        "best_lags_A_to_B": np.zeros((len(set_a), len(set_b))),
        "best_lags_B_to_A": np.zeros((len(set_b), len(set_a))),
    }
    
    for i, a_signal in enumerate(set_a):
        for j, b_signal in enumerate(set_b):
            mi_a_to_b = []
            mi_b_to_a = []
            for lag in lags:
                if lag < 0:
                    lagged_b = np.roll(b_signal, lag)
                    lagged_a = a_signal
                else:
                    lagged_a = np.roll(a_signal, lag)
                    lagged_b = b_signal
                mi_a_to_b.append(calc_MI(lagged_a, lagged_b))
                mi_b_to_a.append(calc_MI(lagged_b, lagged_a))
            results["A_to_B"][i, j, :] = mi_a_to_b
            results["B_to_A"][j, i, :] = mi_b_to_a
            results["best_lags_A_to_B"][i, j] = lags[np.argmax(mi_a_to_b)]
            results["best_lags_B_to_A"][j, i] = lags[np.argmax(mi_b_to_a)]
    
    return results

# Generate synthetic signals
# np.random.seed(42)
# t = np.linspace(0, 10, 1000)

# # Signal sets
# set_A = [np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000),  # A1
#          np.cos(2 * np.pi * t) + 0.1 * np.random.randn(1000)]  # A2

# set_B = [np.roll(set_A[0], 5) + 0.1 * np.random.randn(1000),   # B1
#          np.roll(set_A[1], -10) + 0.1 * np.random.randn(1000)] # B2

# # Compute directionality 
lags9 = np.arange(-100, 100)
results9 = compute_directionality(Act1, Act3, lags9)


### save the result dictionary
with open('004DirC1-C3mg1.pkl', 'wb') as fp:
    pickle.dump(results9, fp)
              

# Aggregate and plot results
A_to_B = results9["A_to_B"]
B_to_A = results9["B_to_A"]

# Plot example pair (A1 -> B1)
plt.figure(figsize=(10, 6))
plt.plot(lags9, A_to_B[0, 0, :], label="MI(A1 -> B1)", color="blue")
plt.plot(lags9, B_to_A[0, 0, :], label="MI(B1 -> A1)", color="red")
plt.axvline(0, color='black', linestyle='--', label='Zero lag')
plt.xlabel("Lag (time steps)")
plt.ylabel("Mutual Information")
plt.title("Directionality Flow: A1 <-> B1")
plt.legend()
plt.show()

# Print best lags for all pairs
print("Best Lags (A -> B):")
print(results9["best_lags_A_to_B"])
print("Best Lags (B -> A):")
print(results9["best_lags_B_to_A"])

print('It took', time.time()-start,'seconds.')

#%%%


"""
      PLOTS MEAN BITS ALONG EVERY CLUSTER INTERACTION

"""
### load the result dictionary

# 

# plt.figure(figsize=(10,6))
# plt.subplot(121)
# plt.imshow(CastAB, aspect = 'auto', vmin = 0, vmax = 1)
# plt.title('Cluster 1 to Cluster 2')
# plt.colorbar()
# plt.grid(False)
# plt.subplot(122)
# plt.imshow(CastBA, aspect = 'auto', vmin = 0, vmax = 1)
# plt.colorbar()
# plt.title('Cluster 2 to Cluster 1')
# plt.grid(False)
# plt.savefig('005DirectMgFree.png')
# ### Boxplots 


### Create a data frame with using the directions inside of each .pkl
with open('004DirC1-C2mg.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
##### 1-2
Df = pd.DataFrame()
Df['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df['Direction'] = '1-2'

Df1 = pd.DataFrame()
Df1['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df1['Direction'] = '2-1'

with open('004DirC2-C3mg.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
######  2-3
Df2 = pd.DataFrame()
Df2['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df2['Direction'] = '2-3'

Df3 = pd.DataFrame()
Df3['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df3['Direction'] = '3-2'

with open('004DirC1-C3mg.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
########  1-3
Df4 = pd.DataFrame()
Df4['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df4['Direction'] = '1-3'

Df5 = pd.DataFrame()
Df5['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df5['Direction'] = '3-1'

#### concat all the times in only one array and name the number of record
af = pd.concat([Df,Df1, Df2, Df3, Df4,Df5])
af['t'] = 1


##########################################################################
with open('004DirC1-C2mg1.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
##### 1-2
Df = pd.DataFrame()
Df['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df['Direction'] = '1-2'

Df1 = pd.DataFrame()
Df1['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df1['Direction'] = '2-1'

with open('004DirC2-C3mg1.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
######  2-3
Df2 = pd.DataFrame()
Df2['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df2['Direction'] = '2-3'

Df3 = pd.DataFrame()
Df3['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df3['Direction'] = '3-2'

with open('004DirC1-C3cero1.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
########  1-3
Df4 = pd.DataFrame()
Df4['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df4['Direction'] = '1-3'

Df5 = pd.DataFrame()
Df5['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df5['Direction'] = '3-1'

bf = pd.concat([Df,Df1, Df2, Df3, Df4,Df5])
bf['t'] = 2

#############################################################################
with open('004DirC1-C2cero3.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
##### 1-2
Df = pd.DataFrame()
Df['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df['Direction'] = '1-2'

Df1 = pd.DataFrame()
Df1['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df1['Direction'] = '2-1'

with open('004DirC2-C3cero3.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
######  2-3
Df2 = pd.DataFrame()
Df2['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df2['Direction'] = '2-3'

Df3 = pd.DataFrame()
Df3['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df3['Direction'] = '3-2'

with open('004DirC1-C3cero3.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)
########  1-3
Df4 = pd.DataFrame()
Df4['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df4['Direction'] = '1-3'

Df5 = pd.DataFrame()
Df5['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df5['Direction'] = '3-1'

cf = pd.concat([Df,Df1, Df2, Df3, Df4,Df5])
cf['t'] = 3

Alldf = pd.concat([af,bf,cf])
# sns.boxplot(x = "Direction", y="MeanMI", data = df)


plt.figure(1)
sns.pointplot(data = Alldf, x="Direction", y = "MeanMI", markers = 'd',
             hue = 't'
              )
plt.title('Mg free evolution 004')

#%%
#   pandas dataframe for evolutionary MI arrays 

with open('005DirC1-C2mg1.pkl', 'rb') as fp:
    resultsP = pickle.load(fp)

CastAB = np.mean(resultsP["A_to_B"], axis = 2)
CastBA = np.mean(resultsP["B_to_A"], axis = 2)

Df12 = pd.DataFrame()
Df12['MeanMI'] = np.abs(CastAB).reshape(len(CastAB[0,:])*len(CastAB[:,0]))
Df12['Direction'] = '1-2'
Df12['Time'] = 'Mg 2'

Df13 = pd.DataFrame()
Df13['MeanMI'] = np.abs(CastBA).reshape(len(CastBA[0,:])*len(CastBA[:,0]))
Df13['Direction'] = '2-1'
Df13['Time'] = 'Mg 2'

df= pd.concat([Df,Df1, Df2, Df3, Df4,Df5, Df6, Df7,Df8, Df9, Df10, Df11, Df12, Df13])

sns.set_theme(style="whitegrid")


sns.violinplot(data = df, x = 'Time', y = 'MeanMI', hue = 'Direction',
               split = True, inner ="quart", fill = False,
               palette = {"1-2": "g", "2-1": ".35"},)

#%%%


import numpy as np
from tqdm import tqdm

def calc_MI(X, Y):
    binsY = np.histogram_bin_edges(Y, bins='sturges', range=(0, 5))
    binsX = np.histogram_bin_edges(X, bins='sturges', range=(0, 5))
    
    c_XY = np.histogram2d(X, Y, bins=[binsX, binsY])[0]
    c_X = np.histogram(X, binsX)[0]
    c_Y = np.histogram(Y, binsY)[0]
    
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    
    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    c_normalized = np.nan_to_num(c / float(np.sum(c)))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))  
    return H

def permutation_test(X, Y, num_permutations=1000):
    observed_mi = calc_MI(X, Y)
    permuted_mis = np.zeros(num_permutations)
    
    for i in range(num_permutations):
        np.random.shuffle(Y)  # Shuffle Y to break any true dependency
        permuted_mis[i] = calc_MI(X, Y)
    
    p_value = np.sum(permuted_mis >= observed_mi) / num_permutations
    return observed_mi, p_value

def compute_directionality(set_a, set_b, lags=np.arange(-50, 51), num_permutations=1000, alpha=0.05):
    """
    Compute directionality between two signal sets (Set A -> Set B and Set B -> Set A) with statistical testing.
    Now includes tqdm progress bars.
    """
    results = {
        "A_to_B": np.zeros((len(set_a), len(set_b), len(lags))),
        "B_to_A": np.zeros((len(set_b), len(set_a), len(lags))),
        "best_lags_A_to_B": np.zeros((len(set_a), len(set_b))),
        "best_lags_B_to_A": np.zeros((len(set_b), len(set_a))),
        "p_values_A_to_B": np.ones((len(set_a), len(set_b), len(lags))),
        "p_values_B_to_A": np.ones((len(set_b), len(set_a), len(lags))),
    }
    
    for i, a_signal in enumerate(tqdm(set_a, desc="Processing Set A")):
        for j, b_signal in enumerate(tqdm(set_b, desc=f"Processing Set B [{i+1}/{len(set_a)}]", leave=False)):
            mi_a_to_b = []
            mi_b_to_a = []
            p_a_to_b = []
            p_b_to_a = []
            
            for lag in tqdm(lags, desc=f"Lags [{i+1},{j+1}]", leave=False):
                if lag < 0:
                    lagged_b = np.roll(b_signal, lag)
                    lagged_a = a_signal
                else:
                    lagged_a = np.roll(a_signal, lag)
                    lagged_b = b_signal
                
                mi_ab, p_ab = permutation_test(lagged_a, lagged_b, num_permutations)
                mi_ba, p_ba = permutation_test(lagged_b, lagged_a, num_permutations)
                
                mi_a_to_b.append(mi_ab)
                mi_b_to_a.append(mi_ba)
                p_a_to_b.append(p_ab)
                p_b_to_a.append(p_ba)
            
            results["A_to_B"][i, j, :] = mi_a_to_b
            results["B_to_A"][j, i, :] = mi_b_to_a
            results["p_values_A_to_B"][i, j, :] = p_a_to_b
            results["p_values_B_to_A"][j, i, :] = p_b_to_a
            
            # Choose best lag based on MI, but only if significant
            significant_mis_A_to_B = np.array(mi_a_to_b) * (np.array(p_a_to_b) < alpha)
            significant_mis_B_to_A = np.array(mi_b_to_a) * (np.array(p_b_to_a) < alpha)
            
            if np.any(significant_mis_A_to_B):
                results["best_lags_A_to_B"][i, j] = lags[np.argmax(significant_mis_A_to_B)]
            else:
                results["best_lags_A_to_B"][i, j] = np.nan  # No significant causality
            
            if np.any(significant_mis_B_to_A):
                results["best_lags_B_to_A"][j, i] = lags[np.argmax(significant_mis_B_to_A)]
            else:
                results["best_lags_B_to_A"][j, i] = np.nan  # No significant causality
    
    return results



# # Compute directionality
lags9 = np.arange(-100, 100)
results9 = compute_directionality(Act1, Act2, lags9)


### save the result dictionary
with open('004DirC1-C2cero1.pkl', 'wb') as fp:
    pickle.dump(results9, fp)
              

# Aggregate and plot results
A_to_B = results9["A_to_B"]
B_to_A = results9["B_to_A"]

# Plot example pair (A1 -> B1)
plt.figure(figsize=(10, 6))
plt.plot(lags9, A_to_B[0, 0, :], label="MI(A1 -> B1)", color="blue")
plt.plot(lags9, B_to_A[0, 0, :], label="MI(B1 -> A1)", color="red")
plt.axvline(0, color='black', linestyle='--', label='Zero lag')
plt.xlabel("Lag (time steps)")
plt.ylabel("Mutual Information")
plt.title("Directionality Flow: A1 <-> B1")
plt.legend()
plt.show()



