import FVD
import GLDAS as gld
import worldclim
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_top_5(matrix):
    top_5 = np.zeros((len(matrix), 5))
    for row in range(len(matrix)):
        top_5[row,:] = (-matrix[row,:]).argsort()[:5]
    return top_5

n_comp = 10
sk_pca = PCA(n_components=n_comp)

pct = 15
val = 1
nrows = 720
ncols = 1440

os.chdir("matfiles")
mask = FVD.loadmat('land_mask.mat')

F = FVD.loadmat('F_' + str(pct) + '.mat')
D = FVD.loadmat('D_' + str(pct) + '.mat')
F[(abs(F)<val) & (abs(D)<val)] = float('nan')
D[(abs(F)<val) & (abs(D)<val)] = float('nan')
full = abs(F) / (abs(F) + abs(D))

#Xa = gld.prepare_PCA_mat_with_time_series(nrows, ncols)

Xa = gld.prepare_mat_with_statistics(nrows, ncols)
print(Xa.shape) # rows are pixels, columns are time series variables (features)

counter = np.arange(Xa.shape[0]) # holds locations of pixels
inds = ~np.isnan(Xa).any(axis=1)
fvd_vec = full.reshape(-1,1)[inds]
Xa = Xa[inds]
counter = counter[inds]
print(Xa.shape)
#X_stan = StandardScaler().fit_transform(Xa) # only need this if using time series
Y = sk_pca.fit_transform(Xa)

var_explained = sk_pca.explained_variance_ratio_
print(np.cumsum(var_explained))
factors = sk_pca.components_
print(abs(factors))
plt.figure()
plt.imshow(abs(factors), cmap='RdBu')
plt.colorbar()
plt.show()

plt.figure(figsize=(5,5))
#plt.plot(var_explained, '-o', c='royalblue', linewidth=1.5, label='individual PC')
plt.plot(np.cumsum(var_explained), '--o', c='black', linewidth=1)#, label='cumulative')
plt.legend(loc='best')
plt.ylabel('cumulative variance explained')
plt.xlabel('PC')
plt.show()

#worldclim.plot_scatter(Y[:,1].reshape(-1,1), fvd_vec, 'pc2')

'''print(os.getcwd())
for comp in range(1, n_comp+1):
    print(comp)
    PC_map = gld.map_determined_PC(Y, comp, counter, nrows, ncols)
    switcher = {
        1: 60, 2: 50, 3: 30, 4: 40, 5: 25, 6: 10, 7: 10, 8: 10, 9: 10, 10: 15
    }
    FVD.savemat(PC_map, 'PC' + str(comp) + '_TS.mat')'''
    #gld.plot_scene(PC_map, np.nanmin(PC_map), switcher.get(comp), mask=mask, name=str(comp))
