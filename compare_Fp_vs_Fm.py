import FVD
import numpy as np
import GLDAS as gld
import os
import glob
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

def plot_violin(data, ylab):
    plt.figure(figsize=(4,4))
    plt.violinplot(data, showmeans=True)
    x = [1, 2]
    labels = ['D- pixels\n(NDVI decreases)', 'D+ pixels\n(NDVI increases)']
    plt.xticks(x, labels)
    plt.ylabel(ylab)
    plt.tight_layout()
    #plt.show()
    plt.savefig('../../plots/violin_' + ylab + '_D.pdf')
    return

def main():
    os.chdir("matfiles")
    mask = FVD.loadmat('land_mask.mat') # mask for land surface
    npx = np.nansum(mask==0)

    pct = 15
    val = 0.1
    nrows = 720
    ncols = 1440

    D = FVD.loadmat('quads_D_v.1_e.25.mat')
    X = FVD.loadmat('X_GLDAS.mat')

    temp = X[:,0].reshape(nrows,ncols)
    rad = X[:,1].reshape(nrows,ncols)
    prec = X[:,2].reshape(nrows,ncols)
    PET = X[:,3].reshape(nrows,ncols)
    LAI = X[:,4].reshape(nrows,ncols)

    M = [temp[D==1], temp[D==2], rad[D==1], rad[D==2], prec[D==1], prec[D==2],
        PET[D==1], PET[D==2], LAI[D==1], LAI[D==2]]

    for num in [1,3,5,7,9]:
        switcher = {1: 'temperature', 3: 'radiation', 5: 'precipitation', 7: 'PET', 9: 'LAI'}
        cleaned_1 = np.copy(M[num-1])
        cleaned_1 = cleaned_1[~np.isnan(M[num-1])]

        cleaned_2 = np.copy(M[num])
        cleaned_2 = cleaned_2[~np.isnan(M[num])]
        plot_violin([cleaned_1, cleaned_2], switcher.get(num))

    return

if __name__ == "__main__":
    main()
