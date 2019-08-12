import numpy as np
import FVD
import os
import glob
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde

def plot_hist(dataBefore, dataDuring, dataAfter, title):
    dataBefore[dataBefore==0] = float('nan')
    dataBefore = dataBefore[np.isfinite(dataBefore)].reshape(-1,1)

    dataDuring[dataDuring==0] = float('nan')
    dataDuring = dataDuring[np.isfinite(dataDuring)].reshape(-1,1)

    dataAfter[dataAfter==0] = float('nan')
    dataAfter = dataAfter[np.isfinite(dataAfter)].reshape(-1,1)

    mx = np.nanmax([np.nanmax(dataBefore), np.nanmax(dataDuring), np.nanmax(dataAfter)])
    mn = np.nanmin([np.nanmin(dataBefore), np.nanmin(dataDuring), np.nanmin(dataAfter)])

    kde_Before = gaussian_kde(dataBefore)
    #kde_Fm = gaussian_kde(dataFm)
    x = np.linspace(mn, mx, 100)

    plt.figure(figsize=(6,3))
    '''plt.hist(dataBefore, bins=30, range=[mn-0.05,mx+0.05],
        color='orangered', ec='white', rwidth=0.92, alpha=0.7, label='Before')
    plt.hist(dataDuring, bins=30, range=[mn-0.05,mx+0.05],
        color='cornflowerblue', ec='white', rwidth=0.92, alpha=0.7, label='During')
    plt.hist(dataAfter, bins=30, range=[mn-0.05,mx+0.05],
        color='limegreen', ec='white', rwidth=0.92, alpha=0.7, label='After')'''
    plt.plot(x, kde_Before(x))
    plt.xlabel('percentile of temperature at pixel')
    plt.ylabel('number of pixels')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return

def main():
    os.chdir("matfiles")

    pct = 15
    val = 0.1
    nrows = 720
    ncols = 1440
    nobs = 12

    temp_1mo_before = FVD.loadmat('temp_1mo_before.mat')
    temp_1mo_after = FVD.loadmat('temp_1mo_after.mat')
    temp_during = FVD.loadmat('temp_during.mat')

    plot_hist(temp_1mo_before, temp_during, temp_1mo_after, 'temperature near soil moisture anomalies')

    return()

if __name__ == "__main__":
    main()
