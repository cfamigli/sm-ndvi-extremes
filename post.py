import FVD
import numpy as np
import os
import glob
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde

def plot_hist(data1, data2, mask):
    data1[mask==0] = float('nan')
    data2[mask==0] = float('nan')
    data1[data1==-9999] = float('nan')
    data2[data2==-9999] = float('nan')
    data1 = data1[np.isfinite(data1)].reshape(-1,)
    data2 = data2[np.isfinite(data2)].reshape(-1,)
    rng = [-0.25,0.25]
    plt.figure(figsize=(4,4))
    #plt.hist(data1, bins=20, range=rng, color='dodgerblue', edgecolor='white', density=True, alpha=0.7)
    plt.hist(data2, bins=20, range=rng, color='crimson', edgecolor='white', density=True, alpha=0.7)
    '''kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)
    plt.plot(np.linspace(rng[0], rng[1], 100), kde1(np.linspace(rng[0], rng[1], 100)), c='dodgerblue')
    plt.plot(np.linspace(rng[0], rng[1], 100), kde2(np.linspace(rng[0], rng[1], 100)), c='crimson')'''
    plt.tight_layout()
    plt.show()
    return

def main():
    os.chdir("matfiles")
    mask = FVD.loadmat('land_mask.mat') # mask for land surface
    npx = np.nansum(mask==0)

    pct = 15
    val = 0.1
    nrows = 720
    ncols = 1440

    sm_den = FVD.loadmat('sm_denominator_monthly.mat')
    sm_num = FVD.loadmat('sm_numerator_monthly.mat')

    npx_low = FVD.loadmat('num_anoms_low.mat')
    npx_high = FVD.loadmat('num_anoms_high.mat')

    #FVD.plot_scene(npx_low, 1,50, mask=mask)
    #FVD.plot_scene(npx_high, 1,50, mask=mask)

    '''temp_den = FVD.loadmat('temp_denominator_ERA5.mat')
    temp_num = FVD.loadmat('temp_numerator_ERA5.mat')

    rad_den = FVD.loadmat('rad_denominator_ERA5.mat')
    rad_num = FVD.loadmat('rad_numerator_ERA5.mat')'''

    '''#FVD.plot_fvd(F, pct, -10, 10, mask=mask, kind='full')
    FVD.plot_fvd(temp_num, pct, -5,5, mask=mask, kind='full')

    #FVD.plot_fvd(D, pct, -10, 10, mask=mask, kind='full')
    FVD.plot_fvd(temp_den, pct, -5,5, mask=mask, kind='full')'''

    '''sm_den_largest = np.copy(sm_den)
    sm_num_largest = np.copy(sm_num)'''

    sm_den_controlled = FVD.loadmat('sm_denominator_monthly_controlled.mat')
    #sm_num_controlled = FVD.loadmat('sm_numerator_monthly_controlled.mat')
    sm_num_controlled = FVD.loadmat('nansum_f_fsub.mat')

    sm_den_controlled_copy = np.copy(sm_den_controlled)
    sm_num_controlled_copy = np.copy(sm_num_controlled)
    #sm_den_controlled_copy[npx_low<5] = float('nan')
    #sm_num_controlled_copy[npx_high<5] = float('nan')
    sm_den_controlled_copy[(abs(sm_den_controlled)<val) & (abs(sm_num_controlled)<val)] = float('nan')
    sm_num_controlled_copy[(abs(sm_den_controlled)<val) & (abs(sm_num_controlled)<val)] = float('nan')

    #npx_F = np.copy(sm_num_controlled)
    #npx_F[abs(sm_num_controlled) < abs(sm_den_controlled)] = float('nan')

    '''sm_den_largest[(abs(sm_den)<abs(temp_den)) | (abs(sm_den)<abs(rad_den))] = float('nan')
    sm_num_largest[(abs(sm_num)<abs(temp_num)) | (abs(sm_num)<abs(rad_num))] = float('nan')

    FVD.plot_fvd(sm_den_controlled_copy/npx_low, pct, -0.1,0.1, mask=mask, kind='full')
    FVD.plot_fvd(sm_num_controlled_copy/npx_high, pct, -0.1,0.1, mask=mask, kind='full')'''
    #FVD.plot_fvd(sm_den_controlled-sm_den, pct, -0.5,0.5, mask=mask, kind='full')
    #FVD.plot_fvd(sm_den, pct, -4,4, mask=mask, kind='full')
    #FVD.plot_fvd(sm_num_controlled-sm_num, pct, -0.5,0.5, mask=mask, kind='full')
    #FVD.plot_fvd(sm_num, pct, -4,4, mask=mask, kind='full')

    #plot_hist(sm_num_controlled-sm_num, sm_den_controlled-sm_den, mask=mask)


    sm_num_controlled_copy[sm_num_controlled_copy==0] = float('nan')
    sm_den_controlled_copy[sm_den_controlled_copy==0] = float('nan')
    lat = np.arange(-90.,90.,0.25)
    avnum = np.nanmean(sm_num_controlled_copy, axis=1)
    avden = np.nanmean(sm_den_controlled_copy, axis=1)

    #FVD.plot_lat(lat, avnum, color='mediumblue')
    #FVD.plot_lat(lat, avden, color='crimson')

    proportion_positive = 1 - FVD.loadmat('proportion_positive.mat')
    quads = FVD.loadmat('quads_F_v.1_e.25.mat')

    #proportion_positive[~np.isfinite(sm_num_controlled_copy)] = float('nan')
    #proportion_positive[proportion_positive==0] = float('nan')
    proportion_positive[quads!=1] = float('nan')

    avprop = np.nanmean(proportion_positive*100., axis=1)
    FVD.plot_lat(lat, avprop, color='mediumblue')

    plt.figure(figsize=(4,4))
    plt.scatter(proportion_positive.reshape(-1,)*100, sm_num_controlled_copy.reshape(-1,), c='mediumblue', s=2)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=50, color='k', linewidth=0.5)
    plt.xlabel('percent of anomalies\nyielding negative ndvi change')
    plt.ylabel('cumulative ndvi change')
    plt.tight_layout()
    #plt.show()

    #FVD.plot_fvd(proportion_positive, 15, 0,1, mask=mask)

    return()

if __name__ == "__main__":
    main()
