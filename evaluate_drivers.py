import numpy as np
import os
import glob
import FVD
import ESACCI as ec
import GIMMS as gm
from netCDF4 import Dataset
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from scipy.stats import skew
from scipy.stats.kde import gaussian_kde
from scipy.optimize import curve_fit

def plot_scene(data, ylab, mask=None, title=None, vmin=None, vmax=None, ):
    # map data
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    if mask is not None:
        data[(~np.isfinite(data)) & (mask<0)] = -9999
        data[mask==0] = float('nan')
    heatmap = ax.pcolor(data, cmap=plt.cm.Spectral_r, vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('lightgray')
    bar = fig.colorbar(heatmap, extend='both')
    bar.ax.set_ylabel(ylab, rotation=270)
    ax.invert_yaxis()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../../plots/drivers_' + title + '.png')
    plt.close(fig)
    return

def plot_hist(dataFp, dataFm, title, xlab=None):
    dataFp[dataFp==-9999] = float('nan')
    dataFp = dataFp[np.isfinite(dataFp)].reshape(-1,1)

    dataFm[dataFm==-9999] = float('nan')
    dataFm = dataFm[np.isfinite(dataFm)].reshape(-1,1)
    print(dataFm)

    mx = np.nanmax([np.nanmax(dataFp), np.nanmax(dataFm)])
    mn = np.nanmin([np.nanmin(dataFp), np.nanmin(dataFm)])

    #kde_Fp = gaussian_kde(dataFp)
    #kde_Fm = gaussian_kde(dataFm)
    x = np.linspace(mn, mx, 100)

    plt.figure(figsize=(6,3))
    plt.hist([dataFm,dataFp], bins=30, range=[mn-0.25,mx+0.25],
        color=['orangered','cornflowerblue'], ec='white', rwidth=0.92, label=['F-','F+'])
    #plt.hist(dataFp, bins=30, range=[mn,mx], color='royalblue', ec='white', rwidth=0.85, alpha=0.5)
    #plt.plot(x, kde_Fp(x))
    if xlab is None:
        plt.xlabel('skewness')
    else:
        plt.xlabel('pearson correlation')
    plt.ylabel('number of pixels')
    if title=='skew_ndvi':
        plt.ylim([0,12000])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('../../plots/drivers_dist_' + title + '.png')
    #plt.show()
    return

def make_subplots(xdata, ydata, quads,label=None):
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(6,6)

    xFp = np.copy(xdata)
    xFp[quads!=2] = float('nan')

    yFp = np.copy(ydata)
    yFp[quads!=2] = float('nan')

    xFm = np.copy(xdata)
    xFm[quads!=1] = float('nan')

    yFm = np.copy(ydata)
    yFm[quads!=1] = float('nan')

    axs[0,0].scatter(xFp[:360,:], yFp[:360,:], c='gray', s=2, marker='.')
    axs[0,0].set_title('NH F+')
    axs[0,0].set_ylim([0,3])
    axs[0,1].scatter(xFp[360:,:], yFp[360:,:], c='gray', s=2, marker='.')
    axs[0,1].set_title('SH F+')
    axs[0,1].set_ylim([0,3])
    axs[1,0].scatter(xFm[:360,:], yFm[:360,:], c='gray', s=2, marker='.')
    axs[1,0].set_title('NH F-')
    axs[1,0].set_ylim([-2,0])
    axs[1,1].scatter(xFm[360:,:], yFm[360:,:], c='gray', s=2, marker='.')
    axs[1,1].set_title('SH F-')
    axs[1,1].set_ylim([-2,0])
    plt.tight_layout()
    plt.show()
    return

def make_subplots_colored_by_lat(xdata, ydata, quads,label):
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios":[1, 1, 0.05]})
    fig.set_size_inches(3.5,6.5)

    xFp = np.copy(xdata)
    xFp[quads!=2] = float('nan')

    yFp = np.copy(ydata)
    yFp[quads!=2] = float('nan')

    xFm = np.copy(xdata)
    xFm[quads!=1] = float('nan')

    yFm = np.copy(ydata)
    yFm[quads!=1] = float('nan')

    lats = np.repeat(np.arange(0,720).reshape(-1,1),1440,axis=1)
    mx = np.nanmax([np.nanmax(xFp), np.nanmax(xFm)])
    mn = np.nanmin([np.nanmin(xFp), np.nanmin(xFm)])

    axs[0].scatter(xFp, yFp, c=lats, cmap=plt.cm.rainbow, s=3, marker='.', alpha=0.7)
    axs[0].set_title('F+', position=(0.9,0.85), fontweight='bold')
    axs[0].set_ylim([0,3])
    axs[0].set_xlim([mn*0.9,mx*1.1])
    axs[0].set_xlabel(label)
    axs[0].set_ylabel('cumulative\nndvi change')
    c = axs[1].scatter(xFm, yFm, c=lats, cmap=plt.cm.rainbow, s=3, marker='.', alpha=0.7)
    axs[1].set_title('F-', position=(0.9,0.05), fontweight='bold')
    axs[1].set_ylim([-2,0])
    axs[1].set_xlim([mn*0.9,mx*1.1])
    axs[1].set_xlabel(label)
    axs[1].set_ylabel('cumulative\nndvi change')

    cbar = fig.colorbar(c, cax=axs[2], orientation='horizontal', pad=0.25, ticks=[100,300,500])
    cbar.ax.set_xticklabels(['65N','15N','35S'])
    cbar.set_label('latitude')
    plt.tight_layout()
    plt.savefig('../../plots/latitude_plots/lat_' + label + '.png', dpi=300)
    plt.close(fig)
    return

def main():
    os.chdir("matfiles")
    mask = FVD.loadmat('land_mask.mat') # mask for land surface
    npx = np.nansum(mask==0)

    pct = 15
    val = 0.1
    nrows = 720
    ncols = 1440
    nobs = 12

    corr_sm_ndvi = FVD.loadmat('corr.mat')
    skew_sm = FVD.loadmat('skew_sm.mat')
    skew_ndvi = FVD.loadmat('skew_ndvi.mat')
    skew_temp = FVD.loadmat('skew_temp.mat')
    skew_rad = FVD.loadmat('skew_rad.mat')
    skew_prec = FVD.loadmat('skew_prec.mat')
    skew_PET = FVD.loadmat('skew_PET.mat')
    skew_vpd = FVD.loadmat('skew_vpd.mat')
    skew_LAI = FVD.loadmat('skew_LAI.mat')

    temp_mean = FVD.loadmat('temp_mean.mat')
    rad_mean = FVD.loadmat('rad_mean.mat')
    prec_mean = FVD.loadmat('prec_mean.mat')
    PET_mean = FVD.loadmat('PET_mean.mat')
    vpd_mean = FVD.loadmat('vpd_mean.mat')
    lai_mean = FVD.loadmat('lai_mean.mat')
    ndvi_mean = FVD.loadmat('ndvi_mean.mat')

    temp_std = FVD.loadmat('temp_std.mat')
    rad_std = FVD.loadmat('rad_std.mat')
    prec_std = FVD.loadmat('prec_std.mat')
    PET_std = FVD.loadmat('PET_std.mat')
    vpd_std = FVD.loadmat('vpd_std.mat')
    lai_std = FVD.loadmat('lai_std.mat')
    ndvi_std = FVD.loadmat('ndvi_std.mat')

    temp_D = FVD.loadmat('temp_D.mat')
    rad_D = FVD.loadmat('rad_D.mat')
    prec_D = FVD.loadmat('prec_D.mat')
    PET_D = FVD.loadmat('PET_D.mat')
    vpd_D = FVD.loadmat('vpd_D.mat')
    lai_D = FVD.loadmat('lai_D.mat')
    ndvi_D = FVD.loadmat('ndvi_D.mat')

    sm_num_controlled = FVD.loadmat('sm_numerator_monthly_controlled_fixed.mat')
    sm_den_controlled = FVD.loadmat('sm_denominator_monthly_controlled_fixed.mat')

    sm_den_controlled_copy = np.copy(sm_den_controlled)
    sm_num_controlled_copy = np.copy(sm_num_controlled)

    sm_den_controlled_copy[(abs(sm_den_controlled)<val) & (abs(sm_num_controlled)<val)] = float('nan')
    sm_num_controlled_copy[(abs(sm_den_controlled)<val) & (abs(sm_num_controlled)<val)] = float('nan')

    sm_num_controlled_copy[sm_num_controlled_copy==0] = float('nan')
    sm_den_controlled_copy[sm_den_controlled_copy==0] = float('nan')

    quads = FVD.loadmat('quads_fixed_F_v.1_e.25.mat')
    sm_num_controlled_copy[~np.isfinite(quads)] = float('nan')

    label = ['corr(SM,NDVI)','skew_sm','skew_ndvi','skew_temp','skew_rad','skew_prec','skew_PET',
        'skew_vpd','skew_LAI','temp_mean','rad_mean','prec_mean','PET_mean','vpd_mean',
        'lai_mean','ndvi_mean','temp_std','rad_std','prec_std','PET_std','vpd_std',
        'lai_std','ndvi_std','temp_D','rad_D','prec_D','PET_D','vpd_D','lai_D','ndvi_D']

    count = 0
    for var in [corr_sm_ndvi, skew_sm, skew_ndvi, skew_temp, skew_rad,
        skew_prec, skew_PET, skew_vpd, skew_LAI, temp_mean, rad_mean,
        prec_mean, PET_mean, vpd_mean, lai_mean, ndvi_mean, temp_std,
        rad_std, prec_std, PET_std, vpd_std, lai_std, ndvi_std, temp_D,
        rad_D, prec_D, PET_D, vpd_D, lai_D, ndvi_D]:

        var[~np.isfinite(quads)] = float('nan')

        #make_subplots(var, sm_num_controlled_copy, quads)
        make_subplots_colored_by_lat(var, sm_num_controlled_copy, quads, label[count])
        count += 1

    return()

if __name__ == "__main__":
    main()
