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

def main():
    os.chdir("matfiles")
    mask = FVD.loadmat('land_mask.mat') # mask for land surface
    npx = np.nansum(mask==0)

    pct = 15
    val = 0.1
    nrows = 720
    ncols = 1440
    nobs = 12

    '''def read_file(filename):
        # read nc file and save as numpy array (not np masked)
        data = Dataset(filename, 'r')
        data.set_auto_mask(False)
        return data

    def combine_sm_files(files, nrows, ncols):
        # reduce daily files to 2-weekly
        # take average of SM observations in each 2 week period
        # return a 3d matrix with 24 layers, one for each period in the year
        dates = [files[i][42:46] for i in range(len(files))]
        inds = [i for i in range(len(dates)) if dates[i].endswith(('01'))]
        inds.append(len(dates))
        mat = np.zeros((len(inds)-1, nrows, ncols))

        for i in range(len(inds)-1):
            print(inds[i])
            print(inds[i+1])
            com = np.zeros((inds[i+1]-inds[i], nrows, ncols))/0
            for j in range(inds[i],inds[i+1]):
                print(files[j])
                print(j-inds[i])
                data = read_file(files[j]).variables['sm'][0,:,:]
                data[data<0] = float('nan')
                com[j-inds[i],:,:] = data
            mat[i,:,:] = np.nanmean(com, axis=0)
        return mat

    def combine_ndvi_files(ncdata, nrows, ncols):
        # reduce 2-weekly file to monthly
        steps = ncdata.shape[0]
        mat = np.zeros((int(steps/2), nrows, ncols))
        print(mat.shape)
        for i in range(0,steps,2):
            mat[i:i+2,:,:] = np.nanmean(ncdata[i:i+2,:,:], axis=0)
        return mat

    # process soil moisture data first
    os.chdir("../../ESACCI/daily_files/COMBINED")
    years = ec.get_subfolders()
    del years[:4] # delete years for which no ndvi data exists
    del years[-3:] #[-34:]
    nyrs = len(years)

    # initialize matrix to hold 2-weekly observations for all years
    sm = np.zeros((nyrs*nobs, nrows, ncols))
    count = 0
    for year in years:
        os.chdir(year)
        files = ec.get_files('nc')
        ymat = combine_sm_files(files, nrows, ncols)
        sm[count*nobs:(count+1)*nobs, :, :] = ymat
        count += 1
        os.chdir("..")

    # process ndvi data
    os.chdir("../../../GIMMS/data")
    files = gm.get_files('nc4')
    files.pop(0)
    #del files[-62:] # remove this line

    # initialize matrix to hold 2-weekly observations for all years
    ndvi = np.zeros((nyrs*nobs, nrows, ncols))
    count = 0
    for file in files:
        print(file)
        year = file[13:18]
        data = read_file(file)

        nd = data['ndvi'][:]
        nd = nd.astype(float)
        nd[nd<0] = float('nan')
        nd = nd / 1e4
        # resample ndvi observation to coarser grid
        reduced_nd = np.zeros((nd.shape[0], nrows, ncols))
        for step in range(0,nd.shape[0]):
            reduced_nd[step,:,:] = block_reduce(nd[step,:,:], block_size=(3,3), func=np.nanmean)
        ndvi[count*6:(count+1)*6, :, :] = combine_ndvi_files(reduced_nd, nrows, ncols)
        count += 1
        print(count)
        data.close()

    assert sm.shape==ndvi.shape
    corr = np.ones((nrows,ncols))*np.nan
    skew_sm = np.ones((nrows,ncols))*np.nan
    skew_ndvi = np.ones((nrows,ncols))*np.nan
    for row in range(nrows):
        print(row)
        for col in range(ncols):
            s = sm[:,row,col]
            n = ndvi[:,row,col]
            nan_inds = np.isnan(s) | np.isnan(n)
            corr[row,col] = np.corrcoef(s[~nan_inds], n[~nan_inds])[1,0]
            skew_sm[row,col] = skew(s[~np.isnan(s)])
            skew_ndvi[row,col] = skew(n[~np.isnan(n)])

    FVD.savemat(corr, 'corr.mat')
    FVD.savemat(skew_sm, 'skew_sm.mat')
    FVD.savemat(skew_ndvi, 'skew_ndvi.mat')'''

    corr_sm_ndvi = FVD.loadmat('corr.mat')
    skew_sm = FVD.loadmat('skew_sm.mat')
    skew_ndvi = FVD.loadmat('skew_ndvi.mat')
    skew_temp = FVD.loadmat('skew_temp.mat')
    skew_rad = FVD.loadmat('skew_rad.mat')
    skew_prec = FVD.loadmat('skew_prec.mat')
    skew_PET = FVD.loadmat('skew_PET.mat')
    skew_vpd = FVD.loadmat('skew_vpd.mat')
    skew_LAI = FVD.loadmat('skew_LAI.mat')

    '''plot_scene(corr_sm_ndvi, mask=mask, title='corr_sm_ndvi', ylab='pearson corr', vmin=-1, vmax=1)
    plot_scene(skew_sm, mask=mask, title='skew_sm', ylab='skewness', vmin=-2, vmax=2)
    plot_scene(skew_ndvi, mask=mask, title='skew_ndvi', ylab='skewness', vmin=-3, vmax=3)
    plot_scene(skew_temp, mask=mask, title='skew_temp', ylab='skewness', vmin=-1, vmax=1)
    plot_scene(skew_rad, mask=mask, title='skew_rad', ylab='skewness', vmin=-1, vmax=1)
    plot_scene(skew_prec, mask=mask, title='skew_prec', ylab='skewness', vmin=-1, vmax=10)
    plot_scene(skew_PET, mask=mask, title='skew_PET', ylab='skewness', vmin=-1, vmax=3)
    plot_scene(skew_vpd, mask=mask, title='skew_vpd', ylab='skewness', vmin=-1, vmax=3)
    plot_scene(skew_LAI, mask=mask, title='skew_LAI', ylab='skewness', vmin=-2, vmax=2)'''

    sm_num_controlled = FVD.loadmat('sm_numerator_monthly_controlled_fixed.mat')
    sm_den_controlled = FVD.loadmat('sm_denominator_monthly_controlled_fixed.mat')

    sm_den_controlled_copy = np.copy(sm_den_controlled)
    sm_num_controlled_copy = np.copy(sm_num_controlled)

    sm_den_controlled_copy[(abs(sm_den_controlled)<val) & (abs(sm_num_controlled)<val)] = float('nan')
    sm_num_controlled_copy[(abs(sm_den_controlled)<val) & (abs(sm_num_controlled)<val)] = float('nan')

    sm_num_controlled_copy[sm_num_controlled_copy==0] = float('nan')
    sm_den_controlled_copy[sm_den_controlled_copy==0] = float('nan')

    quads = FVD.loadmat('quads_F_v.1_e.25.mat')
    sm_num_controlled_copy[~np.isfinite(quads)] = float('nan')
    corr_sm_ndvi[~np.isfinite(quads)] = float('nan')

    sm_num_controlled_copy[quads!=2] = float('nan')
    corr_sm_ndvi[quads!=2] = float('nan')

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    x = np.linspace(np.nanmin(corr_sm_ndvi[360:,:]), np.nanmax(corr_sm_ndvi[360:,:]), 100)
    y = func(x, .05, -7.3, 0.15)

    plt.figure(figsize=(4,4))
    plt.scatter(corr_sm_ndvi[360:,:], sm_num_controlled_copy[360:,:], c='gray', s=2, marker='.')
    plt.plot(x, y, c='black', linewidth=2)
    plt.ylim([0,3])
    plt.show()
    '''plot_hist(corr_sm_ndvi[sm_num_controlled_copy>0], corr_sm_ndvi[sm_num_controlled_copy<0], 'corr_sm_ndvi')
    plot_hist(skew_sm[sm_num_controlled_copy>0], skew_sm[sm_num_controlled_copy<0], 'skew_sm')
    plot_hist(skew_ndvi[sm_num_controlled_copy>0], skew_ndvi[sm_num_controlled_copy<0], 'skew_ndvi')
    plot_hist(skew_temp[sm_num_controlled_copy>0], skew_temp[sm_num_controlled_copy<0], 'skew_temp')
    plot_hist(skew_rad[sm_num_controlled_copy>0], skew_rad[sm_num_controlled_copy<0], 'skew_rad')
    plot_hist(skew_prec[sm_num_controlled_copy>0], skew_prec[sm_num_controlled_copy<0], 'skew_prec')
    plot_hist(skew_PET[sm_num_controlled_copy>0], skew_PET[sm_num_controlled_copy<0], 'skew_PET')
    plot_hist(skew_vpd[sm_num_controlled_copy>0], skew_vpd[sm_num_controlled_copy<0], 'skew_vpd')
    plot_hist(skew_LAI[sm_num_controlled_copy>0], skew_LAI[sm_num_controlled_copy<0], 'skew_LAI')'''

    return()

if __name__ == "__main__":
    main()
