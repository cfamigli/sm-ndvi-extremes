import numpy as np
import os
import glob
import FVD
import ESACCI as ec
from netCDF4 import Dataset
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from skimage.measure import block_reduce

def get_files(ext):
    # input is extension type
    files = [i for i in glob.glob('*.{}'.format(ext)) if len(i)>10]
    files.sort()
    return files

def read_file(filename):
    # read nc file and save as numpy array (not np masked)
    data = Dataset(filename, 'r')
    data.set_auto_mask(False)
    return data

def output_temp_C_full(ncdata, nrows, ncols):
    data = np.vstack([np.ones((120,ncols))*np.nan, ncdata])
    data[data==-9999] = float('nan')
    data -= 273.15
    return data

def calc_clim_avg(mat_all, nyrs, nobs, nrows, ncols):
    # nobs are the number of observations in one year
    # nobs = 12 means observations every month
    mat_avg = np.zeros((nobs, nrows, ncols))
    for i in range(nobs):
        inds = nobs * np.arange(nyrs) + i
        mat_avg[i,:,:] = np.nanmean(mat_all[inds,:,:], axis=0)
    return mat_avg

def calc_clim_std(mat_all, nyrs, nobs, nrows, ncols):
    # nobs are the number of observations in one year
    # returns matrix of shape (nobs, nrows, ncols)
    mat_std = np.zeros((nobs, nrows, ncols))
    for i in range(nobs):
        inds = nobs * np.arange(nyrs) + i
        print(inds)
        mat_std[i,:,:] = np.nanstd(mat_all[inds,:,:], axis=0)
    return mat_std

def combine_files(ncdata, nrows, ncols):
    # reduce 2-weekly file to monthly
    steps = ncdata.shape[0]
    mat = np.zeros((int(steps/2), nrows, ncols))
    print(mat.shape)
    for i in range(0,steps,2):
        mat[i:i+2,:,:] = np.nanmean(ncdata[i:i+2,:,:], axis=0)
    return mat

def plot_scene(data, vmin, vmax, mask=None):
    # map data
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    if mask is not None:
        data[(~np.isfinite(data)) & (mask<0)] = -9999
        data[mask==0] = float('nan')
    heatmap = ax.pcolor(data, cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('lightgray')
    bar = fig.colorbar(heatmap, extend='both')
    bar.ax.set_ylabel('temp', rotation=270)
    ax.invert_yaxis()
    plt.axis('off')
    plt.show()
    plt.close(fig)
    return

def main():

    os.chdir("matfiles")
    mask = FVD.loadmat('land_mask.mat')
    os.chdir("..")

    # output resolution
    nrows = 720
    ncols = 1440
    nobs = 12
    pct = 15.

    # *************************************************************************************************

    os.chdir("../ERA5")
    file = 'ERA5.nc'

    years = np.arange(1982,2016)
    nyrs = len(years)
    data = read_file(file)

    # initialize matrix to hold 2-weekly observations for all years
    temp_orig = data['t2m'][:]-273.15
    #plot_scene(temp_orig[0,:,:])
    temp = np.zeros(temp_orig.shape)
    temp[:,:,0:int(temp_orig.shape[2]/2)] = temp_orig[:,:,int(temp_orig.shape[2]/2):]
    temp[:,:,int(temp_orig.shape[2]/2):] = temp_orig[:,:,0:int(temp_orig.shape[2]/2)]
    temp = np.delete(temp, -1, 1)
    data.close()
    print(temp.shape)

    temp_clim_avg = calc_clim_avg(temp, nyrs, nobs, nrows, ncols)
    print(temp_clim_avg.shape)
    temp_clim_std = calc_clim_std(temp, nyrs, nobs, nrows, ncols)
    print(temp_clim_std.shape)

    temp_pct_thresh_low = ec.anom_percentile(temp, pct)
    temp_pct_thresh_high = ec.anom_percentile(temp, pct, low=False)
    temp_clim_thresh_low = ec.clim_percentile(temp_clim_avg, temp_clim_std, pct)
    temp_clim_thresh_high = ec.clim_percentile(temp_clim_avg, temp_clim_std, pct, low=False)

    temp_low = np.zeros((nyrs*nobs, nrows, ncols))
    temp_low_anom = np.zeros((nyrs*nobs, nrows, ncols))
    temp_high = np.zeros((nyrs*nobs, nrows, ncols))
    temp_high_anom = np.zeros((nyrs*nobs, nrows, ncols))
    for ob in range(nyrs*nobs):
        # isolate anomaly pixels
        temp_low[ob,:,:] = ec.isolate_anomalies(temp[ob,:,:],
            temp_pct_thresh_low,
            temp_pct_thresh_high,
            temp_clim_thresh_low[np.mod(ob,nobs),:,:],
            temp_clim_thresh_high[np.mod(ob,nobs),:,:])
        # get anomaly values
        temp_low_anom[ob,:,:] = np.copy(temp_low[ob,:,:]) - np.copy(temp_clim_avg[np.mod(ob,nobs),:,:])
        # isolate anomaly pixels
        temp_high[ob,:,:] = ec.isolate_anomalies(temp[ob,:,:],
            temp_pct_thresh_low,
            temp_pct_thresh_high,
            temp_clim_thresh_low[np.mod(ob,nobs),:,:],
            temp_clim_thresh_high[np.mod(ob,nobs),:,:], low=False)
        # get anomaly values
        temp_high_anom[ob,:,:] = np.copy(temp_high[ob,:,:]) - np.copy(temp_clim_avg[np.mod(ob,nobs),:,:])
    print(temp_high_anom.shape)

    os.chdir("../GIMMS/data")
    files = get_files('nc4')
    files.pop(0)
    #del files[0:36] # remove this line

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
        ndvi[count*6:(count+1)*6, :, :] = combine_files(reduced_nd, nrows, ncols)
        count += 1
        print(count)
        data.close()

    print(ndvi.shape)
    ndvi_clim_avg = calc_clim_avg(ndvi, nyrs, nobs, nrows, ncols)

    fvd = FVD.calc_fvd(temp_low_anom, temp_high_anom, ndvi, ndvi_clim_avg, nyrs, nobs, nrows, ncols, pct)

    return

if __name__ == "__main__":
    main()
