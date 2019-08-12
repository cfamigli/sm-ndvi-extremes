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

def plot_scene(data, vmin=None, vmax=None, mask=None):
    # map data
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    if mask is not None:
        data[(~np.isfinite(data)) & (mask<0)] = -9999
        data[mask==0] = float('nan')
    heatmap = ax.pcolor(data, cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('lightgray')
    bar = fig.colorbar(heatmap, extend='both')
    bar.ax.set_ylabel('num_anomalies', rotation=270)
    ax.invert_yaxis()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return

def plot_hist(data, mask, color, label, rng):
    if data.ndim==3:
        for step in range(data.shape[0]):
            d = np.copy(data[step,:,:])
            d[mask==0] = float('nan')
            data[step,:,:] = d
    data[data==-9999] = float('nan')
    data = data[np.isfinite(data)].reshape(-1,)
    #plt.figure(figsize=(4,4))
    plt.hist(data, range=rng, bins=20, color=color, edgecolor='white', density=False, alpha=0.5, label=label)
    plt.ylabel('number of pixels')
    plt.xlabel('ndvi change during anomaly')
    plt.tight_layout()
    #plt.show()
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

    # process soil moisture data first
    os.chdir("../ESACCI/daily_files/COMBINED")
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

    sm_clim_avg = ec.calc_clim_avg(sm, nyrs, nobs, nrows, ncols)
    sm_clim_std = ec.calc_clim_std(sm, nyrs, nobs, nrows, ncols)
    sm_pct_thresh_low = ec.anom_percentile(sm, pct)
    sm_pct_thresh_high = ec.anom_percentile(sm, pct, low=False)
    sm_clim_thresh_low = ec.clim_percentile(sm_clim_avg, sm_clim_std, pct)
    sm_clim_thresh_high = ec.clim_percentile(sm_clim_avg, sm_clim_std, pct, low=False)

    '''for week in range(nobs):
        ec.plot_scene(sm_clim_thresh_high[week,:,:]-sm_clim_thresh_low[week,:,:], week=week)'''

    sm_low = np.zeros((nyrs*nobs, nrows, ncols))
    sm_low_anom = np.zeros((nyrs*nobs, nrows, ncols))
    sm_high = np.zeros((nyrs*nobs, nrows, ncols))
    sm_high_anom = np.zeros((nyrs*nobs, nrows, ncols))
    for ob in range(nyrs*nobs):
        # isolate anomaly pixels
        sm_low[ob,:,:] = ec.isolate_anomalies(sm[ob,:,:],
            sm_pct_thresh_low,
            sm_pct_thresh_high,
            sm_clim_thresh_low[np.mod(ob,nobs),:,:],
            sm_clim_thresh_high[np.mod(ob,nobs),:,:])
        # get anomaly values
        sm_low_anom[ob,:,:] = np.copy(sm_low[ob,:,:]) - np.copy(sm_clim_avg[np.mod(ob,nobs),:,:])
        # isolate anomaly pixels
        sm_high[ob,:,:] = ec.isolate_anomalies(sm[ob,:,:],
            sm_pct_thresh_low,
            sm_pct_thresh_high,
            sm_clim_thresh_low[np.mod(ob,nobs),:,:],
            sm_clim_thresh_high[np.mod(ob,nobs),:,:], low=False)
        # get anomaly values
        sm_high_anom[ob,:,:] = np.copy(sm_high[ob,:,:]) - np.copy(sm_clim_avg[np.mod(ob,nobs),:,:])
    # *************************************************************************************************

    os.chdir("../../../ERA5")
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
    print(temp.shape)

    # initialize matrix to hold 2-weekly observations for all years
    rad_orig = data['ssrd'][:]
    #plot_scene(rad_orig[0,:,:])
    rad = np.zeros(rad_orig.shape)
    rad[:,:,0:int(rad_orig.shape[2]/2)] = rad_orig[:,:,int(rad_orig.shape[2]/2):]
    rad[:,:,int(rad_orig.shape[2]/2):] = rad_orig[:,:,0:int(rad_orig.shape[2]/2)]
    rad = np.delete(rad, -1, 1)
    data.close()
    print(rad.shape)

    temp_clim_avg = calc_clim_avg(temp, nyrs, nobs, nrows, ncols)
    print(temp_clim_avg.shape)
    temp_clim_std = calc_clim_std(temp, nyrs, nobs, nrows, ncols)
    print(temp_clim_std.shape)

    rad_clim_avg = calc_clim_avg(rad, nyrs, nobs, nrows, ncols)
    print(rad_clim_avg.shape)
    rad_clim_std = calc_clim_std(rad, nyrs, nobs, nrows, ncols)
    print(rad_clim_std.shape)

    temp_pct_thresh_low = ec.anom_percentile(temp, pct)
    temp_pct_thresh_high = ec.anom_percentile(temp, pct, low=False)
    temp_clim_thresh_low = ec.clim_percentile(temp_clim_avg, temp_clim_std, pct)
    temp_clim_thresh_high = ec.clim_percentile(temp_clim_avg, temp_clim_std, pct, low=False)

    rad_pct_thresh_low = ec.anom_percentile(rad, pct)
    rad_pct_thresh_high = ec.anom_percentile(rad, pct, low=False)
    rad_clim_thresh_low = ec.clim_percentile(rad_clim_avg, rad_clim_std, pct)
    rad_clim_thresh_high = ec.clim_percentile(rad_clim_avg, rad_clim_std, pct, low=False)

    temp_low = np.zeros((nyrs*nobs, nrows, ncols))
    temp_low_anom = np.zeros((nyrs*nobs, nrows, ncols))
    temp_high = np.zeros((nyrs*nobs, nrows, ncols))
    temp_high_anom = np.zeros((nyrs*nobs, nrows, ncols))

    rad_low = np.zeros((nyrs*nobs, nrows, ncols))
    rad_low_anom = np.zeros((nyrs*nobs, nrows, ncols))
    rad_high = np.zeros((nyrs*nobs, nrows, ncols))
    rad_high_anom = np.zeros((nyrs*nobs, nrows, ncols))

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

        # isolate anomaly pixels
        rad_low[ob,:,:] = ec.isolate_anomalies(rad[ob,:,:],
            rad_pct_thresh_low,
            rad_pct_thresh_high,
            rad_clim_thresh_low[np.mod(ob,nobs),:,:],
            rad_clim_thresh_high[np.mod(ob,nobs),:,:])
        # get anomaly values
        rad_low_anom[ob,:,:] = np.copy(rad_low[ob,:,:]) - np.copy(rad_clim_avg[np.mod(ob,nobs),:,:])
        # isolate anomaly pixels
        rad_high[ob,:,:] = ec.isolate_anomalies(rad[ob,:,:],
            rad_pct_thresh_low,
            rad_pct_thresh_high,
            rad_clim_thresh_low[np.mod(ob,nobs),:,:],
            rad_clim_thresh_high[np.mod(ob,nobs),:,:], low=False)
        # get anomaly values
        rad_high_anom[ob,:,:] = np.copy(rad_high[ob,:,:]) - np.copy(rad_clim_avg[np.mod(ob,nobs),:,:])

    print(temp_high_anom.shape)

    #plt.figure(figsize=(4,4))
    #plot_hist(sm_high_anom, mask, 'dodgerblue', 'all anomalies', [0,0.2])
    #plot_hist(sm_high_anom, mask)
    sm_low_anom[(np.isfinite(temp_low_anom)) | (np.isfinite(rad_low_anom)) | (np.isfinite(temp_high_anom)) | (np.isfinite(rad_high_anom))] = float('nan')
    sm_high_anom[(np.isfinite(temp_high_anom)) | (np.isfinite(rad_high_anom)) | (np.isfinite(temp_low_anom)) | (np.isfinite(rad_low_anom))] = float('nan')
    #plot_hist(sm_high_anom, mask, 'crimson', 'isolated anomalies', [0,0.2])
    #plt.legend(loc='best')
    #plt.show()
    #plot_hist(sm_high_anom, mask)

    npx_low = np.nansum(np.isfinite(sm_low_anom), axis=0)*1.
    #FVD.savemat(npx_low, 'num_anoms_low.mat')
    #plot_scene(npx_low, mask=mask)
    npx_high = np.nansum(np.isfinite(sm_high_anom), axis=0)*1.
    #FVD.savemat(npx_high, 'num_anoms_high.mat')
    #plot_scene(npx_high, mask=mask)

        # process ndvi data
    os.chdir("../GIMMS/data")
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

    ndvi_clim_avg = gm.calc_clim_avg(ndvi, nyrs, nobs, nrows, ncols)

    fvd = FVD.calc_fvd(sm_low_anom, sm_high_anom, ndvi, ndvi_clim_avg,
        nyrs, nobs, nrows, ncols, pct, mask, temp)

    return

if __name__ == "__main__":
    main()
