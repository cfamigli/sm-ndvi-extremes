
import ESACCI as ec
import GIMMS as gm
import FVD
import numpy as np
import os
import time
from skimage.measure import block_reduce
from netCDF4 import Dataset

def read_file(filename):
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

def main():
    os.chdir("matfiles")
    mask = FVD.loadmat('land_mask.mat')
    os.chdir("..")

    # process soil moisture data first
    os.chdir("../ESACCI/daily_files/COMBINED")
    years = ec.get_subfolders()
    del years[:4] # delete years for which no ndvi data exists
    del years[-3:] #[-34:]

    # output resolution
    nrows = 720
    ncols = 1440
    nobs = 12
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

    pct = 15.
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

    ndvi_clim_avg = gm.calc_clim_avg(ndvi, nyrs, nobs, nrows, ncols)

    fvd = FVD.calc_fvd(sm_low_anom, sm_high_anom, ndvi, ndvi_clim_avg,
        nyrs, nobs, nrows, ncols, pct)

    return

if __name__ == "__main__":
    main()
