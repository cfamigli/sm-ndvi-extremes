
import ESACCI as ec
import GIMMS as gm
import FVD
import numpy as np
import os
import time
from skimage.measure import block_reduce

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
nobs = 24
nyrs = len(years)

# initialize matrix to hold 2-weekly observations for all years
sm = np.zeros((nyrs*nobs, nrows, ncols))
count = 0
for year in years:
    os.chdir(year)
    files = ec.get_files('nc')
    ymat = ec.combine_files(files, nrows, ncols)
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

'''for week in range(nobs):
    ec.plot_anom(sm_low_anom[(ob-nobs)+week,:,:], week=week)
for week in range(nobs):
    ec.plot_anom(sm_high_anom[(ob-nobs)+week,:,:], low=False, week=week)'''

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
    data = gm.read_file(file)
    t = data.variables['time'][:]
    nsteps = len(t)

    for step in range(nsteps):
        date = gm.time_to_str(t[step], year)
        print(date)
        nd = data.variables['ndvi'][step,:,:]
        nd = nd.astype(float)
        nd[nd<0] = float('nan')
        nd = nd / 1e4
        # resample ndvi observation to coarser grid
        nd = block_reduce(nd, block_size=(3,3), func=np.nanmean)
        ndvi[count, :, :] = nd
        count += 1
        print(count)
    data.close()

ndvi_clim_avg = gm.calc_clim_avg(ndvi, nyrs, nobs, nrows, ncols)

gm.plot_scene(np.nanmean(ndvi, axis=0))
FVD.savemat(np.nanmean(ndvi, axis=0), 'mean_ndvi.mat')

gm.plot_scene(np.nanmean(sm_low_anom, axis=0))
FVD.savemat(np.nanmean(sm_low_anom, axis=0), 'avg_low_anom.mat')

gm.plot_scene(np.nanmean(sm_high_anom, axis=0))
FVD.savemat(np.nanmean(sm_high_anom, axis=0), 'avg_high_anom.mat')

fvd = FVD.calc_fvd(sm_low_anom, sm_high_anom, ndvi, ndvi_clim_avg,
    nyrs, nobs, nrows, ncols, pct)
