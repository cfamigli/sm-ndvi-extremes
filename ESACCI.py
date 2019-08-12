
from netCDF4 import Dataset
import numpy as np
import os
import glob
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def get_files(ext):
    # input is extension type
    files = [i for i in glob.glob('*.{}'.format(ext)) if len(i)>10]
    files.sort()
    return files

def get_subfolders():
    # get list of subfolders
    return sorted([f.name for f in os.scandir(os.curdir) if f.is_dir()])

def read_file(filename):
    # read nc file and save as numpy array (not np masked)
    data = Dataset(filename, 'r')
    data.set_auto_mask(False)
    return data

def combine_files(files, nrows, ncols):
    # reduce daily files to 2-weekly
    # take average of SM observations in each 2 week period
    # return a 3d matrix with 24 layers, one for each period in the year
    dates = [files[i][42:46] for i in range(len(files))]
    inds = [i for i in range(len(dates)) if dates[i].endswith(('01','16'))]
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

def surface_mask(arr):
    mask = np.isfinite(arr) * -9999.
    return mask

def plot_scene(data, mask=None, week=None, low=True):
    # map data
    data = data.astype(float)
    if mask is not None:
        data[(~np.isfinite(data)) & (mask<0)] = -9999
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    #cmap = LinearSegmentedColormap.from_list('name',
        #['peru', 'darkseagreen','lightskyblue','blue'])
    cmap = plt.cm.rainbow
    cmap.set_under(color='lightgray')
    heatmap = ax.pcolor(data, cmap=cmap, vmin=0,#np.nanmin(data),
        vmax=np.nanmax(data))
    bar = fig.colorbar(heatmap)
    bar.ax.set_ylabel('sm', rotation=270, labelpad=20)
    ax.invert_yaxis()
    plt.axis('off')
    if low:
        plt.savefig('../plots/ch_count_low.png')
    else:
        plt.savefig('../plots/ch_count_high.png')
    #plt.show()
    plt.close(fig)
    return

def calc_clim_avg(mat_all, nyrs, nobs, nrows, ncols):
    # nobs are the number of observations in one year
    # nobs = 24 means observations averaged over every 2 weeks,
    # e.g. Jan 1, Jan 15, Feb 1, Feb 15, ...
    # returns matrix of shape (24, nrows, ncols)
    mat_avg = np.zeros((nobs, nrows, ncols))
    for i in range(nobs):
        inds = nobs * np.arange(nyrs) + i
        print(inds)
        mat_avg[i,:,:] = np.nanmean(mat_all[inds,:,:], axis=0)
    return mat_avg

def calc_clim_std(mat_all, nyrs, nobs, nrows, ncols):
    # nobs are the number of observations in one year
    # nobs = 24 means observations averaged over every 2 weeks,
    # e.g. Jan 1, Jan 15, Feb 1, Feb 15, ...
    # returns matrix of shape (24, nrows, ncols)
    mat_std = np.zeros((nobs, nrows, ncols))
    for i in range(nobs):
        inds = nobs * np.arange(nyrs) + i
        print(inds)
        mat_std[i,:,:] = np.nanstd(mat_all[inds,:,:], axis=0)
    return mat_std

def anom_percentile(mat_all, pct, low=True):
    # returns matrix of SM values representing indicated percentile.
    if low:
        pct_mat = np.nanpercentile(mat_all, pct, axis=0)
    else:
        pct = 100 - pct
        pct_mat = np.nanpercentile(mat_all, pct, axis=0)
    print(low)
    return pct_mat

def clim_percentile(mat_avg, mat_std, pct, low=True):
    # returns matrix of "confidence intervals" of same size as mat_avg
    # (24, nrows, ncols)
    switcher = {
        5.: 1.65, 10.: 1.28, 15.: 1.04, 20.: 0.84, 25.: 0.67
    }
    z = switcher.get(pct, "Invalid percentile")
    if low:
        clim_mat = mat_avg - z*mat_std
    else:
        clim_mat = mat_avg + z*mat_std
    return clim_mat

def isolate_anomalies(mat, pl, ph, cl, ch, low=True):
    # set pixels that aren't anomalous to be nan.
    anom_mat = np.copy(mat)
    if low:
        anom_mat[(mat>pl) | (mat>cl)] = float('nan')
    else:
        anom_mat[(mat<ph) | (mat<ch)] = float('nan')
    return anom_mat

def count_thresh(pl, ph, cl, ch, low=True):
    # count how many times clim thresholds are more extreme than percentile thresholds.
    if low:
        cl[cl>=pl] = float('nan')
        counts = np.nansum(np.isfinite(cl), axis=0)*1.
    else:
        ch[ch<=ph] = float('nan')
        counts = np.nansum(np.isfinite(ch), axis=0)*1.
    counts[counts==0] = float('nan')
    return counts

def plot_anom(data, low=True, week=None):
    # map data
    data = data.astype(float)
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    if low:
        cmap = plt.cm.Reds_r
        strpath = '../../plots/sm_low_anom_'
    else:
        cmap = plt.cm.Blues
        strpath = '../../plots/sm_high_anom_'
    heatmap = ax.pcolor(data, cmap=cmap, vmin=np.nanmin(data),
        vmax=np.nanmax(data))
    heatmap.cmap.set_under('black')
    bar = fig.colorbar(heatmap)
    bar.ax.set_ylabel('sm', rotation=270, labelpad=20)
    ax.invert_yaxis()
    plt.axis('off')
    if week is not None:
        plt.savefig(strpath + str(week) +'.png')
    #plt.title(date)
    #plt.show()
    plt.close(fig)
    return
