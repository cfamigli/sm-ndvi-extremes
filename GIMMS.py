
from netCDF4 import Dataset
import numpy as np
import glob
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

def time_to_str(time, year):
    # convert float date to string
    switcher = {
        1.: "January_1", 1.5: "January_15", 2.: "February_1", 2.5: "February_15",
        3.: "March_1", 3.5: "March_15", 4.: "April_1", 4.5: "April_15",
        5.: "May_1", 5.5: "May_15", 6.: "June_1", 6.5: "June_15",
        7.: "July_1", 7.5: "July_15", 8.: "August_1", 8.5: "August_15",
        9.: "September_1", 9.5: "September_15", 10.: "October_1", 10.5: "October_15",
        11.: "November_1", 11.5: "November_15", 12.: "December_1", 12.5: "December_15",
    }
    return switcher.get(time, "Invalid month") + year

def plot_scene(data, vmin, vmax, week=None):
    # map data
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    heatmap = ax.pcolor(data, cmap=plt.cm.BrBG, vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('black')
    bar = fig.colorbar(heatmap, extend='both')
    bar.ax.set_ylabel('ndvi', rotation=270)
    ax.invert_yaxis()
    plt.axis('off')
    #plt.title(date)
    plt.show()
    #if week is not None:
        #plt.savefig('../plots/ndvi_ca_' + str(week) + '.png')
    plt.close(fig)
    return

def get_data(nrows, ncols, nyrs, nobs):
    files = get_files('nc4')
    files.pop(0)
    #del files[-62:] # remove this line

    # initialize matrix to hold 2-weekly observations for all years
    ndvi = np.zeros((nyrs*nobs, nrows, ncols))
    count = 0
    for file in files:
        print(file)
        year = file[13:18]
        data = read_file(file)
        t = data.variables['time'][:]
        nsteps = len(t)

        for step in range(nsteps):
            date = time_to_str(t[step], year)
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
    return ndvi

def calc_clim_avg(mat_all, nyrs, nobs, nrows, ncols):
    # nobs are the number of observations in one year
    # nobs = 24 means observations averaged over every 2 weeks,
    # e.g. Jan 1, Jan 15, Feb 1, Feb 15, ...
    mat_avg = np.zeros((nobs, nrows, ncols))
    for i in range(nobs):
        inds = nobs * np.arange(nyrs) + i
        mat_avg[i,:,:] = np.nanmean(mat_all[inds,:,:], axis=0)
    return mat_avg

def calc_deviation(mat_all, mat_avg, nyrs, nobs):
    anom = np.copy(mat_all)
    for i in range(nobs):
        inds = nobs * np.arange(nyrs) + i
        anom[inds,:,:] = mat_all[inds,:,:] - mat_avg[i,:,:]
    return anom

def plot_anom(data):
    # map data
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    heatmap = ax.pcolor(data, cmap=plt.cm.RdBu, vmin=np.nanmin(data), vmax=np.nanmax(data))
    heatmap.cmap.set_under('black')
    bar = fig.colorbar(heatmap, extend='both')
    bar.ax.set_ylabel('ndvi anomaly', rotation=270)
    ax.invert_yaxis()
    plt.axis('off')
    plt.show()
    plt.close(fig)
    return
