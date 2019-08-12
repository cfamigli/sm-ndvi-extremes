import numpy as np
import os
import glob
from skimage import io
from skimage.measure import block_reduce
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

def read_and_reduce(tif, nrows, ncols):
    im = io.imread(tif)*1.
    im[im>60000] = float('nan')
    im[im<-100] = float('nan')
    im_arr = block_reduce(im, block_size=(3,3), func=np.nanmean)
    assert im_arr.shape==(nrows, ncols)
    return im_arr

def get_files(ext):
    # input is extension type
    files = [i for i in glob.glob('*.{}'.format(ext)) if len(i)>10]
    files.sort()
    return files

def calc_annual_mean(dir, nrows, ncols):
    os.chdir(dir)
    files = get_files('tif')
    avg_arr = np.zeros((nrows, ncols))
    for file in files:
        avg_arr += read_and_reduce(file, nrows, ncols)
    avg_arr = avg_arr/len(files)
    os.chdir('..')
    return avg_arr

def plot_scatter(x, y, xlab):
    idx = (np.isfinite(x) & np.isfinite(y))
    xx = x[idx].reshape(-1,)
    yy = y[idx].reshape(-1,)

    cmap = plt.cm.cubehelix_r
    cmap.set_under(color='white')
    plt.figure(figsize=(7,5.5))
    plt.hist2d(xx, yy, bins=(100,100), cmap=cmap, cmin=1)
    #plt.ylim([0,1])
    plt.ylabel('pc2')
    plt.xlabel(xlab)
    plt.colorbar()
    plt.show()
    #plt.savefig('../../plots/full_' + xlab + '.png')
    plt.close()
    return
