
from netCDF4 import Dataset
import numpy as np
import glob

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

def calc_overall_average(files, nrows, ncols):
    average = np.zeros((nrows,ncols))
    data = read_file(files)
    average = np.nanmean(data.variables['LAI'][:], axis=0)
    return np.flipud(average)

def get_stack(files, nrows, ncols):
    data = read_file(files)
    stack = data.variables['LAI'][:]
    stack = stack[:,::-1,:]
    return stack
