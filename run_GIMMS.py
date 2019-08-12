
import numpy as np
import GIMMS as gm
from netCDF4 import Dataset
import os
import sys
from skimage.measure import block_reduce

np.set_printoptions(threshold=sys.maxsize)

os.chdir("../GIMMS/data")
files = gm.get_files('nc4')

for file in files:
    print(file)
    year = file[13:18]
    data = gm.read_file(file)
    time = data.variables['time'][:]
    print(time)
    nsteps = len(time)

    for step in range(nsteps):
        date = gm.time_to_str(time[step], year)
        print(date)
        nd = data.variables['ndvi'][step,:,:]
        nd = nd.astype(float)
        nd[nd<0] = float('nan')
        nd = nd / 1e4
        # resample ndvi observation to coarser grid
        nd = block_reduce(nd, block_size=(3,3), func=np.nanmean)
        nd = nd[250:300, 700:800]
        print(nd)
        gm.plot_scene(nd)
    data.close()
    break
