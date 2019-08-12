import FVD
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

def plot_scene(data, ylab, mask=None, title=None, vmin=None, vmax=None, ):
    # map data
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    if mask is not None:
        data[(~np.isfinite(data)) & (mask<0)] = -9999
        data[mask==0] = float('nan')
    heatmap = ax.pcolor(data, cmap=plt.cm.YlGn, vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('lightgray')
    bar = fig.colorbar(heatmap, extend='both')
    bar.ax.set_ylabel(ylab, rotation=270, labelpad=40)
    ax.invert_yaxis()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)
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

    p = FVD.loadmat('percent_change_F.mat')
    F = FVD.loadmat('sm_numerator_monthly_controlled_fixed_masked.mat')

    #plot_scene(p, 'cumulative percent\nndvi change', mask=mask, title=None, vmin=1, vmax=500)

    plt.figure(figsize=(4,4))
    plt.scatter(FVD.loadmat('skew_ndvi.mat'), FVD.loadmat('temp_mean.mat'), c='k', s=2, marker='.')
    plt.show()

    return()

if __name__ == "__main__":
    main()
