
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import scipy.io
from scipy.stats import percentileofscore

def savemat(mat3d, matfile):
    # easy way to save 3d arrays, but is very slow for large ones.
    scipy.io.savemat(matfile, mdict={'data': mat3d}, oned_as='row')
    return

def loadmat(matfile):
    data = scipy.io.loadmat(matfile)['data']
    return data

def plot_fvd(fvd, perc, vmin, vmax, mask=None, kind=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    fvd[fvd==0] = float('nan')
    if mask is not None:
        fvd[(~np.isfinite(fvd)) & (mask<0)] = -9999
        fvd[mask==0] = float('nan')
    cmap = LinearSegmentedColormap.from_list('name', ['red', 'palegoldenrod','blue'])
    cmap.set_under(color='lightgray')
    heatmap = ax.pcolor(fvd, cmap=cmap, vmin=vmin,
        vmax=vmax)
    bar = fig.colorbar(heatmap)
    bar.ax.set_ylabel('fvd', rotation=270, labelpad=20)
    ax.invert_yaxis()
    plt.axis('off')
    plt.show()
    #if kind is not None:
        #plt.savefig('../../plots/fvd_' + kind + '_' + str(perc) + '.png')
    plt.close(fig)
    return

def plot_dist_at_pixel(data, x, row, col):
    data_to_plot = data[:, row, col]
    plt.figure(figsize=(4,4))
    plt.hist(data_to_plot, bins=20, color='dodgerblue', edgecolor='white')
    plt.axvline(x=x, c='k', linewidth=0.5, linestyle='--')
    plt.title('row: %d, col: %d' % (row, col))
    plt.xlabel('NDVI change')
    plt.tight_layout()
    plt.show()
    return

def plot_hist(data, mask=None):
    if mask is not None:
        if data.ndim==3:
            for step in range(data.shape[0]):
                d = np.copy(data[step,:,:])
                d[mask==0] = float('nan')
                data[step,:,:] = d
        else:
            data[mask==0] = float('nan')
    data[data==-9999] = float('nan')
    data = data[np.isfinite(data)].reshape(-1,)
    plt.figure(figsize=(4,4))
    plt.hist(data, bins=20, color='dodgerblue', edgecolor='white', density=True, alpha=0.7)
    plt.tight_layout()
    plt.show()
    return


def extract(pixels, data):
    mat = np.zeros((len(pixels), data.shape[0]))
    count = 0
    for pair in pixels:
        mat[count,:] = data[:, pair[0], pair[1]]
        count += 1
    return mat

def calc_fvd(sm_low_anom, sm_high_anom, ndvi, ndvi_clim_avg,
    nyrs, nobs, nrows, ncols, pct, mask, temp):
    # calculates fvd per pixel. returns array of nrows by ncols (fvd map).
    fvd = np.zeros((nrows, ncols))

    d = np.copy(ndvi)
    d[~np.isfinite(sm_low_anom)] = float('nan')
    dsub = np.zeros((nyrs*nobs, nrows, ncols))

    f = np.copy(ndvi)
    f[~np.isfinite(sm_high_anom)] = float('nan')
    fsub = np.zeros((nyrs*nobs, nrows, ncols))

    for ob in range(nyrs*nobs):
        ca = np.copy(ndvi_clim_avg[np.mod(ob,nobs),:,:])
        dsub[ob,:,:] = np.copy(ca)
        dsub[ob,:,:][~np.isfinite(sm_low_anom[ob,:,:])] = float('nan')
        dsub[ob,:,:][~np.isfinite(d[ob,:,:])] = float('nan')
        fsub[ob,:,:] = np.copy(ca)
        fsub[ob,:,:][~np.isfinite(sm_high_anom[ob,:,:])] = float('nan')
        fsub[ob,:,:][~np.isfinite(f[ob,:,:])] = float('nan')

    #pixels = [(400,570),(460,1310),(250,1150),(250,325),(440,800)]
    pixels = [(38,362),(70,316),(71,280),(89,1146),(98,1170)]
    #savemat(extract(pixels, f-fsub), 'extract_F.mat')
    #savemat(extract(pixels, f), 'extract_f_only.mat')
    #savemat(extract(pixels, fsub), 'extract_fsub_only.mat')
    #plot_hist(f-fsub, mask)
    numerator = np.nansum(f,axis=0) - np.nansum(fsub,axis=0)
    #savemat(np.nansum(f,axis=0), 'nansum_f.mat')
    #savemat(np.nansum(fsub,axis=0), 'nansum_fsub.mat')
    stack = f-fsub
    #savemat(np.nansum(f-fsub, axis=0), 'nansum_f_fsub.mat')
    stack[stack==0] = float('nan')
    percent_change = abs(stack)/fsub*100
    savemat(np.nansum(percent_change, axis=0), 'percent_change_F.mat')

    '''temp_1mo_before = np.zeros(4000000)
    temp_during = np.zeros(4000000)
    temp_1mo_after = np.zeros(4000000)

    count = 0
    for layer in range(nyrs*nobs):
        for row in range(nrows):
            for col in range(ncols):
                if np.isnan(stack[layer,row,col]):
                    continue
                elif (layer!=0) & (layer!=nyrs*nobs-1):
                    temp_1mo_before[count] = percentileofscore(temp[:,row,col],temp[layer-1,row,col])
                    temp_during[count] = percentileofscore(temp[:,row,col],temp[layer,row,col])
                    temp_1mo_after[count] = percentileofscore(temp[:,row,col],temp[layer+1,row,col])
                    count += 1
                elif (layer==0):
                    temp_during[count] = percentileofscore(temp[:,row,col],temp[layer,row,col])
                    temp_1mo_after[count] = percentileofscore(temp[:,row,col],temp[layer+1,row,col])
                    count += 1
                elif (layer==nyrs*nobs-1):
                    temp_1mo_before[count] = percentileofscore(temp[:,row,col],temp[layer-1,row,col])
                    temp_during[count] = percentileofscore(temp[:,row,col],temp[layer,row,col])
                    count += 1

    savemat(temp_1mo_before, 'temp_1mo_before.mat')
    savemat(temp_during, 'temp_during.mat')
    savemat(temp_1mo_after, 'temp_1mo_after.mat')'''

    '''stack[stack<0] = 0
    stack[stack>0] = 1
    proportion_positive = np.nansum(stack, axis=0)/np.nansum(np.isfinite(stack), axis=0)
    savemat(proportion_positive, 'proportion_positive.mat')
    '''
    #savemat(np.nansum(f,axis=0), 'F_NDVI_' + str(pct) + '.mat')
    #savemat(np.nansum(fsub,axis=0), 'F_NDVI_sub' + str(pct) + '.mat')
    savemat(numerator, 'sm_numerator_monthly_controlled.mat')
    #plot_fvd(numerator, perc=pct, kind='num')

    savemat(extract(pixels, d-dsub), 'extract_D.mat')
    #plot_hist(d-dsub, mask)
    denominator = np.nansum(d,axis=0) - np.nansum(dsub,axis=0)
    #savemat(np.nansum(d,axis=0), 'D_NDVI_' + str(pct) + '.mat')
    #savemat(np.nansum(dsub,axis=0), 'D_NDVI_sub' + str(pct) + '.mat')
    savemat(denominator, 'sm_denominator_monthly_controlled.mat')
    #plot_fvd(denominator, perc=pct, kind='den')

    '''for pair in [(400,570),(460,1310),(250,1150),(250,325),(440,800)]:
        plot_dist_at_pixel(f-fsub, numerator[pair], pair[0], pair[1])'''

    fvd = abs(numerator) / (abs(numerator)+abs(denominator))
    #savemat(fvd, 'FVD_' + str(pct) + '.mat')
    #plot_fvd(fvd, perc=pct, kind='full')
    return fvd

def match_lc(lc, fvd):
    arr = []
    for type in range(int(np.nanmax(lc))+1):
        subarr = fvd[lc==type]
        subarr = subarr[np.isfinite(subarr)]
        arr.append(subarr)
    return arr

def plot_lc(data):
    plt.figure(figsize=(4,10))
    plt.violinplot(data, widths=0.7, vert=False, showmeans=True, showextrema=False)
    plt.ylabel('F')
    plt.xlim([-4,6])
    plt.yticks(np.arange(0,18))
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gca().xaxis.grid(True)
    plt.show()
    return

def plot_fvd_dist(fvd):
    ub = np.nansum(np.isfinite(fvd))
    fvd = fvd[(fvd>=0) & (fvd<=1)].reshape(-1,1)
    weights = np.ones_like(fvd)/float(len(fvd))
    plt.figure(figsize=(6,5))
    plt.gcf().subplots_adjust(left=0.2)
    plt.hist(fvd, bins=50, weights=weights, color='cornflowerblue', ec='white', linewidth=1)
    plt.xlabel('fvd')
    plt.ylabel('probability')
    plt.show()
    return

def plot_general_dist(data, color):
    ub = np.nansum(np.isfinite(data))
    data[data==-9999] = float('nan')
    print(np.nanmean(data))
    data = data[np.isfinite(data)].reshape(-1,1)
    weights = np.ones_like(data)/float(len(data))
    plt.figure(figsize=(6,5))
    plt.gcf().subplots_adjust(left=0.2)
    plt.hist(data, bins=50, weights=weights, color=color, ec='white', linewidth=1)
    plt.xlabel('net ndvi change')
    plt.ylabel('probability')
    plt.show()
    return

def plot_PCA(Y, fvd):
    plt.figure(figsize=(7,7))
    '''x1 = Y[:,0].reshape(-1,1)
    x2 = Y[:,1].reshape(-1,1)
    plt.scatter(x1[fvd<0.5], x2[fvd<0.5], c='red')
    plt.scatter(x1[fvd>=0.5], x2[fvd>=0.5], c='blue')'''
    plt.scatter(Y[:,0], fvd)
    plt.show()
    return

def plot_quads(data, vmin, vmax, mask=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    data[data==0] = float('nan')
    if mask is not None:
        data[(~np.isfinite(data)) & (mask<0)] = -9999
        data[mask==0] = float('nan')
    #cmap = ListedColormap(['darkgreen','gold','crimson','royalblue'])
    #bounds = [1,2,3,4,5]
    cmap = ListedColormap(['crimson','royalblue'])
    bounds = [1,2,3]
    heatmap = ax.pcolor(data, cmap=cmap, vmin=vmin,
        vmax=vmax, norm=BoundaryNorm(bounds,cmap.N))
    heatmap.cmap.set_under(color='lightgray')
    bar = fig.colorbar(heatmap)
    ax.invert_yaxis()
    plt.axis('off')
    plt.show()
    plt.close(fig)
    return

def define_quadrants(F, D, mask):
    #Q1: F+,D+; Q2: F-,D+; Q3: F-,D-; Q4: F+,D-
    Q = np.copy(mask)

    Q[(F>0) & (D>0)] = 2 #1
    Q[(F<0) & (D>0)] = 1 #2
    Q[(F<0) & (D<0)] = 1 #3
    Q[(F>0) & (D<0)] = 2 #4
    Q[Q<=0] = float('nan')

    print(np.nansum(np.isfinite(Q)))
    #Q[~np.isclose(abs(D),abs(F),0.5)] = float('nan')
    Q[(abs(D)>abs(F)) | (np.isclose(abs(D),abs(F),0.25))] = float('nan') #
    print(np.nansum(np.isfinite(Q)))

    return Q

def plot_scatter_density(x, y, xlab):
    idx = (np.isfinite(x) & np.isfinite(y))
    xx = x[idx]#.reshape(-1,)
    yy = y[idx]#.reshape(-1,)
    print('..')
    print(xx.shape)
    print(yy.shape)

    #ref = np.linspace(0, 1, 20)

    cmap = plt.cm.gist_earth_r#cubehelix_r
    cmap.set_under(color='white')
    plt.figure(figsize=(5,5))
    plt.hist2d(xx, yy, bins=(100,100), cmap=cmap, cmin=0)
    #plt.plot(ref, ref, c='k', linewidth=1, linestyle='--')
    plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)),
        c='k', linewidth=1.5, linestyle='-')
    #plt.ylim([0,1])
    plt.ylabel('net ndvi change')
    plt.xlabel(xlab)
    #plt.colorbar()
    plt.show()
    plt.close()
    return

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

def plot_lat(lat, av, color):
    plt.figure(figsize=(6,15))
    matplotlib.rcParams.update({'font.size': 22})
    plt.plot(np.flipud(av),lat,c=color,linewidth=1.5)
    plt.axvline(x=50., c='k', linewidth=0.5, linestyle='--')
    #plt.xlim([0,102])
    plt.xlim([0,150])
    plt.ylim([-90,90])
    #plt.xlabel('percent of anomalies\nyielding negative ndvi change')
    plt.xlabel('cumulative percent\nndvi change')
    plt.ylabel('latitude')
    plt.tight_layout()
    plt.show()
    return
