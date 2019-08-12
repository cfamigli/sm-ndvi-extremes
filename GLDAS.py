
from netCDF4 import Dataset
import numpy as np
import glob
import os
import LAI as lpy
import FVD
import worldclim
import GIMMS as gmm
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
import itertools
from scipy.stats import skew

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

def calc_overall_average(files, variable, nrows, ncols):
    average = np.zeros((len(files),nrows-120,ncols))
    count = 0
    for file in files:
        data = read_file(file)
        average[count,:,:] = data.variables[variable][:,:,:]
        count += 1
    average = np.nanmean(average, axis=0)
    average = np.vstack([np.zeros((120,ncols))/0, average])
    average[average<-9000] = float('nan')
    if variable=='Tair_f_inst':
        average -= 273.15
    return np.flipud(average)

def plot_scene(data, vmin, vmax, mask=None, name=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(14,6)
    data[data==0] = float('nan')
    if mask is not None:
        data[(~np.isfinite(data)) & (mask<0)] = -9999
        data[mask==0] = float('nan')
    cmap = LinearSegmentedColormap.from_list('name', ['red', 'palegoldenrod','blue'])
    cmap.set_under(color='lightgray')
    heatmap = ax.pcolor(data, cmap=cmap, vmin=vmin,
        vmax=vmax)
    bar = fig.colorbar(heatmap)
    bar.ax.set_ylabel('data', rotation=270, labelpad=20)
    ax.invert_yaxis()
    plt.axis('off')
    plt.show()
    #if name is not None:
        #plt.savefig('../plots/PC' + name + '.png')
    plt.close(fig)
    return

def calc_vpd(temp, q, p):
    # temp in Celsius, q unitless, p in Pa
    esat = 6.1094 * np.exp(17.625 * temp / (temp + 243.04))
    e = q * (p - 0.378) / 0.622 / 100
    vpd = esat - e
    return vpd

def calc_entropy(var_12_month_avg):
    # calculate entropy D = \sum(p log(p/q)) from Feng et al paper
    qm = 1/12
    R = np.nansum(var_12_month_avg, axis=0)
    pm = var_12_month_avg/R
    D = np.nansum(pm * np.log2(pm/qm), axis=0)
    return D

def get_stack(files, variable, nrows, ncols):
    stack = np.zeros((len(files),nrows,ncols))
    count = 0
    for file in files:
        data = read_file(file)
        dv = data.variables[variable][0,:,:]
        dv = np.vstack([np.zeros((120,ncols))/0, dv])
        stack[count,:,:] = dv
        stack[count,:,:] = np.flipud(stack[count,:,:])
        count += 1
    stack[stack<-9000] = float('nan')
    if variable=='Tair_f_inst':
        stack -= 273.15
    return stack

def prepare_PCA_mat_with_avgs(nrows, ncols, wc=False):
    os.chdir('../../GLDAS')
    files = get_files('nc4')
    temp = calc_overall_average(files, 'Tair_f_inst', nrows, ncols)
    rad = calc_overall_average(files, 'SWdown_f_tavg', nrows, ncols)
    prec_rate = calc_overall_average(files, 'Rainf_tavg', nrows, ncols)
    PET = calc_overall_average(files, 'PotEvap_tavg', nrows, ncols)

    os.chdir('../LAI')
    files = get_files('nc4')
    lai = lpy.calc_overall_average(files[0], nrows, ncols)
    lai[lai<0] = float('nan')

    Xw = np.array([])
    if wc:
        os.chdir('../worldclim/bio')
        var_ID = range(1,20)
        for var in var_ID:
            wc = worldclim.read_and_reduce('wc2.0_bio_5m_' + str(var).zfill(2) + '.tif', nrows, ncols)
            switcher = {
                12: 2000, 13: 400, 14: 75, 15: 175, 16: 1000, 17: 350, 18: 800, 19: 450
            }
            if var>=12:
                z = switcher.get(var)
                wc[wc>z] = float('nan')
            Xw = np.hstack((Xw, wc.reshape(-1,1))) if Xw.size else wc.reshape(-1,1)
        os.chdir('..')

    print(temp.reshape(-1,1).shape)
    print(rad.reshape(-1,1).shape)
    print(prec_rate.reshape(-1,1).shape)
    print(PET.reshape(-1,1).shape)
    print(lai.reshape(-1,1).shape)
    X = np.hstack((Xw, temp.reshape(-1,1), rad.reshape(-1,1),
        prec_rate.reshape(-1,1), PET.reshape(-1,1), lai.reshape(-1,1))) if Xw.size else np.hstack((temp.reshape(-1,1),
        rad.reshape(-1,1), prec_rate.reshape(-1,1), PET.reshape(-1,1), lai.reshape(-1,1)))
    return X

def prepare_PCA_mat_with_time_series(nrows, ncols):
    os.chdir('../../GLDAS')
    fg = get_files('nc4')
    temp = get_stack(fg, 'Tair_f_inst', nrows, ncols)
    rad = get_stack(fg, 'SWdown_f_tavg', nrows, ncols)
    prec_rate = get_stack(fg, 'Rainf_tavg', nrows, ncols)
    PET = get_stack(fg, 'PotEvap_tavg', nrows, ncols)
    vpd = calc_vpd(temp, get_stack(fg, 'Qair_f_inst', nrows, ncols),
        get_stack(fg, 'Psurf_f_inst', nrows, ncols))

    os.chdir('../LAI')
    fl = get_files('nc4')
    lai = lpy.get_stack(fl[0], nrows, ncols)
    lai[lai<0] = float('nan')

    os.chdir('../GIMMS/data')
    ndvi = gmm.get_data(nrows, ncols, nyrs=34, nobs=24)
    ndvi[ndvi<-1] = float('nan')

    full_stack = np.concatenate((temp, rad, prec_rate, PET, vpd, lai, ndvi), axis=0)
    X = np.zeros((nrows*ncols, full_stack.shape[0]))
    mask = np.arange(nrows*ncols).reshape(nrows,ncols)
    for row in range(nrows):
        for col in range(ncols):
            X[mask[row,col], :] = full_stack[:, row, col]
    return X

def save_skewness_mats(nrows, ncols):
    os.chdir('../GLDAS')
    fg = get_files('nc4')
    temp = get_stack(fg, 'Tair_f_inst', nrows, ncols)
    rad = get_stack(fg, 'SWdown_f_tavg', nrows, ncols)
    prec_rate = get_stack(fg, 'Rainf_tavg', nrows, ncols)
    PET = get_stack(fg, 'PotEvap_tavg', nrows, ncols)
    vpd = calc_vpd(temp, get_stack(fg, 'Qair_f_inst', nrows, ncols),
        get_stack(fg, 'Psurf_f_inst', nrows, ncols))

    os.chdir('../LAI')
    fl = get_files('nc4')
    lai = lpy.get_stack(fl[0], nrows, ncols)
    lai[lai<0] = float('nan')

    skew_temp = np.ones((nrows,ncols))*np.nan
    skew_rad = np.ones((nrows,ncols))*np.nan
    skew_prec = np.ones((nrows,ncols))*np.nan
    skew_PET = np.ones((nrows,ncols))*np.nan
    skew_vpd = np.ones((nrows,ncols))*np.nan
    skew_LAI = np.ones((nrows,ncols))*np.nan

    for row in range(nrows):
        print(row)
        for col in range(ncols):
            t = temp[:,row,col]
            r = rad[:,row,col]
            pr = prec_rate[:,row,col]
            pt = PET[:,row,col]
            v = vpd[:,row,col]
            l = lai[:,row,col]

            skew_temp[row,col] = skew(t[~np.isnan(t)])
            skew_rad[row,col] = skew(r[~np.isnan(r)])
            skew_prec[row,col] = skew(pr[~np.isnan(pr)])
            skew_PET[row,col] = skew(pt[~np.isnan(pt)])
            skew_vpd[row,col] = skew(v[~np.isnan(v)])
            skew_LAI[row,col] = skew(l[~np.isnan(l)])

    FVD.savemat(skew_temp, 'skew_temp.mat')
    FVD.savemat(skew_rad, 'skew_rad.mat')
    FVD.savemat(skew_prec, 'skew_prec.mat')
    FVD.savemat(skew_PET, 'skew_PET.mat')
    FVD.savemat(skew_vpd, 'skew_vpd.mat')
    FVD.savemat(skew_LAI, 'skew_LAI.mat')
    return

def prepare_mat_with_statistics(nrows, ncols):
    os.chdir('../GLDAS')
    fg = get_files('nc4')
    temp = get_stack(fg, 'Tair_f_inst', nrows, ncols)
    rad = get_stack(fg, 'SWdown_f_tavg', nrows, ncols)
    prec_rate = get_stack(fg, 'Rainf_tavg', nrows, ncols)
    PET = get_stack(fg, 'PotEvap_tavg', nrows, ncols)
    vpd = calc_vpd(temp, get_stack(fg, 'Qair_f_inst', nrows, ncols),
        get_stack(fg, 'Psurf_f_inst', nrows, ncols))

    temp += 273.15

    os.chdir('../LAI')
    fl = get_files('nc4')
    lai = lpy.get_stack(fl[0], nrows, ncols)
    lai[lai<0] = float('nan')

    os.chdir('../GIMMS/data')
    ndvi = gmm.get_data(nrows, ncols, nyrs=34, nobs=24)
    ndvi[ndvi<-1] = float('nan')

    temp_mean = np.nanmean(temp, axis=0)
    rad_mean = np.nanmean(rad, axis=0)
    prec_mean = np.nanmean(prec_rate, axis=0)
    PET_mean = np.nanmean(PET, axis=0)
    vpd_mean = np.nanmean(vpd, axis=0)
    lai_mean = np.nanmean(lai, axis=0)
    ndvi_mean = np.nanmean(ndvi, axis=0)

    temp_std = np.nanstd(temp, axis=0)
    rad_std = np.nanstd(rad, axis=0)
    prec_std = np.nanstd(prec_rate, axis=0)
    PET_std = np.nanstd(PET, axis=0)
    vpd_std = np.nanstd(vpd, axis=0)
    lai_std = np.nanstd(lai, axis=0)
    ndvi_std = np.nanstd(ndvi, axis=0)

    obs = temp.shape[0]
    nmo = 12
    temp_reduced = np.zeros((nmo,nrows,ncols))
    rad_reduced = np.zeros((nmo,nrows,ncols))
    prec_reduced = np.zeros((nmo,nrows,ncols))
    PET_reduced = np.zeros((nmo,nrows,ncols))
    vpd_reduced = np.zeros((nmo,nrows,ncols))
    ndvi_reduced = np.zeros((nmo,nrows,ncols))
    for month in range(nmo):
        print(np.arange(month,obs,nmo))
        temp_reduced[month,:,:] = np.nanmean(temp[np.arange(month,obs,nmo),:,:], axis=0)
        rad_reduced[month,:,:] = np.nanmean(rad[np.arange(month,obs,nmo),:,:], axis=0)
        prec_reduced[month,:,:] = np.nanmean(prec_rate[np.arange(month,obs,nmo),:,:], axis=0)
        PET_reduced[month,:,:] = np.nanmean(PET[np.arange(month,obs,nmo),:,:], axis=0)
        vpd_reduced[month,:,:] = np.nanmean(vpd[np.arange(month,obs,nmo),:,:], axis=0)
        ndvi_reduced[month,:,:] = np.nanmean(ndvi[np.arange(month,obs,nmo),:,:], axis=0)

    '''temp_rad_corr = np.zeros((nrows,ncols))
    temp_prec_corr = np.zeros((nrows,ncols))
    temp_PET_corr = np.zeros((nrows,ncols))
    temp_vpd_corr = np.zeros((nrows,ncols))
    temp_lai_corr = np.zeros((nrows,ncols))
    rad_prec_corr = np.zeros((nrows,ncols))
    rad_PET_corr = np.zeros((nrows,ncols))
    rad_vpd_corr = np.zeros((nrows,ncols))
    rad_lai_corr = np.zeros((nrows,ncols))
    prec_PET_corr = np.zeros((nrows,ncols))
    prec_vpd_corr = np.zeros((nrows,ncols))
    prec_lai_corr = np.zeros((nrows,ncols))
    PET_vpd_corr = np.zeros((nrows,ncols))
    PET_lai_corr = np.zeros((nrows,ncols))
    vpd_lai_corr = np.zeros((nrows,ncols))
    for row in range(nrows):
        print(row)
        for col in range(ncols):
            temp_rad_corr[row, col] = np.correlate(temp_reduced[:,row,col], rad_reduced[:,row,col])
            temp_prec_corr[row, col] = np.correlate(temp_reduced[:,row,col], prec_reduced[:,row,col])
            temp_PET_corr[row, col] = np.correlate(temp_reduced[:,row,col], PET_reduced[:,row,col])
            temp_vpd_corr[row, col] = np.correlate(temp_reduced[:,row,col], vpd_reduced[:,row,col])
            temp_lai_corr[row, col] = np.correlate(temp_reduced[:,row,col], lai[:,row,col])
            rad_prec_corr[row, col] = np.correlate(rad_reduced[:,row,col], prec_reduced[:,row,col])
            rad_PET_corr[row, col] = np.correlate(rad_reduced[:,row,col], PET_reduced[:,row,col])
            rad_vpd_corr[row, col] = np.correlate(rad_reduced[:,row,col], vpd_reduced[:,row,col])
            rad_lai_corr[row, col] = np.correlate(rad_reduced[:,row,col], lai[:,row,col])
            prec_PET_corr[row, col] = np.correlate(prec_reduced[:,row,col], PET_reduced[:,row,col])
            prec_vpd_corr[row, col] = np.correlate(prec_reduced[:,row,col], vpd_reduced[:,row,col])
            prec_lai_corr[row, col] = np.correlate(prec_reduced[:,row,col], lai[:,row,col])
            PET_vpd_corr[row, col] = np.correlate(PET_reduced[:,row,col], vpd_reduced[:,row,col])
            PET_lai_corr[row, col] = np.correlate(PET_reduced[:,row,col], lai[:,row,col])
            vpd_lai_corr[row, col] = np.correlate(vpd_reduced[:,row,col], lai[:,row,col])'''

    temp_D = calc_entropy(temp_reduced)
    rad_D = calc_entropy(rad_reduced)
    prec_D = calc_entropy(prec_reduced)
    PET_D = calc_entropy(PET_reduced)
    vpd_D = calc_entropy(vpd_reduced)
    lai_D = calc_entropy(lai)
    ndvi_D = calc_entropy(ndvi_reduced)

    '''full_stack = np.array([temp_mean,
        rad_mean,
        prec_mean,
        PET_mean,
        vpd_mean,
        lai_mean])
        #temp_std,rad_std,prec_std,PET_std,vpd_std,lai_std,ndvi_std,temp_D,
        #rad_D,prec_D,PET_D,vpd_D,lai_D,ndvi_D])
        #temp_rad_corr, temp_prec_corr, temp_PET_corr, temp_vpd_corr, temp_lai_corr,
        #rad_prec_corr, rad_PET_corr, rad_vpd_corr, rad_lai_corr, prec_PET_corr,
        #prec_vpd_corr, prec_lai_corr, PET_vpd_corr, PET_lai_corr, vpd_lai_corr])
    print(full_stack.shape)
    X = np.zeros((nrows*ncols, full_stack.shape[0]))
    print(X.shape)
    mask = np.arange(nrows*ncols).reshape(nrows,ncols)
    for row in range(nrows):
        for col in range(ncols):
            X[mask[row,col], :] = full_stack[:, row, col]

    #X_stan = StandardScaler().fit_transform(X)'''

    FVD.savemat(temp_mean, 'temp_mean.mat')
    FVD.savemat(rad_mean, 'rad_mean.mat')
    FVD.savemat(prec_mean, 'prec_mean.mat')
    FVD.savemat(PET_mean, 'PET_mean.mat')
    FVD.savemat(vpd_mean, 'vpd_mean.mat')
    FVD.savemat(lai_mean, 'lai_mean.mat')
    FVD.savemat(ndvi_mean, 'ndvi_mean.mat')

    FVD.savemat(temp_std, 'temp_std.mat')
    FVD.savemat(rad_std, 'rad_std.mat')
    FVD.savemat(prec_std, 'prec_std.mat')
    FVD.savemat(PET_std, 'PET_std.mat')
    FVD.savemat(vpd_std, 'vpd_std.mat')
    FVD.savemat(lai_std, 'lai_std.mat')
    FVD.savemat(ndvi_std, 'ndvi_std.mat')

    FVD.savemat(temp_D, 'temp_D.mat')
    FVD.savemat(rad_D, 'rad_D.mat')
    FVD.savemat(prec_D, 'prec_D.mat')
    FVD.savemat(PET_D, 'PET_D.mat')
    FVD.savemat(vpd_D, 'vpd_D.mat')
    FVD.savemat(lai_D, 'lai_D.mat')
    FVD.savemat(ndvi_D, 'ndvi_D.mat')
    return

def compute_corr_with_FVD(X, fvd):
    corr_vec = np.zeros(X.shape[1])
    fvd = fvd.reshape((-1,))
    for column in range(X.shape[1]):
        print(X[:,column].shape)
        print(fvd.shape)
        print(column)
        corr_vec[column] = np.corrcoef(X[:,column], fvd)[0,1]
    return corr_vec

def select_subset_from_corr(X, inds):
    return X[:,inds]

def map_determined_PC(PC_mat, PC_component, counter, nrows, ncols):
    mask = np.arange(nrows*ncols)*1.
    run = 0
    for i in range(len(mask)):
        if mask[i] not in counter:
            mask[i] = float('nan')
        else:
            print(run)
            mask[i] = PC_mat[run, PC_component-1]
            run += 1
    return mask.reshape(nrows,ncols)

'''def list_uncorrelated_subsets(corr_mat):
    num_features = len(corr_mat)
    for feature in range(num_features):
        corr_vec = corr_mat[:,feature]
        corr_vec[corr_vec>0.5] = float('nan')
        subset = np.argwhere(~np.isnan(corr_vec))
        for pair in itertools.combinations(subset,r=2):
            if corr_mat[pair]>0.5:

    return list_of_subsets'''
