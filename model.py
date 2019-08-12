import FVD
import GLDAS as gld
import numpy as np
import pandas as pd
import worldclim
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_scatter_density(x, y, xlab):
    idx = (np.isfinite(x) & np.isfinite(y))
    xx = x[idx].reshape(-1,)
    yy = y[idx].reshape(-1,)

    ref = np.linspace(0, 1, 20)

    cmap = plt.cm.gist_earth_r#cubehelix_r
    cmap.set_under(color='white')
    plt.figure(figsize=(5,5))
    plt.hist2d(xx, yy, bins=(100,100), cmap=cmap, cmin=0, vmin=1)
    plt.plot(ref, ref, c='k', linewidth=1, linestyle='--')
    plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)),
        c='k', linewidth=1.5, linestyle='-')
    plt.ylim([0,1])
    plt.ylabel('Predicted FVD')
    plt.xlabel(xlab)
    #plt.colorbar()
    plt.show()
    plt.close()
    return

def remove_correlated_features(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    print(corr_matrix.to_string())

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    print(upper)

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]
    print(to_drop)

    # Drop features
    df = df.drop(df.columns[to_drop], axis=1)
    upper = upper.drop(upper.columns[to_drop], axis=1)
    print(upper)
    return df.values

pct = 15
val = 1
nrows = 720
ncols = 1440
n_comp = 2

os.chdir("matfiles")
mask = FVD.loadmat('land_mask.mat')

F = FVD.loadmat('F_' + str(pct) + '.mat')
D = FVD.loadmat('D_' + str(pct) + '.mat')
F[(abs(F)<val) & (abs(D)<val)] = float('nan')
D[(abs(F)<val) & (abs(D)<val)] = float('nan')
full = abs(F) / (abs(F) + abs(D))
full = full.reshape(-1,1)
#np.random.shuffle(full)

'''X = np.array([])
for comp in range(1, n_comp+1):
    X = np.append(X, FVD.loadmat('PC' + str(comp)
        + '_TS.mat').reshape(-1,1), axis=1) if X.size else FVD.loadmat('PC'
        + str(comp) + '_TS.mat').reshape(-1,1)'''
X = gld.prepare_mat_with_statistics(nrows, ncols)
print(X.shape)

X[X==-9999] = float('nan')
inds_X = ~np.isnan(X).any(axis=1)
inds_y = ~np.isnan(full).any(axis=1)
X = X[(inds_X) & (inds_y)]
y = full[(inds_X) & (inds_y)]
#X = FVD.loadmat('X_statistics.mat')
#y = FVD.loadmat('FVD_statistics.mat')
print(X)
print(X.shape)
print(gld.compute_corr_with_FVD(X, y))
X = gld.select_subset_from_corr(X, [4,10,14])
plt.figure()
plt.imshow(np.corrcoef(X.T), cmap='RdBu')
plt.colorbar()
plt.show()
X = remove_correlated_features(pd.DataFrame(X))
print(X.shape)
os.chdir("..")

reg = LinearRegression().fit(X, y)
pred = reg.predict(X)
print(reg.score(X, y))
plot_scatter_density(y, pred, 'obs')

#####################################

rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(X, y.reshape((-1,)))
pred = rf.predict(X)
print(rf.score(X, y))
print(np.sqrt(metrics.mean_squared_error(y.reshape((-1,)), pred)))
plot_scatter_density(y.reshape((-1,)), pred, 'Observed FVD')

# Get numerical feature importances
print(rf.feature_importances_)
print(np.sum(rf.feature_importances_))

plt.figure(figsize=(4,4))
plt.scatter(X[:,0], X[:,1], c=pred, s=4, cmap = LinearSegmentedColormap.from_list(
    'name', ['red', 'palegoldenrod','blue']), alpha=0.2)
plt.show()

#####################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(X_train, y_train.reshape((-1,)))
pred = rf.predict(X_test)
print(rf.score(X_test, y_test))
print(np.sqrt(metrics.mean_squared_error(y_test.reshape((-1,)), pred)))
plot_scatter_density(y_test.reshape((-1,)), pred, 'Observed FVD')

print(rf.feature_importances_)

plt.figure(figsize=(4,4))
plt.scatter(X_test[:,0], X_test[:,1], c=pred, s=4, cmap = LinearSegmentedColormap.from_list(
    'name', ['red', 'palegoldenrod','blue']), alpha=0.3)
plt.show()

#FVD.plot_fvd_dist(pred)'''
