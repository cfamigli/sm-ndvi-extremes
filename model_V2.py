import FVD
import pandas as pd
import GLDAS as gld
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def plot_ROC(y_test, pred):
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()
    plt.show()
    return

def main():
    pct = 15
    nrows = 720
    ncols = 1440

    os.chdir("matfiles")
    mask = FVD.loadmat('land_mask.mat')

    for cols in [[1,2,3,4], [0,2,3,4], [0,1,3,4], [0,1,2,4], [0,1,2,3]]:
        y = FVD.loadmat('quads_D_v.1_e.25.mat').reshape(-1,1)
        y[y==1] = 0 # F-
        y[y==2] = 1 # F+
        X = FVD.loadmat('X_GLDAS.mat')
        print(cols)
        X = np.delete(X, cols, 1)
        inds = (np.isfinite(y).any(axis=1) & np.isfinite(X).all(axis=1))
        X = X[inds]
        y = y[inds]

        X_stan = StandardScaler().fit_transform(X)
        #print(np.cov(X_stan.T))
        X_train, X_test, y_train, y_test = train_test_split(X_stan, y, test_size=0.2, random_state=42)

        logr = LogisticRegression(penalty='l2').fit(X_train, y_train)
        pred_logr = logr.predict(X_test)
        print(logr.score(X_test, y_test)) # fraction of correct predictions
        print(np.exp(logr.coef_))
        print(logr.intercept_)
        print(metrics.confusion_matrix(y_test, pred_logr))
        print(metrics.classification_report(y_test, pred_logr))
        print('............................')
        #####################################

        plot_ROC(y_test, pred_logr)

    print('............................')
    print('............................')
    print('............................')
    print('............................')

    return

if __name__ == "__main__":
    main()
