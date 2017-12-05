
# coding: utf-8

'''
This module implements three methods to impute missing data.
1. randomForestRegression imputes continuous, ordinal or binary features
   by performing random forest regression on nearby 2000 points.
2. randomForestClassifier imputes categorical features by
   performing random forest classification on nearby 2000 points.
3. nearbyAverage imputes continuous, ordinal or binary features, if the
   missing values account for too much in a feature, by averaging
   the nearby 100 points.
params:
    train: training data, pandas dataframe
    data: data that has missing values, pandas dataframe
    columns_miss: column names that have missing values, list of string
    col: column that needs to be imputed, string
    n_neighbors: number of nearby points being fitted, int
    fill_train: if training data is imputed, then only nearby points with
                the same label are considered as nearby points. If validation
                or test data is imputed, then labels are ignored. a boolean.
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def randomForestRegression(train, data, columns_miss, col, n_neighbors=2000, fill_train=True):
    temp1 = train.drop(columns_miss, axis=1)
    temp2 = data.drop(columns_miss, axis=1)
    
    rows_miss = np.where(data[col] == -1)
    for row in rows_miss[0]:
        if fill_train:
        # if we are filling training data, then only consider same label;
        # else if we are filling validation or test data, then ignore label
            label = data.loc[row, 'target']
            known = temp1[(temp1[col] != -1) & (temp1['target'] == label)]
            unknown = temp2.iloc[row].drop(col)
        else:
            known = temp1[temp1[col] != -1].drop('target', axis=1)
            unknown = temp2.iloc[row].drop([col, 'target'])
            
        X, y = known.drop([col], axis=1), known[col]
       
        # find nearby points
        neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
        neigh.fit(X)
        ind_nei = neigh.kneighbors(unknown.values.reshape(1, -1), return_distance=False)
        X, y = X.iloc[ind_nei[0]], y.iloc[ind_nei[0]]
        
        # perform regression fit on nearby points
        regr = RandomForestRegressor(n_estimators=2000, random_state=0, n_jobs=-1)
        regr.fit(X, y)
        predicted = regr.predict(unknown.values.reshape(1, -1))
        data.loc[row, col] = predicted
        
        
def randomForestClassifier(train, data, columns_miss, col, n_neighbors=2000, fill_train=True):
    temp1 = train.drop(columns_miss, axis=1)
    temp2 = data.drop(columns_miss, axis=1)
    
    rows_miss = np.where(data[col] == -1)
    for row in rows_miss[0]:
        if fill_train:
        # if we are filling training data, then only consider same label;
        # else if we are filling validation or test data, then ignore label
            label = data.loc[row, 'target']
            known = temp1[(temp1[col] != -1) & (temp1['target'] == label)]
            unknown = temp2.iloc[row].drop(col)
        else:
            known = temp1[temp1[col] != -1].drop('target', axis=1)
            unknown = temp2.iloc[row].drop([col, 'target'])
            
        X, y = known.drop([col], axis=1), known[col]
        
        # find nearby points
        neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
        neigh.fit(X)
        ind_nei = neigh.kneighbors(unknown.values.reshape(1, -1), return_distance=False)
        X, y = X.iloc[ind_nei[0]], y.iloc[ind_nei[0]]
        
        # perform classfication on nearby points
        clf = RandomForestClassifier(n_estimators=2000, max_features=0.6, \
                                     min_samples_split=10, min_samples_leaf=10, \
                                     random_state=0, n_jobs=-1)
        clf.fit(X, y)
        predicted = clf.predict(unknown.values.reshape(1, -1))
        data.loc[row, col] = predicted
        

def nearbyAverage(train, data, columns_miss, col, n_neighbors=100, fill_train=True):
    temp1 = train.drop(columns_miss, axis=1)
    temp2 = data.drop(columns_miss, axis=1)
    
    rows_miss = np.where(data[col] == -1)
    for row in rows_miss[0]:
        if fill_train:
        # if we are filling training data, then only consider same label;
        # else if we are filling validation or test data, then ignore label
            label = data.loc[row, 'target']
            known = temp1[(temp1[col] != -1) & (temp1['target'] == label)]
            unknown = temp2.iloc[row].drop(col)
        else:
            known = temp1[temp1[col] != -1].drop('target', axis=1)
            unknown = temp2.iloc[row].drop([col, 'target'])
            
        X, y = known.drop([col], axis=1), known[col]
        
        # find nearby points
        neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
        neigh.fit(X)
        ind_nei = neigh.kneighbors(unknown.values.reshape(1, -1), return_distance=False)
        y = y.iloc[ind_nei[0]]
        
        # average the values of 100 y
        data.loc[row, col] = y.mean()