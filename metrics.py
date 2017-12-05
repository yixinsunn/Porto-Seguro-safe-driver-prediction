
# coding: utf-8

import numpy as np

def gini(actual, pred):
    """ 
    This function computes gini coefficient.
    Reference from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
    """
    assert(len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1*all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.0
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)