# SVM Regression Model

import time

from sklearn import svm

from models.RapidLinear import *


class RapidSVR(RapidLinear):
    def __init__(self, file_path=''):
        RapidLinear.__init__(self, "SVR", file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.scale(X)
            self.model = svm.SVR(gamma='auto')
            time1 = time.time()
            self.model.fit(self.scaler.transform(X), Y)
            time2 = time.time()
            return time2 - time1
        return -1
