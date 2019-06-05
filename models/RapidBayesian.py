# ElasticNet Regression Model

import time

from sklearn import linear_model

from models.RapidLinear import *


class RapidBayesian(RapidLinear):
    def __init__(self, file_path=''):
        RapidLinear.__init__(self, "BR", file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.model = linear_model.BayesianRidge()
            time1 = time.time()
            self.model.fit(X, Y)
            time2 = time.time()
            return time2 - time1
        return -1
