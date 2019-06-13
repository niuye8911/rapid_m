# ElasticNet Regression Model

import time

from sklearn.linear_model import ElasticNetCV

from models.RapidLinear import *


class RapidEN(RapidLinear):
    def __init__(self, file_path=''):
        RapidLinear.__init__(self, "EN", file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.scale(X)
            self.model = ElasticNetCV(cv=3, max_iter=1000000)
            time1 = time.time()
            self.model.fit(self.scaler.transform(X), Y)
            time2 = time.time()
            return time2 - time1
        return -1
