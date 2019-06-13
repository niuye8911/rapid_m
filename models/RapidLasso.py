# Simple Linear Regression Model

import time

from sklearn.linear_model import LassoCV

from models.RapidLinear import *


class RapidLasso(RapidLinear):
    def __init__(self, file_path=''):
        RapidLinear.__init__(self, "LS", file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.scale(X)
            self.model = LassoCV(cv=3, max_iter=1000000)
            time1 = time.time()
            self.model.fit(self.scaler.transform(X), Y)
            time2 = time.time()
            return time2 - time1
        return -1
