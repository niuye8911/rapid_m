# Simple Linear Regression Model

from sklearn import linear_model
from sklearn.linear_model import LassoCV
from models.RapidLinear import *
import time

class RapidLasso(RapidLinear):
    def __init__(self, file_path=''):
        RapidLinear.__init__(self, "LassoCV", file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.model = LassoCV(cv=3, max_iter=1000000)
            time1 = time.time()
            self.model.fit(X, Y)
            time2 = time.time()
            return time2 - time1
        return -1
