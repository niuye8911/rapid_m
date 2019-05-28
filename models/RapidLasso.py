# Simple Linear Regression Model

from sklearn import linear_model
from sklearn.linear_model import LassoCV
from models.Linear import *


class RapidLasso(Linear):

    def __init__(self, file_path=''):
        Linear.__init__(self,"LassoCV",file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.model = LassoCV(cv=3, max_iter=1000000)
            self.model.fit(X,Y)
