# ElasticNet Regression Model

from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV
from models.Linear import *


class RapidEN(Linear):

    def __init__(self, file_path=''):
        Linear.__init__(self,"EN",file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.model = ElasticNetCV(cv=3, max_iter=1000000)
            self.model.fit(X,Y)
