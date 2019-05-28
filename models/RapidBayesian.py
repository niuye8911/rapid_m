# ElasticNet Regression Model

from sklearn import linear_model
from models.Linear import *


class RapidBayesian(Linear):

    def __init__(self, file_path=''):
        Linear.__init__(self,"BayesianRidge",file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.model = linear_model.BayesianRidge()
            self.model.fit(X,Y)
