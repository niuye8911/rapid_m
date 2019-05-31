# ElasticNet Regression Model

from sklearn import linear_model
from models.RapidLinear import *


class RapidBayesian(RapidLinear):

    def __init__(self, file_path=''):
        RapidLinear.__init__(self,"BayesianRidge",file_path)

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.model = linear_model.BayesianRidge()
            self.model.fit(X,Y)
