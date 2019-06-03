# Simple Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from models.RapidModel import *
import time
import pickle


class RapidLinear(RapidModel):
    def __init__(self, name='LinearRegression', file_path=''):
        RapidModel.__init__(self, name, file_path)
        if file_path == '':
            self.model = None

    def fromFile(self, file_path):
        pickle.load(open(file_path, 'rb'))

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.model = LinearRegression()
            time1 = time.time()
            self.model.fit(X, Y)
            time2 = time.time()
            return time2 - time1
        return -1

    def predict(self, x):
        ''' predict the result '''
        if self.model is not None:
            return self.model.predict(x)

    def validate(self, X, Y):
        ''' validate the trained model '''
        if self.model is None:
            return -1
        pred = self.model.predict(X)
        r2 = r2_score(Y, pred)
        return r2

    def save(self, file_path_prefix):
        pickle.dump(self.model, open(file_path_prefix + '.pkl', 'wb'))
