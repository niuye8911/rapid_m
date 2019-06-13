# Simple Linear Regression Model

import pickle
import time

from sklearn.linear_model import LinearRegression

from models.RapidModel import *


class RapidLinear(RapidModel):
    def __init__(self, name='LR', file_path=''):
        RapidModel.__init__(self, name, file_path)
        if file_path == '':
            self.model = None

    def fromFile(self, file_path):
        self.loadScaler(file_path)
        self.model = pickle.load(open(file_path + '.pkl', 'rb'))

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.scale(X)
            self.model = LinearRegression()
            time1 = time.time()
            self.model.fit(self.scaler.transform(X), Y)
            time2 = time.time()
            return time2 - time1
        return -1

    def predict(self, x):
        ''' predict the result '''
        if self.model is not None:
            return self.model.predict(self.scaler.transform(x))

    def save(self, file_path_prefix):
        self.saveScaler(file_path_prefix)
        pickle.dump(self.model, open(file_path_prefix + '.pkl', 'wb'))
