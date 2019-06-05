# The base class for Models used in Rapid-M

import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score


class RapidModel:
    def __init__(self, name='', file_path=''):
        self.name = name
        self.train_x = None
        self.train_y = None
        self.model = None
        if file_path is not '':
            self.fromFile(file_path)

    def fromFile(self, file_path):
        pass

    def fit(self, X, Y):
        ''' train the model '''
        pass

    def predict(self, x):
        ''' predict the result '''
        pass

    def validate(self, X, Y):
        ''' validate the trained model '''
        if self.model is None:
            return -1
        pred = self.model.predict(X)
        r2 = r2_score(Y, pred)
        mse = metrics.mean_squared_error(Y, pred)
        diff = np.mean(np.abs((Y - pred) / Y)) * 100
        return r2, mse, diff

    def save(self, file_path):
        pass
