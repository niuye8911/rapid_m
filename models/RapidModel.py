# The base class for Models used in Rapid-M

import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


class RapidModel:
    SCALER_POSTFIX = '_scaler.pkl'

    def __init__(self, name='', file_path=''):
        self.name = name
        self.train_x = None
        self.train_y = None
        self.model = None
        self.scaler = StandardScaler()
        if file_path is not '':
            self.fromFile(file_path)

    def fromFile(self, file_path):
        pass

    def scale(self, X):
        ''' scale the data '''
        self.scaler.fit(X)

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

    def loadScaler(self, file_path):
        self.scaler = joblib.load(file_path + RapidModel.SCALER_POSTFIX)

    def saveScaler(self, file_path):
        joblib.dump(self.scaler, file_path + RapidModel.SCALER_POSTFIX)

    def save(self, file_path):
        pass
