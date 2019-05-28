# Simple Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from models.RapidModel import *
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
            self.model.fit(X, Y)

    def predict(self, x):
        ''' predict the result '''
        if self.model is not None:
            return self.model.predict(x)

    def validate(self, X, Y):
        ''' validate the trained model '''
        if self.model is None:
            return -1
        pred = self.model.predict(X)
        r2_score = r2_score(Y, pred)
        return r2_score

    def save(self,file_path):
        pickle.dump(self.model, open(file_path, 'wb'))
