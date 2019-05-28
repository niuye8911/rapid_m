# The base class for Models used in Rapid-M

import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from pathlib import Path
from functools import reduce


class RapidModel:
    def __init__(self, name='', file_path=''):
        self.name = ''
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
        pass

    def save(self, file_path):
        pass
