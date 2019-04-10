# The base class for profiling data in RAPID_M

import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from pathlib import Path
from functools import reduce


class RapidProfile:
    # pre_determined excluded system footprint that won't affect perf
    EXCLUDED_FEATURES = {
        "ACYC",
        'C0res%',
        'C10res%',
        'C1res%',
        'C2res%',
        'C3res%',
        'C6res%',
        'C7res%',
        'C8res%',
        'C9res%',
        'Proc Energy (Joules)',
        'Configuration',
        'TIME(ticks)',
        'SLOWDOWN',
        # some features that can be calculated by others
        'PhysIPC',
        'L2MISS',
        'L3MISS',
        'INST'
    }

    SCALAR_PATH = './RapidScalar.pkl'

    def __init__(self, df):
        self.dataFrame = df
        # default x = first N-1 row
        self.x = df.columns.values.tolist()
        # default y = last column
        self.y = []

    def setXLabel(self, x):
        '''determine the X vector(features)'''
        self.x = x

    def setYLabel(self, y):
        '''determine the Y vector(observations)'''
        self.y = y

    def getXData(self):
        return self.dataFrame[self.x]

    def getYData(self):
        return self.dataFrame[self.y]

    def cleanLabelByExactName(self, excludes):
        '''
        @param excludes: a vector containing all unwanted feature string
        note that the 'x' has already been cleaned up by cleanData()
        '''
        match_func = lambda feature: reduce((lambda x, y: (y == feature) or x), excludes, False)
        self.x = list(filter(lambda feature: not match_func(feature), self.x))
        return

    def createScalar(self, writeout=False):
        ''' create a persistent scaler for all data '''
        data = self.dataFrame[self.x]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.scalar = min_max_scaler.fit(data)
        if writeout:
            joblib.dump(self.scalar, 'RapidScalar.pkl')

    def loadScalar(self, use_exist=False):
        ''' load an existing scalar '''
        if not Path(RapidProfile.SCALAR_PATH).is_file():
            self.createScalar()
        self.scalar = joblib.load('RapidScalar.pkl')


    def scale(self):
        self.createScalar()
        self.dataFrame[self.x] = pd.DataFrame(
            self.scalar.transform(self.dataFrame[self.x]))

    def cleanData(self):
        ''' clean the PCM data to correct form '''
        # re-calculate the numerical value
        # 1) INST
        #TODO: WHY INST IS SO IMPORTANT
        self.dataFrame['INST'] = self.dataFrame['ACYC'].div(
            self.dataFrame['INST'])
        # 2) INSTnom%
        self.dataFrame['INSTnom%'] = self.dataFrame['INSTnom%'].apply(lambda x: x/100.0)
        # 3) PhysIPC%
        self.dataFrame['PhysIPC%'] = self.dataFrame['PhysIPC%'].apply(lambda x: x/100.0)
        # 2) READ / WRITE
        #self.dataFrame['READ'] = self.dataFrame['READ'].mul(4200.) / (
        #        self.dataFrame['TIME(ticks)'])
        #        self.dataFrame['WRITE'] = self.dataFrame['WRITE'].mul(4200.) / (
        #            self.dataFrame['TIME(ticks)'])
