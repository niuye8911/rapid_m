# The base class for profiling data in RAPID_M

import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from pathlib import Path
from functools import reduce


class RapidProfile:
    # pre_determined excluded system footprint that won't affect perf
    EXCLUDED_FEATURES = {
        "ACYC", "AFREQ", "FREQ", "INSTnom", "L2MISS", "L3MISS", "PhysIPC",
        'C0res%', 'C10res%', 'C1res%', 'C2res%', 'C3res%', 'C6res%', 'C7res%',
        'C8res%', 'C9res%', 'Configuration'
    }

    SCALAR_PATH = './RapidScalar.pkl'

    def __init__(self, csv_file):
        self.raw_file = csv_file
        self.dataFrame = pd.read_csv(csv_file)
        # default x = first N-1 row
        self.x = pd.read_csv(self.raw_file, nrows=1).columns.tolist()[1:-1]
        # default y = last column
        self.y = pd.read_csv(self.raw_file, nrows=1).columns.tolist()[-1]
        self.cleanData()

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
        match_func = lambda feature: reduce((lambda x, y: (y in feature) or x), excludes, False)
        self.x = list(filter(lambda feature: not match_func(feature), self.x))
        return

    def createScalar(self):
        ''' create a persistent scaler for all data '''
        data = self.dataFrame[self.x].values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.scalar = min_max_scaler.fit(data)
        joblib.dump(self.scalar, 'RapidScalar.pkl')

    def loadScalar(self):
        ''' load an existing scalar '''
        if not Path(RapidProfile.SCALAR_PATH).is_file():
            self.createScalar()
        self.scalar = joblib.load('RapidScalar.pkl')

    def scale(self):
        self.loadScalar()
        self.dataFrame[self.x] = pd.DataFrame(
            self.scalar.transform(self.dataFrame[self.x]))

    def cleanData(self):
        ''' clean the PCM data to correct form '''
        # drop the excluded column
        #self.dataFrame.drop(
        #        list(RapidProfile.EXCLUDED_FEATURES), axis=1, inplace=True)
        self.cleanLabelByExactName(RapidProfile.EXCLUDED_FEATURES)
        # re-calculate the numerical value
        # 1) INST
        self.dataFrame['INST'] = self.dataFrame['INST'].div(
            self.dataFrame['TIME(ticks)'])

        # 2) READ / WRITE
        self.dataFrame['READ'] = self.dataFrame['READ'].div(
            self.dataFrame['TIME(ticks)']).mul(1000)
        self.dataFrame['WRITE'] = self.dataFrame['WRITE'].div(
            self.dataFrame['TIME(ticks)']).mul(1000)

    def writeOut(self, outfile):
        ''' write the cleaned dataframe to csv '''
        indexs = self.x + [self.y]
        indexs = ['Configuration'] + indexs
        self.dataFrame[indexs].to_csv(outfile, sep=',', index=False)
