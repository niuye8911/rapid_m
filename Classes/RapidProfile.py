import pandas as pd


class RapidProfile:
    def __init__(self, csv_file):
        self.raw_data = csv_file
        self.dataFrame = pd.read_csv(csv_file)
        self.x = []
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

    def cleanLabelByName(self, excludes):
        '''
        @param excludes: a vector containing all unwanted feature string
        '''
        match_func = lambda feature: reduce((lambda x, y: (y in feature) or x),
                                            excludes, False)
        self.x = filter(lambda feature: not match_func(feature), self.x)
        return

    def cleanData(self):
        ''' clean the PCM data to correct form '''
