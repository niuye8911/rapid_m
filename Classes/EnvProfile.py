# This is the parser for an App's profile

import pandas as pd
from functools import reduce


class EnvProfile:
    EXCLUDED_FEATURES = {"ACYC"}

    def __init__(self, csv_file, host_name):
        self.raw_data = csv_file
        self.dataFrame = pd.read_csv(csv_file)
        self.features = self.getFeatures()
        self.hostName = host_name

    def cleanFeatures(self):
        self.dataFrame.drop(
            self.dataFrame.columns[list(EnvProfile.EXCLUDED_FEATURES)],
            axis=1,
            inplace=True)

    def getFeatures(self):
        # the first line contains all the features
        header = pd.read_csv(self.raw_data, nrows=1).columns.tolist()
        # clean the features
        updated_header = [x for x in header if not EnvProfile.match(x)]
        # 2/3 of the csv are features
        featureLen = len(header) * 2 / 3
        print(header)
        print(updated_header)
        return header[0:int(featureLen)]

    @staticmethod
    def match(feature):
        return reduce((lambda x, y: ((y + "-") in feature) or x),
                      EnvProfile.EXCLUDED_FEATURES, False)

    def getYLabel(self):
        # the first line contains all the features
        header = pd.read_csv(self.raw_data, nrows=1).columns.tolist()
        # 2/3 of the csv are features
        featureLen = len(header) * 1 / 3
        return header[int(featureLen):]

    def getX(self):
        return self.dataFrame[self.features]

    def getY(self):
        return self.dataFrame[self.getYLabel()]
