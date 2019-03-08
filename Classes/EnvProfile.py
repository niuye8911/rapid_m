# This is the parser for an App's profile

import pandas as pd


class EnvProfile:
    def __init__(self, csv_file, host_name):
        self.raw_data = csv_file
        self.dataFrame = pd.read_csv(csv_file)
        self.features = self.getFeatures()
        self.hostName = host_name

    def getFeatures(self):
        # the first line contains all the features
        header = pd.read_csv(self.raw_data, nrows=1).columns.tolist()
        # 2/3 of the csv are features
        featureLen = len(header) * 2 / 3
        return header[0:int(featureLen)]

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
