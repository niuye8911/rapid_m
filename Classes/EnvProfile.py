# This is the parser for an App's profile

import pandas as pd
from functools import reduce
from Classes.RapidProfile import RapidProfile


class EnvProfile:
    def __init__(self, csv_file, host_name):
        RapidProfile.__init__(self, csv_file)
        self.hostName = host_name

    def partitionData(self):
        self.x = pd.read_csv(self.raw_file, nrows=1).columns.tolist()
        length = len(self.x)
        self.sys1DF = RapidProfile(self.dataFrame(self.x[0:length * 1 / 3]))
        self.sys2DF = RapidProfile(
            self.dataFrame(self.x[length * 1 / 3:length * 2 / 3]))
        self.combinedDF = RapidProfile(self.dataFrame(self.x[length * 2 / 3:]))

    def cleanFeatures(self):
        self.sys1DF.cleanLabelByExactName(
            map(lambda x: x + '-1', RapidProfile.EXCLUDED_FEATURES))
        self.sys2DF.cleanLabelByExactName(
            map(lambda x: x + '-2', RapidProfile.EXCLUDED_FEATURES))
        self.combinedDF.cleanLabelByExactName(
            map(lambda x: x + '-C', RapidProfile.EXCLUDED_FEATURES))

    def cleanData(self):
        self.sys1DF.cleanData()
        self.sys2DF.cleanData()
        self.combinedDF.cleanData()

    def scaleAll(self):
        self.sys1DF.scale()
        self.sys2DF.scale()
        self.combinedDF.scale()

    def getFeatures(self):
        return self.sys1DF.x + self.sys2DF.x

    def getYLabel(self):
        return self.combinedDF.x

    def getX(self):
        return pd.concaat(
            [self.sys1DF.dataFrame[self.sys1DF.x], self.sys2DF[self.sys2DF.x]],
            axis=1,
            join="inner")

    def getY(self):
        return self.combinedDF.dataFrame[self.combinedDF.x]
