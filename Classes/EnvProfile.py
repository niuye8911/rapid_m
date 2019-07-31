# This is the parser for an App's profile

from collections import OrderedDict

import pandas as pd

from Classes.RapidProfile import RapidProfile
from DataUtil import *

class EnvProfile(RapidProfile):
    def __init__(self, df, host_name):
        RapidProfile.__init__(self, df)
        self.hostName = host_name
        self.partitionData()
        self.cleanData()
        self.cleanFeatures()

    def partitionData(self):
        ind_len = int(len(self.x) / 3)
        self.sys1DF = RapidProfile(self.dataFrame[self.x[0:ind_len]])
        self.sys2DF = RapidProfile(self.dataFrame[self.x[ind_len:2 * ind_len]])
        self.combinedDF = RapidProfile(self.dataFrame[self.x[2 * ind_len:]])
        # create the scalar using the combined df
        # self.combinedDF.scale(True)
        # self.sys1DF.scale()
        # self.sys2DF.scale()

    def cleanFeatures(self):
        self.sys1DF.cleanLabelByExactName(
            list(map(lambda x: x + '-1', RapidProfile.EXCLUDED_FEATURES)))
        self.sys2DF.cleanLabelByExactName(
            list(map(lambda x: x + '-2', RapidProfile.EXCLUDED_FEATURES)))
        self.combinedDF.cleanLabelByExactName(
            list(map(lambda x: x + '-C', RapidProfile.EXCLUDED_FEATURES)))

    def cleanData(self):
        self.sys1DF.cleanData('-1')
        self.sys2DF.cleanData('-2')
        self.combinedDF.cleanData('-C')

    def scaleAll(self):
        self.sys1DF.scale()
        self.sys2DF.scale()
        self.combinedDF.scale()

    def getFeatures(self):
        return self.sys1DF.x + self.sys2DF.x

    def getYLabel(self):
        return self.combinedDF.x

    def getX(self):
        first_df = pd.concat([
            self.sys1DF.dataFrame[self.sys1DF.x],
            self.sys2DF.dataFrame[self.sys2DF.x]
        ],
            axis=1)
        concated_df = reformat_dfs(self.sys1DF.dataFrame[self.sys1DF.x],
                                        self.sys2DF.dataFrame[self.sys2DF.x])
        # second_df = pd.concat([
        #        self.sys2DF.dataFrame[self.sys2DF.x],
        #        self.sys1DF.dataFrame[self.sys1DF.x]
        #    ],
        #  axis=1)
        # concated_df = first_df.append(second_df, ignore_index=True, sort=False)
        concated_df.to_csv('./cleaned_machine.csv')
        return concated_df

    def getY(self):
        forward_df = self.combinedDF.dataFrame[self.combinedDF.x]
        # get a dictionary of [feature, df]
        result_Y = OrderedDict()
        features = list(map(lambda x: x[0:-2], self.combinedDF.x))
        for feature in features:
            values = self.combinedDF.dataFrame[feature + '-C']
            result_Y[feature] = values
        return result_Y, forward_df
