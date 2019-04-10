# This is the parser for an App's profile

import pandas as pd
from functools import reduce
from Classes.RapidProfile import RapidProfile


class EnvProfile(RapidProfile):
    def __init__(self, df, host_name):
        RapidProfile.__init__(self, df)
        self.hostName = host_name
        #self.cleanData()
        self.partitionData()
        self.cleanFeatures()

    def partitionData(self):
        ind_len = int(len(self.x) / 3)
        self.sys1DF = RapidProfile(self.dataFrame[self.x[0:ind_len]])
        self.sys2DF = RapidProfile(self.dataFrame[self.x[ind_len:2 * ind_len]])
        self.combinedDF = RapidProfile(self.dataFrame[self.x[2 * ind_len:]])

    def cleanFeatures(self):
        self.sys1DF.cleanLabelByExactName(
            list(map(lambda x: x + '-1', RapidProfile.EXCLUDED_FEATURES)))
        self.sys2DF.cleanLabelByExactName(
            list(map(lambda x: x + '-2', RapidProfile.EXCLUDED_FEATURES)))
        self.combinedDF.cleanLabelByExactName(
            list(map(lambda x: x + '-C', RapidProfile.EXCLUDED_FEATURES)))

    def cleanData(self):
        self.dataFrame['INST-1'] = self.dataFrame['ACYC-1'].div(
            self.dataFrame['INST-1'])
        self.dataFrame['INST-2'] = self.dataFrame['ACYC-2'].div(
            self.dataFrame['INST-2'])
        self.dataFrame['INST-C'] = self.dataFrame['ACYC-C'].div(
            self.dataFrame['INST-C'])

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
        concated_df = self.reformat_dfs(self.sys1DF.dataFrame[self.sys1DF.x],
                          self.sys2DF.dataFrame[self.sys2DF.x])
        #second_df = pd.concat([
    #        self.sys2DF.dataFrame[self.sys2DF.x],
    #        self.sys1DF.dataFrame[self.sys1DF.x]
    #    ],
                            #  axis=1)
        #concated_df = first_df.append(second_df, ignore_index=True, sort=False)
        return concated_df

    def reformat_dfs(self,df1, df2):
        ''' reformat the dfs so that we get a symetric matrix where each pair
        is represented by [smaller, sum] '''
        # the first row
        columns = df1.columns.values
        columns = list(map(lambda x: x[:-2], columns))
        #first_columns = list(map(lambda x: x+'-smaller', columns))
        #second_columns = list(map(lambda x: x+'-sum', columns))
        #columns = first_columns + second_columns
        # reorder the data
        combined_df = pd.concat([
            df1,
            df2
        ],
                             axis=1)
        for index, row in combined_df.iterrows():
            for feature in columns:
                f1 = row[feature+'-1']
                f2 = row[feature+'-2']
                row[feature+'-1'] = min(f1,f2)
                row[feature+'-2'] = max(f1,f2)
        return combined_df

    def getY(self):
        forward_df = self.combinedDF.dataFrame[self.combinedDF.x]
        #print(result_df.shape)
        #return pd.concat([result_df,result_df],axis=0)
        concated_df = forward_df.append(
            forward_df, ignore_index=True, sort=False)
        return forward_df
