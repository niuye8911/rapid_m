from Classes.RapidProfile import RapidProfile
import pandas as pd


class AppSysProfile(RapidProfile):
    def __init__(self, df, app_name):
        RapidProfile.__init__(self, df)
        self.appName = app_name
        self.setXLabel(self.x[1:])
        self.cleanLabelByExactName(RapidProfile.EXCLUDED_FEATURES)
        
        #self.scale()

    def getSysByConfig(self, config):
        return self.dataFrame.loc[self.dataFrame['Configuration'].
                                  apply(lambda x: x == config), self.x]

    def getSubFrameByConfigs(self, configs):
        return self.dataFrame.loc[self.dataFrame['Configuration'].apply(
            lambda x: x in configs)]

    def getData(self):
        return self.dataFrame[self.x]

    def getConfigs(self):
        return self.dataFrame['Configuration'].values

    def getX(self):
        return self.x

    def writeOut(self, outfile):
        ''' write the cleaned dataframe to csv '''
        self.dataFrame[['Configuration'].append(self.x)].to_csv(
            outfile, sep=',', index=False)
