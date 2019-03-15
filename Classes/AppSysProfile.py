from Classes.RapidProfile import RapidProfile
import pandas as pd

class AppSysProfile(RapidProfile):
    def __init__(self, df, app_name):
        RapidProfile.__init__(self, df)
        self.appName = app_name
        self.cleanLabelByExactName(RapidProfile.EXCLUDED_FEATURES)
        #self.cleanData()
        self.scale()

    def getSysByConfig(self,config):
        return self.dataFrame.loc[self.dataFrame['Configuration'].apply(
            lambda x: x == config), self.x]

    def getData(self):
        return self.dataFrame[self.x]

    def getConfigs(self):
        return self.dataFrame['Configuration'].values
