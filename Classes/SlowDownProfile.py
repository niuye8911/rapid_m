# This is the parser for an App's profile

import pandas as pd
from Classes.RapidProfile import *


class SlowDownProfile(RapidProfile):
    INDEX = {"Configuration"}

    def __init__(self, df, app_name):
        self.appName = app_name
        RapidProfile.__init__(self, df)
        self.setYLabel('SLOWDOWN')
        # scale the data using the scalar
        self.cleanLabelByExactName(RapidProfile.EXCLUDED_FEATURES)
        self.cleanData()
        self.scale()

    def getFeatures(self):
        return self.x

    def getSubdata(self, config_list):
        if config_list is None or config_list == []:
            return self.dataFrame
        return self.dataFrame.loc[self.dataFrame['Configuration'].apply(
            lambda x: x in config_list)]
