# This is the parser for an App's profile

import pandas as pd
from RapidProfile import RapidProfile


class SlowDownProfile(RapidProfile):
    INDEX = {"Configuration"}

    def __init__(self, csv_file, app_name):
        self.appName = app_name
        RapidProfile.__init__(self, csv_file)
        self.setYLabel('SLOWDOWN')
        # scale the data using the scalar
        self.cleanData()
        self.scale()

    def getFeatures(self):
        return self.x

    def getSubdata(self, config_list):
        if config_list is None or config_list == []:
            return self.dataFrame
        return self.dataFrame.loc[self.dataFrame['Configuration'].apply(
            lambda x: x in config_list)]
