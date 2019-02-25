# This is the parser for an App's profile

import pandas as pd


class SlowDownProfile:
    def __init__(self, csv_file, app_name):
        self.raw_data = csv_file
        self.dataFrame = pd.read_csv(csv_file)
        self.features = self.getFeatures()
        self.appName = app_name

    def getFeatures(self):
        # the first line contains all the features
        # the first column contains the
        return pd.read_csv(self.raw_data, nrows=1).columns.tolist()[1:-1]

    def getSubdata(self, config_list):
        if config_list is None or config_list == []:
            return self.dataFrame
        return self.dataFrame.loc[self.dataFrame['Configuration'].apply(
            lambda x: x in config_list)]
