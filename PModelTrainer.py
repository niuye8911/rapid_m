"""
 Clustering app's configs based on prediction accuracy of slow-down
  Author:
    Rajanya Dhar / Liu Liu
    12/2018
"""

import os

from Classes.App import *
from Classes.PModel import *
from Classes.SlowDownProfile import *
from Utility import *


class PModelTrainer:
    def __init__(self, app_name, slowDownProfile, cluster_list=[]):
        '''
        Train a performance model based on the measurement
        :param app_name: the name of the app
        :param slowDownProfile: formatted data of slow-down
        :param cluster_list: list of cluster
        '''
        self.app_name = app_name
        self.slowDownProfile = slowDownProfile
        self.cluster_list = cluster_list
        self.p_models = []

    def updateCluster(self, cluster_list):
        self.cluster_list = cluster_list

    def train(self):
        '''
        Train a performance model based on the measurement
        :return: accuracy, mse, mae
        '''

        # init the p-models object with output file
        clusterDFs = list(
            map(lambda x: self.slowDownProfile.getSubdata(x),
                self.cluster_list))

        features = self.slowDownProfile.getFeatures()
        if clusterDFs is None:  #no cluster provided
            clusterDFs = slowDownProfile.getSubdata([])

        # train all clustered data frame
        self.p_models = list(
            map(lambda x: self.pModelTrain(x, features), clusterDFs))

    def getDiff(self):
        diffs = list(map(lambda x: x.diff, self.p_models))
        if diffs is None or diffs == []:
            return -1.
        return sum(diffs) / len(diffs)

    def getMSE(self):
        mses = list(map(lambda x: x.mse, self.p_models))
        if mses is None or mses == []:
            return -1.
        return sum(mses) / len(mses)

    def getMAE(self):
        maes = list(map(lambda x: x.mae, self.p_models))
        if maes is None or maes == []:
            return -1.
        return sum(maes) / len(maes)

    def getR2(self):
        r2s = list(map(lambda x: x.r2, self.p_models))
        if r2s is None or r2s == []:
            return -1.
        return sum(r2s) / len(r2s)

    def pModelTrain(self, df, features):
        pModel = PModel()
        pModel.setDF(df, features)
        pModel.train()
        pModel.validate()
        return pModel

    def write_to_file(self, dir_name=''):
        id = 1
        if dir_name != '' and not os.path.isdir(dir_name):
            # create the dir if not exist
            os.mkdir(dir_name)

        for model in self.p_models:
            model.write_to_file(dir_name + "/" + self.app_name + str(id) +
                                ".pkl")
            id += 1

    def dump_into_app(self, app):
        id = 1
        for model in self.p_models:
            model.dump_into_app(app, app.name + str(id))
            print(model.model.coef_)
            id += 1
