"""
 Clustering app's configs based on prediction accuracy of slow-down
  Author:
    Rajanya Dhar / Liu Liu
    12/2018
"""

import os
import scipy.stats
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

    def minButNotSingle(self, metrics):
        id = -1
        max = 0.0
        for i in range(0, len(metrics)):
            if metrics[i] >= max and len(self.cluster_list[i]) > 1:
                id = i
                max = metrics[i]
        return id

    def getDiff(self):
        diffs = list(map(lambda x: x.diff, self.p_models))
        if diffs is None or diffs == []:
            return [-1, -1]
        return diffs, self.minButNotSingle(diffs)

    def getMSE(self):
        mses = list(map(lambda x: x.mse, self.p_models))
        if mses is None or mses == []:
            return [-1]
        return mses, mses.index(max(mses))

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
            model.drawPrediction(dir_name + "/" + self.app_name + str(id) +
                                 ".png")
            model.printPrediction(dir_name + "/" + self.app_name + str(id) +
                                  ".csv")
            id += 1
        self.printCI(dir_name)

    def printCI(self, dir_name=''):
        diffs = []
        diff_list = list(map(lambda x: x.diffs, self.p_models))
        for i in diff_list:
            diffs = diffs + i
        m, ci_upp, ci_low = cal_ci(diffs)
        output = open(dir_name + "/" + self.app_name + "_ci.csv",'w')
        line = [self.app_name, str(m), str(ci_low), str(ci_upp)]
        output.write(",".join(line))
        output.close()

    def dump_into_app(self, app):
        id = 1
        for model in self.p_models:
            model.dump_into_app(app, app.name + str(id))
            #print(model.model.coef_)
            id += 1
