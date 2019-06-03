"""
 Train an machine to predict the combined system profile
  Author:
    Liu Liu
    03/2019
"""

import os
import scipy.stats

from Classes.MModel import *
from Classes.Machine import *
from Utility import *


class MModelTrainer:
    def __init__(self, host_name, machineProfile, TEST=False):
        '''
        :param host_name: the name of the machine
        :param machineProfile: formatted data of environment
        '''
        self.host_name = host_name
        self.machineProfile = machineProfile
        self.m_model = None
        self.TEST = TEST

    def train(self):
        '''
        Train a machine model based on the measurement
        '''
        # train all data frame
        X = self.machineProfile.getX()
        Y_dict, Y_all = self.machineProfile.getY()
        self.m_model = self.mModelTrain(X, Y_dict, Y_all)

    def getMSE(self):
        return self.m_model.mse

    def getMAE(self):
        return self.m_model.mae

    def getR2(self):
        return self.m_model.r2

    def mModelTrain(self, X, Y_dict, Y_all):
        mModel = MModel()
        mModel.setX(X)
        mModel.setYDict(Y_dict)
        mModel.setYAll(Y_all)
        mModel.setYLabel(self.machineProfile.getYLabel())
        mModel.train(self.TEST)
        mModel.validate()
        return mModel

    def printCI(self, dir_name=''):
        # write diffs raw data to a csv
        diff_list = pd.DataFrame(data=self.m_model.diffs)
        diff_list.dropna(thresh=1)
        diff_list.to_csv(dir_name + '/' + self.host_name + '_diff.csv',
                         index=False)
        # calculate the CI
        output = open(dir_name + "/" + self.host_name + "_ci.csv", 'w')
        for feature in self.m_model.diffs.keys():
            diff = diff_list[feature].values.tolist()
            m, ci_upp, ci_low = cal_ci(diff)
            line = [feature, str(m), str(ci_low), str(ci_upp)]
            output.write(','.join(line))
            output.write('\n')
        output.close()
        # draw the ci
        draw_ci(dir_name + "/" + self.host_name + "_ci.csv",
                dir_name + '/' + self.host_name + "_mmodel")

    def write_to_file(self, dir_name=''):
        if dir_name != '' and not os.path.isdir(dir_name):
            # create the dir if not exist
            os.mkdir(dir_name)

        self.m_model.write_to_file(dir_name + "/" + self.host_name)
        self.printCI(dir_name)

    def dump_into_machine(self, machine_file):
        self.m_model.dump_into_machine(machine_file)
