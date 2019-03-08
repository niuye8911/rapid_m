"""
 Train an machine to predict the combined system profile
  Author:
    Liu Liu
    03/2019
"""

import os

from Classes.MModel import *
from Classes.Machine import *
from Utility import *


class MModelTrainer:
    def __init__(self, host_name, machineProfile):
        '''
        :param host_name: the name of the machine
        :param machineProfile: formatted data of environment
        '''
        self.host_name = host_name
        self.machineProfile = machineProfile
        self.m_model = None

    def train(self):
        '''
        Train a machine model based on the measurement
        '''
        # train all data frame
        X = self.machineProfile.getX()
        Y = self.machineProfile.getY()
        self.m_model = self.mModelTrain(X, Y)

    def getMSE(self):
        return self.m_model.mse

    def getMAE(self):
        return self.m_model.mae

    def getR2(self):
        return self.m_model.r2

    def mModelTrain(self, X, Y):
        mModel = MModel()
        mModel.setX(X)
        mModel.setY(Y)
        mModel.train()
        mModel.validate()
        return mModel

    def write_to_file(self, dir_name=''):
        id = 1
        if dir_name != '' and not os.path.isdir(dir_name):
            # create the dir if not exist
            os.mkdir(dir_name)

        self.m_model.write_to_file(dir_name + "/" + self.host_name + ".pkl")

    def dump_into_machine(self, machine_file):
        self.m_model.dump_into_machine(machine_file)
