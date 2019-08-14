"""
 This is an Machine Init-er
  Author:
    Liu Liu
    03/2019
"""

import json

import pandas as pd

from Rapid_M_Classes.EnvProfile import EnvProfile
from Rapid_M_Classes.Machine import Machine
from MModelTrainer import *
from Utility import *


def trainEnv(machine_file,
             machine_measurements,
             directory,
             DRAW=True,
             TEST=False):
    # load in the file
    machine = Machine(machine_file)
    # check if the machine is trained
    if not machine.isTrained():
        RAPID_info("Training for ", machine.host_name)
        # read in the machine measurement file
        envProfile = EnvProfile(
            pd.read_csv(machine_measurements), machine.host_name)
        mModelTrainer = MModelTrainer(machine.host_name, envProfile, TEST)
        mModelTrainer.train()

        # write mModel to file
        mModelTrainer.write_to_file(directory)

        # write model to machine file
        mModelTrainer.dump_into_machine(machine)

        machine.TRAINED = True
        # write the machine to file
        write_to_file(machine_file, machine)


def write_to_file(machine_file, machine):
    with open(machine_file, 'w') as output:
        json.dump(machine.__dict__, output, indent=2)
