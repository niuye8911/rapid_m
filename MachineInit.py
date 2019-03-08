"""
 This is an Machine Init-er
  Author:
    Liu Liu
    03/2019
"""

from Classes.Machine import *
from Classes.MModel import *
from Classes.EnvProfile import *
from MModelTrainer import *
from Utility import *


def trainEnv(machine_file, machine_measurements, directory, DRAW=True):
    # load in the file
    machine = Machine(machine_file)
    # check if the machine is trained
    if not machine.isTrained():
        RAPID_info("Training for ", machine.host_name)
        # read in the machine measurement file
        envProfile = EnvProfile(machine_measurements, machine.host_name)
        mModelTrainer = MModelTrainer(machine.host_name, envProfile)
        mModelTrainer.train()

        # write mModel to file
        mModelTrainer.write_to_file(directory)

        # write model to machine file
        mModelTrainer.dump_into_machine(machine)

        machine.TRAINED = True
        # write the machine to file
        write_to_file(machine_file, machine)


def write_to_file(app_file, app):
    with open(app_file, 'w') as output:
        json.dump(app.__dict__, output, indent=2)
