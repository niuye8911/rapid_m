from Classes.Machine import *
from Utility import *


def train(machine_file):
    '''
    Train the Machine Model using a proper model
    :param machine_file: the path to the machine file (string)
    :param measurements: a csv file containing all the combined system
    profile measurements
    :return: void, but write the model to the file
    '''

    # only training the current machine

    machine = Machine(machine_file)
    if not machine.isTrained():

        # TODO:Abdall's work goes here*******

        model, accuracy = dummyTrainer()

        # TODO:END of Abdall's work*******

        # machine file
        machine.TRAINED = True
        with open(machine_file, 'w') as output:
            json.dump(machine.__dict__, output, indent=2)

        RAPID_info("environment prediction for " + machine.name, str(accuracy))
        return accuracy


def dummyTrainer():
    # I do nothing
    dummy_model = {
        'attribute1': {
            'a': 0.2,
            'b': 3
        },
        'attribute2': {
            'a': 0.3,
            'b': 1
        }
    }
    dummy_accuracy = 0.2
    return dummy_model, dummy_accuracy
