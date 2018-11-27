from Classes.App import *
from Utility import *


def train(machine_file, measurements):
    '''
    Train the Machine Model using a proper model
    :param machine_file: the path to the machine file (string)
    :param measurements: a csv file containing all the combined system profile measurements
    :return: void, but write the model to the file
    '''

    if not machine.isTrained():
        machine['params'] = {}

        # TODO:Abdall's work goes here*******

        model, accuracy = dummyTrainer(measurements)

        # TODO:END of Abdall's work*******

        # Note: the final goal of this function is to write the model back to app file
        app['params'] = model
        app['TRAINED'] = True
        app['accuracy'] = accuracy
        with open(app_file, 'w') as output:
            json.dump(app, output, indent=2)

        RAPID_info("slow-down prediction for " + app['name'], str(accuracy))
        return accuracy


def dummyTrainer(measurements):
    # I do nothing
    dummy_model = {'attribute1': {'a': 0.2, 'b': 3}, 'attribute2': {'a': 0.3, 'b': 1}}
    dummy_accuracy = 0.2
    return dummy_model, dummy_accuracy
