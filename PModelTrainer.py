"""
 Clustering app's configs based on prediction accuracy of slow-down
  Author:
    Rajanya Dhar / Liu Liu
    12/2018
"""

from Classes.App import *
from Utility import *


def train(app_file, measurements):
    '''
    Train the app using a proper model
    :param app_file: the path to the app file (string)
    :param measurements: a csv file containing all the slow-down measurements
    :return: void, but write the model to the file
    '''

    # get the app object
    app = App(app_file)
    if not app.isTrained():
        app['params'] = {}

        # TODO:Rajanya's work goes here*******

        model, accuracy = dummyTrainer(measurements)

        # TODO:END of Rajanya's work*******

        # Note: the final goal of this function is to write the model back to
        # app file
        app['params'] = model
        app['TRAINED'] = True
        app['accuracy'] = accuracy
        with open(app_file, 'w') as output:
            json.dump(app, output, indent=2)

        RAPID_info("slow-down prediction for " + app['name'], str(accuracy))
        return accuracy


def dummyTrainer(measurements):
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
