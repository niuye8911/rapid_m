'''
    Class for a machine profile
    A Machine profile contains information:
    1) host_name (string)
    2) TRAINED (boolean)
    3) model_params<attribute, value>
'''

import json
from collections import OrderedDict


class Machine:
    def __init__(self, filePath=""):
        self.host_name = -1
        self.TRAINED = False
        self.model_params = OrderedDict()
        self.features = []
        if filePath:
            self.fromFile(filePath)

    def fromFile(self, file):
        with open(file) as machine_json:
            machine = json.load(machine_json)
            self.host_name = machine['host_name']
            self.TRAINED = machine['TRAINED']
            if 'model_params' in machine:
                self.model_params = machine['model_params']

    def isTrained(self):
        return self.TRAINED
