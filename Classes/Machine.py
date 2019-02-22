'''
    Class for a machine profile
    A Machine profile contains information:
    1) name (string)
    2) id (int)
    3) TRAINED (boolean)
    4) model_type (enum MModelType)
    5) model_params<attribute, value> (dict<string,list<float,float>>)
'''

import json
from enum import Enum
import hashlib
import socket


class Machine:
    def __init__(self, filePath=""):
        self.name = ""
        self.machine_id = -1
        self.TRAINED = False
        if filePath:
            self.fromFile(filePath)

    def fromFile(self, file):
        with open(file) as machine_json:
            machine = json.load(machine_json)
            self.name = machine['name']
            if self.name == "":
                self.name = socket.gethostname()
            self.machine_id = hashlib.sha1(self.name.encode(
                'utf-8')).hexdigest() if 'id' not in machine else machine['id']
            self.TRAINED = machine['TRAINED']

    def isTrained(self):
        return self.TRAINED
