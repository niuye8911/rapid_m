'''
    Class for an app
    An App contains information:
    1) name (string)
    2) machine_id (int)
    3) model_type (enum PModelType)
    4) model_params<attribute, value> (dict<string,float>)
'''

import json
from enum import Enum

from rapid_m_backend_server import Utility as util


class PModelType(Enum):
    LINEAR = "LINEAR"
    LOGISTIC = "LOGISTIC"


class App:
    def __init__(self, filePath=""):
        self.name = ""
        self.machine_id = -1
        self.model_type = None
        self.model_params = dict()
        if filePath:
            self.fromFile(filePath)

    def fromFile(self, file):
        with open(file) as app_json:
            app = json.load(app_json)
            self.name = app['name']
            self.machine_id = app['machine_id']
            self.model_type = filter(lambda x: x.value is app['model_type'], PModelType)
            self.model_params = self.readParams(app['params'])

    def readParams(self, params_json):
        params = json.load(params_json)
        for attr, value in params:
            if attr in self.model_params.keys():
                util.RAPID_warn("duplicated attribute", attr)
            self.model_params[attr] = float(value)
