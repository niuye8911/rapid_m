'''
    Class for an app
    An App contains information:
    1) name (string)
    2) machine_id (int)
    3) TRAINED (boolean)
    4) model_type (enum PModelType)
    5) model_params<attribute, value> (dict<string,float>)
    6) CLUSTERED (boolean)
    7) num_of_cluster (int)
    8) cluster_info
'''
import json
from collections import OrderedDict
from enum import Enum

import Utility as util


class PModelType(Enum):
    LINEAR = "LINEAR"
    LOGISTIC = "LOGISTIC"


class App:
    def __init__(self, filePath=""):
        self.name = ""
        self.machine_id = -1
        self.model_type = None
        self.model_params = OrderedDict()
        self.TRAINED = False
        self.CLUSTERED = False
        self.num_of_cluster = -1
        self.cluster_info = dict()
        if filePath:
            self.fromFile(filePath)

    def fromFile(self, file):
        with open(file) as app_json:
            app = json.load(app_json)
            self.name = app['name']
            self.machine_id = app['machine_id']
            self.model_type = None if 'model_type' not in app else filter(
                lambda x: x.value == app['model_type'], PModelType)
            self.TRAINED = app['TRAINED']
            self.CLUSTERED = app['CLUSTERED']
            if app['TRAINED']:
                self.model_params = self.readParams(app['params'])
                self.TRAINED = True
            if app['CLUSTERED']:
                self.num_of_cluster = app['num_of_cluster']
                self.cluster_info = self.readClusterInfo(app['cluster_info'])

    def readParams(self, params_json):
        params = json.load(params_json)
        for attr, value in params:
            if attr in self.model_params.keys():
                util.RAPID_warn("duplicated attribute", attr)
            self.model_params[attr] = float(value)

    def readClusterInfo(self, cluster_json):
        pass

    def isTrained(self):
        return self.TRAINED

    def isClustered(self):
        return self.CLUSTERED
