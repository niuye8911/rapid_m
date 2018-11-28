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

from rapid_m_backend_server import Utility as util


class MModelType(Enum):
    LINEAR = "IDONTKNOW"


class Machine:
    def __init__(self, filePath=""):
        self.name = ""
        self.machine_id = -1
        self.model_type = None
        self.model_params = dict()
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
            self.model_type = filter(lambda x: x.value is app['model_type'],
                                     PModelType)
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
