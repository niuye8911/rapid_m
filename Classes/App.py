'''
    Class for an app
    An App contains information:
    1) name (string)
    2) machine_id (int)
    3) TRAINED (boolean)
    5) model_params<attribute, value> (dict<string,float>)
    6) CLUSTERED (boolean)
    7) num_of_cluster (int)
    8) cluster_info
'''
import json
from collections import OrderedDict


class App:
    def __init__(self, filePath="", overwrite=False):
        '''if overwrite, the file will be overwritten by new content'''
        self.name = ""
        self.machine_id = -1
        self.model_params = OrderedDict()
        self.TRAINED = False
        self.CLUSTERED = False
        self.num_of_cluster = -1
        self.cluster_info = dict()
        self.overwrite = overwrite
        self.maxes = {}
        if filePath:
            self.fromFile(filePath)

    def fromFile(self, file):
        with open(file) as app_json:
            app = json.load(app_json)
            self.name = app['name']
            self.machine_id = app['machine_id']
            self.TRAINED = app['TRAINED']
            self.CLUSTERED = app['CLUSTERED']
            self.maxes = {} if 'maxes' not in app else app['maxes']
            if app['TRAINED']:
                self.readParams(app['model_params'])
                self.TRAINED = True
            if app['CLUSTERED']:
                self.num_of_cluster = app['num_of_cluster']
                self.cluster_info = app['cluster_info']
                self.CLUSTERED = True

    def readParams(self, params_json):
        for bucket_name, bucket_info in params_json.items():
            self.model_params[bucket_name] = bucket_info

    def isTrained(self):
        return self.TRAINED

    def isClustered(self):
        return self.CLUSTERED
