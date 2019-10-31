"""
 This is supposed to be the entry of backend solver
  Author:
    Liu Liu
    11/2018
"""

import optparse
import sys
import os
from enum import Enum

import ClusterTrainer, json
import PModelTrainer
import warnings
from AppInit import init
from BucketSelector import bucketSelect, genBuckets
from MachineInit import trainEnv
from Utility import not_none, writeSelectionToFile
from sklearn.exceptions import DataConversionWarning
import socket
import functools
import itertools
import json, time

import pandas as pd

from Rapid_M_Classes.App import App
from Rapid_M_Classes.Bucket import Bucket
from Rapid_M_Classes.MModel import MModel
from Rapid_M_Classes.PModel import PModel
from Utility import *
from DataUtil import *

DEBUG = False

# ignore the TF debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# ignore the data conversion
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

s = socket.socket()
host = "localhost"
ALL_APP_FILE = '/var/www/html/rapid_server/storage/data_machine_algaesim.txt'
MACHINE_FILE = '/home/liuliu/Research/rapid_m_backend_server/examples/example_machine_empty.json'
m_model = None
p_models = {}
buckets = {}
port = 12345
s.bind((host, port))

s.listen(5)


def getAllApps():
    apps = []
    with open(ALL_APP_FILE, 'r') as file:
        active_apps = json.load(file)
        applications = active_apps['applications']
        for app in applications:
            application = App(app['dir'] + '/' + 'profile' + '.json')
            apps.append({'app': application, 'dir': app['dir']})
    return apps


def getPModels(apps):
    models = {}
    for app in apps:
        app = app['app']
        models[app.name] = {}
        for bucket_name, info in app.model_params.items():
            models[app.name][bucket_name] = PModel(info, app.maxes)
        RAPID_info("Model Loader", "loaded P-Models for " + app.name)
    return models


def prepare_models_and_buckets():
    global m_model, p_models, buckets
    apps = getAllApps()
    # prepare mmodel
    m_model = MModel(MACHINE_FILE)
    print("mmodel loads done")
    # prepare p_model
    p_models = getPModels(apps)
    print("pmodels loads done")
    # prepare buckets
    buckets = genBuckets(apps, p_models)
    print("all buckets loads done")


prepare_models_and_buckets()

while True:
    c, addr = s.accept()
    data = c.recv(1024)
    if data:
        json_data = json.loads(data)
        mode = str(json_data['mode'])
        data_file = json_data['data']
        result_file = json_data['result']
        print("getting data,", mode, data_file, result_file)
        result, return_buckets = bucketSelect(data_file,
                                              SELECTOR=mode,
                                              P_MODELS=p_models,
                                              M_MODEL=m_model,
                                              BUCKETS=buckets)
        if result == None and return_buckets == None:
            # no active apps
            writeSelectionToFile(result_file, data_file, None, None, None,
                                 None, None)
        else:
            writeSelectionToFile(result_file, data_file, result[0], result[1],
                                 result[2], result[3], buckets)
    c.close()
