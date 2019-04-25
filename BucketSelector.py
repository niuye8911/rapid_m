"""
 This is the optimizer to select the correct bucket for each active apps
  Author:
    Liu Liu
    04/2019
"""

from Classes.App import *
from Classes.PModel import *
from Classes.SlowDownProfile import *
from Classes.AppSysProfile import *
from Classes.Bucket import *
from Classes.MModel import *
from Utility import *
import pandas as pd
import itertools
import json

MACHINE_FILE = '/home/liuliu/Research/rapid_m_backend_server/examples/example_machine_empty.json'


def bucketSelect(active_apps_file, SELECTOR="P_M"):
    with open(active_apps_file, 'r') as file:
        active_apps = json.load(file)

        if SELECTOR == "P_M":
            return pmSelect(active_apps)


def pmSelect(active_apps):
    # get the M-Model
    m_model = MModel(MACHINE_FILE)
    RAPID_info('M-Model Loader: ', m_model.TRAINED)
    # get all the apps
    apps = getActiveApps(active_apps)
    # get all the P_models {app_name: {bucket_name: model }}
    models = loadAppModels(apps)
    # convert apps to buckets
    buckets = genBuckets(apps, models)
    # get all combinations of buckets
    bucket_combs = getBucketCombs(buckets)
    # predict the overall envs for each comb
    combined_envs = getEnvs(bucket_combs, m_model)


def getEnvs(bucket_combs, mmodel):
    ''' generate the combined env using M-Model '''
    for comb in bucket_combs:
        env_dicts = list(map(lambda x: formatEnv(x.rep_env, mmodel.features), comb))
        print(env_dicts)


def formatEnv(env, features):
    result = []
    features = list(map(lambda x: x[:-2], features)) # remove the '-C'
    for feature in features:
        if feature == 'MEM':
            result.append(env['READ'] + env['WRITE'])
        elif feature == 'INST':
            result.append(env['ACYC'] / env['INST'])
        elif feature == 'INSTnom' or feature == 'PhysIPC%':
            result.append(env[feature] / 100.0)
        else:
            result.append(env[feature])
    return list(map(lambda x: float(x),result))

def getBucketCombs(buckets):
    bucket_lists = buckets.values()
    combs = list(itertools.product(*bucket_lists))
    return combs
    #printBucketCombs(combs)


def printBucketCombs(combs):
    names = list(map(lambda x: list(map(lambda y: y.b_name, x)), combs))
    print(names)


def genBuckets(apps, models):
    buckets = {}
    for app in apps:
        dir = app['dir']
        app = app['app']
        buckets[app.name] = []
        for bucket_name, info in app.cluster_info.items():
            bucket = Bucket(app.name, bucket_name, info['cluster'],
                            models[app.name][bucket_name], dir + '/cost.csv',
                            dir + '/mv.csv', info['env'])
            buckets[app.name].append(bucket)
        RAPID_info("Bucket Loader", "loaded Buckets for " + app.name)
    return buckets


def getActiveApps(active_apps):
    '''return a list of App instances from the profile'''
    apps = []
    applications = active_apps['applications']
    for app in applications:
        status = app['status']
        if status == 0:
            # inactive apps
            continue
        application = App(app['dir'] + '/profile.json')
        apps.append({
            'app': application,
            'budget': app['budget'],
            'dir': app['dir']
        })
    return apps


def loadAppModels(apps):
    models = {}
    for app in apps:
        app = app['app']
        models[app.name] = {}
        for bucket_name, info in app.model_params.items():
            models[app.name][bucket_name] = PModel(info)
        RAPID_info("Model Loader", "loaded P-Models for " + app.name)
    return models
