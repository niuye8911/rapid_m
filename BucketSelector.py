"""
 This is the optimizer to select the correct bucket for each active apps
  Author:
    Liu Liu
    04/2019
"""

import functools
import itertools
import json

import pandas as pd

from Classes.App import *
from Classes.Bucket import *
from Classes.MModel import *
from Classes.PModel import *
from Utility import *
from DataUtil import *

MACHINE_FILE = '/home/liuliu/Research/rapid_m_backend_server/examples/example_machine_empty.json'
DELIMITER = ","  # bucket comb delimiter


def bucketSelect(active_apps_file, SELECTOR="P_M", env=[]):
    with open(active_apps_file, 'r') as file:
        active_apps = json.load(file)
        # get all the apps
        apps = getActiveApps(active_apps)
        if SELECTOR == "P_M":
            return pmSelect(apps)
        if SELECTOR == "P":
            if env == []:
                exit(1)
            return pSelect(apps, env)


def pmSelect(apps):
    # get the M-Model
    m_model = MModel(MACHINE_FILE)
    features = m_model.features
    RAPID_info('M-Model Loader: ', m_model.TRAINED)
    # get all the P_models {app_name: {bucket_name: model }}
    p_models = loadAppModels(apps)
    # convert apps to buckets
    buckets = genBuckets(apps, p_models)
    # get all combinations of buckets
    bucket_combs = getBucketCombs(buckets)
    # predict the overall envs for each comb
    combined_envs = getEnvs(bucket_combs, mmodel=m_model)
    # predict the per-app slow-down
    slowdowns = getSlowdowns(combined_envs, p_models, features)
    # get the bucket selection based on slow-down
    PPRINT(slowdowns)
    return getSelection(slowdowns, apps, buckets)


def pSelect(apps, measured_env):
    # get all the P_models {app_name: {bucket_name: model }}
    p_models = loadAppModels(apps)
    # convert apps to buckets
    buckets = genBuckets(apps, p_models)
    # get all combinations of buckets
    bucket_combs = getBucketCombs(buckets)
    # formulate the measured combined envs
    combined_envs = getEnvs(bucket_combs, P_ONLY=True, env=measured_env)
    # predict the per-app slow-down
    slowdowns = getSlowdowns(combined_envs, p_models, features)
    # get the slowdown
    PPRINT(slowdowns)
    return getSelection(slowdowns, apps, buckets)


def getSelection(slowdowns, apps, buckets):
    ''' get the final selection of configs and buckets '''
    selection = {}
    best_mv = 0.0
    for bucketlist, appslowdowns in slowdowns.items():
        total_mv = 0.0
        configs = {}
        bucket_list = bucketlist.split(',')
        app_names = list(appslowdowns.keys())
        # get overall mv per bucket list
        for app in app_names:
            bucket_name = list(filter(lambda x: app in x, bucket_list))[0]
            bucket = list(
                filter(lambda x: x.b_name == bucket_name, buckets[app]))[0]
            slow_down = appslowdowns[app]
            budget = list(filter(lambda x: x['app'].name == app,
                                 apps))[0]['budget']
            config, mv, SUCCESS = bucket.getOptimal(budget, slow_down)
            # TODO: what to use when SUCCESS is false
            configs[app] = config[0]
            total_mv += mv[0]
        if total_mv > best_mv:
            selection = configs.copy()
    PPRINT(selection)
    return selection


def getSlowdowns(combined_envs, p_models, features):
    ''' return the slow-down for each app with slow-down '''
    slowDownTable = {}
    for comb_name, env in combined_envs.items():
        buckets = comb_name.split(DELIMITER)
        comb_slowdown = {}
        for bucket in buckets:
            app_name = bucket[:-1]
            p_model = p_models[app_name][bucket]
            # tranform the vector to dataframe
            slowdown = p_model.predict(env_to_frame(env, features))
            comb_slowdown[app_name] = slowdown[0]
        slowDownTable[comb_name] = comb_slowdown
    return slowDownTable


def getEnvs(bucket_combs, mmodel=None, P_ONLY=False, env=[]):
    ''' generate the combined env using M-Model '''
    result = {}
    for comb in bucket_combs:
        comb_name = DELIMITER.join((list(map(lambda x: x.b_name, comb))))
        if not P_ONLY:
            env_dicts = list(
                map(lambda x: formatEnv(x.rep_env, mmodel.features), comb))
            # cumulatively apply M-Model
            final_env = functools.reduce(lambda x, y: mReducer(x, y, mmodel),
                                         env_dicts)
        else:
            # P_ONLY
            final_env = env
        # print(final_env)
        result[comb_name] = final_env
    return result


def mReducer(env1, env2, mmodel):
    result = mmodel.predict(env1, env2)
    return list(result.values[0])


def getBucketCombs(buckets):
    bucket_lists = buckets.values()
    combs = list(itertools.product(*bucket_lists))
    return combs
    # printBucketCombs(combs)


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
            models[app.name][bucket_name] = PModel(info, app.maxes)
        RAPID_info("Model Loader", "loaded P-Models for " + app.name)
    return models
