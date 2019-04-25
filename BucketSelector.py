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
from Utility import *
import pandas as pd
import json


def bucketSelect(active_apps_file, SELECTOR="P_M"):
    with open(active_apps_file, 'r') as file:
        active_apps = json.load(file)

        if SELECTOR == "P_M":
            return pmSelect(active_apps)


def pmSelect(active_apps):
    # get all the apps
    apps = getActiveApps(active_apps)
    # get all the P_models {app_name: {bucket_name: model }}
    models = loadAppModels(apps)
    # convert apps to buckets
    buckets = genBuckets(apps, models)


def genBuckets(apps, models):
    buckets = {}
    for app in apps:
        dir = app['dir']
        app = app['app']
        buckets[app.name]={}
        for bucket_name, info in app.cluster_info.items():
            bucket = Bucket(info['cluster'], models[app.name][bucket_name],
                            dir + '/cost.csv', dir + '/mv.csv', info['env'])
            buckets[app.name][bucket_name] = bucket
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
