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
import pandas as pd
import json


def bucketSelect(active_apps_file, SELECTOR="P_M"):
    with open(active_apps_file, 'r') as file:
        active_apps = json.load(file)
        if SELECTOR == "P_M":
            return pmSelect(active_apps)


def pmSelect(active_apps):
    apps = getActiveApps(active_apps)
    for app in apps:
        print(app['app'].name)


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
        apps.append({'app': application, 'budget': app['budget']})
    return apps
