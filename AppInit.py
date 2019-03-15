"""
 This is an App Init-er
  Author:
    Liu Liu
    11/2018
"""

from Classes.App import *
from Classes.PModel import *
from Classes.SlowDownProfile import *
from Classes.AppSysProfile import *
from ClusterTrainer import *
from PModelTrainer import *
from Utility import *
import pandas as pd

MAX_ITERATION = 5
SLOWDOWN_THRESHOLD = .07


def init(app_file, performance_file, profile_file, directory, DRAW=True):
    # load in the file
    app = App(app_file)
    # check if the app is clustered
    if not app.isClustered():
        RAPID_info("clustering for ", app.name)
        # read in the slow-down file
        slowDownProfile = SlowDownProfile(
            pd.read_csv(performance_file), app.name)
        appSysProfile = AppSysProfile(pd.read_csv(profile_file), app.name)

        pModelTrainer, cluster_list, Z = determine_k(
            slowDownProfile, appSysProfile, directory, app.name)
        exit(1)
        # write cluster info to app
        write_cluster_info(app, cluster_list)

        # write pModels to file
        pModelTrainer.write_to_file(directory)

        # write slow-down model to app
        pModelTrainer.dump_into_app(app)

        app.TRAINED = True
        # write the app to file
        write_to_file(app_file, app)
        # whether to show the cluster result
        if DRAW:
            draw(Z)
        # write the scaled perf-file to disk
        slowDownProfile.writeOut(app.name + "-perf_scaled.csv")


def write_to_file(app_file, app):
    with open(app_file, 'w') as output:
        json.dump(app.__dict__, output, indent=2)


def determine_k(slowDownProfile, appSysProfile, directory, app_name):
    # iterate through different cluster numbers
    pModelTrainer = PModelTrainer(app_name, slowDownProfile)
    for num_of_cluster in range(1, MAX_ITERATION + 1):
        # get the clusters
        cluster_list, c, Z = get_k_cluster(
            appSysProfile, num_of_cluster)
        RAPID_info("Partition Lvl:", str(num_of_cluster))
        pModelTrainer.updateCluster(cluster_list)
        pModelTrainer.train()
        diff = pModelTrainer.getDiff()
        RAPID_info("average DIFF:", str(diff))
        if diff <= SLOWDOWN_THRESHOLD:
            break
    return pModelTrainer, cluster_list, Z
