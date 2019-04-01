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

INCREMENTAL = "incremental"
DIRECT_K = "direct_k"


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

        pModelTrainer, cluster_list, Z = determine_k_incremental(
            slowDownProfile, appSysProfile, directory, app.name)

        # write cluster info to app
        write_cluster_info(app, cluster_list)

        # write pModels to file
        pModelTrainer.write_to_file(directory)

        # write slow-down model to app
        pModelTrainer.dump_into_app(app)

        app.TRAINED = True
        # write the app to file
        #write_to_file(app_file, app)
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
        # partition the cluster with the hist error
        cluster_list, Z = get_k_cluster(appSysProfile, num_of_cluster)
        RAPID_info("Partition Lvl:", str(num_of_cluster))
        pModelTrainer.updateCluster(cluster_list)
        pModelTrainer.train()
        diff, largest_id = pModelTrainer.getDiff()
        r2, largest_r2_id = pModelTrainer.getMSE()
        RAPID_info("average DIFF/MSE:", diff)
        RAPID_info("largest_id:", largest_id)
        RAPID_info("size:", str(list(map(lambda x: len(x), cluster_list))))
        print("\n")
        if sum(diff) / len(diff) <= SLOWDOWN_THRESHOLD:
            break
    return pModelTrainer, cluster_list, Z


def determine_k_incremental(slowDownProfile, appSysProfile, directory,
                            app_name):
    # iterate through different cluster numbers
    pModelTrainer = PModelTrainer(app_name, slowDownProfile)
    cluster_list = []
    target_id = -1
    for num_of_cluster in range(1, MAX_ITERATION + 1):
        cluster_list, Z = increment_cluster(appSysProfile, cluster_list,
                                            target_id)
        RAPID_info("Partition Lvl:", str(num_of_cluster))
        pModelTrainer.updateCluster(cluster_list)
        pModelTrainer.train()
        diff, target_id = pModelTrainer.getDiff()
        r2, tmp_id = pModelTrainer.getMSE()
        RAPID_info("average DIFF/MSE:", str(diff) + '/' + str(r2))
        RAPID_info("largest_id:", target_id)
        RAPID_info("size:", str(list(map(lambda x: len(x), cluster_list))))
        print("\n")
        if sum(diff) / len(diff) <= SLOWDOWN_THRESHOLD:
            break
    return pModelTrainer, cluster_list, Z
