"""
 This is an App Init-er
  Author:
    Liu Liu
    11/2018
"""

from Classes.App import *
from Classes.SlowDownProfile import *
from ClusterTrainer import cluster, parseProfile
from Utility import *

MAX_ITERATION = 5


def init(app_file, performance_file, profile_file):
    app = App(app_file)
    # check if the app is clustered
    if not app.isClustered():
        RAPID_info("clustering for ", app.name)
        # read in the slow-down file
        slowDownProfile = SlowDownProfile(performance_file, app.name)
        determine_k(profile_file)


def write_to_file(app_file, app):
    with open(app_file, 'w') as output:
        json.dump(app.__dict__, output, indent=2)


def determine_k(profile_file):
    # iterate through different cluster numbers
    accuracy = []
    observations, data = parseProfile(profile_file)
    for num_of_cluster in range(1, MAX_ITERATION):
        # get the clusters
        observations, cluster_list, c = cluster(observations, data,
                                                num_of_cluster)
        # observations: <config_name, profile>
        # cluster_list:[[cluster_list]]
        # c: score
        # link Rajanya's work here

    return num_of_cluster
