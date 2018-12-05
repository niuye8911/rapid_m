"""
 This is an App Init-er
  Author:
    Liu Liu
    11/2018
"""

from Classes.App import *
from Classes.PModel import *
from Classes.SlowDownProfile import *
from ClusterTrainer import *
from Utility import *

MAX_ITERATION = 5
SLOWDOWN_THRESHOLD = .06


def init(app_file, performance_file, profile_file):
    # load in the file
    app = App(app_file)
    # check if the app is clustered
    if not app.isClustered():
        RAPID_info("clustering for ", app.name)
        # read in the slow-down file
        slowDownProfile = SlowDownProfile(performance_file, app.name)
        model_list, cluster_list = determine_k(slowDownProfile, profile_file)
        # write cluster info to app
        write_cluster_info(app, cluster_list)
        # write slow-down model to app
        id = 1
        for model in model_list:
            model.write_pmodel_info(app, get_cluster_name(app.name, id))
            id += 1
        app.TRAINED = True
        # write the app to file
        write_to_file(app_file, app)


def write_to_file(app_file, app):
    with open(app_file, 'w') as output:
        json.dump(app.__dict__, output, indent=2)


def determine_k(slowDownProfile, profile_file):
    # iterate through different cluster numbers
    observations, data = parseProfile(profile_file)
    model_list = []
    for num_of_cluster in range(1, MAX_ITERATION):
        # get the clusters
        observations, cluster_list, c = get_k_cluster(observations,
                                                      data,
                                                      num_of_cluster)
        # observations: <config_name, profile>
        # cluster_list:[[cluster_list]]
        # c: score
        id = 1
        accuracy = []
        model_list = []
        for cluster in cluster_list:
            # create model file
            tmp_model_file = "./tmp" + str(id) + ".pkl"
            id += 1
            # prepare data for training and validating
            clusterDF = slowDownProfile.getSubdata(cluster)
            pModel = PModel(tmp_model_file)
            pModel.setDF(clusterDF, slowDownProfile.getFeatures())
            pModel.train()
            mse, r2 = pModel.validate()
            model_list.append(pModel)
            accuracy.append(mse)
        average_accuracy = sum(accuracy) / len(accuracy)
        if average_accuracy <= SLOWDOWN_THRESHOLD:
            break
    return model_list, cluster_list
