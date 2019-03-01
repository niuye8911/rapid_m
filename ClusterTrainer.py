"""
 Clustering app's configs based on prediction accuracy of slow-down
  Author:
    Ashley Dunn / Liu Liu
    12/2018
"""
from collections import OrderedDict

import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist


# parse the profile
def parseProfile(measurements):
    with open(measurements, 'r') as f:
        # get all the features
        features = f.readline().strip(',\n').split(",")[1:]
        # init the clustering data
        data = np.empty((0, len(features)))
        observations = OrderedDict()

        for line in f:
            observation = line.partition(",")
            config_name = observation[0]
            observation_data = list(
                map(lambda x: float(x),
                    observation[2].strip(',\n').split(",")))
            observations[config_name] = observation_data
            data = np.append(data, [observation_data], axis=0)

    if data.size == 0:
        print("error reading csv file")

    return observations, data


def get_k_cluster(observations, data, k):
    '''
    Train the app using a proper model
    :param app_profile: a csv file containing all the configuration with
    their measurements
    :param k: number of clusters to cluster
    :return: void, but write the clusters to the file
    '''

    # Z: the linkage matrix
    # c: the coefficient distance
    observations, Z, c = hCluster(observations, data)
    # get the clusters
    clusters = fcluster(Z, k, criterion='maxclust')
    # get the cluster list
    cluster_list = get_cluster_list(clusters, observations, k)
    return observations, cluster_list, c, Z


def get_cluster_list(clusters, observations, k):
    observations = list(observations.keys())
    cluster_info_list = []
    for i in range(1, k + 1):
        cluster_info_list.append([])
    for i in range(0, len(clusters)):
        cluster_info_list[clusters[i] - 1].append(observations[i])
    return cluster_info_list


def hCluster(observations, data):
    # hierarchal clustering
    Z = linkage(data, 'ward')
    c, coph_dists = cophenet(Z, pdist(data))
    return observations, Z, c


def write_cluster_info(app, cluster_info_list):
    k = len(cluster_info_list)
    cluster_info = OrderedDict()
    for i in range(1, k + 1):
        cluster_info[get_cluster_name(app.name,
                                      str(i))] = cluster_info_list[i - 1]
    app.cluster_info = cluster_info
    app.num_of_cluster = k
    app.CLUSTERED = True
    app.model_type = "LINEAR"


def get_cluster_name(app_name, id):
    return app_name + str(id)
