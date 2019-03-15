"""
 Clustering app's configs based on prediction accuracy of slow-down
  Author:
    Ashley Dunn / Liu Liu
    12/2018
"""
import numpy as np
from collections import OrderedDict
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist


def get_k_cluster(appSysProfile, k):
    '''
    Train the app using a proper model
    :param app_profile: a csv file containing all the configuration with
    their measurements
    :param k: number of clusters to cluster
    :return: void, but write the clusters to the file
    '''

    # Z: the linkage matrix
    # c: the coefficient distance
    Z, c = hCluster(appSysProfile.getData())
    # get the clusters
    clusters = fcluster(Z, k, criterion='maxclust')
    # get the cluster list
    cluster_list = get_cluster_list(clusters, appSysProfile, k)
    return cluster_list, c, Z


def get_cluster_list(clusters, appSysProfile, k):
    observations = appSysProfile.getConfigs()
    cluster_info_list = []
    for i in range(1, k + 1):
        cluster_info_list.append([])
    for i in range(0, len(clusters)):
        cluster_info_list[clusters[i] - 1].append(observations[i])
    return cluster_info_list


def hCluster(data):
    # hierarchal clustering
    Z = linkage(data, 'ward')
    c, coph_dists = cophenet(Z, pdist(data))
    return Z, c


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
