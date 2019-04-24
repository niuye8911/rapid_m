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


def increment_cluster(appSysProfile, cluster_list, target_id):
    '''
    Rather than getting k clusteres, partition only the cluster with the highest
    Error
    '''
    Z, c = hCluster(appSysProfile.getData())
    # the first cut
    if target_id == -1:
        return get_cluster_list(
            fcluster(Z, 1, criterion='maxclust'), appSysProfile.dataFrame,
            1), Z
    # else, cluster the target cluster into 2 parts
    subFrame = appSysProfile.getSubFrameByConfigs(cluster_list[target_id])
    x = appSysProfile.getX()
    subZ, c = hCluster(subFrame[x])
    clusters = fcluster(subZ, 2, criterion='maxclust')
    updated_list = get_cluster_list(clusters, subFrame, 2)
    # replace the original list by two updated list
    del cluster_list[target_id]
    cluster_list.insert(target_id, updated_list[0])
    cluster_list.insert(target_id + 1, updated_list[1])
    return cluster_list, Z


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
    cluster_list = get_cluster_list(clusters, appSysProfile.dataFrame, k)
    return cluster_list, Z


def get_cluster_list(clusters, df, k):
    observations = df['Configuration'].values
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


def df_to_dict(rep_env):
    dicts = rep_env.to_dict()
    return dicts


def write_cluster_info(app, cluster_info_list, rep_envs):
    k = len(cluster_info_list)
    cluster_info = OrderedDict()
    for i in range(1, k + 1):
        cluster_info[get_cluster_name(app.name, str(i))] = {
            'cluster': cluster_info_list[i - 1],
            'env': df_to_dict(rep_envs[i - 1])
        }
    app.cluster_info = cluster_info
    app.num_of_cluster = k
    app.CLUSTERED = True
    app.model_type = "LINEAR"


def get_cluster_name(app_name, id):
    return app_name + str(id)
