"""
 Clustering app's configs based on prediction accuracy of slow-down
  Author:
    Ashley Dunn / Liu Liu
    12/2018
"""
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
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


def draw(Z):
    # view of basic Dendrogram with all clusters
    plt.figure(figsize=(25, 10))
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        truncate_mode='lastp',
    )
    plt.show()

    # view of Dendrogram with only the last 30 clusters and distances marked
    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('(cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'],
                               ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate(
                        "%.3f" % y, (x, y),
                        xytext=(0, -5),
                        textcoords='offset points',
                        va='top',
                        ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
    )
    plt.show()

    # elbow method, need to look at better
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want
    # 2 clusters
    print
    "clusters:", k
