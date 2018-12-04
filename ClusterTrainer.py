"""
 Clustering app's configs based on prediction accuracy of slow-down
  Author:
    Ashley Dunn / Liu Liu
    12/2018
"""

import csv
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

from Classes.App import *
from Utility import *


def cluster(app_file, app_profile):
    '''
    Train the app using a proper model
    :param app_file: the path to the app file (string)
    :param app_profile: a csv file containing all the configuration with
    their measurements
    :return: void, but write the clusters to the file
    '''

    # get the app object
    app = App(app_file)
    if not app.isClustered():
        # Z: the linkage matrix
        # c: the coefficient distance
        observations, Z, c = hCluster(app_profile)
        k = determine_k(Z, observations)
        clusters = fcluster(Z, k, criterion='maxclust')

        # write result to object
        write_cluster_info(app, clusters, observations, k)

        with open(app_file, 'w') as output:
            json.dump(app.__dict__, output, indent=2)

        RAPID_info("clustering for " + app.name, str(c))


def determine_k(Z, observations):
    # iterate through different cluster numbers
    for num_of_cluster in range(1, len(Z) + 2):
        clusters = fcluster(Z, num_of_cluster, criterion='maxclust')
        cluster_list = get_cluster_list(clusters, observations, num_of_cluster)
        # check the average accuracy

    return num_of_cluster


def get_cluster_list(clusters, observations, k):
    cluster_info_list = []
    for i in range(1, k + 1):
        cluster_info_list[i] = []
    for i in range(0, len(clusters)):
        cluster_info_list[clusters[i] - 1].append(observations[i])
    return cluster_info_list


def write_cluster_info(app, clusters, observations, k):
    # init cluster_info
    cluster_info = OrderedDict()
    cluster_info_list = get_cluster_list(clusters, observations, k)
    for i in range(1, k + 1):
        cluster_info[app.name + str(i)] = cluster_info_list[i - 1]
    app.cluster_info = cluster_info
    app.num_of_cluster = k
    app.CLUSTERED = True


def hCluster(measurements):
    with open(measurements, 'r') as f:
        csv_data = csv.reader(f, delimiter=',')
        # get all the features
        features = f.readline().strip(',\n').split(",")
        # init the clustering data
        data = np.empty((0, len(features)))
        observations = []

        for line in f:
            observation = line.partition(",")
            config_name = observation[0]
            observation_data = map(lambda x: float(x), observation[
                2].strip(',\n').split(","))
            observations.append(config_name)
            data = np.append(data, [observation_data], axis=0)

    if data.size == 0:
        print("error reading csv file")

    # hierarchal clustering
    Z = linkage(data, 'ward')
    c, coph_dists = cophenet(Z, pdist(data))
    return observations, Z, c


def draw(Z):
    # view of basic Dendrogram with all clusters
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
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
                    plt.annotate("%.3f" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
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
