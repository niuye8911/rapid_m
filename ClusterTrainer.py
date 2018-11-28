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
    if not app.isTrained():
        app['params'] = {}

        # TODO:Ashley's work goes here*******

        cluster_distance, cluster_info, num_of_cluster = dummyCluster(
            app_profile)

        # TODO:END of Rajanya's work*******

        # Note: the final goal of this function is to write the cluster-info
        # back to app file
        app['CLUSTERED'] = True
        app['cluster_info'] = cluster_info
        app['num_of_cluster'] = num_of_cluster
        with open(app_file, 'w') as output:
            json.dump(app, output, indent=2)

        RAPID_info("clustering for " + app['name'], str(cluster_distance))


def dummyCluster(measurements):
    # I do nothing
    dummy_cluster_info = {"first": list("config1", "config2"),
                          "second": list("config3", "config4")}
    dummy_num_of_cluster = 2
    dummy_cluster_distance = 0.2
    return dummy_cluster_distance, dummy_cluster_info, dummy_num_of_cluster
