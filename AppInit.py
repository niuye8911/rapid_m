"""
 This is an App Init-er
  Author:
    Liu Liu
    11/2018
"""

from ClusterTrainer import *
from PModelTrainer import *
from Utility import *
from Classes.App import App
from Classes.SlowDownProfile import SlowDownProfile
from Classes.AppSysProfile import AppSysProfile

MAX_ITERATION = 10
SLOWDOWN_THRESHOLD = .07

INCREMENTAL = "incremental"
DIRECT_K = "direct_k"


def init(app_file,
         performance_file,
         profile_file,
         directory,
         test=True,
         app_name='',
         DRAW=True):
    # load in the file
    app = App(app_file, not test)
    if not app_name == '':
        app.name = app_name
    # check if the app is clustered
    if not app.isClustered():
        RAPID_info("clustering for ", app.name)
        # read in the slow-down file
        slowDownProfile = SlowDownProfile(pd.read_csv(performance_file),
                                          app.name)
        appSysProfile = AppSysProfile(pd.read_csv(profile_file), app.name)
        # get the maxes
        maxes = getMaxes(slowDownProfile.dataFrame[slowDownProfile.x])
        app.maxes = maxes
        pModelTrainer, cluster_list, Z = determine_k_incremental(
            slowDownProfile, appSysProfile, directory, app)
        # calculate the average envs
        rep_env = gen_rep_env(profile_file, cluster_list)

        # write cluster info to app
        write_cluster_info(app, cluster_list, rep_env)

        # write pModels to file
        pModelTrainer.write_to_file(directory)

        # write slow-down model to app
        pModelTrainer.dump_into_app()

        app.TRAINED = True

        # write the app to file
        output_file = app_file if app.overwrite else directory + '/' + app.name + ".json"
        write_to_file(output_file, app)
        # whether to show the cluster result
        if DRAW:
            draw(Z)
        # write the scaled perf-file to disk
        slowDownProfile.writeOut(app.name + "-perf_scaled.csv")


def getMaxes(X):
    ''' scale the data '''
    maxes = {}
    for col in X.columns:
        # take the maximum number of two vectors per feature
        maxes[col] = X.max()[col]
    return maxes


def gen_rep_env(sys_file, cluster_list):
    sys = pd.read_csv(sys_file)
    avgs = []
    for cluster in cluster_list:
        rows = sys.loc[sys['Configuration'].isin(cluster)]
        avg = rows.mean(axis=0)
        avgs.append(avg)
    return avgs


def write_to_file(app_file, app):
    with open(app_file, 'w') as output:
        json.dump(app.__dict__, output, indent=2)


def determine_k(slowDownProfile, appSysProfile, directory, app):
    # iterate through different cluster numbers
    pModelTrainer = PModelTrainer(app, slowDownProfile)
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


def determine_k_incremental(slowDownProfile,
                            appSysProfile,
                            directory,
                            app,
                            UPDATE_THROUGH_P=True):
    # iterate through different cluster numbers
    pModelTrainer = PModelTrainer(app, slowDownProfile)
    cluster_list = []
    target_id = -1
    cluster_list, Z = first_cut(appSysProfile)
    k = len(cluster_list)
    print('based on the criterion, clustered into', k)
    pModelTrainer.updateCluster(cluster_list)
    pModelTrainer.train()
    if not UPDATE_THROUGH_P:
        return pModelTrainer, cluster_list, Z
    for num_of_cluster in range(k, MAX_ITERATION + 1):
        # if any cluster cannot be separated to another cluster
        if [] in cluster_list:
            # go for traditional cluster
            cluster_list, Z = get_k_cluster(appSysProfile, num_of_cluster)

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
        cluster_list, Z = increment_cluster(appSysProfile, cluster_list,
                                            target_id)
    return pModelTrainer, cluster_list, Z
