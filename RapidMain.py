"""
 This is supposed to be the entry of backend solver
  Author:
    Liu Liu
    11/2018
"""

import optparse
import sys
import os
from enum import Enum

import ClusterTrainer
import PModelTrainer
import warnings
from AppInit import init
from BucketSelector import bucketSelect, getActiveApps
from MachineInit import trainEnv
from Utility import not_none, writeSelectionToFile
from sklearn.exceptions import DataConversionWarning

DEBUG = False

# ignore the TF debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# ignore the data conversion
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# flows supported by this learner
class Flow(Enum):
    TRAIN_SLOWDOWN = "TRAIN_SLOWDOWN"  # SLOWDOWN performance model
    TRAIN_ENV = "TRAIN_ENV"  # M-Model
    TRAIN_CLUSTER = "TRAIN_CLUSTER"  # Hierarchical Cluster
    GET_BUCKETS = "GET_BUCKETS"  # runtime Cluster selection
    INIT = "INIT"  # CLUSTER based on SLOWDOWN


def main(argv):
    global DEBUG
    parser = genParser()
    options, args = parser.parse_args()
    DEBUG = options.debug
    # determine modes and pass params to different routine
    flow = next(filter(lambda x: x.value == options.flow, Flow))

    # validate the parameters
    checkParams(flow, options)

    if flow is Flow.TRAIN_SLOWDOWN:
        # train the slow-down by calling Rajanya's work
        PModelTrainer.train(options.app_file, options.app_measurements)
        # do something about the accuracy

    elif flow is Flow.TRAIN_CLUSTER:
        # cluster the slow-down by calling Asheley's work
        ClusterTrainer.cluster(options.app_file, options.app_profiles)
        # do something about the number

    elif flow is Flow.TRAIN_ENV:
        # train the environment predictor by calling Liu's work
        trainEnv(options.machine_file,
                 options.machine_measurements,
                 options.dir,
                 TEST=DEBUG)
        # do something about the accuracy

    elif flow is Flow.INIT:
        test = options.test
        # cluster the app profile and check accuracuy
        init(options.app_file, options.app_measurements, options.app_profiles,
             options.dir, options.test, options.appname)

    elif flow is Flow.GET_BUCKETS:
        # find the optimal bucket selection for each active application
        result, buckets = bucketSelect(options.active_apps,
                                       SELECTOR=options.mode)
        if result == None and buckets == None:
            # no active apps
            writeSelectionToFile(options.result_file, options.active_apps,
                                 None, None, None, None, None)
        else:
            writeSelectionToFile(options.result_file, options.active_apps,
                                 result[0], result[1], result[2], result[3],
                                 buckets)


def checkParams(flow, options):
    if flow is Flow.TRAIN_SLOWDOWN:  # Rajanya's work starts here
        return not_none([options.app_file, options.app_measurements])
    elif flow is Flow.TRAIN_ENV:  # Liu's work starts here
        return not_none([options.machine_file, options.machine_measurements])
    elif flow is Flow.TRAIN_CLUSTER:  # Asheley's work starts here
        return not_none([options.app_profiles])
    elif flow is Flow.GET_BUCKETS:  # runtime compute
        return not_none([options.active_apps, options.result_file])
    elif flow is Flow.INIT:  # All our work starts here
        return not_none([
            options.app_file, options.app_measurements, options.app_profiles,
            options.dir
        ])


def genParser():
    parser = optparse.OptionParser()
    # for slow-down training
    parser.add_option('--path2app', dest="app_file")
    parser.add_option('--appdata', dest="app_measurements")
    # for environment training
    parser.add_option('--path2machine', dest="machine_file")
    parser.add_option('--envdata', dest="machine_measurements")
    # for clustering
    parser.add_option('--apppfs', dest="app_profiles")
    # for bucket selection
    parser.add_option('--apps', dest="active_apps")
    parser.add_option('--result', dest='result_file')
    parser.add_option('--mode', dest='mode', default='P_M')
    # for mode
    parser.add_option('--flow', dest="flow")
    # for server maintainance
    parser.add_option('--dir', dest="dir")
    parser.add_option(
        '--test', dest="test", action="store_true",
        default=False)  # if it's test, then won't modify the app file
    parser.add_option('--app', dest="appname", default='')
    parser.add_option(
        '-d', dest="debug", action="store_true",
        default=False)  # if it's test, then won't modify the app file
    return parser


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
