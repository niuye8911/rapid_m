"""
 This is supposed to be the entry of backend solver
 functionality provided:
    1) trainSlowDown(): train an app's slow-down model given a list of environment + slow-down measurement
    2) trainEnvironment(): train the environment predictor given a list of pairs of environment
    3) getCluster(): provide a list of clusters given a list of profiles
    4) getBucket(): this is the core function that selects buckets during runtime
"""

import optparse
import sys
from enum import Enum

import ClusterTrainer
import PModelTrainer
from Utility import *


# modes supported by this learner
class Mode(Enum):
    TRAIN_SLOWDOWN = "TRAIN_SLOWDOWN"
    TRAIN_ENV = "TRAIN_ENV"
    TRAIN_CLUSTER = "TRAIN_CLUSTER"
    GET_BUCKETS = "GET_BUCKETS"


def main(argv):
    parser = genParser()
    options, args = parser.parse_args()

    # determine modes and pass params to different routine
    mode = filter(lambda x: x.value is options.mode, Mode)
    # validate the parameters
    checkParams(mode, options)

    if mode is Mode.TRAIN_SLOWDOWN:
        # train the slow-down by calling Rajanya's work
        accuracy = PModelTrainer.train(options.app_file, options.app_measurements)

    elif mode is Mode.TRAIN_CLUSTER:
        # cluster the slow-down by calling Asheley's work
        num_of_cluster = ClusterTrainer.cluster(options.app_file, options.app_profiles)


def checkParams(mode, options):
    if mode is Mode.TRAIN_SLOWDOWN:  # Rajanya's work starts here
        return not_none(options.app_file, options.app_measurements)
    elif mode is Mode.TRAIN_ENV:  # Abdal's work starts here
        return not_none(options.env_measurements)
    elif mode is Mode.GET_CLUSTER:  # Asheley's work starts here
        return not_none(options.app_profiles)
    elif mode is Mode.GET_BUCKETS:  # All our work starts here
        return not_none(options.active_apps)


def genParser():
    parser = optparse.OptionParser()
    # for slow-down training
    parser.add_option('--path2app', dest="app_file")
    parser.add_option('--appdata', dest="app_measurements")
    # for environment training
    parser.add_option('--envdata', dest="env_measurements")
    # for clustering
    parser.add_option('--apppfs', dest="app_profiles")
    # for bucket selection
    parser.add_option('--apps', dest="active_apps")
    # for mode
    parser.add_option('--mode', dest="mode")
    return parser


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
