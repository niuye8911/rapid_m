"""
 This is supposed to be the entry of backend solver
  Author:
    Liu Liu
    11/2018
"""

import optparse
import sys
from enum import Enum

import ClusterTrainer
import MModelTrainer
import PModelTrainer
from AppInit import init
from MachineInit import trainEnv
from Utility import not_none


# flows supported by this learner
class Flow(Enum):
    TRAIN_SLOWDOWN = "TRAIN_SLOWDOWN"  # SLOWDOWN performance model
    TRAIN_ENV = "TRAIN_ENV"  # M-Model
    TRAIN_CLUSTER = "TRAIN_CLUSTER"  # Hierarchical Cluster
    GET_BUCKETS = "GET_BUCKETS"  # runtime Cluster selection
    INIT = "INIT"  # CLUSTER based on SLOWDOWN


def main(argv):
    parser = genParser()
    options, args = parser.parse_args()
    # determine modes and pass params to different routine
    flow = next(filter(lambda x: x.value == options.flow, Flow))

    # validate the parameters
    checkParams(flow, options)

    if flow is Flow.TRAIN_SLOWDOWN:
        # train the slow-down by calling Rajanya's work
        p_accuracy = PModelTrainer.train(options.app_file,
                                         options.app_measurements)
        # do something about the accuracy

    elif flow is Flow.TRAIN_CLUSTER:
        # cluster the slow-down by calling Asheley's work
        num_of_cluster = ClusterTrainer.cluster(options.app_file,
                                                options.app_profiles)
        # do something about the number

    elif flow is Flow.TRAIN_ENV:
        # train the environment predictor by calling Liu's work
        trainEnv(options.machine_file, options.machine_measurements,
                 options.dir)
        # do something about the accuracy

    elif flow is Flow.INIT:
        # cluster the app profile and check accuracuy
        init(options.app_file, options.app_measurements, options.app_profiles,
             options.dir)


def checkParams(flow, options):
    if flow is Flow.TRAIN_SLOWDOWN:  # Rajanya's work starts here
        return not_none([options.app_file, options.app_measurements])
    elif flow is Flow.TRAIN_ENV:  # Liu's work starts here
        return not_none([options.machine_file, options.machine_measurements])
    elif flow is Flow.TRAIN_CLUSTER:  # Asheley's work starts here
        return not_none([options.app_profiles])
    elif flow is Flow.GET_BUCKETS:  # runtime compute
        return not_none([options.active_apps])
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
    # for mode
    parser.add_option('--flow', dest="flow")
    # for server maintainance
    parser.add_option('--dir', dest="dir")
    return parser


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
