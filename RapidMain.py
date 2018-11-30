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
from Utility import *


# flows supported by this learner
class Flow(Enum):
    TRAIN_SLOWDOWN = "TRAIN_SLOWDOWN"
    TRAIN_ENV = "TRAIN_ENV"
    TRAIN_CLUSTER = "TRAIN_CLUSTER"
    GET_BUCKETS = "GET_BUCKETS"


def main(argv):
    parser = genParser()
    options, args = parser.parse_args()

    # determine modes and pass params to different routine
    flow = filter(lambda x: x.value is options.flow, Flow)
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
        # train the environment predictor by calling Abdall's work
        m_accuracy = MModelTrainer.train(options.env_measurements)
        # do something about the accuracy


def checkParams(flow, options):
    if flow is Flow.TRAIN_SLOWDOWN:  # Rajanya's work starts here
        return not_none([options.app_file, options.app_measurements])
    elif flow is Flow.TRAIN_ENV:  # Abdal's work starts here
        return not_none([options.env_measurements, options.machine_file])
    elif flow is Flow.GET_CLUSTER:  # Asheley's work starts here
        return not_none([options.app_profiles])
    elif flow is Flow.GET_BUCKETS:  # All our work starts here
        return not_none([options.active_apps])


def genParser():
    parser = optparse.OptionParser()
    # for slow-down training
    parser.add_option('--path2app', dest="app_file")
    parser.add_option('--appdata', dest="app_measurements")
    # for environment training
    parser.add_option('--machine', dest="machine_file")
    parser.add_option('--envdata', dest="env_measurements")
    # for clustering
    parser.add_option('--apppfs', dest="app_profiles")
    # for bucket selection
    parser.add_option('--apps', dest="active_apps")
    # for mode
    parser.add_option('--flow', dest="flow")
    return parser


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
