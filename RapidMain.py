"""
 This is supposed to be the entry of backend solver
 functionality provided:
    1) predictEnvironment(): predict the combined environment given two system profile
    2) predictSlowDown(): predict the slow-down factor given a profile + an app's model
    3) trainSlowDown(): train an app's slow-down model given a list of environment + slow-down measurement
    4) trainEnvironment(): train the environment predictor given a list of pairs of environment
    5) getCluster(): provide a list of clusters given a list of profiles
"""

import optparse
import sys


def main(argv):
    pass


def genParser():
    parser = optparse.OptionParser()
    # for slow-down training
    parser.add_option('--path2app', dest="app_file")
    parser.add_option('--appdata', dest="app_measurements")
    # for slow-down prediction
    parser.add_option('--env', dest="environment")
    # for environment training
    parser.add_option('--envdata', dest="env_measurements")
    # for environment prediction
    parser.add_option('--envpfs', dest="active_profiles")
    # for clustering
    parser.add_option('--apppfs', dest="app_profiles")
    return parser


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
