''' view the result of static experiments '''

import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
# read in all files

modes = ['P_M', 'INDIVIDUAL', 'N']
budgets = [1.0]
ids = [0, 1]
apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']

mvs = OrderedDict()
successes = OrderedDict()
reconfigs = OrderedDict()
utils = OrderedDict()
rejects = OrderedDict()


def list_avg(l):
    if len(l) == 0:
        return 0
    else:
        return sum(l) / len(l)


def readFile(dir):
    for budget in budgets:
        mvs[budget] = {}
        successes[budget] = {}
        reconfigs[budget] = {}
        rejects[budget] = {}
        utils[budget] = {}
        # read in all files
        for mode in modes:
            overall_success = []
            overall_reject = []
            overall_reconfig = []
            overall_mv = []
            overall_util = []
            for id in ids:
                file_name = dir + '/execution_' + mode + '_' + str(
                    budget) + '_' + str(id) + '.log'
                success = 0
                reject = 0
                reconfig = 0
                budget_util = []
                mv = 0.0
                with open(file_name) as json_file:
                    f = json.load(json_file)
                    total_runs = len(f)
                    for entry in f:
                        # check if it successes
                        suc = entry['success']
                        reconfig += entry['total_reconfig']
                        if suc == '1':
                            success += 1
                            mv += entry['mv']
                            budget_util.append(
                                min(1.0, entry['elapsed'] / entry['budget']))
                        elif suc == '2':
                            reject += 1
                overall_success.append(success / total_runs)
                overall_reject.append(reject / total_runs)
                overall_reconfig.append(reconfig / success)
                overall_util.append(list_avg(budget_util))
                overall_mv.append(mv)
            mvs[budget][mode] = list_avg(overall_mv)
            successes[budget][mode] = list_avg(overall_success)
            reconfigs[budget][mode] = list_avg(overall_reconfig)
            rejects[budget][mode] = list_avg(overall_reject)
            utils[budget][mode] = list_avg(overall_util)


readFile('./')
print('reconfigs', reconfigs)
print('raw_mvs', mvs)
print('success_rate', successes)
print('util', utils)
print('reject', rejects)
