''' view the result of static experiments '''

import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
# read in all files

modes = ['P_M', 'INDIVIDUAL', 'N']
budgets = [1.0]
ids = [0,1]
apps = ['swaptions','ferret','bodytrack','svm','nn','facedetect']

mvs = OrderedDict()
successes = OrderedDict()
reconfigs = OrderedDict()
utils = OrderedDict()
rejects = OrderedDict()

def list_avg(l):
    if len(l)==0:
        return 0
    else:
        return sum(l)/len(l)

def readFile(dir):
    for budget in budgets:
        mvs[budget] = {}
        successes[budget] = {}
        reconfigs[budget] = {}
        rejects[budget]={}
        utils[budget] = {}
            # read in all files
        for mode in modes:
            overall_success = []
            overall_reject = []
            overall_reconfig = []
            overall_mv = []
            overall_util = []
            for id in ids:
                file_name = dir + '/execution_'+mode+'_'+str(budget)+'_'+str(id)+'.log'
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
                        if suc == '1':
                            success+=1
                            mv+=entry['mv']
                            reconfig+=entry['total_reconfig']
                            budget_util.append(max(1.0,entry['elapsed']/entry['budget']))
                        elif suc == '2':
                            reject+=1
                overall_success.append(success/total_runs)
                overall_reject.append(reject/total_runs)
                overall_reconfig.append(reconfig/success)
                overall_util.append(list_avg(budget_util))
                overall_mv.append(mv)
            mvs[budget][mode] = list_avg(overall_mv)
            successes[budget][mode] = list_avg(overall_success)
            reconfigs[budget][mode] = list_avg(overall_reconfig)
            rejects[budget][mode]=list_avg(overall_reject)
            utils[budget][mode] = list_avg(overall_util)




def draw():
    sub_graphs = ['exceed', 'miss_pred', 'mv','exceed_rate']
    for budget in budgets:
        id = 0
        fig, axes = plt.subplots(nrows=4)
        fig.tight_layout()
        bar_width = 0.2
        opacity = 0.8
        index = np.arange(n_groups)
        plt.xlabel('Num Of Apps')
        #plt.xticks(index + bar_width, range(2, n_groups + 2))
        rects = []
        for data in sub_graphs:
            result = datas[data]
            # assign bars
            i = 0
            axes[id].set_ylim([0.0, 1.0])
            axes[id].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if data=='exceed_rate':
                axes[id].set_ylim([0.0,0.5])
                axes[id].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            rects_done = len(rects) > 0
            for mode in modes:
                rect = axes[id].bar(index + i * bar_width,
                                    result[budget][mode],
                                    bar_width,
                                    alpha=opacity)
                if not rects_done:
                    rects.append(rect)
                i += 1
            axes[id].set(ylabel=data)
            axes[id].yaxis.grid(which="major", linestyle='--', linewidth=0.7)
            plt.sca(axes[id])
            plt.xticks(index + bar_width, range(2, n_groups + 2))
            id += 1
        #axes[0].axis("off")
        fig.legend(rects, modes, loc='upper center', ncol=4)
        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9)
        plt.savefig('result_' + str(budget) + '.png')

readFile('./')
print(reconfigs)
print(mvs)
print(successes)
print(utils)
print(rejects)
