''' view the result of static experiments '''

import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
# read in all files

modes = ['P', 'P_M', 'INDIVIDUAL', 'N']
budgets = [0.8, 1.0, 1.5]
apps = []
n_groups = 0

mvs = OrderedDict()
exceeds = OrderedDict()
misses = OrderedDict()
datas = {'mv': mvs, 'exceed': exceeds, 'miss_pred': misses}


def readFile(dir):
    global n_groups, apps
    for budget in budgets:
        mvs[budget] = {}
        exceeds[budget] = {}
        misses[budget] = {}
        # read in all files
        for data, result in datas.items():
            for mode in modes:
                result[budget][mode] = []
                file_name = dir + '/' + data + '_' + mode + '_' + str(
                    budget) + '.json'
                with open(file_name) as json_file:
                    f = json.load(json_file, object_pairs_hook=OrderedDict)
                    for num, values in f.items():
                        if apps == []:
                            apps = list(values.keys())
                            n_groups = len(apps) - 1
                        c = list(values.values())
                        result[budget][mode].append(sum(c) / len(c))


def draw():
    sub_graphs = ['exceed', 'miss_pred', 'mv']
    for budget in budgets:
        id = 0
        fig, axes = plt.subplots(nrows=3)
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


readFile('./10_1_without_svm')
draw()
