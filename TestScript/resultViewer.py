''' view the result of static experiments '''

import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
# read in all files

modes = ['N', 'INDIVIDUAL', 'P', 'P_M']
budgets = [0.8, 1.0, 1.5]
apps = []
n_groups = 0

mvs = OrderedDict()
exceeds = OrderedDict()
misses = OrderedDict()
exceed_rates = OrderedDict()
datas = {
    'mv': mvs,
    'exceed': exceeds,
    'miss_pred': misses,
    'exceed_rate': exceed_rates
}

mode_mapping = {
    'INDIVIDUAL': 'CO',
    'N': 'ES',
    'P_M': 'RM',
    'P': "AS",
    'P_M_RUSH': 'RM_RUSH',
    'P_SAVING': 'LOW'
}

color = {
    'INDIVIDUAL': 'black',
    'N': 'r',
    'P_M': 'g',
    'P': "grey",
    'P_M_RUSH': 'blue',
    'P_SAVING': 'c'
}


def readFile(dir):
    global n_groups, apps
    for budget in budgets:
        mvs[budget] = {}
        exceeds[budget] = {}
        misses[budget] = {}
        exceed_rates[budget] = {}
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
    #sub_graphs = ['exceed', 'miss_pred', 'mv','exceed_rate']
    sub_graphs = ['exceed', 'miss_pred', 'exceed_rate']
    for budget in budgets:
        id = 0
        fig, axes = plt.subplots(nrows=len(sub_graphs))
        fig.tight_layout()
        bar_width = 0.2
        opacity = 0.8
        index = np.arange(n_groups)
        plt.xlabel('Num Of Apps', fontsize=16)
        #plt.xticks(index + bar_width, range(2, n_groups + 2))
        rects = []
        for data in sub_graphs:
            result = datas[data]
            # assign bars
            i = 0
            axes[id].set_ylim([0.0, 1.0])
            axes[id].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if data == 'exceed_rate':
                axes[id].set_ylim([0.0, 0.75])
                axes[id].set_yticks([0.0, 0.25, 0.5, 0.75])
            rects_done = len(rects) > 0
            for mode in modes:
                rect = axes[id].bar(index + i * bar_width,
                                    result[budget][mode],
                                    bar_width,
                                    alpha=opacity,
                                    color=color[mode])
                if not rects_done:
                    rects.append(rect)
                i += 1
            if data == 'exceed':
                data = 'violation'
            axes[id].set_ylabel(data, fontsize=14)
            #axes[id].set(ylabel=data,fontsize=12)
            axes[id].yaxis.grid(which="major", linestyle='--', linewidth=0.7)
            plt.sca(axes[id])
            plt.xticks(index + bar_width, range(2, n_groups + 2), fontsize=12)
            id += 1
        #axes[0].axis("off")
        legends = [mode_mapping[k] for k in modes]
        fig.legend(rects, legends, loc='upper center', ncol=4, fontsize=14)
        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9)
        plt.savefig('result_' + str(budget) + '.png')


readFile('./10_4')
draw()
