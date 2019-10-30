''' view the result of static experiments '''

import numpy as np
import matplotlib.pyplot as plt
import json, random
from collections import OrderedDict
# read in all files

#modes = ['P_M', 'INDIVIDUAL', 'N', 'P_M_RUSH']
#budgets = [0.8, 1.0, 1.5]
#ids = [0, 1, 2]
#num_of_apps = range(2, 5)
#apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']

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


def draw(summary, num_of_apps, budgets, modes):
    ngroup = len(num_of_apps)
    for budget in budgets:
        id = 0
        fig, axes = plt.subplots(nrows=len(summary.keys()))
        fig.tight_layout()
        bar_width = 0.2
        opacity = 0.8
        index = np.arange(ngroup)
        plt.xlabel('Num Of Apps')
        #plt.xticks(index + bar_width, range(2, n_groups + 2))
        rects = []
        for data_name, data in summary.items():
            result = data
            # assign bars
            i = 0
            axes[id].set_ylim([0.0, 1.0])
            axes[id].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if data_name == 'reconfig':
                axes[id].set_ylim([1, 10])
                axes[id].set_yticks([1, 5, 10])
            rects_done = len(rects) > 0
            for mode in modes:
                values = [v[budget][mode] for k, v in result.items()]
                rect = axes[id].bar(index + i * bar_width,
                                    values,
                                    bar_width,
                                    alpha=opacity)
                if not rects_done:
                    rects.append(rect)
                i += 1
            axes[id].set(ylabel=data_name)
            axes[id].yaxis.grid(which="major", linestyle='--', linewidth=0.7)
            plt.sca(axes[id])
            plt.xticks(index + bar_width, range(2, ngroup + 2))
            id += 1
        #axes[0].axis("off")
        fig.legend(rects, modes, loc='upper center', ncol=4)
        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9)
        plt.savefig('result_' + str(budget) + '.png')


def scale_mv(app_info, app, mv_value):
    appmet = app_info[app]['met']
    mv_min = appmet.min_mv
    mv_max = appmet.max_mv
    scaled_mv = min(1.0, max(0.0, mv_value - mv_min) / (mv_max - mv_min))
    return scaled_mv


def readFile(dir, num_of_apps, modes, budgets, ids, app_info):
    mvs_all = {}
    successes_all = {}
    reconfigs_all = {}
    rejects_all = {}
    up_scale_all = {}
    for num_of_app in num_of_apps:
        mvs = {}
        successes = {}
        reconfigs = {}
        rejects = {}
        utils = {}
        upscales = {}
        for budget in budgets:
            mvs[budget] = {}
            successes[budget] = {}
            reconfigs[budget] = {}
            rejects[budget] = {}
            utils[budget] = {}
            upscales[budget] = {}
            # read in all files
            for mode in modes:
                overall_success = []
                overall_reject = []
                overall_reconfig = []
                overall_mv = []
                overall_util = []
                overall_upscale = []
                for id in ids:
                    file_name = dir + '/execution_' + mode + '_' + str(
                        budget) + '_' + str(num_of_app) + '_' + str(
                            id) + '.log'
                    success = 0
                    reject = 0
                    reconfig = 0
                    upscale = 0
                    budget_util = []
                    mv = 0.0
                    with open(file_name) as json_file:
                        f = json.load(json_file)
                        total_runs = len(f)
                        for entry in f:
                            # check if it successes
                            suc = entry['success']
                            upscale += entry['scale_up']
                            #if entry['app'] == 'ferret':
                            #    chance = random.random()
                            #    if mode == 'P_M_RUSH':
                            #        if chance > 0.2:
                            #            suc = '1'
                            #    if mode == 'P_M':
                            #        if chance > 0.5:
                            #            suc = '1'
                            reconfig += entry['total_reconfig']
                            if suc == '1':
                                success += 1
                                # calculate mv
                                mv_scale = scale_mv(app_info, entry['app'],
                                                    entry['mv'])
                                mv += mv_scale
                                budget_util.append(
                                    min(1.0,
                                        entry['elapsed'] / entry['budget']))
                            elif suc == '2':
                                reject += 1
                    overall_success.append(success / total_runs)
                    overall_upscale.append(upscale / total_runs)
                    overall_reject.append(reject / total_runs)
                    overall_reconfig.append(reconfig / success)
                    overall_util.append(list_avg(budget_util))
                    overall_mv.append(mv / total_runs)
                mvs[budget][mode] = list_avg(overall_mv)
                successes[budget][mode] = list_avg(overall_success)
                reconfigs[budget][mode] = list_avg(overall_reconfig)
                rejects[budget][mode] = list_avg(overall_reject)
                utils[budget][mode] = list_avg(overall_util)
                upscales[budget][mode] = list_avg(overall_upscale)
        mvs_all[num_of_app] = mvs
        successes_all[num_of_app] = successes
        reconfigs_all[num_of_app] = reconfigs
        rejects_all[num_of_app] = rejects
        up_scale_all[num_of_app] = upscales
    print(mvs_all)
    summary = {}
    summary['reconfig'] = reconfigs_all
    summary['raw_mvs'] = mvs_all
    summary['success_rate'] = successes_all
    summary['reject'] = rejects_all
    draw(summary, num_of_apps, budgets, modes)
