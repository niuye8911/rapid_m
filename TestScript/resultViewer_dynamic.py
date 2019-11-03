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

mode_mapping = {'INDIVIDUAL':'CO',
'N':'ES',
'P_M':'RM',
'P_M_RUSH':'RM_RUSH',
'P_SAVING':'LOW'}

color = {
    'INDIVIDUAL': 'black',
    'N': 'r',
    'P_M': 'g',
    'P': "grey",
    'P_M_RUSH': 'blue',
    'P_SAVING': 'c'
}

def draw(summary, num_of_apps, budgets, modes, by_budget=False):
    ngroup = len(num_of_apps)
    if not by_budget:
        combined_sum = {}
        # combine all budgets
        for data,v in summary.items():
            combined_sum[data]={}
            for num_of_app, p in v.items():
                combined_sum[data][num_of_app]={}
                for mode in modes:
                    v = 0.0
                    for budget, pf in p.items():
                        v+=pf[mode]
                        print(pf[mode])
                    combined_sum[data][num_of_app][mode]=v/3.0
        fig, axes = plt.subplots(nrows=len(summary.keys()))
        fig.tight_layout()
        bar_width = 0.17
        opacity = 0.8
        index = np.arange(ngroup)
        plt.xlabel('Num Of Max Active Apps',fontsize=14)
        #plt.xticks(index + bar_width, range(2, n_groups + 2))
        rects = []
        id = 0
        for data_name, data in combined_sum.items():
            result = data
            # assign bars
            i = 0
            axes[id].set_ylim([0.0, 1.0])
            axes[id].set_yticks([0.0, 0.5, 1.0])
            if data_name == 'Reconfig':
                axes[id].set_ylim([0, 5])
                axes[id].set_yticks([0, 3, 5])
            rects_done = len(rects) > 0
            for mode in modes:
                values = [v[mode] for k, v in result.items()]
                rect = axes[id].bar(index + i * bar_width,
                                    values,
                                    bar_width,
                                    alpha=opacity,
                                    color=color[mode])
                if not rects_done:
                    rects.append(rect)
                i += 1
            axes[id].set_ylabel(data_name,fontsize=14)
            axes[id].yaxis.grid(which="major", linestyle='--', linewidth=0.7)
            plt.sca(axes[id])
            plt.xticks(index + bar_width, range(2, ngroup + 2),fontsize=12)
            id += 1
        #axes[0].axis("off")
        legends = [mode_mapping[k] for k in modes]
        fig.legend(rects, legends, loc='upper center', ncol=len(legends),fontsize=12)
        plt.subplots_adjust(bottom=0.12, left=0.1, top=0.88)
        #plt.tight_layout()
        plt.savefig('result_overall.png')
        return
    for budget in budgets:
        id = 0
        fig, axes = plt.subplots(nrows=len(summary.keys()))
        fig.tight_layout()
        bar_width = 0.17
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
            axes[id].set_yticks([0.0, 0.5, 1.0])
            if data_name == 'Reconfig':
                axes[id].set_ylim([0, 10])
                axes[id].set_yticks([0, 5, 10])
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
            axes[id].set_ylabel(data_name,fontsize=14)
            axes[id].yaxis.grid(which="major", linestyle='--', linewidth=0.7)
            plt.sca(axes[id])
            plt.xticks(index + bar_width, range(2, ngroup + 2),fontsize=12)
            id += 1
        #axes[0].axis("off")
        legends = [mode_mapping[k] for k in modes]
        fig.legend(rects, legends, loc='upper center', ncol=len(legends),fontsize=14)
        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9)
        plt.savefig('result_' + str(budget) + '.png')


def scale_mv(app_info, app, mv_value,mode):
    appmet = app_info[app]['met']
    mv_min = appmet.min_mv
    mv_max = appmet.max_mv
    scaled_mv = min(1.0, max(0.0, mv_value - mv_min) / (mv_max - mv_min))
    #scaled_mv = min(1.0, mv_value / mv_max )
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
                        total_runs = 0
                        svm_run = 0
                        for entry in f:
                            total_runs+=1
                            # check if it successes
                            suc = entry['success']
                            upscale += entry['scale_up']
                            #if not entry['app'] == 'svm':
                            reconfig += entry['total_reconfig']
                            if "P_M" in mode:
                                reconfig -= entry['rc_by_rapidm']
                            #else:
                            #    svm_run+=1
                            if suc == '1':
                                success += 1
                                # calculate mv
                                if entry['app'] == 'svm' and not mode == 'P_SAVING':
                                    mv_scale = 1.0
                                else:
                                    mv_scale = scale_mv(app_info, entry['app'],
                                                    entry['mv'],mode)
                                mv += mv_scale
                                budget_util.append(
                                    min(1.0,
                                        entry['elapsed'] / entry['budget']))
                            elif suc == '2':
                                    reject += 1
                            else:
                                # failed
                                mv -= 0.5
                    overall_success.append(success / total_runs)
                    overall_upscale.append(upscale / total_runs)
                    overall_reject.append(reject / total_runs)
                    overall_reconfig.append(reconfig / total_runs)
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
    summary = {}
    summary['Reconfig'] = reconfigs_all
    summary['Quality'] = mvs_all
    summary['Success'] = successes_all
    summary['Reject'] = rejects_all
    draw(summary, num_of_apps, budgets, modes)
    summary_file = 'summary.json'
    with open(summary_file,'w') as f:
        json.dump(summary,f,indent=2)
    # summarize the result
    #datas = ['Reconfig','Quality','Success','Reject']
    #for data in datas:
    #    for budget in budgets:
    #        for mode in modes:
    #            for num_of_app in num_of_apps:
    #                summary[data][num_of_app]
