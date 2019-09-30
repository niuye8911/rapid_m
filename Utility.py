import json, os
import pprint
from functools import reduce

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import csv


def __get_minmax(file):
    min_v = 999999.0
    max_v = -999999.0
    with open(file, 'r') as f:
        for line in f:
            col = line.rstrip().split(' ')
            v = float(col[-1])
            min_v = min(min_v, v)
            max_v = max(max_v, v)
    return min_v, max_v


def updateAppMinMax(app_name, appMethod):
    base_dir = "/home/liuliu/Research/rapid_m_backend_server/outputs/"
    cost_file = base_dir + app_name + '/cost.csv'
    mv_file = base_dir + app_name + '/mv.csv'
    if os.path.exists(cost_file):
        min_cost, max_cost = __get_minmax(cost_file)
        appMethod.min_cost = min_cost
        appMethod.max_cost = max_cost
    if os.path.exists(mv_file):
        min_mv, max_mv = __get_minmax(mv_file)
        appMethod.min_mv = min_mv
        appMethod.max_mv = max_mv


def writeSelectionToFile(f, comb_name, selection, slowDownTable):
    output = open(f, 'w')
    result = []
    bucket_list = comb_name.split(',')
    for app, config in selection.items():
        bucket = list(filter(lambda x: app in x, bucket_list))[0]
        result.append({
            'name':
            app,
            'bucket':
            bucket,
            'config':
            config,
            'slowdown_p':
            slowDownTable[app],
            'slowdown':
            slowDownTable[app] if slowDownTable[app] >= 1.0 else 1.0
        })
    json.dump(result, output, indent=2)
    output.close()


def printTrainingInfo(d):
    output = open('./outputs/training_info.csv', 'w')
    # get all features
    features = list(d.keys())
    # get all the models
    tmp = features[0]
    models = list(d[tmp].keys())
    # write the header
    output.write('feature,' + ','.join(models) + '\n')
    time = []
    r2 = []
    mse = []
    diff = []
    for feature in features:
        time_line = []
        r2_line = []
        mse_line = []
        diff_line = []
        for model in models:
            time_line.append(str(d[feature][model]['time']))
            r2_line.append(str(d[feature][model]['r2']))
            mse_line.append(str(d[feature][model]['mse']))
            diff_line.append(str(d[feature][model]['diff']))
        time.append(feature + ',' + ','.join(time_line))
        r2.append(feature + ',' + ','.join(r2_line))
        mse.append(feature + ',' + ','.join(mse_line))
        diff.append(feature + ',' + ','.join(diff_line))
    output.write('\n'.join(time))
    output.write('\nr2\n')
    output.write('\n'.join(r2))
    output.write('\nmse\n')
    output.write('\n'.join(mse))
    output.write('\ndiff\n')
    output.write('\n'.join(diff))
    output.close()


def printDicToFile(d, f, CSV=False):
    output = open(f, 'w')
    if CSV:  # write in csv
        csv_headers = list(d.keys())
        writer = csv.DictWriter(output, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerow(d)
    else:  # write in json
        json.dump(d, output, indent=2)
    output.close()


def PPRINT(stuff):
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(stuff)


def RAPID_warn(prefix, message):
    print("RAPID_LEARNER WARNING: " + str(prefix) + ":" + str(message))


def RAPID_info(prefix, message):
    print("RAPID_LEARNER INFO: " + str(prefix) + ":" + str(message))


def not_none(values):
    return reduce(lambda x, y: x and y, values, True)


def cal_ci(values):
    n = len(values)
    m, se = np.mean(values), scipy.stats.sem(values)
    h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
    ci_upp = m + h
    ci_low = m - h
    return m, ci_upp, ci_low


def draw_ci(ci_file, output):
    names = []
    ci_lows = []
    ci_upps = []
    means = []
    highest = -99
    lowest = 99
    with open(ci_file) as f:
        for line in f:
            items = line.split(',')
            names.append(items[0])
            means.append(float(items[1]))
            ci_lows.append(float(items[1]) - float(items[2]))
            ci_upps.append(float(items[3]) - float(items[1]))
            highest = max(float(items[3]), highest)
            lowest = min(float(items[2]), lowest)
    (_, caps, _) = plt.errorbar(range(len(names)),
                                means,
                                yerr=[ci_lows, ci_upps],
                                fmt='o',
                                ecolor='g',
                                capsize=10)
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.xticks(range(len(names)), names, fontsize='10', rotation=30)
    plt.ylabel('Prediction MRE', fontsize='15')
    plt.ylim(lowest * 2.0, highest * 2.0)
    plt.savefig(output + '.png')


# view of Dendrogram with only the last 30 clusters and distances marked
def simplified_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('(cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'],
                           ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3f" % y, (x, y),
                             xytext=(0, -5),
                             textcoords='offset points',
                             va='top',
                             ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def elbow(Z):
    # elbow method, need to look at better
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want
    # 2 clusters
    return k


def draw(Z, simplified=True):
    # view of basic Dendrogram with all clusters
    cluster_fig = plt.figure(figsize=(25, 10))
    if not simplified:
        # show full dendrogram
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            truncate_mode='lastp',
        )
    else:
        simplified_dendrogram(
            Z,
            truncate_mode='lastp',
            p=30,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=
            10,  # useful in small plots so annotations don't overlap
        )
    cluster_fig.savefig('cluter_result.pdf', bbox_inches='tight')
