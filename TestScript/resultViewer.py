''' view the result of static experiments '''

import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
# read in all files

modes = ['P','P_M','INDIVIDUAL','N']
budgets = [0.8,1.0,1.5]
apps = ['swaptions','ferret','bodytrack','svm','nn','facedetect']

mvs = OrderedDict()
exceeds = OrderedDict()
misses = OrderedDict()
datas = {'mv':mvs,'exceed':exceeds,'miss_pred':misses}

n_groups = 5


def readFile(dir):
    for budget in budgets:
        mvs[budget] = {}
        exceeds[budget] = {}
        misses[budget] = {}
        # read in all files
        for data,result in datas.items():
            for mode in modes:
                result[budget][mode] = []
                file_name = dir+'/'+data+'_'+mode+'_'+str(budget)+'.json'
                with open(file_name) as json_file:
                    f = json.load(json_file,object_pairs_hook=OrderedDict)
                    for num,values in f.items():
                        c = list(values.values())
                        result[budget][mode].append(sum(c)/len(c))

def draw():
    for budget in budgets:
        id = 0
        fig,axes = plt.subplots(3)
        bar_width=0.2
        opacity = 0.8
        index = np.arange(n_groups)
        plt.xlabel('Num Of Apps')
        plt.xticks(index+bar_width,range(2,n_groups+2))
        for data, result in datas.items():
            # assign bars
            i = 0
            for mode in modes:
                rect = axes[id].bar(index+i*bar_width, result[budget][mode], bar_width, alpha=opacity,label=mode)
                i+=1
            axes[id].set(ylabel=data)
            id += 1
        plt.legend(loc='upper right')
        plt.savefig('result_'+str(budget)+'.png')


readFile('./aug18')
draw()
