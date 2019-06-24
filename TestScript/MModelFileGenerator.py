''' generate a MModel file based on
1) Perf(feature-c)
2) Mperf(feature-2) and
3)sys
'''

import pandas as pd

BASE_DIR = '/home/liuliu/Research/rapid_m_backend_server/testData/halfandhalf/'
APPS = ['ferret', 'swaptions', 'bodytrack', 'svm', 'nn']
RESULT = '../testData/mmodelfile_w_info.csv'

HEADER_DONE = False


def getMModelFile(sysFile, mperfFile, perfFile, result):
    # get the column names
    columns = pd.read_csv(sysFile).columns.values.tolist()[1:]
    # get all the configs
    configs = pd.read_csv(sysFile)['Configuration'].tolist()
    # get all the config sys
    config_footprint = getConfigFootprint(sysFile, columns)
    # get all the added environment to the configs
    added_env = getAddedEnv(mperfFile, columns, mperf = True)
    # get all the overall environment according to the slowdown
    overall_env = getOverallEnv(perfFile, columns)
    # assemble everything into a dataFrame
    assembleAlltoFile(columns, config_footprint, added_env, overall_env,
                      result)


def assembleAlltoFile(columns, config_footprint, added_envs, overall_envs,
                      result):
    global HEADER_DONE
    # write the header
    if not HEADER_DONE:
        result.write(','.join(map(lambda x: x + '-1', columns)))
        result.write(',')
        result.write(','.join(map(lambda x: x + '-2', columns)))
        result.write(',')
        result.write(','.join(map(lambda x: x + '-C', columns)))
        # add two columns
        result.write(',cpu_load,mem_load')
        result.write('\n')
        HEADER_DONE = True
    # write the observations config by config
    for config in config_footprint.keys():
        lines = map(
            lambda x: ','.join(x) + '\n',
            assemblePerConfig(config, config_footprint, added_envs,
                              overall_envs))
        for line in lines:
            result.write(line)


def assemblePerConfig(config, config_footprint, added_envs, overall_envs):
    sys1 = config_footprint[config]
    lines = []
    for slowdown, added_env in added_envs[config].items():
        sys2 = added_env['env']
        cpu_load = added_env['load'][0]
        mem_load = added_env['load'][1]
        if slowdown not in overall_envs[config]:
            # broken in perf
            continue
        sysC = overall_envs[config][slowdown]['env']
        line = sys1 + sys2 + sysC
        line = list(map(lambda x: str(x), line))
        # add load info
        line += [str(cpu_load), str(mem_load)]
        lines.append(line)
    return lines


def getConfigFootprint(sysFile, columns):
    config_fps = {}
    for index, row in pd.read_csv(sysFile).iterrows():
        config_fps[row['Configuration']] = row[columns].values.tolist()
    return config_fps


def getOverallEnv(perfFile, columns):
    return getAddedEnv(perfFile, columns)


def getAddedEnv(mperfFile, columns, mperf=False):
    ''' for each config, indexed by the slow-down '''
    added_envs = {}
    for index, row in pd.read_csv(mperfFile).iterrows():
        config = row['Configuration']
        slowDown = float(row['SLOWDOWN'])
        added_env = row[columns].values.tolist()

        if -1 in added_env or len(added_env) != len(columns):
            # broken line
            continue
        if config not in added_envs:
            added_envs[config] = {}
        cpu_load = -1
        mem_load = -1
        if mperf:
            cpu_load, mem_load = getStresser(row['stresser'])
        added_envs[config][slowDown] = {
            'env': added_env,
            'load': [cpu_load, mem_load]
        }
    return added_envs


def getStresser(info):
    # 0 = low, 1 = med, 2 = high
    stresser = info.split(':')[1]
    cpu = -1
    mem = -1
    if 'cpu' in stresser:
        # this is a streetool instance
        stresser = stresser.split('_')
        cpu_num = int(stresser[1])
        io_num = int(stresser[3])
        vm_num = int(stresser[5])
        bytes = stresser[7]
        if cpu_num > 3:
            cpu = 3
        else:
            cpu = 2
        if vm_num > 2 and bytes == '1M':
            mem = 3
        else:
            mem = 2
    else:
        cpu = 1
        mem = 1
    return cpu, mem


result = open(RESULT, 'w')
for app in APPS:
    # for type in ['small','big']:
    #    sys_file = BASE_DIR+app+"-sys-"+type+".csv"
    #    mperf_file = BASE_DIR+app+"-mperf-"+type+".csv"
    #    perf_file = BASE_DIR+app+"-perf-"+type+".csv"
    #    getMModelFile(sys_file, mperf_file,perf_file,result)
    sys_file = BASE_DIR + app + "-sys.csv"
    mperf_file = BASE_DIR + app + "-mperf.csv"
    perf_file = BASE_DIR + app + "-perf.csv"
    getMModelFile(sys_file, mperf_file, perf_file, result)
result.close()
