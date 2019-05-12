''' generate a MModel file based on
1) Perf(feature-c)
2) Mperf(feature-2) and
3)sys
'''

import pandas as pd

BASE_DIR = '/home/liuliu/Research/rapid_m_backend_server/testData/appself/'
APPS = ['ferret','swaptions','bodytrack','facedetect']
RESULT = './mmodelfile.csv'

HEADER_DONE = False

def getMModelFile(sysFile, mperfFile, perfFile, result):
    # get the column names
    columns = pd.read_csv(sysFile).columns.values.tolist()[1:]
    # get all the configs
    configs = pd.read_csv(sysFile)['Configuration'].tolist()
    # get all the config sys
    config_footprint = getConfigFootprint(sysFile, columns)
    # get all the added environment to the configs
    added_env = getAddedEnv(mperfFile, columns)
    # get all the overall environment according to the slowdown
    overall_env = getOverallEnv(perfFile, columns)
    # assemble everything into a dataFrame
    assembleAlltoFile(columns, config_footprint, added_env, overall_env, result)


def assembleAlltoFile(columns, config_footprint, added_envs, overall_envs, result):
    global HEADER_DONE
    #write the header
    if not HEADER_DONE:
        result.write(','.join(map(lambda x: x + '-1', columns)))
        result.write(',')
        result.write(','.join(map(lambda x: x + '-2', columns)))
        result.write(',')
        result.write(','.join(map(lambda x: x + '-C', columns)))
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
        sys2 = added_env
        sysC = overall_envs[config][slowdown]
        line = sys1 + sys2 + sysC
        line = list(map(lambda x: str(x), line))
        lines.append(line)
    return lines


def getConfigFootprint(sysFile, columns):
    config_fps = {}
    for index, row in pd.read_csv(sysFile).iterrows():
        config_fps[row['Configuration']] = row[columns].values.tolist()
    return config_fps


def getOverallEnv(perfFile, columns):
    return getAddedEnv(perfFile, columns)


def getAddedEnv(mperfFile, columns):
    ''' for each config, indexed by the slow-down '''
    added_envs = {}
    for index, row in pd.read_csv(mperfFile).iterrows():
        config = row['Configuration']
        slowDown = float(row['SLOWDOWN'])
        added_env = row[columns].values.tolist()
        if config not in added_envs:
            added_envs[config] = {}
        added_envs[config][slowDown] = added_env
    return added_envs

result = open(RESULT,'w')
for app in APPS:
    for type in ['small','big']:
        sys_file = BASE_DIR+app+"-sys-"+type+".csv"
        mperf_file = BASE_DIR+app+"-mperf-"+type+".csv"
        perf_file = BASE_DIR+app+"-perf-"+type+".csv"
        getMModelFile(sys_file, mperf_file,perf_file,result)
result.close()
