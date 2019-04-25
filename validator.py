# validate the accuracy of applying M+P models
import optparse
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from Classes.App import *
from sklearn.metrics import r2_score
from sklearn import metrics
from Classes.Machine import *
from Classes.SlowDownProfile import *
from Classes.AppSysProfile import *


def main(argv):
    parser = genParser()
    options, args = parser.parse_args()
    app_summary = getApp(options.app_summary)
    app_name = app_summary.name
    m_models = getMModel(options.machine_summary)
    observation = getObservation(options.observation, app_name)
    appsys = getAppSys(options.app_sys, app_name)
    validate(observation, app_summary, appsys, m_models)


def validate(observation, app_summary, appsys, m_models):
    '''validate the M+P model
    observation: the observated slowdown given an env
    app_summary: containing all the P_model for a specific bucket
    m_model: the machine model
    '''
    dataFrame = observation.getSubdata()
    configs = list(dataFrame['Configuration'])
    y_gt = np.array(dataFrame['SLOWDOWN'].values.tolist())
    # generate the predicted slow-down
    y_pred = np.array(genPred(observation, app_summary, appsys, m_models))
    # validate the diff
    mse = np.sqrt(metrics.mean_squared_error(y_gt, y_pred))
    mae = metrics.mean_absolute_error(y_gt, y_pred)
    r2 = r2_score(y_gt, y_pred)
    # relative error
    diff = abs(y_gt - y_pred) / y_gt
    diff = sum(diff) / len(diff)
    print(mae)


def genPred(observation, app_summary, appsys, m_models):
    ''' generate the prediction of slow-down'''
    dataFrame = observation.getSubdata()
    y_pred = []
    debug_file = open('./tmp.csv', 'w')
    debug_file.write(','.join(observation.x) + '\n')
    for index, row in dataFrame.iterrows():
        # take the config
        config = row['Configuration']
        # take the pmodel
        p_model, features, isPoly = getPModel(config, app_summary)
        # take the added env
        added_env = row[observation.x]
        # take the app's footprint
        app_env = appsys.getSysByConfig(config)
        # predict the combined env
        env1 = list(added_env)
        env2 = app_env.values.tolist()[0]
        env = combineEnvs(env1, env2)

        # predic the values feature by feature
        pred_env = []
        for feature in observation.x:
            if feature not in features:
                # pass if the feature is not needed
                pred_env.append('-')
                continue
            model = m_models[feature]['model']['model']
            env_isPoly = m_models[feature]['poly']
            if env_isPoly:
                input_env = PolynomialFeatures(degree=2).fit_transform([env])
            else:
                input_env = [env]
            value = model.predict(input_env)[0]
            pred_env.append(value)
        writeEnvsToDebug(debug_file,env1,env2,pred_env)

        # filter out the unwanted env
        # predict the slowdown
        # prepare data for the P model
        filtered_pred_env = list(filter(lambda x: x!='-', pred_env))
        data_x = [filtered_pred_env]
        if isPoly:
            data_x = PolynomialFeatures(degree=2).fit_transform(data_x)
        pred_slowdown = p_model.predict(data_x)
        y_pred.append(pred_slowdown[0])
        debug_file.write(str(row['SLOWDOWN']) + ',')
        debug_file.write(str(pred_slowdown) + '\n\n')
    debug_file.close()
    return y_pred

def combineEnvs(env1, env2):
    env = env2 + env1
    # format the data
    distance = int(len(env) / 2)
    for i in range(0, distance):
        f1 = env[i]
        f2 = env[i + distance]
        env[i] = min(f1, f2)
        env[i + distance] = max(f1, f2)
    return env

def writeEnvsToDebug(debug_file,env1,env2,env3):
    debug_file.write(','.join(map(lambda x: str(x), env1)))
    debug_file.write('\n')
    debug_file.write(','.join(map(lambda x: str(x), env2)))
    debug_file.write('\n')
    debug_file.write(','.join(map(lambda x: str(x), env3)))
    debug_file.write('\n')

def getPModel(config, summary):
    '''fetch the P-model for a configuration'''
    buckets = summary.cluster_info
    models = summary.model_params
    p_model_file = ''
    for name, configs in buckets.items():
        if config in configs['cluster']:
            # this is the correct bucket
            for target, params in models.items():
                if target == name:
                    p_model_file = params['file']
                    features = params['feature']
                    poly = params['poly']
                    break
    if p_model_file == '':
        print("WARNING: no p model found for config:"+config)
        exit(0)
    return pickle.load(open(p_model_file, 'rb')), features, poly


def getMModel(summary_file):
    machine = Machine(summary_file)
    mmodelfiles = machine.model_params['MModelfile']
    mmodelpoly = machine.model_params['MModelPoly']
    mmodels = {}
    for feature, fileloc in mmodelfiles.items():
        mmodels[feature] = {'model':pickle.load(open(fileloc, 'rb')),'poly':mmodelpoly[feature]}
    return mmodels


def getApp(summary_file):
    '''load in the application summary'''
    return App(summary_file)


def getAppSys(sys_file, name):
    return AppSysProfile(pd.read_csv(sys_file), name)


def getObservation(obs_file, name):
    '''load in the observation files'''
    return SlowDownProfile(pd.read_csv(obs_file), name)


def genParser():
    parser = optparse.OptionParser()
    # for slow-down training
    parser.add_option('--macsum', dest="machine_summary")
    parser.add_option('--appsum', dest="app_summary")
    parser.add_option('--appsys', dest="app_sys")
    parser.add_option('--obs', dest="observation")
    return parser


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
