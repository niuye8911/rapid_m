# validate the accuracy of applying M+P models
import optparse
import sys

from Classes.App import *
from Classes.AppSysProfile import *
from Classes.MModel import *

#TODO: use average sys in bucket

def main(argv):
    parser = genParser()
    options, args = parser.parse_args()
    app_summary = getApp(options.app_summary)
    app_name = app_summary.name
    m_model = getMModel(options.machine_summary)
    observation = getObservation(options.observation, app_name)
    appsys = getAppSys(options.app_sys, app_name)
    validate(observation, app_summary, appsys, m_model)


def validate(observation, app_summary, appsys, m_model):
    '''validate the M+P model
    observation: the observated slowdown given an env
    app_summary: containing all the P_model for a specific bucket
    m_model: the machine model
    '''
    dataFrame = observation
    configs = list(dataFrame['Configuration'])
    y_gt = np.array(dataFrame['SLOWDOWN'].values.tolist())
    # generate the predicted slow-down
    y_pred = np.array(genPred(observation, app_summary, appsys, m_model))
    # validate the diff
    mse = np.sqrt(metrics.mean_squared_error(y_gt, y_pred))
    mae = metrics.mean_absolute_error(y_gt, y_pred)
    r2 = r2_score(y_gt, y_pred)
    # relative error
    diff = abs(y_gt - y_pred) / y_gt
    diff = sum(diff) / len(diff)
    print(mae)


def genPred(observation, app_summary, appsys, m_model):
    ''' generate the prediction of slow-down'''
    dataFrame = observation
    y_pred = []
    features = m_model.features
    debug_file = open('./tmp.csv', 'w')
    debug_file.write(','.join(features) + '\n')
    for index, row in dataFrame.iterrows():
        # take the config
        config = row['Configuration']
        # take the pmodel
        p_model, p_features, isPoly = getPModel(config, app_summary)
        # take the added env
        added_env = row
        # take the app's footprint
        app_env = appsys.getSysByConfig(config)
        # predict the combined env
        env1 = formatEnv(added_env, features)
        env2 = app_env[features].values[0]
        # print(len(env1),env1,len(env2),env2)
        pred_env = m_model.predict(env1, env2)
        writeEnvsToDebug(debug_file, env1, env2, pred_env.values[0])

        # filter out the unwanted env
        # predict the slowdown
        # prepare data for the P model
        data_x = pred_env[p_features]
        if isPoly:
            data_x = PolynomialFeatures(degree=2).fit_transform(data_x)
        pred_slowdown = p_model.predict(data_x)
        y_pred.append(pred_slowdown[0])
        debug_file.write(str(row['SLOWDOWN']) + ',')
        debug_file.write(str(pred_slowdown) + '\n\n')
    debug_file.close()
    return y_pred


def formatEnv(env, features):
    result = []
    for feature in features:
        if feature == 'MEM':
            result.append(env['READ'] + env['WRITE'])
        elif feature == 'INST':
            inst = env['ACYC'] / env['INST']
            result.append(inst)
        elif feature == 'INSTnom%' or feature == 'PhysIPC%':
            result.append(env[feature] / 100.0)
        else:
            result.append(env[feature])
    return list(map(lambda x: float(x), result))


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


def writeEnvsToDebug(debug_file, env1, env2, env3):
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
        print("WARNING: no p model found for config:" + config)
        exit(0)
    return pickle.load(open(p_model_file, 'rb')), features, poly


def getMModel(summary_file):
    machine = MModel(summary_file)
    return machine


def getApp(summary_file):
    '''load in the application summary'''
    return App(summary_file)


def getAppSys(sys_file, name):
    return AppSysProfile(pd.read_csv(sys_file), name)


def getObservation(obs_file, name):
    '''load in the observation files'''
    return pd.read_csv(obs_file)


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
