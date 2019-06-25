# validate the accuracy of applying M+P models
import optparse
import sys

from Classes.App import *
from Classes.AppSysProfile import *
from Classes.MModel import *
from BucketSelector import *

MACHINE_FILE = '/home/liuliu/Research/rapid_m_backend_server/examples/example_machine_empty.json'


def main(argv):
    parser = genParser()
    options, args = parser.parse_args()
    if options.mode == 'SLOWDOWN':
        #TODO: use average sys in bucket
        app_summary = getApp(options.app_summary)
        app_name = app_summary.name
        m_model = getMModel(options.machine_summary)
        observation = getObservation(options.observation, app_name)
        appsys = getAppSys(options.app_sys, app_name)
        validate_per_app(observation, app_summary, appsys, m_model)
    if options.mode == 'SELECTION':
        if options.active_apps == '':
            print("no input active apps file")
            exit(1)
        with open(options.active_apps, 'r') as file:
            active_apps = json.load(file)
            apps = getActiveApps(active_apps)
            # init all models
            p_models = loadAppModels(apps)
            # get M-Model
            m_model = MModel(MACHINE_FILE)
            # get the buckets
            buckets = genBuckets(apps, p_models)
            # get the highest and lowest budget
            range = getBudgetRange(apps)
            # split the budets
            budgets = splitBudget(range)
            validate_selection(apps, budgets, buckets, m_model, p_models)


def validate_selection(apps, budgets, buckets, m_model, p_models):
    # TODO: check the env inputted to the P-Model
    scenarios = pd.read_csv('./testData/mmodelfile_w_info.csv')
    load1_ids = [x for x in scenarios.columns if x[-2:] == '-1']
    load2_ids = [x for x in scenarios.columns if x[-2:] == '-2']
    loadC_ids = [x for x in scenarios.columns if x[-2:] == '-C']
    # iterate through all scenarios
    features = m_model.features
    for index, row in scenarios.iterrows():
        target_app = row['load1'].split(':')[0]
        target_config = row['load1'].split(':')[1]
        load1 = row[load1_ids].values.tolist()
        load2 = row[load2_ids].values.tolist()
        loadC = row[loadC_ids].values.tolist()
        # get the belonging bucket
        target_bucket = list(
            filter(lambda x: target_config in x.configurations,
                   buckets[target_app]))[0]
        p_model = target_bucket.p_model
        # get the P-ONLY selection
        slowdown_p = p_model.predict(env_to_frame(loadC, features))
        # get the M-P selection
        pred_env = m_model.predict(load1, load2)
        slowdown_mp = p_model.predict(pred_env)
        print(pred_env,loadC,slowdown_p,slowdown_mp)
        exit(1)


def splitBudget(b_range):
    budgets = []
    for i in range(1, 11):
        budgets.append({app: getBudget(x, i) for app, x in b_range.items()})
    return budgets


def getBudget(range, i):
    return (range[1] - range[0]) / 10.0 * i


def getBudgetRange(apps):
    range = {}
    for app in apps:
        dir = app['dir']
        cost_file = open(dir + '/cost.csv', 'r')
        highest = 0.0
        lowest = 99999999
        for line in cost_file:
            cost = float(line.split(' ')[-1])
            highest = max(highest, cost)
            lowest = min(lowest, cost)
        range[app['app'].name] = [lowest, highest]
    return range


def validate_per_app(observation, app_summary, appsys, m_model):
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
    parser.add_option('-m', dest="mode", default='SLOWDOWN')
    parser.add_option('--apps', dest="active_apps")
    return parser


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
