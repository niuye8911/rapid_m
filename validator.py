# validate the accuracy of applying M+P models
import optparse
import sys

from Classes.App import *
from Classes.AppSysProfile import *
from Classes.MModel import *
from BucketSelector import *

MACHINE_FILE = '/home/liuliu/Research/rapid_m_backend_server/examples/example_machine_empty.json'

TOLERANCE = 0.05


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
    result = {}
    # iterate through apps
    for app in apps:
        target_app = app['app'].name
        # select the frame
        app_df = scenarios.loc[scenarios['load1'].str.contains(target_app)]
        if target_app not in result:
            result[target_app] = []
        p_correct = [0] * 10
        pm_correct = [0] * 10
        p_mv_dist = [0.0] * 10
        pm_mv_dist = [0.0] * 10
        # get the total number of data for this app
        total = app_df.shape[0]
        # iterate through budget
        id = 0
        for budget in budgets:
            app_budget = budget[target_app]
            # get df and p model by buckets
            bucket_dfs = {}
            for bucket in buckets[target_app]:
                env_df = app_df.loc[app_df['load1'].apply(lambda x: x.split(
                    ":")[1] in bucket.configurations)]
                bucket_dfs[bucket.b_name] = {
                    'env': env_df,
                    'p': bucket.p_model
                }
                # get the corresponding env
                env_p = formatEnv_df(env_df[loadC_ids], m_model.features, '-C')
                env_pm = m_model.predict_batch(env_df[load1_ids],
                                               env_df[load2_ids])
                # get the P-ONLY slow-down
                slowdown_pm = bucket.p_model.predict(env_pm)
                # get the P+M slow-down
                slowdown_p = bucket.p_model.predict(env_p)
                # get the REAL slowdown
                slowdown_gt = env_df['slowdown'].tolist()
                # get the optimal solution
                p_sel = list(
                    map(lambda x: bucket.getOptimal(app_budget, x),
                        slowdown_p))
                pm_sel = list(
                    map(lambda x: bucket.getOptimal(app_budget, x),
                        slowdown_pm))
                gt_sel = list(
                    map(lambda x: bucket.getOptimal(app_budget, x),
                        slowdown_p))
                gt_range_sel = list(
                    map(lambda x: bucket.getOptimal(app_budget, x, TOLERANCE),
                        slowdown_gt))
                # calculate the hit rate on selection range
                hr_p = [
                    x[0][0] in gt_range_sel[i][0] for i, x in enumerate(p_sel)
                ]
                hr_pm = [
                    x[0][0] in gt_range_sel[i][0] for i, x in enumerate(pm_sel)
                ]
                # calculate the MV loss
                mv_loss_p = [
                    abs(x[1][0] - gt_sel[i][1][0]) / gt_sel[i][1][0]
                    for i, x in enumerate(p_sel)
                ]
                mv_loss_pm = [
                    abs(x[1][0] - gt_sel[i][1][0]) / gt_sel[i][1][0]
                    for i, x in enumerate(pm_sel)
                ]
                # record the hit rate and the mv loss
                p_correct[id] += sum(hr_p)
                pm_correct[id] += sum(hr_pm)
                p_mv_dist[id] += sum(mv_loss_p)
                pm_mv_dist[id] += sum(mv_loss_pm)
            # summarize the bucket in this budget
            p_correct[id] = float(p_correct[id]) / float(total)
            pm_correct[id] = float(pm_correct[id]) / float(total)
            p_mv_dist[id] = float(p_mv_dist[id]) / float(total)
            pm_mv_dist[id] = float(pm_mv_dist[id]) / float(total)
            # go to the next budget id
            id += 1
        result[target_app] = {
            'P': p_correct,
            'PM': pm_correct,
            'P_MV': p_mv_dist,
            'PM_MV': pm_mv_dist
        }
        RAPID_info('Valid Selector', "finished:" + target_app)
        printSelection(result)


def printSelection(selection):
    output = open('./selection.csv', 'w')
    # write the header
    header = ['app'] + list(map(lambda x: str(x) + '%', range(10, 100, 10)))
    output.write(','.join(header) + '\n')
    # write app by app
    p = []
    pm = []
    p_mv = []
    pm_mv = []
    for app, results in selection.items():
        p_line = [app] + list(map(lambda x: str(x), results['P']))
        pm_line = [app] + list(map(lambda x: str(x), results['PM']))
        p_mv_line = [app] + list(map(lambda x: str(x), results['P_MV']))
        pm_mv_line = [app] + list(map(lambda x: str(x), results['PM_MV']))
        p.append(p_line)
        pm.append(pm_line)
        p_mv.append(p_mv_line)
        pm_mv.append(pm_mv_line)
    # write out
    for line in p:
        output.write(','.join(line) + '\n')
    output.write('\nP_M\n')
    for line in pm:
        output.write(','.join(line) + '\n')
    output.write('\nP_mv\n')
    for line in p_mv:
        output.write(','.join(line) + '\n')
    output.write('\nPM_mv\n')
    for line in pm_mv:
        output.write(','.join(line) + '\n')
    output.close()


def splitBudget(b_range):
    budgets = []
    for i in range(1, 11):
        budgets.append({app: getBudget(x, i) for app, x in b_range.items()})
    return budgets


def getBudget(range, i):
    return range[0] + (range[1] - range[0]) / 10.0 * i


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
