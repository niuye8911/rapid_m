"""
 This is the optimizer to select the correct bucket for each active apps
  Author:
    Liu Liu
    04/2019
"""

import functools
import itertools
import json

import pandas as pd

from Rapid_M_Classes.App import App
from Rapid_M_Classes.Bucket import Bucket
from Rapid_M_Classes.MModel import MModel
from Rapid_M_Classes.PModel import PModel
from Utility import *
from DataUtil import *

MACHINE_FILE = '/home/liuliu/Research/rapid_m_backend_server/examples/example_machine_empty.json'
DELIMITER = ","  # bucket comb delimiter


def bucketSelect(active_apps_file, SELECTOR="P_M", env=[]):
    with open(active_apps_file, 'r') as file:
        active_apps = json.load(file)
        # get all the apps
        apps = getActiveApps(active_apps)
        if len(apps) == 0:
            # no active apps
            return None, None
        if SELECTOR == 'INDIVIDUAL':
            return indiSelect(apps)
        if SELECTOR == "P_M" or SELECTOR == "P_M_RUSH":
            return pmSelect(apps)
        if SELECTOR == "N":
            return nSelect(apps)
        if SELECTOR == "P":
            return pSelect(apps, env)


def nSelect(apps):
    num_of_apps = len(apps)
    return indiSelect(apps, float(num_of_apps))


def indiSelect(apps, f_slowdown=1.0):
    p_models = loadAppModels(apps)
    # convert apps to buckets
    buckets = genBuckets(apps, p_models)
    bucket_selection = []
    configs = {}
    slowdowns = {}
    expectation = {}
    successes = {}
    for app in apps:
        bucket, config, success, slow, expected = single_app_select(
            app['app'], app['budget'], buckets, fixed_slowdown=f_slowdown)
        bucket_selection.append(bucket)
        configs[app['app'].name] = config[app['app'].name]
        slowdowns[app['app'].name] = f_slowdown
        successes[app['app'].name] = success[app['app'].name]
        expectation[app['app'].name] = expected[app['app'].name]
    return [
        ','.join(bucket_selection), configs, successes, slowdowns, expectation
    ], buckets


def pmSelect(apps):
    # get the M-Model
    m_model = MModel(MACHINE_FILE)
    features = m_model.features
    RAPID_info('M-Model Loader: ', m_model.TRAINED)
    # get all the P_models {app_name: {bucket_name: model }}
    p_models = loadAppModels(apps)
    # convert apps to buckets
    buckets = genBuckets(apps, p_models)
    if len(apps) == 1:
        selections = getSelection_batch(None, apps, buckets, single_app=True)
        return selections, buckets
    # get all combinations of buckets
    bucket_combs = getBucketCombs(buckets)
    # predict the overall envs for each comb
    combined_envs = getEnvs_batch(bucket_combs, mmodel=m_model)
    # predict the per-app slow-down
    slowdowns = getSlowdowns_batch(combined_envs, p_models)
    # get the bucket selection based on slow-down
    selections = getSelection_batch(slowdowns, apps, buckets)
    return selections, buckets


def pSelect(apps, measured_env):
    # get all the P_models {app_name: {bucket_name: model }}
    p_models = loadAppModels(apps)
    # convert apps to buckets
    buckets = genBuckets(apps, p_models)
    # get all combinations of buckets
    bucket_combs = getBucketCombs(buckets)
    # formulate the measured combined envs
    combined_envs = getEnvs_batch(bucket_combs, P_ONLY=True, env=measured_env)
    # predict the per-app slow-down
    slowdowns = getSlowdowns_batch(combined_envs, p_models)
    return getSelection_batch(slowdowns, apps, buckets), buckets


def _get_expected(config_dicts, buckets, comb_name):
    expected = {}
    bucket_names = comb_name.split(',')
    for app_name, config in config_dicts.items():
        bucket_name = list(filter(lambda x: app_name in x, bucket_names))[0]
        bucket = list(
            filter(lambda x: x.b_name == bucket_name, buckets[app_name]))[0]
        expected[app_name] = bucket.profile[config]['cost']
    return expected


def getSelection_batch(slowdowns, apps, buckets, single_app=False):
    ''' get the final selection of configs and buckets
    @param slowdowns: a slowdown df with column: [comb_name, app_1, ... app_n]
    '''
    if single_app:
        # only 1 app active
        return single_app_select(apps[0]['app'], apps[0]['budget'], buckets)
    results = slowdowns.apply(lambda x: mv_per_row(x, apps, buckets), axis=1)
    mv_col = results.apply(lambda x: x['mv'])
    configs_col = results.apply(lambda x: x['configs'])
    success_col = results.apply(lambda x: x['success'])
    all_success_col = results.apply(lambda x: x['all_success'])
    run_success_col = results.apply(lambda x: x['run_success'])
    slowdowns['success'] = success_col
    slowdowns['mv'] = mv_col
    slowdowns['configs'] = configs_col
    slowdowns['all_success'] = all_success_col
    slowdowns['run_success'] = run_success_col
    # first try to get a combination that all apps successes
    all_success_rows = slowdowns.loc[slowdowns['all_success'] == True]
    if not all_success_rows.empty:
        max_row = all_success_rows.loc[all_success_rows['mv'].idxmax()]
    else:
        # check if there's any combination that maintains the running apps
        run_success_rows = slowdowns.loc[slowdowns['run_success'] == True]
        if not run_success_rows.empty:
            max_row = run_success_rows.loc[run_success_rows['mv'].idxmax()]
        else:
            # sacrifice some apps
            max_row = slowdowns.loc[slowdowns['mv'].idxmax()]
    # the result
    comb_name = max_row['comb_name']
    config_dict = max_row['configs']
    success_dict = max_row['success']
    slowdown_t = slowdown_table(max_row, apps)
    expected_exec = _get_expected(config_dict, buckets, comb_name)
    return [comb_name, config_dict, success_dict, slowdown_t, expected_exec]


def single_app_select(app, budget, buckets, fixed_slowdown=1.0):
    #only 1 app active
    configs = {}
    app_name = app.name
    max_mv = -10
    bucket_selection = ''
    found = False
    failed_selection = ''
    failed_cost = 9999
    for bucket in buckets[app_name]:
        bucket_name = bucket.b_name
        slow_down = fixed_slowdown
        budget = float(budget)
        config, mv, SUCCESS = bucket.getOptimal(budget, slow_down)
        if SUCCESS:
            found = True
            if mv[0] > max_mv:
                configs[app_name] = config[0]
                max_mv = mv[0]
                bucket_selection = bucket.b_name
        # if not succeed, no result
        else:
            if not found:
                # find the minimum cost config
                if bucket.profile[config[0]]['cost'] < failed_cost:
                    failed_selection = config[0]
                    failed_cost = bucket.profile[config[0]]['cost']
                    bucket_selection = bucket.b_name
                    configs[app_name] = config[0]
    expected_exec = _get_expected(configs, buckets, bucket_selection)
    if found:
        return bucket_selection, configs, {
            app_name: True
        }, {
            app_name: fixed_slowdown
        }, expected_exec
    else:
        return bucket_selection, {
            app_name: failed_selection
        }, {
            app_name: False
        }, {
            app_name: fixed_slowdown
        }, expected_exec


def slowdown_table(row, apps):
    slowdown = {}
    for app in apps:
        app_name = app['app'].name
        slowdown[app_name] = row[app_name]
    return slowdown


def mv_per_row(row, apps, buckets):
    bucket_list = row['comb_name'].split(',')
    total_mv = 0.0
    configs = {}
    successes = {}
    statuss = {}
    for app in apps:
        successes[app['app'].name] = False
        statuss[app['app'].name] = app['status']
    for bucket_name in bucket_list:
        app_name = bucket_name[:-1]  #!!!there should not be > 10 buckets
        bucket = list(
            filter(lambda x: x.b_name == bucket_name, buckets[app_name]))[0]
        slow_down = 1.2 if float(row[app_name]) < 1.0 else float(row[app_name])
        slow_down = min(slow_down,3.0)
        budget = list(filter(lambda x: x['app'].name == app_name,
                             apps))[0]['budget']
        status = list(filter(lambda x: x['app'].name == app_name,
                             apps))[0]['status']
        config, mv, SUCCESS = bucket.getOptimal(float(budget),
                                                float(slow_down))
        configs[app_name] = config[0]
        successes[app_name] = SUCCESS
        total_mv = (total_mv + mv[0]) if SUCCESS else total_mv
    all_success = True
    running_apps_all_success = True
    for app, suc in successes.items():
        all_success = all_success and suc
        if statuss[app] == 2:  # running app
            running_apps_all_success = running_apps_all_success and suc
    return {
        'mv': total_mv,
        'configs': configs,
        'success': successes,
        'all_success': all_success,
        'run_success': running_apps_all_success
    }


def getSelection(slowdowns, apps, buckets):
    ''' get the final selection of configs and buckets '''
    selection = {}
    best_mv = 0.0
    for bucketlist, appslowdowns in slowdowns.items():
        total_mv = 0.0
        configs = {}
        bucket_list = bucketlist.split(',')
        app_names = list(appslowdowns.keys())
        # get overall mv per bucket list
        for app in app_names:
            bucket_name = list(filter(lambda x: app in x, bucket_list))[0]
            bucket = list(
                filter(lambda x: x.b_name == bucket_name, buckets[app]))[0]
            slow_down = appslowdowns[app]
            budget = list(filter(lambda x: x['app'].name == app,
                                 apps))[0]['budget']
            config, mv, SUCCESS = bucket.getOptimal(budget, slow_down)
            # TODO: what to use when SUCCESS is false
            configs[app] = config[0]
            total_mv += mv[0]
        if total_mv > best_mv:
            selection = configs.copy()
    PPRINT(selection)
    return selection


def getSlowdowns_batch(combined_envs, p_models):
    ''' return the slow-down for each app with slow-down
    @param combined_envs: a df with all comb_name and envs(processed)
    '''
    slowDownTable = {}
    env_ids = list(combined_envs.columns)
    env_ids.remove('comb_name')
    return_ids = ['comb_name']
    num = combined_envs.shape[0]
    # calculate the slow-down with all p_models
    for app in p_models.keys():
        return_ids.append(app)
        combined_envs[app] = [0.0] * num
        for bucket, p_model in p_models[app].items():
            # get the df containing the bucekt name
            sub_df = combined_envs.loc[combined_envs['comb_name'].str.contains(
                bucket)]
            sub_df_only_one = combined_envs.loc[combined_envs['comb_name'] ==
                                                bucket]
            # apply the p_model to get the slowdown
            slowdown = p_model.predict(sub_df[env_ids])
            # set the corresponding rows to slowdown
            combined_envs.loc[sub_df.index.values, [app]] = slowdown
            combined_envs.loc[sub_df_only_one.index.values, [app]] = 1.0
    return combined_envs


def getSlowdowns(combined_envs, p_models, features):
    ''' return the slow-down for each app with slow-down '''
    slowDownTable = {}
    for comb_name, env in combined_envs.items():
        buckets = comb_name.split(DELIMITER)
        comb_slowdown = {}
        for bucket in buckets:
            app_name = bucket[:-1]
            p_model = p_models[app_name][bucket]
            # tranform the vector to dataframe
            slowdown = p_model.predict(env_to_frame(env, features))
            comb_slowdown[app_name] = slowdown[0]
        slowDownTable[comb_name] = comb_slowdown
    return slowDownTable


def getEnvs_batch(bucket_combs, mmodel=None, P_ONLY=False, env=[]):
    result = {}
    if not P_ONLY:
        envs = []
        combs = {}
        for comb in bucket_combs:
            length = len(comb)
            if length not in combs:
                combs[length] = []
            combs[length].append(comb)
        for num, comb in combs.items():
            envs_of_num = mmodel.predict_seq(comb)
            envs.append(envs_of_num)
        overall_envs = pd.concat(envs, ignore_index=True)
        return overall_envs
    else:
        comb_names = list(
            map(lambda comb: (list(map(lambda x: x.b_name, comb))),
                bucket_combs))
        comb_name_strs = list(map(lambda x: ",".join(x), comb_names))
        # convert the env to df
        env_formatted = formatEnv_df(env, env.columns.tolist())
        # num of comb
        num_of_comb = len(comb_names)
        # generate the fake df
        measured_env = OrderedDict()
        for col in env_formatted.columns:
            if col == 'comb':
                continue
            measured_env[col] = list(env_formatted[col].values) * num_of_comb
        envs = pd.DataFrame(data=measured_env)
        envs['comb_name'] = comb_name_strs
    return envs


def getEnvs(bucket_combs, mmodel=None, P_ONLY=False, env=[]):
    ''' generate the combined env using M-Model '''
    result = {}
    for comb in bucket_combs:
        comb_name = DELIMITER.join((list(map(lambda x: x.b_name, comb))))
        if not P_ONLY:
            env_dicts = list(
                map(lambda x: formatEnv(x.rep_env, mmodel.features), comb))
            # cumulatively apply M-Model
            final_env = functools.reduce(lambda x, y: mReducer(x, y, mmodel),
                                         env_dicts)
        else:
            # P_ONLY
            final_env = env
        # print(final_env)
        result[comb_name] = final_env
    return result


def mReducer(env1, env2, mmodel):
    result = mmodel.predict(env1, env2)
    return list(result.values[0])


def getBucketCombs(buckets):
    bucket_lists = buckets.values()
    #combs = list(itertools.product(*bucket_lists))
    #return combs
    # need to redo this for selection
    bucket_lists_with_none = list(map(lambda x: x + [None], bucket_lists))
    combs = list(itertools.product(*bucket_lists_with_none))
    combs_with_none = [list(filter(lambda x: not x == None, y)) for y in combs]
    combs_with_none.remove([])
    return combs_with_none


def printBucketCombs(combs):
    names = list(map(lambda x: list(map(lambda y: y.b_name, x)), combs))
    print(names)


def genBuckets(apps, models):
    buckets = {}
    for app in apps:
        dir = app['dir']
        app = app['app']
        buckets[app.name] = []
        for bucket_name, info in app.cluster_info.items():
            pmodel = models[
                app.name][bucket_name] if models is not None else None
            bucket = Bucket(app.name, bucket_name, info['cluster'], pmodel,
                            dir + '/cost.csv', dir + '/mv.csv', info['env'])
            buckets[app.name].append(bucket)
        RAPID_info("Bucket Loader", "loaded Buckets for " + app.name)
    return buckets


def getActiveApps(active_apps):
    '''return a list of App instances from the profile'''
    apps = []
    applications = active_apps['applications']
    for app in applications:
        status = app['status']
        if status == 0 or status == 4:
            # inactive apps
            continue
        application = App(app['dir'] + '/' + 'profile' + '.json')
        apps.append({
            'app': application,
            'budget': app['budget'],
            'dir': app['dir'],
            'status': status
        })
    return apps


def loadAppModels(apps):
    models = {}
    for app in apps:
        app = app['app']
        models[app.name] = {}
        for bucket_name, info in app.model_params.items():
            models[app.name][bucket_name] = PModel(info, app.maxes)
        RAPID_info("Model Loader", "loaded P-Models for " + app.name)
    return models
