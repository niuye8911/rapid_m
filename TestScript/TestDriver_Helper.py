''' helper function for cleaning up '''
import os, json, csv
import pandas as pd
from Utility import updateAppMinMax

SD_FILE_COLUMNS = [
    'num', 'app', 'budget', 'ind_exec', 'exec', 'config', 'slowdown_p',
    'slowdown_gt', 'success_p', 'success_gt', 'raw_qos', 'qos'
]

MINMAX_FILE = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/outputs/app_min_max.json'


def readMinMaxMV():
    minmax = {}
    with open(MINMAX_FILE) as minmax_mv_file:
        data = json.load(minmax_mv_file)
        for app, values in data.items():
            minmax[app] = {'min': values[0], 'max': values[1]}
    return minmax


minmax = readMinMaxMV()


def scale_mv(app, mv):
    global minmax
    min_mv = minmax[app]['min']
    max_mv = minmax[app]['max']
    scaled_mv = min(1.0, max(0.0, (mv - min_mv) / (max_mv - min_mv)))
    return scaled_mv


def scale_mv_by_appmet(app, appMet, mv):
    min_mv = appMet.min_mv
    max_mv = appMet.max_mv
    scaled_mv = min(1.0, max(0.0, (mv - min_mv) / (max_mv - min_mv)))
    return scaled_mv


def genBudgets(app_info, scale=1.0):
    result = {}
    for app_name, info in app_info.items():
        updateAppMinMax(app_name, info['met'])
        budget = info['met'].training_units * scale * info[
            'met'].max_cost / 1000.0
        result[app_name] = budget
    return result


def getMV(app_name, app_info):
    app_dir = app_info[app_name]['dir']
    app_method = app_info[app_name]['met']
    # calculate qos
    # prepare the gt data path
    os.chdir(app_dir)
    if app_name in ['swaptions', 'ferret', 'bodytrack']:
        app_method.gt_path = './groundtruth_small.txt'
    elif app_name == 'facedetect':
        app_method.gt_path = app_method.train_grond_truth_path
        # get the qos
    if app_name in ['bodytrack', 'svm', 'nn', 'swaptions']:
        raw_qos = app_method.getQoS()
    elif app_name in ['facedetect', 'ferret']:
        raw_qos = app_method.getQoS()
        raw_qos = raw_qos[2]
    # scale the qos
    qos = scale_mv_by_appmet(app_name, app_info[app_name]['met'], raw_qos)
    return raw_qos, qos


def clean_up(data, sd_entries):
    qos_s = {}
    for sd_entry in sd_entries:
        # check if succeed
        app_name = sd_entry['app']
        success = sd_entry['success']
        qos = sd_entry['qos']
        raw_qos = sd_entry['raw_qos']
        exec_time = sd_entry['exec']
        # write them to data
        if app_name not in data.keys():
            data[app_name] = {'qos': [], 'time': [], 'raw_qos': []}
        data[app_name]['qos'].append(qos)
        data[app_name]['time'].append({
            'success': success,
            'real_time': exec_time
        })
        data[app_name]['raw_qos'].append(raw_qos)
    return


def writeSlowDown(slowdowns):
    os.chdir('/home/liuliu/Research/rapid_m_backend_server/TestScript/')
    # write slowdowns
    file_name = './slowdown_validator.csv'
    with open(file_name, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=SD_FILE_COLUMNS)
        writer.writeheader()
        for data in slowdowns:
            writer.writerow(data)
    return file_name


def summarize_data(slowdown_file):
    df = pd.read_csv(slowdown_file)
    exceeds = {}
    mvs = {}
    miss_preds = {}
    num_of_apps = df.num.unique()
    apps = df.app.unique()
    for num_of_app in num_of_apps:
        # get the sub_df with num == num_of_app
        sub_df = df.loc[df['num'] == num_of_app]
        exceeds[str(num_of_app)] = {}
        mvs[str(num_of_app)] = {}
        miss_preds[str(num_of_app)] = {}
        for app_name in apps:
            # calculate the exceeding rate
            app_df = sub_df.loc[sub_df['app'] == app_name]
            max_budget = app_df['budget'].tolist()[0]
            exceed = app_df.apply(lambda row: 1 if row['success_p'] and
                                  not row['success_gt'] else 0,
                                  axis=1).tolist()
            miss_pred = app_df.apply(lambda row: 1 if not row['success_p'] and
                                     row['success_gt'] else 0,
                                     axis=1).tolist()
            mv = app_df.apply(lambda row: float(row['qos']) if row['success_p']
                              and row['success_gt'] else 0.0,
                              axis=1).tolist()
            exceeds[str(num_of_app)][app_name] = float(sum(exceed)) / float(
                len(exceed))
            miss_preds[str(num_of_app)][app_name] = float(
                sum(miss_pred)) / float(len(miss_pred))
            # check mvs
            mvs[str(num_of_app)][app_name] = float(sum(mv)) / float(len(mv))
    os.chdir('/home/liuliu/Research/rapid_m_backend_server/TestScript/')
    file = open('./exceed.json', 'w')
    json.dump(exceeds, file, indent=2)
    file.close()
    file = open('./miss_pred.json', 'w')
    json.dump(miss_preds, file, indent=2)
    file.close()
    file = open('./mv.json', 'w')
    json.dump(mvs, file, indent=2)
    file.close()
