''' helper function for cleaning up '''
import os, json, csv

SD_FILE_COLUMNS = [
    'num', 'app', 'config', 'exec', 'ind_exec', 'slowdown_p', 'slowdown_gt',
    'success'
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
    scaled_mv =  min(1.0,max(0.0, (mv - min_mv) / (max_mv - min_mv)
                          ))
    return scaled_mv

def genBudgets(app_info, scale=1.0):
    result = {}
    for app_name, info in app_info.items():
        budget = info['met'].training_units * scale * info[
            'met'].max_cost / 1000.0
        result[app_name] = budget
    return result


def clean_up(data, app_info, exec_time, sd_entry, budgets):
    qos_s = {}
    for app_name, real_exec in exec_time.items():
        # check if succeed
        success = list(filter(lambda x: x['app'] == app_name,
                              sd_entry))[0]['success']
        if success and budgets[app_name] * 1.1 > real_exec:
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
                qos = app_method.getQoS()
            elif app_name in ['facedetect', 'ferret']:
                qos = app_method.getQoS()
                qos = qos[2]
            # scale the qos
            qos = scale_mv(app_name, qos)
        else:
            qos = 0.0

        # write them to data
        if app_name not in data.keys():
            data[app_name] = {'qos': [], 'time': []}
        data[app_name]['qos'].append(qos)
        data[app_name]['time'].append({
            'success': success,
            'real_time': exec_time[app_name]
        })
    return


def summarize_data(data, app_info, budget, slowdowns):
    exceeds = {}
    mvs = {}
    miss_preds = {}
    for num_of_app, per_data in data.items():
        exceeds[num_of_app] = {}
        mvs[num_of_app] = {}
        miss_preds[num_of_app] = {}
        for app_name, d in per_data.items():
            app_method = app_info[app_name]['met']
            max_budget = budget[app_name]
            times = d['time']
            qoss = d['qos']
            # check exceeds
            exc = list(
                map(
                    lambda x: 1 if x['success'] and x['real_time'] > (
                        1.1 * max_budget) else 0, times))
            miss_pred = list(
                map(
                    lambda x: 1 if ((not x['success']) and x['real_time'] <
                                    max_budget) else 0, times))
            exceeds[num_of_app][app_name] = float(sum(exc)) / float(len(exc))
            miss_preds[num_of_app][app_name] = float(sum(miss_pred)) / float(
                len(miss_pred))
            # check mvs
            mvs[num_of_app][app_name] = sum(qoss) / float(len(qoss))
    # write it to file
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
    # write slowdowns
    with open('./slowdown_validator.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=SD_FILE_COLUMNS)
        writer.writeheader()
        for data in slowdowns:
            writer.writerow(data)