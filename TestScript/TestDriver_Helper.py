''' helper function for cleaning up '''
import os, json, csv

SD_FILE_COLUMNS = ['app','config','exec','slowdown']

def genBudgets(app_info):
    result = {}
    for app_name, info in app_info.items():
        budget = info['met'].training_units * info['met'].max_cost / 1000.0
        result[app_name] = budget
    return result

def clean_up(data, app_info, exec_time):
    qos_s = {}
    for app_name, info in exec_time.items():
        app_dir = app_info[app_name]['dir']
        app_method = app_info[app_name]['met']
        # calculate qos
            # prepare the gt data path
        os.chdir(app_dir)
        if app_name in ['swaptions', 'ferret','bodytrack']:
            app_method.gt_path ='./groundtruth_small.txt'
        elif app_name == 'facedetect':
            app_method.gt_path = app_method.train_grond_truth_path
            # get the qos
        if app_name in ['bodytrack','svm','nn','swaptions']:
            qos = app_method.getQoS()
        elif app_name in ['facedetect','ferret']:
            qos = app_method.getQoS()
            qos = qos[2]
        # write them to data
        if app_name not in data.keys():
            data[app_name]={'qos':[],'time':[]}
        data[app_name]['qos'].append(qos)
        data[app_name]['time'].append(exec_time[app_name])
    return

def summarize_data(data, app_info, budget, slowdowns):
    exceeds = {}
    mvs = {}
    for num_of_app, per_data in data.items():
        exceeds[num_of_app] = {}
        mvs[num_of_app] = {}
        for app_name, d in per_data.items():
            app_method = app_info[app_name]['met']
            max_budget = budget[app_name]
            times = d['time']
            qoss = d['qos']
            # check exceeds
            exc = list(map(lambda x: 1 if x>max_budget else 0, times))
            exceeds[num_of_app][app_name] = float(sum(exc)) / float(len(exc))
            # check mvs
            mvs[num_of_app][app_name] = sum(qoss)/ float(len(qoss))
    # write it to file
    os.chdir('/home/liuliu/Research/rapid_m_backend_server/TestScript/')
    file = open('./exceed.json','w')
    json.dump(exceeds, file, indent=2)
    file.close()
    file = open('./mv.json','w')
    json.dump(mvs, file, indent=2)
    file.close()
    # write slowdowns
    with open('./slowdown_validator.csv','w') as file:
        writer = csv.DictWriter(file, fieldnames = SD_FILE_COLUMNS)
        writer.writeheader()
        for data in slowdowns:
            writer.writerow(data)
