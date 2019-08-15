''' Test driver to run multiple apps together '''
import os, sys, imp, subprocess, json
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor as Pool

RAPIDS_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/'
RAPIDS_SOURCE_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/'
RAPIDS_M_DIR = '/home/liuliu/Research/rapid_m_backend_server/'
sys.path.append(os.path.dirname(RAPIDS_DIR))
sys.path.append(os.path.dirname(RAPIDS_M_DIR))
sys.path.append(os.path.dirname(RAPIDS_SOURCE_DIR))

import Rapids.Classes
import Rapids.App.AppSummary
from BucketSelector import bucketSelect
from Rapid_M_Thread import Rapid_M_Thread, rapid_callback, rapid_worker
from TestDriver_Helper import *
import warnings
from sklearn.exceptions import DataConversionWarning

# ignore the TF debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# ignore the data conversion
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']
#apps = ['nn', 'swaptions']
APP_MET_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/appExt/'
TEST_APP_FILE = '/home/liuliu/Research/rapid_m_backend_server/TestScript/test_app_file.txt'

app_info = {}

STRAWMANS = ['P_M']


# preparation
def genInfo():
    for app in apps:
        # init all appmethods
        file_loc = APP_MET_PREFIX + app + "Met.py"
        module = imp.load_source("", file_loc)
        appMethod = module.appMethods("", app)
        # prepare run_dir for all apps
        run_dir = os.getcwd() + "/run/" + app
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        app_info[app] = {'met': appMethod, 'dir': run_dir}


# gen apps' gt
def genGT():
    processes = []
    for app, info in app_info.items():
        if app in ["svm", "nn", "facedetect"]:
            continue
        cmd = info['met'].getCommand(qosRun=False)
        working_dir = info['dir']
        # change working dir
        os.chdir(working_dir)
        # run all command and get the output
        p = subprocess.Popen(" ".join(cmd), shell=True)
        processes.append(p)
    exit_codes = [p.wait() for p in processes]


# generate the app running instances
def genRunComb():
    num_of_app = len(apps)
    combs = {}
    total = 0
    for i in range(2, num_of_app + 1):
        combs[i] = list(combinations(apps, i))
        total += len(combs[i])
    print('total num:', total)
    return combs


def run_a_comb(apps, mode):
    '''comb is a list of app names'''
    progress_map = {}
    thread_list = []
    selections = bucketSelect(TEST_APP_FILE, SELECTOR=mode)
    slowdowns = selections[2]
    configs = selections[1]
    expect_finish_times = selections[3]
    for app in apps:
        progress_map[app] = -1
        appMethod = app_info[app]['met']
        run_dir = app_info[app]['dir']
        app_cmd = appMethod.getCommandWithConfig(configs[app], qosRun=False)
        app_thread = Rapid_M_Thread(name=app + "_thread",
                                    target=rapid_worker,
                                    dir=run_dir,
                                    cmd=app_cmd,
                                    app_time=progress_map,
                                    app=app,
                                    callback=rapid_callback,
                                    callback_args=(app, app_cmd, progress_map))
        thread_list.append(app_thread)
    # kick off all threads
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    # assemble a slowdown entry for slow-down validation
    sd_entry = []
    for app, slowdown in slowdowns.items():
        ind_time = float(expect_finish_times[app]) / 1000.0 * float(
            app_info[app]['met'].training_units)
        sd_entry.append({
            'num': len(apps),
            'app': app,
            'config': configs[app],
            'exec': progress_map[app],
            'slowdown_p': slowdowns[app],
            'ind_exec': ind_time,
            'slowdown_gt': float(progress_map[app]) / ind_time
        })
    return progress_map, sd_entry, expect_finish_times


def run(combs):
    for mode in STRAWMANS:
        qos = {}
        data = {}
        slowdowns = []
        budgets = genBudgets(app_info)
        for num_of_app, comb in combs.items():
            per_data = {}
            counter = 0
            for apps in comb:
                update_app_file(apps)
                progress_map, sd_entry, expect_exec = run_a_comb(apps, mode)
                slowdowns = slowdowns + sd_entry
                clean_up(per_data, app_info, progress_map)
            data[num_of_app] = per_data

        summarize_data(data, app_info, budgets, slowdowns)
        os.chdir('/home/liuliu/Research/rapid_m_backend_server/TestScript')
        os.rename('mv.json', 'mv_' + mode + ".json")
        os.rename('exceed.json', 'exceed_' + mode + ".json")
        os.rename('slowdown_validator.csv',
                  'slowdown_validator_' + mode + ".csv")


def update_app_file(apps):
    active_apps = {}
    with open(TEST_APP_FILE, 'r') as file:
        active_apps = json.load(file)
        applications = active_apps['applications']
        for app in applications:
            app_name = app['id']
            if app_name not in apps:
                app['status'] = 0
            else:
                app['status'] = 1
                app['budget'] = app_info[app_name]['met'].max_cost
    with open(TEST_APP_FILE, 'w') as file:
        json.dump(active_apps, file, indent=2)


genInfo()
combs = genRunComb()
run(combs)
