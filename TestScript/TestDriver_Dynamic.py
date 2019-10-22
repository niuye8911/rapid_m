''' Test driver to run multiple apps together and reconfig dynamically'''
import os, sys, imp, subprocess, json, time, requests
import pandas as pd
from itertools import combinations, permutations
from concurrent.futures import ThreadPoolExecutor as Pool

RAPIDS_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/'
RAPIDS_SOURCE_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/'
RAPIDS_M_DIR = '/home/liuliu/Research/rapid_m_backend_server/'
sys.path.append(os.path.dirname(RAPIDS_DIR))
sys.path.append(os.path.dirname(RAPIDS_M_DIR))
sys.path.append(os.path.dirname(RAPIDS_SOURCE_DIR))

import random
import Rapids.App.AppSummary
from Rapids.util import __get_minmax
from BucketSelector import bucketSelect
from Rapid_M_Thread import Rapid_M_Thread, rapid_callback, rapid_worker, rapid_dynamic_callback, rapid_dynamic_worker
from TestDriver_Helper import *
import warnings, time
from random import randint
from sklearn.exceptions import DataConversionWarning

MAX_WAIT_TIME = 10  # wait at most 10 second until the new app be inited
MISSION_TIME = 60 * 5  # 10 minutes run

# ignore the TF debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# ignore the data conversion
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']
APP_MET_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/appExt/'
APP_OUT_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/outptus/'

app_info = {}

metric_df = None

STRAWMANS = ['P_M']  # strawmans to use
#BUDGET_SCALE = [0.8, 1.0, 1.5]
BUDGET_SCALE = [1.0]


#preparation
def reset_server():
    # reset the rapid_m server
    URL = "http://algaesim.cs.rutgers.edu/rapid_server/reset.php"
    try:
        response = requests.get(URL, timeout=3)
        response.raise_for_status()  # Raise error in case of failure
    except requests.exceptions.HTTPError as httpErr:
        print("Http Error:", httpErr)
        return False
    except requests.exceptions.ConnectionError as connErr:
        print("Error Connecting:", connErr)
        return False
    except requests.exceptions.Timeout as timeOutErr:
        print("Timeout Error:", timeOutErr)
        return False
    except requests.exceptions.RequestException as reqErr:
        print("Something Else:", reqErr)
        return False
    print("Rapid_M Server Reset Done")
    return True


def genInfo():
    for app in apps:
        # appmethods
        file_loc = APP_MET_PREFIX + app + "Met.py"
        module = imp.load_source("", file_loc)
        appMethod = module.appMethods(app, app)
        # run_dir
        run_dir = os.getcwd() + "/run/" + app
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        app_info[app] = {'met': appMethod, 'dir': run_dir}


def updateAppMinMaxFake(appMethod, app):
    cost_file = APP_OUT_PREFIX + app + '/' + app + "-cost.fact"
    mv_file = APP_OUT_PREFIX + app + '/' + app + "-mv.fact"
    if os.path.exists(cost_file):
        min_cost, max_cost = __get_minmax(cost_file)
        appMethod.min_cost = min_cost
        appMethod.max_cost = max_cost
    if os.path.exists(mv_file):
        min_mv, max_mv = __get_minmax(mv_file)
        appMethod.min_mv = min_mv
        appMethod.max_mv = max_mv


# get a random app
def getNewApp(active_apps):
    active_app_list = []
    for k, v in active_apps.items():
        if v == True:
            active_app_list.append(k)
    remaining_apps = list(set(apps) - set(active_app_list))
    return random.choice(remaining_apps)


def getAppRunConfig(app):
    base_dir = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/outputs/'
    return base_dir + app + '/' + app + '_run.config'


def getEnvByComb(apps):
    # numerate different names of comb
    possible_names = list(permutations(apps))
    possible_names = list(map(lambda x: '+'.join(x), possible_names))
    row = metric_df.loc[metric_df['comb'].isin(possible_names)]
    return row


def writeMissionToFile(mission_log):
    with open('./dynamic_mission.log', 'w') as fout:
        json.dump(mission_log, fout, indent=2)


def run_a_mission(target_num_apps, budgets):
    '''run at most target_num_apps, each will be executed repeat_times '''
    progress_map = {}
    thread_list = []
    # first get all apps' command
    commands = {}
    active_apps = {}  # record how many apps are active
    for app in apps:
        # update the run_config
        run_config = getAppRunConfig(app)
        appmet = app_info[app]['met']
        updateAppMinMaxFake(appmet, app)
        appmet.setRunConfigFile(run_config)
        appmet.updateRunConfig(budgets[app], rapid_m=True, debug=True)
        commands[app] = app_info[app]['met'].getRapidsCommand()
        active_apps[app] = False
    # log the mission progress
    mission_log = []
    start_time = time.time()
    while (time.time() - start_time <= MISSION_TIME):
        num_of_active_apps = sum(active_apps.values())
        if num_of_active_apps < target_num_apps:
            print("current active apps:", num_of_active_apps)
            # some app has finished, get a new app
            new_app = getNewApp(active_apps)
            print("get a new app", new_app)
            app_cmd = commands[new_app]
            appmet = app_info[new_app]['met']
            run_dir = app_info[new_app]['dir']
            # wait for a random time
            time.sleep(randint(1, MAX_WAIT_TIME))
            new_app_start_time = time.time() - start_time
            log_entry = {
                'global_start': start_time,
                'start_time': new_app_start_time,
                'app': new_app,
                'budget': budgets[new_app]
            }
            active_apps[new_app] = True
            print("starting new app:xxxxxxxxxxx", start_time, new_app,
                  new_app_start_time)
            app_thread = Rapid_M_Thread(
                name=new_app + "_thread",
                target=rapid_dynamic_worker,
                dir=run_dir,
                cmd=app_cmd,
                app_time=progress_map,
                app=new_app,
                callback=rapid_dynamic_callback,
                callback_args=(new_app, appmet, run_dir, log_entry,
                               mission_log, active_apps))
            thread_list.append(app_thread)
            app_thread.start()
    # clear up the thread when all jobs are done
    print("waiting for all remaning apps to finish")
    for t in thread_list:
        if not t.isAlive():
            t.handled = True
    thread_list = [t for t in thread_list if not t.handled]
    for t in thread_list:
        t.join()
    # write to file
    os.chdir('/home/liuliu/Research/rapid_m_backend_server/TestScript/')
    writeMissionToFile(mission_log)


def run(target_num_apps):
    for scale in BUDGET_SCALE:
        for mode in STRAWMANS:
            qos = {}
            data = {}
            budgets = genBudgets(app_info, scale)
            run_a_mission(target_num_apps, budgets)


if not reset_server():
    exit(1)
genInfo()
#for i in range(2, len(apps) + 1):
#    run(i)
run(4)
