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

import random, glob
import Rapids.App.AppSummary
from Rapids.util import __get_minmax
from BucketSelector import bucketSelect
from Rapid_M_Thread import Rapid_M_Thread, rapid_callback, rapid_worker, rapid_dynamic_callback, rapid_dynamic_worker
from TestDriver_Helper import *
import warnings, time
from random import randint
from sklearn.exceptions import DataConversionWarning
from resultViewer_dynamic import *
from shutil import copyfile

MAX_WAIT_TIME = 15  # wait at most 10 second until the new app be inited
MISSION_TIME = 60 * 10  # 10 minutes run
SERVER_MODE_FILE = '/home/liuliu/SERVER_MODE'
REPEAT = 3
MISSION_DIR = "./mission_oct31/"
MISSION_PREFIX = MISSION_DIR+"mission_"
EXECUTION_PREFIX = MISSION_DIR+"execution_"
APP_RANGE = range(2, 7)

# ignore the TF debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# ignore the data conversion
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']
APP_MET_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/appExt/'
APP_OUT_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/outputs/'

app_info = {}
commands = {}

metric_df = None

STRAWMANS = ['P_SAVING','N','INDIVIDUAL',  'P_M', 'P_M_RUSH']  # strawmans to use
#STRAWMANS = ['P_M','P_M_RUSH']  # strawmans to use
#BUDGET_SCALE = [0.8, 1.0, 1.5]

BUDGET_SCALE = [0.6,0.8,1.0]


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


def resetRunDir():
    for app in apps:
        run_dir = app_info[app]['dir']
        filelist = [f for f in os.listdir(run_dir)]
        for f in filelist:
            os.remove(os.path.join(run_dir, f))


def genInfo():
    for app in apps:
        # appmethods
        file_loc = APP_MET_PREFIX + app + "Met.py"
        module = imp.load_source("", file_loc)
        appMethod = module.appMethods(app, app)
        # run_dir
        run_dir = os.getcwd() + "/run/" + app + '/'
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        # update the run_config
        run_config = getAppRunConfig(app)
        appMethod.setRunConfigFile(run_config)
        appMethod.setRunDir(run_dir)
        cmd = appMethod.getRapidsCommand()
        commands[app] = cmd
        updateAppMinMaxFake(appMethod, app)
        app_info[app] = {'met': appMethod, 'dir': run_dir}
        # generate the ground truth first
        appMethod.runGT(True)


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

# update the mission value based on real execution
def get_mv_from_file(file_name):
    result = {}
    with open(file_name) as json_file:
        f = json.load(json_file)
        for entry in f:
            # check if it successes
            app = entry['app']
            if not app in result:
                result[app] = entry['mv']
    return result

def update_mvs(max_mv,min_mv):
    for app in max_mv.keys():
        mv_max = max_mv[app]
        mv_min = min_mv[app]
        app_info[app]['met'].max_mv = mv_max
        app_info[app]['met'].min_mv = mv_min

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


def writeMissionToFile(mission_log, log_name):
    with open(log_name, 'w') as fout:
        json.dump(mission_log, fout, indent=2)


def genMissionFromFile(mission_log_file):
    mission = []
    with open(mission_log_file, 'r') as mission_log_json:
        mission_log = json.load(mission_log_json)
        for entry in mission_log:
            app = entry['app']
            start_time = entry['start_time']
            budget = entry['budget']
            mission.append({
                'app': app,
                'start_time': start_time,
                'budget': budget
            })
    mission = sorted(mission, key=lambda x: x['start_time'])
    return mission


def execute_mission(mission, num_app, mode, budgets, id, budget_scale):
    log_name = EXECUTION_PREFIX + mode + '_' + str(budget_scale) + '_' + str(
        num_app) + '_' + str(id) + '.log'
    if os.path.exists(log_name):
        return
    # save some time, if it's executing the same mission for PS, just use the old data
    if mode == 'P_SAVING':
        first_log = EXECUTION_PREFIX + mode + '_0.6' + '_' + str(
            num_app) + '_' + str(id) + '.log'
        if os.path.exists(first_log):
            copyfile(first_log, log_name)
            return
    # change the server mode
    with open(SERVER_MODE_FILE, 'w') as mode_file:
        print(mode)
        mode_file.write(mode)
    mission_log = []
    active_apps = {}
    start_time = time.time()
    thread_list = []
    progress_map = {}
    for entry in mission:
        app = entry['app']
        app_start_time = entry['start_time']
        budget = budgets[app]
        # execute the app with target
        app_cmd = commands[app]
        appmet = app_info[app]['met']
        run_dir = app_info[app]['dir']
        if mode == "P_SAVING":
            appmet.updateRunConfig(budget,rapid_m=False,debug=True,power_saving=True)
        elif mode == 'P_M_RUSH':
            print('using rush to end')
            appmet.updateRunConfig(budget,
                                   rapid_m=True,
                                   debug=True,
                                   rush_to_end=True)
        else:
            appmet.updateRunConfig(budget, rapid_m=True, debug=True)
        # sleep until the target time
        cur_time = time.time() - start_time
        sleep_time = max(app_start_time - cur_time, 0)
        time.sleep(sleep_time)
        app_real_start_time = time.time() - start_time
        print("starting new app:xxxxxxxxxxx", app, app_real_start_time)
        log_entry = {
            'global_start': start_time,
            'start_time_process': app_real_start_time,
            'app': app,
            'budget': budget
        }
        app_thread = Rapid_M_Thread(name=app + "_thread",
                                    target=rapid_dynamic_worker,
                                    dir=run_dir,
                                    cmd=app_cmd,
                                    app_time=log_entry,
                                    app=app,
                                    callback=rapid_dynamic_callback,
                                    callback_args=(app, appmet, run_dir,
                                                   log_entry, mission_log,
                                                   active_apps))
        thread_list.append(app_thread)
        app_thread.start()
    wait_and_finish(thread_list, mission_log, log_name)


def genMission(target_num_apps, id):
    # change the server mode
    with open(SERVER_MODE_FILE, 'w') as mode_file:
        mode_file.write('INDIVIDUAL')
    '''run at most target_num_apps, each will be executed repeat_times '''
    log_name = MISSION_PREFIX + str(target_num_apps) + '_' + str(id) + ".log"
    if os.path.exists(log_name):
        print("mission done before:", target_num_apps, str(id))
        return
    progress_map = {}
    thread_list = []
    # first get all apps' command
    active_apps = {}  # record how many apps are active
    for app in apps:
        appmet = app_info[app]['met']
        # maximum budget to reserve slot
        appmet.updateRunConfig(999999, rapid_m=True, debug=True)
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
                'start_time_from_process': new_app_start_time,
                'app': new_app,
                'budget': 999999
            }
            active_apps[new_app] = True
            print("starting new app:xxxxxxxxxxx", start_time, new_app,
                  new_app_start_time)
            app_thread = Rapid_M_Thread(
                name=new_app + "_thread",
                target=rapid_dynamic_worker,
                dir=run_dir,
                cmd=app_cmd,
                app_time=log_entry,
                app=new_app,
                callback=rapid_dynamic_callback,
                callback_args=(new_app, appmet, run_dir, log_entry,
                               mission_log, active_apps))
            thread_list.append(app_thread)
            app_thread.start()
        time.sleep(10)  # wait for 10 seconds for the next check
    # clear up the thread when all jobs are done
    wait_and_finish(thread_list, mission_log, log_name)
    print("*****************MISSION_GENERATE_DONE************")


def wait_and_finish(thread_list, mission_log, log_name):
    print("waiting for all remaning apps to finish")
    for t in thread_list:
        if not t.handled:
            print('WAITING FOR THREAD:' + t.name)
            t.join(timeout=300)  # 5 mins maximum wait time
    # write to file
    os.chdir('/home/liuliu/Research/rapid_m_backend_server/TestScript/')
    writeMissionToFile(mission_log, log_name)


def main(argv):
    if not os.path.exists(MISSION_DIR):
        os.mkdir(MISSION_DIR)
    genInfo()
    # estimate the overall time
    overall_time = len(APP_RANGE) * REPEAT * len(BUDGET_SCALE) * len(
        STRAWMANS) * MISSION_TIME / 60
    print("estimating time in minutes:", overall_time)
    for num_app in APP_RANGE:
        for i in range(0, REPEAT):
            # reset the server
            if not reset_server():
                exit(1)
            # reset the run dir
            resetRunDir()
            # generate a mission
            genMission(num_app, id=i)
            for budget in BUDGET_SCALE:
                budgets = genBudgets(app_info, budget)
                mission = genMissionFromFile(MISSION_PREFIX + str(num_app) + '_' +
                                             str(i) + ".log")
                # follow the mission and run different strawmans
                for mode in STRAWMANS:
                    # reset the server
                    if not reset_server():
                        exit(1)
                    # reset the run dir
                    resetRunDir()
                    execute_mission(mission, num_app, mode, budgets, i, budget)
    max_mvs = get_mv_from_file(MISSION_DIR+'mission_2_0.log')
    min_mvs = get_mv_from_file(MISSION_DIR+'execution_P_SAVING_0.6_2_0.log')
    print(max_mvs,min_mvs)
    update_mvs(max_mvs,min_mvs)
    readFile(MISSION_DIR, APP_RANGE, STRAWMANS, BUDGET_SCALE,
             range(0, REPEAT), app_info)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
