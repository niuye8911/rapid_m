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
#apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']
apps = ['swaptions', 'bodytrack']
APP_MET_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/appExt/'
TEST_APP_FILE = './test_app_file.txt'

app_info = {}


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
    for app, info in app_info.items():
        if app is not "facedetect":
            pass
        cmd = info['met'].getCommand(qosRun=True)
        working_dir = info['dir']
        # change working dir
        os.chdir(working_dir)
        # run all command and get the output
        stresser = subprocess.Popen(" ".join(cmd),
                                    shell=True,
                                    preexec_fn=os.setsid)


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


def run_a_comb(apps):
    '''comb is a list of app names'''
    progress_map = {}
    thread_list = []
    configs = bucketSelect(TEST_APP_FILE, SELECTOR="INDIVIDUAL")
    configs = configs[1]
    for app in apps:
        progress_map[app] = -1
        appMethod = app_info[app]['met']
        run_dir = app_info[app]['dir']
        max_budget = appMethod.max_cost * appMethod.fullrun_units

        app_cmd = appMethod.getCommandWithConfig(configs[app], qosRun=True)
        print(app_cmd)
        exit(1)
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
    print("all workers started")
    for t in thread_list:
        t.join()
    print(progress_map)


def run(combs, mode="INDIVIDUAL"):
    qos = {}
    exec_time = {}
    if mode == "INDIVIDUAL":
        for num_of_app, comb in combs.items():
            for apps in comb:
                update_app_file(apps)
                run_a_comb(apps)
                exit(1)


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
