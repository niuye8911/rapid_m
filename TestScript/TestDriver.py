''' Test driver to run multiple apps together '''
import os, sys, imp, subprocess, json, time
import pandas as pd
from itertools import combinations, permutations
from concurrent.futures import ThreadPoolExecutor as Pool

RAPIDS_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/'
RAPIDS_SOURCE_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/'
RAPIDS_M_DIR = '/home/liuliu/Research/rapid_m_backend_server/'
sys.path.append(os.path.dirname(RAPIDS_DIR))
sys.path.append(os.path.dirname(RAPIDS_M_DIR))
sys.path.append(os.path.dirname(RAPIDS_SOURCE_DIR))

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
#apps = ['bodytrack', 'swaptions', 'ferret']
APP_MET_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/appExt/'
TEST_APP_FILE = '/home/liuliu/Research/rapid_m_backend_server/TestScript/test_app_file.txt'

app_info = {}

STRAWMANS = ['P', 'P_M', 'N', 'INDIVIDUAL']
#STRAWMANS = ['P_M']

GEN_SYS = False

PCM_COMMAND = [
    'sudo /home/liuliu/Research/pcm/pcm.x', '0.5', '-nc', '-ns', '2>/dev/null',
    '-i=5', '-csv=tmp.csv'
]  #monitor for 2 seconds

metric_df = None


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


def getEnvByComb(apps):
    # numerate different names of comb
    possible_names = list(permutations(apps))
    possible_names = list(map(lambda x: '+'.join(x), possible_names))
    row = metric_df.loc[metric_df['comb'].isin(possible_names)]
    return row


def run_a_comb(apps, budgets, mode):
    '''comb is a list of app names
    GEN_SYS: generate environment if True'''
    progress_map = {}
    thread_list = []
    if not GEN_SYS:
        if mode == 'P':
            row = getEnvByComb(apps)
            selections = bucketSelect(TEST_APP_FILE, SELECTOR='P', env=row)
        else:
            selections = bucketSelect(TEST_APP_FILE, SELECTOR=mode)
        slowdowns = selections[3]
        configs = selections[1]
        successes = selections[2]
        expect_finish_times = selections[4]
    for app in apps:
        progress_map[app] = -1
        appMethod = app_info[app]['met']
        run_dir = app_info[app]['dir']
        if not GEN_SYS:
            app_cmd = appMethod.getCommandWithConfig(configs[app],
                                                     qosRun=False,
                                                     fullRun=False)
        else:
            app_cmd = appMethod.getCommand()  # default command
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
    if GEN_SYS:
        # start the monitor to get system environment
        time.sleep(1)
        monitor = __start_monitor()
    for t in thread_list:
        t.join()
    # assemble a slowdown entry for slow-down validation
    sd_entry = []
    if not GEN_SYS:
        for app, slowdown in slowdowns.items():
            ind_time = float(expect_finish_times[app]) / 1000.0 * float(
                app_info[app]['met'].training_units)
            if not successes[app]:
                raw_qos = 0.0
                qos = 0.0
            else:
                raw_qos, qos = getMV(app, app_info)
            sd_entry.append({
                'num':
                len(apps),
                'app':
                app,
                'config':
                configs[app],
                'exec':
                progress_map[app],
                'slowdown_p':
                slowdown,
                'success_p':
                successes[app],
                'success_gt':
                progress_map[app] < 1.1 * budgets[app],
                'ind_exec':
                ind_time,
                'budget':
                budgets[app],
                'slowdown_gt':
                float(progress_map[app]) / ind_time,
                'raw_qos':
                raw_qos,
                'qos':
                qos
            })
        return sd_entry
    else:  # return the recorded ENV
        os.chdir('/home/liuliu/Research/rapid_m_backend_server/TestScript')
        env = app_info['swaptions']['met'].parseTmpCSV()
        return env


def __start_monitor():
    os.chdir('/home/liuliu/Research/rapid_m_backend_server/TestScript')
    os.system(" ".join(PCM_COMMAND))


def run(combs):
    global metric_df
    budget_scale = [0.8, 1.0, 1.5]
    for scale in budget_scale:
        for mode in STRAWMANS:
            if mode == 'P':
                metric_df = pd.read_csv('./ALL_METRIC.csv')
            qos = {}
            data = {}
            slowdowns = []
            metrics = {}
            budgets = genBudgets(app_info, scale)
            for num_of_app, comb in combs.items():
                break
                per_data = {}  # all data for current num_of_app
                counter = 0
                for apps in comb:
                    if GEN_SYS:
                        metric = run_a_comb(apps, budgets, mode)
                        metrics["+".join(apps)] = metric
                    else:
                        update_app_file(apps, scale)
                        # sd_entry: all app in current comb
                        sd_entry = run_a_comb(apps, budgets, mode)
                        slowdowns = slowdowns + sd_entry
                if not GEN_SYS:
                    data[num_of_app] = per_data

            if not GEN_SYS:
                # write the slowdowns to a csv
                slowdown_file = writeSlowDown(slowdowns)
                summarize_data(slowdown_file)
                os.chdir(
                    '/home/liuliu/Research/rapid_m_backend_server/TestScript')
                os.rename('mv.json', 'mv_' + mode + "_" + str(scale) + ".json")
                os.rename('miss_pred.json',
                          'miss_pred_' + mode + "_" + str(scale) + ".json")
                os.rename('exceed.json',
                          'exceed_' + mode + "_" + str(scale) + ".json")
                os.rename('raw_mv.json',
                          'raw_mv' + mode + "_" + str(scale) + ".json")
                os.rename(
                    'slowdown_validator.csv',
                    'slowdown_validator_' + mode + "_" + str(scale) + ".csv")
            else:
                gen_metric_csv(metrics)


def gen_metric_csv(metrics):
    # write the header
    file = open('./metric.csv', 'w')
    file.write(list(metrics.values())[0].printAsHeader(',', leading="comb,"))
    file.write('\n')
    for comb_name, metric in metrics.items():
        file.write(comb_name + ',')
        file.write(metric.printAsCSVLine(','))
        file.write('\n')
    file.close()


def update_app_file(apps, scale=1.0):
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
                app['budget'] = app_info[app_name]['met'].max_cost * scale
    with open(TEST_APP_FILE, 'w') as file:
        json.dump(active_apps, file, indent=2)


genInfo()
combs = genRunComb()
run(combs)
