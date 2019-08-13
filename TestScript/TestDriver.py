''' Test driver to run multiple apps together '''
import os, sys, imp, subprocess
RAPIDS_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/'
RAPIDS_SOURCE_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/'
sys.path.append(os.path.dirname(RAPIDS_DIR))
sys.path.append(os.path.dirname(RAPIDS_SOURCE_DIR))

import Rapids.Classes
import Rapids.App.AppSummary

apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']
APP_MET_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/appExt/'

# preparation
app_info = {}
for app in apps:
    # init all appmethods
    file_loc = APP_MET_PREFIX + app + "Met.py"
    module = imp.load_source("", file_loc)
    appMethod = module.appMethods("", app)
    # prepare run_dir for all apps
    run_dir = os.getcwd()+"/run/" + app
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    app_info[app] = {'met': appMethod, 'dir': run_dir}

# gen apps' gt
for app,info in app_info.items():
    if app is not "bodytrack":
        pass
    cmd = info['met'].getCommand(qosRun=True)
    working_dir = info['dir']
    # change working dir
    os.chdir(working_dir)
    # run all command and get the output
    stresser = subprocess.Popen(" ".join(cmd),
                                shell=True,
                                preexec_fn=os.setsid)
