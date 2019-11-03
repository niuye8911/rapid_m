import pandas as pd
import imp,sys,os

RAPIDS_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/'
RAPIDS_SOURCE_DIR = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/'
RAPIDS_M_DIR = '/home/liuliu/Research/rapid_m_backend_server/'
sys.path.append(os.path.dirname(RAPIDS_DIR))
sys.path.append(os.path.dirname(RAPIDS_M_DIR))
sys.path.append(os.path.dirname(RAPIDS_SOURCE_DIR))

apps = ['swaptions','bodytrack','facedetect','ferret','svm','nn']
app_mets = {}

COST_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/Rapids/outputs/'
APP_MET_PREFIX = '/home/liuliu/Research/rapidlib-linux/modelConstr/appExt/'

def getCost(app):
    file = COST_PREFIX+app+'/'+app+'-cost.fact'
    cost_map = {}
    with open(file,'r') as cost_file:
        for line in cost_file:
            col = line.split(' ')
            v = float(col[-1])
            config = '-'.join(col[:-1])
            cost_map[config] = v
    return cost_map

def genInfo():
    for app in apps:
        # appmethods
        file_loc = APP_MET_PREFIX + app + "Met.py"
        module = imp.load_source("", file_loc)
        appMethod = module.appMethods(app, app)
        # run_dir
        app_mets[app]=appMethod

def getTrainingCol(row,cost):
    sd = row['SLOWDOWN']
    config = row['Configuration']
    stresser_alone = 0 # base cost
    app_alone = app_mets[app].training_units * cost[config] / 1000
    app_stresser = app_mets[app].training_units * cost[config] * sd / 1000
    return (stresser_alone+app_alone+app_stresser)/60.0

genInfo()
for app in apps:
    df = pd.read_csv(app+'-mperf.csv')
    cost_file = getCost(app)
    training_time = df.apply(lambda x: getTrainingCol(x, cost_file), axis=1)
    print(app,sum(training_time))
