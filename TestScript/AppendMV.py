''' append the mv to the end of each slowdown_valid... '''

import pandas as pd

MODES = ['INDIVIDUAL','N','P','P_M']
BUDGETS = [0.8,1.0,1.5]
FOLDER = '/home/liuliu/Research/rapid_m_backend_server/TestScript/sep30/'

def getMVbyConfig(app,config):
    mvfile = '/home/liuliu/Research/rapid_m_backend_server/outputs/'+app+'/mv.csv'
    target_configs = config.split('-')
    with open(mvfile,'r') as  mv:
        for line in mv:
            mv_line = line.rstrip().split(' ')
            right = all(elem in mv_line for elem in target_configs)
            if right:
                return float(mv_line[-1])

def getMVbyRow(row):
    app = row['app']
    config = row['config']
    return getMVbyConfig(app, config)

def update_csv(mode, budget):
    csv_file = FOLDER+'slowdown_validator_'+mode+"_"+str(budget)+'.csv'
    new_csv_file=FOLDER+'slowdown_validator_'+mode+"_"+str(budget)+'_withmv.csv'
    df = pd.read_csv(csv_file)
    df['mv']=df.apply(lambda row:getMVbyRow(row),axis=1)
    df.to_csv(new_csv_file)

for budget in BUDGETS:
    for mode in MODES:
        update_csv(mode, budget)
