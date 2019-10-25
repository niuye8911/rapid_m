''' draw the mission as line graph '''
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from collections import OrderedDict
start_time = 0.0
last_end_time = 0.0

apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']
modes = ['N','INDIVIDUAL','P_M']
budgets = [1.0]
num_of_apps = [3]
ids = [0,1]

APP_COLOR = {
    'swaptions': 'b',
    'ferret': 'g',
    'bodytrack': 'c',
    'svm': 'y',
    'nn': 'm',
    'facedetect': 'r'
}

APP_Y = {
    'swaptions': 1.0,
    'ferret': 2.0,
    'bodytrack': 3.0,
    'svm': 4.0,
    'nn': 5.0,
    'facedetect': 6.0
}

def draw_a_mission(num_of_app, buget,id):
    global last_end_time, start_time
    data_files = OrderedDict()
    # read in files
    mission_file = './mission/mission_'+str(num_of_app)+'_'+str(id)+'.log'
    data_files['mission']=mission_file
    for mode in modes:
        exec_file = './mission/execution_'+mode+'_'+str(budget)+'_'+str(num_of_app)+'_'+str(id)+'.log'
        data_files[mode]=exec_file
    # plot graphs
    sub_graphs = ['mission'] + modes
    fig,ax = plt.subplots(2,2,sharex=True, sharey=True,gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
    fig.tight_layout()
    fig.text(0.5, 0.02, 'Time (seconds)', ha='center',fontsize=12)
    rowid = 0
    colid = 0
    lines={}
    # first draw mission_
    for mode,file in data_files.items():
        # reset time
        last_end_time = 0.0
        start_time = 0.0
        with open(file) as log:
            log_json = json.load(log)
            success = 0
            fail = 0
            # update end time
            for entry in log_json:
                status = 1
                if entry['success'] == "1":
                    success += 1
                elif entry['success']=='2':
                    #reject
                    fail += 1
                    status = 2
                else:
                    #fail
                    fail+=1
                    status = 0
                end_time = entry['start_time'] + entry['elapsed']
                last_end_time = max(last_end_time, end_time)
                left=entry['start_time']
                right=entry['start_time'] + entry['elapsed']
                if status==1:
                    ax[rowid,colid].plot([left,right],[APP_Y[entry['app']],APP_Y[entry['app']]],color=APP_COLOR[entry['app']])
                elif status==2:
                    # reject
                    ax[rowid,colid].plot([left],[APP_Y[entry['app']]],marker='x',color=APP_COLOR[entry['app']])
                else:
                    #fail during the middle
                    ax[rowid,colid].plot([left,right],[APP_Y[entry['app']],APP_Y[entry['app']]],'--',color=APP_COLOR[entry['app']])
                    ax[rowid,colid].plot([right],[APP_Y[entry['app']]],marker='X',color=APP_COLOR[entry['app']])
                ax[rowid,colid].axvline(x=entry['start_time'],
                           color='grey',
                           linewidth=0.5,
                           linestyle='--')
                ax[rowid,colid].axvline(x=entry['start_time'] + entry['elapsed'],
                           color='grey',
                           linewidth=0.5,
                           linestyle='--')
            ax[rowid,colid].set_title(mode,fontsize=10)
            colid+=1
            if colid==2:
                colid=0
                rowid+=1
    ax[0,0].set_yticks(range(0,len(apps)+1))
    ax[0,1].set_yticks(range(0,len(apps)+1))
    ax[1,1].set_yticks(range(0,len(apps)+1))
    ax[0,0].set_yticklabels(['']+apps,rotation=45)
    ax[1,0].set_yticks(range(0,len(apps)+1))
    ax[1,0].set_yticklabels(['']+apps,rotation=45)
    plt.show()

for num_of_app in num_of_apps:
    for budget in budgets:
        for id in ids:
            draw_a_mission(num_of_app,budget,id)
