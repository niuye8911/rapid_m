''' draw the mission as line graph '''
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from collections import OrderedDict
start_time = 0.0
last_end_time = 0.0

apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']
modes = ['N','INDIVIDUAL','P_M','P_M_RUSH']
budgets = [1.0]
num_of_apps = [3]
ids = [0,1]

mode_ax = {}

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

def draw_a_mission(num_of_app, budget,id):
    global last_end_time, start_time
    data_files = OrderedDict()
    # read in files
    mission_file = './mission/mission_'+str(num_of_app)+'_'+str(id)+'.log'
    data_files['mission']=mission_file
    for mode in modes:
        exec_file = './mission/execution_'+mode+'_'+str(budget)+'_'+str(num_of_app)+'_'+str(id)+'.log'
        data_files[mode]=exec_file
    # plot graphs
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    mode_ax['mission'] = fig.add_subplot(gs[0, :])
    mode_ax['mission'].set_title('Mission',fontsize=10)
    plt.xlabel('Time (Seconds)')
    mode_ax['INDIVIDUAL'] = fig.add_subplot(gs[1, 0])
    mode_ax['INDIVIDUAL'].set_title('INDIVIDUAL',fontsize=10)
    mode_ax['N'] = fig.add_subplot(gs[1, 1])
    mode_ax['N'].set_title('N',fontsize=10)
    mode_ax['P_M'] = fig.add_subplot(gs[2, 0])
    mode_ax['P_M'].set_title('P_M',fontsize=10)
    mode_ax['P_M_RUSH'] = fig.add_subplot(gs[2,1])
    mode_ax['P_M_RUSH'].set_title('P_M_RUSH',fontsize=10)
    plt.tight_layout()
    # first draw mission_
    print(data_files)
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
                ax = mode_ax[mode]
                if status==1:
                    ax.plot([left,right],[APP_Y[entry['app']],APP_Y[entry['app']]],color=APP_COLOR[entry['app']])
                elif status==2:
                    # reject
                    ax.plot([left],[APP_Y[entry['app']]],marker='x',color=APP_COLOR[entry['app']])
                else:
                    #fail during the middle
                    ax.plot([left,right],[APP_Y[entry['app']],APP_Y[entry['app']]],'--',color=APP_COLOR[entry['app']])
                    ax.plot([right],[APP_Y[entry['app']]],marker='X',color=APP_COLOR[entry['app']])
                ax.axvline(x=entry['start_time'],
                           color='grey',
                           linewidth=0.5,
                           linestyle='--')
                ax.axvline(x=entry['start_time'] + entry['elapsed'],
                           color='grey',
                           linewidth=0.5,
                           linestyle='--')
                if not mode == 'mission':
                    ax.yaxis.set_visible(False)
                ax.set_xlim(0,700)
    mode_ax['mission'].set_yticks(range(0,len(apps)+1))
    mode_ax['mission'].set_yticklabels(['']+apps,rotation=45)
    plt.show()


draw_a_mission(4,1.0,0)
