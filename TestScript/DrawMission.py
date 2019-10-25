''' draw the mission as line graph '''
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle

start_time = 0.0
last_end_time = 0.0

apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']
modes = ['N','INDIVIDUAL','P_M']
budgets = [1.0]
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

def draw_a_mission(input,output):
    global last_end_time, start_time
    with open(input) as log:
        log_json = json.load(log)
        fig, ax = plt.subplots()
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
                plt.plot([left,right],[APP_Y[entry['app']],APP_Y[entry['app']]],color=APP_COLOR[entry['app']])
            elif status==2:
                # reject
                plt.plot([left],[APP_Y[entry['app']]],marker='x',color=APP_COLOR[entry['app']])
            else:
                #fail during the middle
                plt.plot([left,right],[APP_Y[entry['app']],APP_Y[entry['app']]],'--',color=APP_COLOR[entry['app']])
                plt.plot([right],[APP_Y[entry['app']]],marker='X',color=APP_COLOR[entry['app']])
            #ax.hlines(y=APP_Y[entry['app']],
            #          xmin=left,
            #          xmax=right,
            #          '-gD',
        #              markevery = [xmain],
        #              color=APP_COLOR[entry['app']])
            ax.axvline(x=entry['start_time'],
                       color='grey',
                       linewidth=0.5,
                       linestyle='--')
            ax.axvline(x=entry['start_time'] + entry['elapsed'],
                       color='grey',
                       linewidth=0.5,
                       linestyle='--')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Applications')
        plt.yticks(range(1, len(apps) + 1), apps, rotation=45)
        plt.savefig(output)

for id in ids:
        # draw the base mission
    slot_log = './mission_slot_'+str(id)+'.log'
    slot_out = 'trace_demo_'+str(id)+'.png'
    draw_a_mission(slot_log,slot_out)
    last_end_time = 0.0
    start_time = 0.0
    for budget in budgets:
        for mode in modes:
            mission_log = './execution_'+mode+"_"+str(budget)+"_"+str(id)+'.log'
            output = 'trace_execution_'+mode+"_"+str(budget)+"_"+str(id)+'.png'
            draw_a_mission(mission_log,output)
            last_end_time = 0.0
            start_time = 0.0
