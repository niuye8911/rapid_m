''' draw the mission as line graph '''
import json
import matplotlib.pyplot as plt

start_time = 0.0
last_end_time = 0.0

apps = ['swaptions', 'ferret', 'bodytrack', 'svm', 'nn', 'facedetect']

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

with open('./dynamic_mission.log') as log:
    log_json = json.load(log)
    fig, ax = plt.subplots()
    # update end time
    for entry in log_json:
        end_time = entry['start_time'] + entry['elapsed']
        last_end_time = max(last_end_time, end_time)
        ax.hlines(y=APP_Y[entry['app']],
                  xmin=entry['start_time'],
                  xmax=entry['start_time'] + entry['elapsed'],
                  color=APP_COLOR[entry['app']])
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
    plt.savefig('pm_1_0.png')
