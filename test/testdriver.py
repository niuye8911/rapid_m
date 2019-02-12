import optparse
import imp
import json
import sys
import threading
import os

from subprocess import Popen

num_of_apps = 1
all_apps = []

def main(argv):
    parser = declareParser()
    options, args = parser.parse_args()
    parseCMD(options)
    runAll()

def runAll():
    tasks = []
    root_dir = os.getcwd()
    for app in all_apps:
        command = app['method'].getFullRunCommand(app['budget'], app['xml'])
        # create run dir
        rundir = root_dir+'/'+app['name']+"_rundir"
        if not os.path.isdir(rundir):
            os.mkdir(rundir)
        delay = int(app['startTime'])/1000
        tasks.append(threading.Timer(delay, run, [command, rundir]))
    # run all tasks with delay
    for task in tasks:
        task.start()

def run(command, rundir):
    os.chdir(rundir)
    print " ".join(command)
    os.system(" ".join(command))

def declareParser():
    parser = optparse.OptionParser()
    parser.add_option('--run', dest="runfile")
    return parser

def parseCMD(options):
    run_file = options.runfile
    parseRunFile(run_file)

def parseRunFile(run_file):
    global num_of_apps, all_apps
    with open(run_file) as run_json:
        apps = json.load(run_json)
        num_of_apps = len(apps)
        for app in apps['apps']:
            methods_path = app['appMet']
            obj_path = app['appPath']
            name = app['name']
            budget = float(app['budget'])
            appxml = app['appXML']
            start_time = int(app['startTime'])
            module = imp.load_source("", methods_path)
            appMethod = module.appMethods(name, obj_path)
            all_apps.append({'name':name, 'method':appMethod, 'xml':appxml, 'budget':budget, 'startTime':start_time})


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
