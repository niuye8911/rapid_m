''' a thread with subprocess callback'''
import threading
import time
import os
import pandas as pd
import subprocess


class Rapid_M_Thread(threading.Thread):
    def __init__(self, callback=None, callback_args=None, *args, **kwargs):
        target = kwargs.pop('target')
        dir = kwargs.pop('dir')
        cmd = kwargs.pop('cmd')
        app_time = kwargs.pop('app_time')
        app = kwargs.pop('app')
        thread_name = kwargs.pop('name')
        super(Rapid_M_Thread, self).__init__(target=self.target_with_callback,
                                             args=(app_time, dir, cmd, app))
        self.name = thread_name
        self.callback = callback
        self.method = target
        self.callback_args = callback_args
        self.handled = False

    def target_with_callback(self, app_time, dir, cmd, app):
        self.method(dir, app_time, cmd, app)
        if self.callback is not None:
            self.callback(*self.callback_args)
            self.handled = True


def rapid_worker(dir, app_time, cmd, app):
    # run command under dir and record time
    start_time = time.time()
    p = subprocess.Popen(" ".join(cmd), shell=True, cwd=dir)
    p.wait()
    exec_time = time.time() - start_time
    app_time[app] = exec_time


def rapid_callback(app, cmd, app_time):
    print(app, '*****************finished')
    # this is run after your thread end
    while -1 in app_time.values():
        p = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            cwd='/home/liuliu/Research/rapid_m_backend_server/TestScript/tmp/')
        p.wait()
        print(app, '****************waiting')
    return


def rapid_dynamic_worker(dir, log_entry, cmd, app):
    # run command under dir and record time
    print(app, '*****************started')
    log_entry['start_time'] = time.time() - log_entry['global_start']
    p = subprocess.Popen(" ".join(cmd), shell=True, cwd=dir)
    p.wait()


def rapid_dynamic_callback(app, appmet, rundir, log_entry, mission_logs,
                           active_apps):
    # tell the server it's finished
    URL = "http://algaesim.cs.rutgers.edu/rapid_server/end.php?machine=algaesim&app=" + app
    payload = "budget=0"
    finish_s_time = time.time()
    try:
        response = requests.post(URL, data=payload, timeout=30)
    except:
        print("telling server from script failed")
    finishing_time = time.time() - finish_s_time
    # get the qos, summarize the result, write to log
    print(app, '*****************finished')
    # this runs after your thread end
    mv = appmet.getQoS()
    if type(mv) is list:
        mv = mv[-1]
    try:
        mission_log = appmet.parseLog()
    except:
        print(app, "Mission log missing")
        log_entry['success'] = False
    log_entry['finishing_time'] = finishing_time
    log_entry['success'] = mission_log['success']
    log_entry['mv'] = mv
    log_entry['elapsed'] = mission_log['runtime']/1000.0
    log_entry['rc_by_budget'] = mission_log['rc_by_budget']
    log_entry['rc_by_rapidm'] = mission_log['rc_by_rapidm']
    log_entry['total_reconfig'] = mission_log['totReconfig']
    log_entry['scale_up'] = mission_log['slowdown_scale']
    log_entry['failed_reason'] = mission_log['failed_reason']
    mission_logs.append(log_entry)
    active_apps[app] = False
    return
