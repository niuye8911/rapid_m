''' a thread with subprocess callback'''
import threading
import time
import os
import subprocess


class Rapid_M_Thread(threading.Thread):
    def __init__(self, callback=None, callback_args=None, *args, **kwargs):
        target = kwargs.pop('target')
        dir = kwargs.pop('dir')
        cmd = kwargs.pop('cmd')
        app_time = kwargs.pop('app_time')
        app = kwargs.pop('app')
        super(Rapid_M_Thread, self).__init__(target=self.target_with_callback,
                                             args=(app_time, dir, cmd, app))
        self.callback = callback
        self.method = target
        self.callback_args = callback_args

    def target_with_callback(self, app_time, dir, cmd, app):
        self.method(dir, app_time, cmd, app)
        if self.callback is not None:
            self.callback(*self.callback_args)


def rapid_worker(dir, app_time, cmd, app):
    # run command under dir and record time
    start_time = time.time()
    #os.chdir(dir)
    #os.system(" ".join(cmd))
    p = subprocess.Popen(" ".join(cmd),shell=True, cwd=dir)
    p.wait()
    exec_time = time.time() - start_time
    app_time[app] = exec_time


def rapid_callback(app, cmd, app_time):
    print(app, '*****************finished')
    # this is run after your thread end
    while -1 in app_time.values():
        os.chdir(
            '/home/liuliu/Research/rapid_m_backend_server/TestScript/tmp/')
        print(app, '****************waiting')
        os.system(" ".join(cmd))
    return
