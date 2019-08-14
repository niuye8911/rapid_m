'''
    Class for a system state
    A SystemState contains information:
    1) machine_id (int)
    1) attributes <attribute,value> (dict<string,float>)
'''


class SystemState:
    EXCLUDED_METRIC = {"Date", "Time"}

    def __init__(self, machine_id, ):
        self.machine_id = machine_id
        self.metrics = dict()
        self.metric_names = []

    def add_metric(self, name, value):
        if name not in self.EXCLUDED_METRIC:
            self.metrics[name] = value
            self.metric_names.append(name)
            self.metric_names = sorted(self.metric_names)
