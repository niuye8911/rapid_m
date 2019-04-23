# the class for bucket
# a bucket contains a P-model instance and a list of configurations along with
# their cost / quality

class Bucket:
    def __init__(self, configurations, p_model, cost_fact, mv_fact):
        self.configurations = configurations
        self.p_model = p_model

    def genSubProfile(self):
        ''' genrate the cost / quality profile for configurations '''
        pass

    def getOptimal(self, budget, slowdown):
        ''' use the slowdown and the cost to find the best one '''
        pass
