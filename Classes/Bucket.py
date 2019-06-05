# the class for bucket
# a bucket contains a P-model instance and a list of configurations along with
# their cost / quality


class Bucket:
    def __init__(self, app_name, b_name, configurations, p_model, cost_fact,
                 mv_fact, rep_env):
        self.app_name = app_name
        self.b_name = b_name
        self.configurations = configurations
        self.p_model = p_model
        self.profile = {}
        self.rep_env = rep_env
        self.genSubProfile(cost_fact, mv_fact)

    def genSubProfile(self, cost_fact_file, mv_fact_file):
        ''' genrate the cost / quality profile for configurations '''
        cost_fact = self.readFact(cost_fact_file)
        mv_fact = self.readFact(mv_fact_file)
        for configuration, cost in cost_fact.items():
            self.profile[configuration] = {}
            self.profile[configuration]['cost'] = cost_fact[configuration]
            self.profile[configuration]['mv'] = mv_fact[configuration]

    def readFact(self, factfile):
        fact_dict = {}
        end_of_config = -1
        with open(factfile) as fact:
            for line in fact:
                columns = line.split()
                if end_of_config == -1:
                    end_of_config = self.getEndOfConfig(columns)
                configuration = "-".join(columns[:end_of_config])
                value = float(columns[len(columns) - 1])
                fact_dict[configuration] = value
        return fact_dict

    def getEndOfConfig(self, columns):
        ''' return the index of the first element in the column that does not
        belong to the configuration '''
        for i in range(0, len(columns), 2):
            if self.isfloat(columns[i]):
                return i

    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def getSlowDown(self, combined_env):
        return self.p_model.predict(combined_env)

    def getOptimal(self, budget, slowdown):
        ''' use the slowdown and the cost to find the best one '''
        cur_mv = -1
        selected = ''
        for config, metrics in self.profile.items():
            if metrics['cost'] * slowdown <= budget and metrics['mv'] > cur_mv:
                cur_mv = metrics['mv']
                selected = config
        return selected, cur_mv
