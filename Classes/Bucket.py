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
        # convert rep_env back to float
        for f,v in self.rep_env.items():
            self.rep_env[f]=float(v)

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

    def getParetos(self, profile_dict):
        paretos = []
        profile = [[k, v] for k, v in profile_dict.items()]
        # sorted by cost
        profile = sorted(profile, key=lambda x: x[1]['cost'])
        # remove non-pareto (greater cost, lower mv)
        highest = -1
        for c in profile:
            cur_cost = c[1]['cost']
            cur_mv = c[1]['mv']
            if cur_mv > highest:
                highest = cur_mv
                paretos.append(c)
        return paretos

    def getOptimal(self, budget, slowdown, tol=0.0):
        ''' use the slowdown and the cost to find the best one '''
        selected = []
        # filter out those non-pareto-optimal points
        paretos = self.getParetos(self.profile)
        # find the minimal config
        min_id = -1
        for i in range(0, len(paretos)):
            if paretos[i][1]['cost'] * slowdown <= budget * (1 - tol):
                min_id = i
            else:
                break
        # find the maximal config
        max_id = -1
        for i in reversed(range(0, len(paretos))):
            if paretos[i][1]['cost'] * slowdown <= budget * (1 + tol):
                max_id = i
                break
        # budget too small, max_id = -1
        if max_id == -1:
            selected = []
        elif min_id == -1:
            selected = paretos[0:max_id + 1]
        else:
            # both min and max != -1
            selected = paretos[min_id:max_id + 1]
        # clean up selected
        selected_conf = list(map(lambda x: x[0], selected))
        selected_mv = list(map(lambda x: float(x[1]['mv']), selected))
        SUCCESS = True
        if selected_conf == []:
            SUCCESS = False
            selected_conf = [paretos[0][0]]
            selected_mv = [paretos[0][1]['mv']]
        return selected_conf, selected_mv, SUCCESS
