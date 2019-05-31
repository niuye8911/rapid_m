from models.RapidBayesian import *
from models.RapidEN import *
from models.RapidLasso import *
from models.RapidLinear import *

class ModelPool:
    #CANDIDATE_MODELS = ['linear','EN','lassoCV','Bayesian','NN']
    CANDIDATE_MODELS = ['linear','EN','lassoCV','Bayesian']
    def getModel(self, name):
        if name not in self.CANDIDATE_MODELS:
            print('not supported model:'+name)
            return None
        if name == 'linear':
            return RapidLinear()
        elif name == 'EN':
            return RapidEN()
        elif name == 'lassoCV':
            return RapidLasso()
        elif name == 'Bayesian':
            return RapidBayesian()
        elif name == 'NN':
            return None # todo