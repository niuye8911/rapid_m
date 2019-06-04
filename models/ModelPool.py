from models.RapidBayesian import *
from models.RapidEN import *
from models.RapidLasso import *
from models.RapidLinear import *
from models.RapidNN import *


class ModelPool:
    CANDIDATE_MODELS = ['LR', 'LS', 'EN','BR', 'NN']

    def getModel(self, name, path=''):
        if name not in self.CANDIDATE_MODELS:
            print('not supported model:' + name)
            return None
        if name == 'LR':
            return RapidLinear(file_path=path)
        elif name == 'EN':
            return RapidEN(file_path=path)
        elif name == 'LS':
            return RapidLasso(file_path=path)
        elif name == 'BR':
            return RapidBayesian(file_path=path)
        elif name == 'NN':
            return RapidNN(file_path=path)
