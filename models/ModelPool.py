from collections import OrderedDict

from sklearn.preprocessing import PolynomialFeatures

from models.RapidBayesian import *
from models.RapidEN import *
from models.RapidLasso import *
from models.RapidLinear import *
from models.RapidNN import *
from models.RapidSVR import *
from Utility import *


class ModelPool:
    CANDIDATE_MODELS = ['LR', 'LS', 'EN', 'BR', 'NN', 'SVR']

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
        elif name == 'SVR':
            return RapidSVR(file_path=path)

    def selectFeature(self, x_train, y_train, x_test, y_test, model_name):
        ''' select a feature based on the selected model '''


    def selectModel(self, x_train, y_train, x_test, y_test, TEST=False):
        ''' choose a proper model with the lowest mre,
        if TEST is set to True, then use all models '''
        min_mse = 99
        min_diff = 99
        selected_model = None
        poly = False
        training_time = OrderedDict()
        x_train_poly = PolynomialFeatures(degree=2).fit_transform(x_train)
        x_test_poly = PolynomialFeatures(degree=2).fit_transform(x_test)
        for model_name in ModelPool.CANDIDATE_MODELS:
            if model_name is not 'NN':
                tmp_linear_model = self.getModel(model_name)
                tmp_poly_model = self.getModel(model_name)

                time_linear = tmp_linear_model.fit(x_train, y_train)
                time_high = tmp_poly_model.fit(x_train_poly, y_train)
                r2_linear, mse_linear, diff_linear = tmp_linear_model.validate(
                    x_test, y_test)
                r2_poly, mse_poly, diff_poly = tmp_poly_model.validate(
                    x_test_poly, y_test)
                training_time[model_name + '-1'] = {
                    'time': time_linear,
                    'r2': r2_linear,
                    'mse': mse_linear,
                    'diff': diff_linear
                }
                training_time[model_name + '-2'] = {
                    'time': time_high,
                    'r2': r2_poly,
                    'mse': mse_poly,
                    'diff': diff_poly
                }
                if mse_linear < mse_poly and mse_linear < min_mse:
                    selected_model = tmp_linear_model
                    poly = False
                    min_mse = mse_linear
                elif mse_poly < mse_linear and mse_poly < min_mse:
                    selected_model = tmp_poly_model
                    poly = True
                    min_mse = mse_poly
                min_diff = min(min_diff, min(diff_linear, diff_poly))
            else:
                # if accuracy is enough, skip NN
                if min_diff < 10 and not TEST:
                    RAPID_info('ModelPool', "Accuracy Reached, no need for NN")
                    continue
                # NN does not need to check high order
                nn_model = self.getModel('NN')
                time_nn = nn_model.fit(x_train, y_train)
                r2, mse, diff = nn_model.validate(x_test, y_test)
                training_time['nn'] = {
                    'time': time_nn,
                    'r2': r2,
                    'mse': mse,
                    'diff': diff
                }
                if min_diff >= 10:
                    selected_model = nn_model
                    poly = False
                    min_diff = diff
        # print the training time into the debug file
        return selected_model, poly, training_time
