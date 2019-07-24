from collections import OrderedDict

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from models.RapidBayesian import *
from models.RapidEN import *
from models.RapidLasso import *
from models.RapidLinear import *
from models.RapidNN import *
from models.RapidSVR import *
from Utility import *

import pandas as pd


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
        elif name == 'SVR':
            return RapidSVR(file_path=path)
        elif name == 'NN':
            return RapidNN(file_path=path)

    def selectFeature(self, xdf, ydf, x_train, x_test, y_train, y_test,
                      model_name, isPoly, speedup=True):
        ''' select a feature based on the selected model '''
        print("selecting features:", model_name)
        features = [x for x in xdf.columns]
        target, current = self.__getCorr(xdf, ydf)
        if speedup:
            first = current[0][:-2]
            prefix='1' if current[0][-1]=='2' else '2'
            current.append(first+'-'+prefix)
        best_model, mse, r2 = self.__avgmser2(model_name, current, x_train,
                                              x_test, y_train, y_test, isPoly)

        id = 1
        while r2 < 0.92:
            bestr2 = 0
            bestnew = "NONE"
            bestmse = 0.0
            id += 1
            tried = []
            for addition in features:
                if addition in current or addition in tried:
                    continue
                tempcurrent = current.copy()
                addition_prefix = addition[:-2]
                if speedup:
                    tempcurrent.append(addition_prefix+'-1')
                    tempcurrent.append(addition_prefix+'-2')
                else:
                    tempcurrent.append(addition)
                model, tempmse, tempr2 = self.__avgmser2(
                    model_name, tempcurrent, x_train, x_test, y_train, y_test,
                    isPoly)
                if tempr2 > bestr2:
                    bestr2 = tempr2
                    if not speedup:
                        bestnew = addition
                    else:
                        bestnew = addition_prefix
                    bestmse = tempmse
                    best_model = model
                if speedup:
                    tried.append(bestnew+"-1")
                    tried.append(bestnew+"-2")
                else:
                    tried.append(bestnew)
            if bestnew == "NONE":
                break
            if speedup:
                current.append(bestnew+'-1')
                current.append(bestnew+'-2')
            else:
                current.append(bestnew)
            r2 = bestr2
            mse = bestmse
        return best_model, current

    def selectModel(self, x_train, x_test, y_train, y_test, TEST=False):
        ''' choose a proper model with the lowest mre,
        if TEST is set to True, then use all models '''
        min_mse = 999999
        min_diff = 999999
        max_r2 = 0
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
                min_mse = min(min_mse, min(mse_linear, mse_poly))
                max_r2 = max(max_r2, max(r2_linear, r2_poly))
            else:
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
                if r2 >= max_r2:
                    selected_model = nn_model
                    poly = False
                    min_diff = diff
                    max_r2 = r2
                    min_mse = mse
        # print the training time into the debug file
        return selected_model, poly, training_time

    def __avgmser2(self,
                   model_name,
                   factorlist,
                   X_train,
                   X_test,
                   y_train,
                   y_test,
                   isPoly,
                   k=1):
        mses = []
        r2s = []
        X_train = X_train[factorlist]
        X_test = X_test[factorlist]
        X_train = X_train if not isPoly else PolynomialFeatures(
            degree=2).fit_transform(X_train)
        X_test = X_test if not isPoly else PolynomialFeatures(
            degree=2).fit_transform(X_test)
        for i in range(k):
            # init the model
            model = self.getModel(model_name)
            # train the model
            model.fit(X_train, y_train)
            # validate the model
            r2, mse, diff = model.validate(X_test, y_test)
            mses.append(mse)
            r2s.append(r2)
        avg_r2 = sum(r2s) / len(r2s)
        avg_mse = sum(mses) / len(mses)
        return model, avg_mse, avg_r2

    def __getCorr(self, xdf, ydf):
        ndf = pd.concat([xdf, ydf], axis=1)
        max_corr = ndf.corr(
        ).loc[[x
               for x in xdf.columns], [x for x in ndf.columns
                                       if x[-1] == "C"]].abs().idxmax()
        factors = {k: [v] for k, v in max_corr.iteritems()}
        return list(factors.keys())[0], list(factors.values())[0]
