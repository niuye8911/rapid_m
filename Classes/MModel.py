import numpy as np
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.preprocessing import PolynomialFeatures
from Utility import *
from sklearn import linear_model
from collections import OrderedDict
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import pandas as pd
from Utility import *
from sklearn.base import clone

CANDIDATE_MODELS = {
    'linear': LinearRegression(),
    'EN': ElasticNetCV(cv=3, max_iter=1000000),
    'lassoCV': LassoCV(cv=3, max_iter=1000000),
    'Bayesian': linear_model.BayesianRidge()
}


class MModel:
    def __init__(self):
        self.models = OrderedDict()
        self.TRAINED = False
        self.mse = -1.
        self.mae = -1.
        self.r2 = -1.
        self.output_loc = ''

    def setX(self, X):
        self.xDF = X

    def setYDict(self, Y):
        self.yDFs = Y

    def setYAll(self, Y):
        self.yDF = Y

    def setYLabel(self, features):
        self.features = features

    def chooseModelAndPoly(self, feature):
        max_r2 = -99
        selected_model = None
        poly = False
        name = ''
        for model_name, model in CANDIDATE_MODELS.items():
            tmp_linear_model = clone(model)
            tmp_poly_model = clone(model)
            tmp_linear_model.fit(self.x_train, self.y_train[feature + '-C'])
            tmp_poly_model.fit(self.x_train_poly, self.y_train[feature + '-C'])

            linear_pred = tmp_linear_model.predict(self.x_test)
            poly_pred = tmp_poly_model.predict(self.x_test_poly)
            r2_linear = r2_score(self.y_test[feature + '-C'], linear_pred)
            r2_poly = r2_score(self.y_test[feature + '-C'], poly_pred)
            if r2_linear > r2_poly and r2_linear > max_r2:
                selected_model = tmp_linear_model
                poly = False
                max_r2 = r2_linear
                name = model_name
            elif r2_poly > r2_linear and r2_poly > max_r2:
                selected_model = tmp_poly_model
                poly = True
                max_r2 = r2_poly
                name = model_name
        return selected_model, poly, name

    def trainSingleFeature(self, feature):
        RAPID_info("TRAINING", feature)
        return self.chooseModelAndPoly(feature)

    def train(self):
        x = self.xDF
        y = self.yDFs
        y_all = self.yDF
        # first get the test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y_all, test_size=0.3, random_state=101)
        self.x_test_poly = PolynomialFeatures(degree=2).fit_transform(
            self.x_test)
        self.x_train_poly = PolynomialFeatures(degree=2).fit_transform(
            self.x_train)
        # train the model for each feature individually
        for feature, values in y.items():
            model, isPoly, model_name = self.trainSingleFeature(feature)
            self.models[feature] = {
                'model': model,
                'isPoly': isPoly,
                'name': model_name
            }
        self.TRAINED = True

    def validate(self):
        debug_file = open('./mmodel_valid.csv', 'w')
        # generate the y_pred for each feature
        y_pred = {}
        for feature, values in self.yDFs.items():
            # check if it's poly
            isPoly = self.models[feature]['isPoly']
            input_feature = self.x_test_poly if isPoly else self.x_test
            y_pred_feature = self.models[feature]['model'].predict(
                input_feature)
            y_pred[feature] = y_pred_feature
        features = y_pred.keys()
        y_pred = pd.DataFrame(data=y_pred)
        # get the CI for feature
        self.getDiffPerFeature(y_pred, self.y_test, features)
        self.mse = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))
        self.mae = metrics.mean_absolute_error(self.y_test, y_pred)
        self.r2 = r2_score(self.y_test, y_pred)
        diffs, avg_diff = self.diffOfTwoMatrix(y_pred, self.y_test)
        self.diff = diffs
        self.avg_diff = avg_diff

    def getDiffPerFeature(self, y_pred, y_test, features):
        diffs = {}
        for feature in features:
            diffs[feature] = (y_pred[feature] - y_test[feature + '-C']) / (
                y_test[feature + '-C'])
        self.diffs = diffs

    def diffOfTwoMatrix(self, y_pred, y_test):
        diffs = []
        feature_diffs = OrderedDict()
        for i in range(0, y_test.shape[0]):
            yPred = y_pred.iloc[i].values
            yTest = y_test.iloc[i].values
            diff = [
                abs((test - pred) / test) if test != 0 else 0
                for pred, test in zip(yPred, yTest)
            ]
            diffs.append(diff)
        feature_diff = np.average(np.matrix(diffs), axis=0)
        average_diff = np.average(feature_diff)
        feature_diff = feature_diff.tolist()[0]
        # arrage the array into a dict
        for i in range(0, len(self.features)):
            feature_diffs[self.features[i]] = feature_diff[i]
        return feature_diffs, average_diff

    def loadFromFile(self, model_file):
        self.model = pickle.load(open(model_file, 'rb'))
        self.TRAINED = True

    def predict(self, two_vecs):
        pred_vec = self.model.predict(two_vecs)
        return pred_vec

    def write_to_file(self, output_prefix):
        # save the model to disk
        self.outfile = OrderedDict()
        for feature, model in self.models.items():
            outfile = output_prefix + '_' + feature + '.pkl'
            pickle.dump(model, open(outfile, 'wb'))
            self.outfile[feature] = outfile

    def dump_into_machine(self, machine):
        machine.model_params['MModel'] = dict()
        machine.model_params['MModel']["mse"] = self.mse
        machine.model_params['MModel']["mae"] = self.mae
        machine.model_params['MModel']["r2"] = self.r2
        machine.model_params['MModel']["avg_diff"] = self.avg_diff
        machine.model_params['MModel']["diff"] = self.diff
        machine.model_params['MModelfile'] = dict()
        for feature, outfile in self.outfile.items():
            machine.model_params['MModelfile'][feature] = outfile
        machine.model_params['MModelPoly'] = dict()
        for feature, outfile in self.outfile.items():
            machine.model_params['MModelPoly'][feature] = self.models[feature][
                'isPoly']
