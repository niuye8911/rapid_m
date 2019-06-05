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
import json
from Utility import *
from sklearn.base import clone
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from models.ModelPool import *

FEATURES = [
    'AFREQ', 'EXEC', 'FREQ', 'INST', 'INSTnom', 'IPC', 'L2HIT', 'L2MPI',
    'L3HIT', 'PhysIPC%', 'MEM'
]


class MModel:
    def __init__(self, file_loc=""):
        self.models = OrderedDict()
        self.TRAINED = False
        self.mse = -1.
        self.mae = -1.
        self.r2 = -1.
        self.output_loc = ''
        self.features = []
        self.modelPool = ModelPool()
        if file_loc != "":
            self.loadFromFile(file_loc)

    def loadFromFile(self, model_file):
        # TODO, load model
        with open(model_file, 'r') as file:
            mmodel = json.load(file)
            # host name
            self.host_name = mmodel['host_name']
            if not mmodel['TRAINED']:
                return
            model_params = mmodel['model_params']
            # features
            self.features = mmodel['features']
            for feature in self.features:
                file_loc = model_params['Meta'][feature]['filepath']
                self.models[feature] = {
                    'model':
                    self.modelPool.getModel(
                        model_params['Meta'][feature]['name'], file_loc),
                    'isPoly':
                    model_params['Meta'][feature]['isPoly'],
                    'name':
                    model_params['Meta'][feature]['name']
                }
            self.TRAINED = True
        return

    def dump_into_machine(self, machine):
        # write the features
        machine.features = self.features
        machine.model_params['Metric'] = OrderedDict()
        machine.model_params['Meta'] = OrderedDict()
        machine.model_params['Metric']["avg_diff"] = self.avg_diff
        for feature in self.features:
            # write the metric
            machine.model_params['Metric'][feature] = self.diff[feature]
            # write the metadata
            machine.model_params['Meta'][feature] = OrderedDict({
                'name':
                self.models[feature]['name'],
                'isPoly':
                self.models[feature]['isPoly'],
                'filepath':
                self.outfile[feature]
            })

    def setX(self, X):
        self.xDF = X

    def setYDict(self, Y):
        self.yDFs = Y

    def setYAll(self, Y):
        self.yDF = Y

    def setYLabel(self, features):
        self.features = list(map(lambda x: x[:-2], features))

    def getModel(self,feature, TEST=False):
        return self.modelPool.selectModel(self.x_train, self.y_train[feature+'-C'], self.x_test, self.y_test[feature+'-C'], TEST)

    def chooseModelAndPoly(self, feature, TEST=False):
        ''' if TEST is set to True, then use all models '''
        min_diff = 99
        selected_model = None
        poly = False
        training_time = OrderedDict()
        for model_name in self.modelPool.CANDIDATE_MODELS:
            if model_name is not 'NN':
                tmp_linear_model = self.modelPool.getModel(model_name)
                tmp_poly_model = self.modelPool.getModel(model_name)

                time_linear = tmp_linear_model.fit(
                    self.x_train, self.y_train[feature + '-C'])
                time_high = tmp_poly_model.fit(self.x_train_poly,
                                               self.y_train[feature + '-C'])
                r2_linear, mse_linear, diff_linear = tmp_linear_model.validate(
                    self.x_test, self.y_test[feature + '-C'])
                r2_poly, mse_poly, diff_poly = tmp_poly_model.validate(
                    self.x_test_poly, self.y_test[feature + '-C'])
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
                if diff_linear < diff_poly and diff_linear < min_diff:
                    selected_model = tmp_linear_model
                    poly = False
                    min_diff = diff_linear
                elif diff_poly > diff_linear and diff_poly > min_diff:
                    selected_model = tmp_poly_model
                    poly = True
                    min_diff = diff_poly
            else:
                # if accuracy is enough, skip NN
                if min_diff < 10 and not TEST:
                    RAPID_info('M-Model', "Accuracy Reached, no need for NN")
                    continue
                # NN does not need to check high order
                nn_model = self.modelPool.getModel('NN')
                time_nn = nn_model.fit(self.x_train,
                                       self.y_train[feature + '-C'])
                r2, mse, diff = nn_model.validate(self.x_test,
                                                  self.y_test[feature + '-C'])
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

    def nnTrain(self, X, Y):
        # use neural net for training
        seed = 7
        np.random.seed(seed)
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp',
                           KerasRegressor(build_fn=self.initNNModel,
                                          epochs=30,
                                          batch_size=5,
                                          verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=10, random_state=seed)
        pipeline.fit(X, Y)
        scores = pipeline.evaluate(X, Y)
        return scores
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    def initNNModel(self):
        model = Sequential()
        model.add(
            Dense(40,
                  input_dim=20,
                  kernel_initializer='normal',
                  activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def trainSingleFeature(self, feature, TEST=False):
        RAPID_info("TRAINING", feature)
        return self.getModel(feature, TEST)

    def train(self, TEST=False):
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
        debug_info = OrderedDict()
        for feature, values in y.items():
            model, isPoly, training_time = self.trainSingleFeature(
                feature, TEST)
            self.models[feature] = {
                'model': model,
                'isPoly': isPoly,
                'name': model.name
            }
            debug_info[feature] = training_time
        printTrainingInfo(debug_info)
        self.TRAINED = True

    def validate(self):
        debug_file = open('./mmodel_valid.csv', 'w')
        # generate the y_pred for each feature
        y_pred = OrderedDict()
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
            pred = y_pred[feature].values.tolist()
            test = y_test[feature + '-C'].values.tolist()
            diffs[feature] = [(p - t) / t for p, t in zip(pred, test)]
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

    def predict(self, vec1, vec2):
        if len(vec1) != len(vec2):
            RAPID_warn('M-Model', "two vecs with different lengths")
        # assemble the two vecs into a single vec
        try:
            vec = list(vec1) + list(vec2)
        except:
            RAPID_warn(
                'M-Model',
                "Caught Unexpected Exception, vec1 and vec2 cannot be combined"
            )
            print('vec1', vec1)
            print('vec2', vec2)
            exit(1)
        # format the vec
        for i in range(0, len(vec1)):
            smaller = min(vec1[i], vec2[i])
            bigger = max(vec1[i], vec2[i])
            vec[i] = smaller
            vec[i + len(vec1)] = bigger
        # predict per-feature
        vec = [vec]
        vec_poly = PolynomialFeatures(degree=2).fit_transform(vec)
        pred = OrderedDict()
        features = self.features
        id = 0
        for feature in features:
            model = self.models[feature]['model']
            input = vec
            if self.models[feature]['isPoly']:
                input = vec_poly
            combined_feature = model.predict(input)
            if combined_feature < 0:
                combined_feature = (vec1[id] + vec2[id]) / 2.0
            pred[feature] = combined_feature
            id += 1
        return pd.DataFrame(data=pred)

    def write_to_file(self, output_prefix):
        # save the model to disk
        self.outfile = OrderedDict()
        for feature in self.features:
            model = self.models[feature]['model']
            outfile = output_prefix + '_' + feature
            model.save(outfile)
            self.outfile[feature] = outfile
