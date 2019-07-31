import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split

from Utility import *
from DataUtil import *
from models.ModelPool import *


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
        self.maxes = {}
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
                    model_params['Meta'][feature]['name'],
                    'features':
                    model_params['Meta'][feature]['features']
                }
            self.maxes = mmodel['maxes']
            self.TRAINED = True
        return

    def dump_into_machine(self, machine):
        # write the features
        machine.features = self.features
        machine.model_params['Metric'] = OrderedDict()
        machine.model_params['Meta'] = OrderedDict()
        machine.maxes = self.maxes
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
                'features':
                self.models[feature]['features'],
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

    def getModel(self, feature, TEST=False):
        # first pass: select model
        model, isPoly, training_time = self.modelPool.selectModel(
            self.x_train, self.x_test, self.y_train[feature + '-C'],
            self.y_test[feature + '-C'], TEST)
        print("selected Model:", model.name, "ispoly:", isPoly)
        # second pass: select feature
        model, min_features = self.modelPool.selectFeature(
            self.xDF_scaled, self.yDF[feature + '-C'], self.x_train,
            self.x_test, self.y_train[feature + '-C'],
            self.y_test[feature + '-C'], model.name, isPoly)
        # third phase: finalize model
        final_model = self.modelPool.getModel(model.name)
        x = self.x_train[min_features]
        x_test = self.x_test[min_features]
        if isPoly:
            x = PolynomialFeatures(degree=2).fit_transform(x)
            x_test = PolynomialFeatures(degree=2).fit_transform(x_test)
        elapsed_time = final_model.fit(x, self.y_train[feature + '-C'])
        r2, mse, diff = final_model.validate(x_test,
                                             self.y_test[feature + '-C'])
        training_time['final'] = {
            'time': elapsed_time,
            'r2': r2,
            'mse': mse,
            'diff': diff
        }
        return model, min_features, isPoly, training_time

    def trainSingleFeature(self, feature, TEST=False):
        RAPID_info("TRAINING", feature)
        return self.getModel(feature, TEST)

    def preprocess(self, X):
        ''' scale the data '''
        for col in X.columns:
            # take the maximum number of two vectors per feature
            if col[-1] == "C":
                continue
            if col[:-2] not in self.maxes:
                self.maxes[col[:-2]] = X.max()[col]
            if X.max()[col] > self.maxes[col[:-2]]:
                self.maxes[col[:-2]] = X.max()[col]
        scaled_X = pd.DataFrame()
        for col in X.columns:
            if col[-1] == 'C':
                scaled_X[col] = X[col]
            scaled_X[col] = X[col] / self.maxes[col[:-2]]
        return scaled_X

    def train(self, TEST=False):
        x = self.preprocess(self.xDF)
        self.xDF_scaled = x
        y = self.yDFs
        y_all = self.yDF
        # first get the test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y_all, test_size=0.3, random_state=101)
        # train the model for each feature individually
        debug_info = OrderedDict()
        for feature, values in y.items():
            model, min_features, isPoly, training_time = self.trainSingleFeature(
                feature, TEST)
            self.models[feature] = {
                'model': model,
                'isPoly': isPoly,
                'name': model.name,
                'features': min_features
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
            x_test = self.x_test[self.models[feature]['features']]
            if isPoly:
                x_test = PolynomialFeatures(degree=2).fit_transform(x_test)
            y_pred_feature = self.models[feature]['model'].predict(x_test)
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

    def validate_batch(self):
        debug_file = open('./mmodel_valid.csv', 'w')
        id1_s = list(map(lambda x: x+'-1', self.features))
        id1_s = list(map(lambda x: x+'-2', self.features))
        load1 = self.x_test[id1_s]
        load2 = self.x_test[id2_s]
        y_pred = self.predict_batch(self, load1, load2)
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

    def predict_batch(self, load1, load2):
        # clean up the data
        load1 = formatEnv_df(load1, self.features, '-1', REMOVE_POSTFIX=False)
        load2 = formatEnv_df(load2, self.features, '-2', REMOVE_POSTFIX=False)
        x = reformat_dfs(load1, load2)
        # scale X
        scaled_x = pd.DataFrame()
        for col in x.columns:
            scaled_x[col] = x[col] / self.maxes[col[:-2]]
        y_pred = OrderedDict()
        for feature in self.features:
            # check if it's poly
            isPoly = self.models[feature]['isPoly']
            # filter out input
            x_input = scaled_x[self.models[feature]['features']]
            if isPoly:
                x_input = PolynomialFeatures(degree=2).fit_transform(x_input)
            y_pred_feature = self.models[feature]['model'].predict(x_input)
            y_pred[feature] = y_pred_feature
        return pd.DataFrame(data=y_pred)

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
        vec1 = self.__scaleInput(vec1)
        vec2 = self.__scaleInput(vec2)
        # swap to format the vec
        for i in range(0, len(vec1)):
            smaller = min(vec1[i], vec2[i])
            bigger = max(vec1[i], vec2[i])
            vec[i] = smaller
            vec[i + len(vec1)] = bigger
        # predict per-feature
        pred = OrderedDict()
        features = self.features
        id = 0
        for feature in features:
            model = self.models[feature]['model']
            active_features = self.models[feature]['features']
            # filter the input
            input = [self.__filterInput(vec, active_features)]
            if self.models[feature]['isPoly']:
                input = PolynomialFeatures(degree=2).fit_transform(input)
            combined_feature = model.predict(input)
            # small fix: if predicted is smaller than 0, use the maximum one
            if combined_feature < 0:
                combined_feature = vec2[id]
            pred[feature] = combined_feature
            id += 1
        return pd.DataFrame(data=pred)

    def __filterInput(self, vec, active_features):
        all_features = list(map(lambda f: f + '-1', self.features)) + list(
            map(lambda f: f + '-2', self.features))
        feature_ids = list(
            map(lambda f: all_features.index(f), active_features))
        filtered = list(map(lambda id: vec[id], feature_ids))
        return filtered

    def __scaleInput(self, vec):
        id = 0
        output = []
        for feature in self.features:
            raw_value = vec[id]
            scaled_value = raw_value / self.maxes[feature]
            output.append(scaled_value)
            id += 1
        return output

    def write_to_file(self, output_prefix):
        # save the model to disk
        self.outfile = OrderedDict()
        for feature in self.features:
            model = self.models[feature]['model']
            outfile = output_prefix + '_' + feature
            model.save(outfile)
            self.outfile[feature] = outfile
