import numpy as np
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from Utility import *
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.base import clone
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class PModel:
    CANDIDATE_MODELS = {
        'linear': LinearRegression(),
        'EN': ElasticNetCV(cv=3, max_iter=1000000),
        'lassoCV': LassoCV(cv=3, max_iter=1000000),
        'Bayesian': linear_model.BayesianRidge()
    }

    def __init__(self, p_info=''):
        self.model = None
        self.TRAINED = False
        self.mse = -1.
        self.mae = -1.
        self.r2 = -1.
        self.output_loc = ''
        if p_info is not '':
            # read from a file
            self.fromInfo(p_info)

    def fromInfo(self, info):
        self.model = pickle.load(open(info['file'], 'rb'))
        self.TRAINED = True
        self.polyFeature = info['poly']
        self.model_type = info['model_type']
        self.feature = info['feature']

    def setDF(self, dataFrame, feature):
        self.df = dataFrame
        self.features = feature

    def train(self):
        x = self.df[self.features]
        y = self.df['SLOWDOWN']
        x_train, self.x_test, y_train, self.y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        RAPID_info("TRAINED", x_train.shape[0])
        x_train_poly = PolynomialFeatures(degree=2).fit_transform(x_train)
        self.x_test_poly = PolynomialFeatures(degree=2).fit_transform(
            self.x_test)
        # select the model and features
        #self.selectModelAndFeature(x_train, y_train)
        self.nnTrain(x,y)
        self.TRAINED = True

    def nnTrain(self, X, Y):
        # use neural net for training
        seed = 7
        np.random.seed(seed)
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(build_fn=self.initNNModel, epochs=100, batch_size=5, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=10, random_state=seed)
        pipeline.fit(X,Y)
        scores = pipeline.evaluate(X,Y)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        #results = cross_val_score(pipeline, X,Y,cv=kfold)
        #print(results)
        #print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    def initNNModel(self):
        model = Sequential()
        model.add(
            Dense(12,
                  input_dim=11,
                  kernel_initializer='normal',
                  activation='relu'))
        model.add(Dense(1,kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def selectModelAndFeature(self, x_train, y_train):
        # use the validate process to pick the most important 10 linear features
        # scale the data
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scalar = min_max_scaler.fit(x_train)
        x_train_scaled = scalar.transform(x_train)
        # iterate through all models
        selected_features = []
        selected_poly = False
        min_mse = 99999
        selected_model_name = ''
        for model_name, model in PModel.CANDIDATE_MODELS.items():
            rfe = RFE(model, 10)
            fit = rfe.fit(x_train_scaled, y_train)
            # get the feature names:
            feature_names = list(
                filter(lambda f: fit.ranking_[self.features.index(f)] == 1,
                       self.features))
            selected_x_train = x_train[feature_names]
            selected_x_train_poly = PolynomialFeatures(
                degree=2).fit_transform(selected_x_train)
            selected_x_test_poly = PolynomialFeatures(degree=2).fit_transform(
                self.x_test[feature_names])
            # train 1st order
            linear_model = clone(model)
            linear_model.fit(selected_x_train, y_train)
            diff, mse = self.validate(self.x_test[feature_names], linear_model)
            if mse < min_mse:
                selected_features = feature_names
                selected_poly = False
                selected_model_name = model_name
                min_mse = mse
                self.model = linear_model
            # train the 2nd order
            high_model = clone(model)
            high_model.fit(selected_x_train_poly, y_train)
            diff, mse = self.validate(selected_x_test_poly, high_model)
            if mse < min_mse:
                selected_poly = True
                min_mse = mse
                self.model = high_model


# set all members
        self.polyFeature = selected_poly
        self.features = selected_features
        self.modelType = selected_model_name
        if selected_poly:
            self.x_test_selected = selected_x_test_poly
        else:
            self.x_test_selected = self.x_test[selected_features]

    def validate(self, x=None, model=None):
        if x is None:
            x = self.x_test_selected
        if model is None:
            model = self.model
        self.y_pred = model.predict(x)
        self.mse = np.sqrt(metrics.mean_squared_error(self.y_test,
                                                      self.y_pred))
        self.mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)
        # relative error
        self.diffs = list(abs(self.y_test - self.y_pred) / self.y_test)
        self.diff = sum(self.diffs) / len(self.diffs)
        return self.diff, self.mse

    def loadFromFile(self, model_file):
        self.model = pickle.load(open(model_file, 'rb'))
        self.TRAINED = True

    def formulate_env(self, env):
        ''' given a df (env), filtered out the unwanted feature and get poly '''
        input = env[self.feature]
        if self.polyFeature:
            input = PolynomialFeatures(degree=2).fit_transform(input)
        return input

    def predict(self, system_profile):
        ''' the input is a df with all features '''
        pred_slowdown = self.model.predict(self.formulate_env(system_profile))
        return pred_slowdown

    def write_to_file(self, output):
        # save the model to disk
        pickle.dump(self.model, open(output, 'wb'))
        self.output_loc = output

    def dump_into_app(self, app, name):
        app.model_params[name] = dict()
        app.model_params[name]["file"] = self.output_loc
        app.model_params[name]["mse"] = self.mse
        app.model_params[name]["mae"] = self.mae
        app.model_params[name]["diff"] = self.diff
        app.model_params[name]["r2"] = self.r2
        app.model_params[name]["feature"] = self.features
        app.model_params[name]["poly"] = self.polyFeature
        app.model_params[name]["model_type"] = self.modelType

    def drawPrediction(self, output):
        predictions = self.y_pred
        observations = self.y_test
        normed_pred = (predictions - min(observations)) / (max(observations) -
                                                           min(observations))
        normed_obs = (observations - min(observations)) / (max(observations) -
                                                           min(observations))
        # plot the base line
        x = [0, 1]
        y = [0, 1]
        plt.plot(x, y, 'r-')
        plt.plot(normed_obs, normed_pred, 'x', color='black')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('SlowDown Observation')
        plt.ylabel('Prediction')
        plt.savefig(output)

    def printPrediction(self, outfile):
        result = pd.DataFrame({'GT': self.y_test, 'Pred': self.y_pred})
        overall = pd.concat([self.x_test, result], axis=1)
        overall.to_csv(outfile)
