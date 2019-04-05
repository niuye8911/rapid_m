import numpy as np
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

from Utility import *
from sklearn import linear_model


class PModel:
    def __init__(self):
        self.model = None
        self.TRAINED = False
        self.mse = -1.
        self.mae = -1.
        self.r2 = -1.
        self.output_loc = ''

    def setDF(self, dataFrame, feature):
        self.df = dataFrame
        self.features = feature

    def train(self):
        x = self.df[self.features]
        y = self.df['SLOWDOWN']
        x_train, self.x_test, y_train, self.y_test = train_test_split(
            x, y, test_size=0.3, random_state=101)
        RAPID_info("TRAINED", x_train.shape[0])
        #self.model = LinearRegression()
        self.model = linear_model.Lasso(alpha=0.1, max_iter=100000)
        #self.model = linear_model.BayesianRidge()
        x_train_poly = PolynomialFeatures(degree=2).fit_transform(x_train)
        self.x_test_poly = PolynomialFeatures(degree=2).fit_transform(
            self.x_test)
        self.model.fit(x_train_poly, y_train)
        self.TRAINED = True

    def validate(self):
        self.y_pred = self.model.predict(self.x_test_poly)
        self.mse = np.sqrt(metrics.mean_squared_error(self.y_test, self.y_pred))
        self.mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)
        # relative error
        self.diffs = abs(self.y_test - self.y_pred) / self.y_test
        self.diff = sum(self.diffs) / len(self.diffs)
        return self.diff, self.r2

    def loadFromFile(self, model_file):
        self.model = pickle.load(open(model_file, 'rb'))
        self.TRAINED = True

    def predict(self, system_profile):
        pred_slowdown = self.model.predict(system_profile)[0]
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

    def drawPrediction(self, output):
        predictions = self.y_pred
        observations = self.y_test
        normed_pred = (predictions - min(observations)) / (
            max(observations) - min(observations))
        normed_obs = (observations - min(observations)) / (
            max(observations) - min(observations))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('SlowDown Observation')
        plt.ylabel('Prediction')
        # plot the base line
        x = [0, 1]
        y = [0, 1]
        plt.plot(x, y, 'r-')
        plt.plot(normed_obs, normed_pred, 'x', color='black')
        plt.savefig(output)

    def printPrediction(self,outfile):
        result = pd.DataFrame({'GT':self.y_test, 'Pred':self.y_pred})
        overall = pd.concat([self.x_test, result],axis=1)
        overall.to_csv(outfile)
