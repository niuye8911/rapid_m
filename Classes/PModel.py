import numpy as np
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from Utility import *


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

        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        self.TRAINED = True

    def validate(self):
        y_pred = self.model.predict(self.x_test)
        self.mse = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))
        self.mae = metrics.mean_absolute_error(self.y_test, y_pred)
        self.r2 = r2_score(self.y_test, y_pred)
        # relative error
        diff = abs(self.y_test - y_pred) / self.y_test
        self.diff = sum(diff) / len(diff)
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
