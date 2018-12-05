import pickle

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class PModel:
    def __init__(self, model_file):
        self.model_file = model_file
        self.model = None
        self.TRAINED = False

    def setDF(self, dataFrame, feature):
        self.df = dataFrame
        self.features = feature

    def train(self):
        x = self.df[self.features]
        y = self.df['SLOWDOWN']

        x_train, self.x_test, y_train, self.y_test = train_test_split(x, y,
                                                                      test_size=0.4,
                                                                      random_state=101)

        self.model = LinearRegression()
        self.model.fit(x_train, y_train)

        # save the model to disk
        pickle.dump(self.model, open(self.model_file, 'wb'))
        self.TRAINED = True

    def validate(self):
        # load the model from disk
        if not self.TRAINED:
            self.model = self.loadFromFile(self.model_file, 'rb')
        y_pred = self.model.predict(self.x_test)
        self.mse = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))
        self.r2 = r2_score(self.y_test, y_pred)
        return self.mse, self.r2

    def loadFromFile(self, model_file):
        self.model = pickle.load(open(model_file, 'rb'))
        self.TRAINED = True

    def predict(self, system_profile):
        if not self.TRAINED:
            self.model = self.loadFromFile(self.model_file, 'rb')
        pred_slowdown = self.model.predict(system_profile)[0]
        return pred_slowdown

    def write_pmodel_info(self, app, name):
        app.model_params[name] = dict()
        app.model_params[name]["file"] = self.model_file
        app.model_params[name]["mse"] = self.mse
        app.model_params[name]["r2"] = self.r2
