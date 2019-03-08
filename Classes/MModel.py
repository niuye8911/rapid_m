import numpy as np
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from Utility import *


class MModel:
    def __init__(self):
        self.model = None
        self.TRAINED = False
        self.mse = -1.
        self.mae = -1.
        self.r2 = -1.
        self.output_loc = ''

    def setX(self, X):
        self.xDF = X

    def setY(self, Y):
        self.yDF = Y

    def train(self):
        x = self.xDF
        y = self.yDF

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
        diffs = self.diffOfTwoMatrix(y_pred, self.y_test)
        self.diff = sum(diffs) / len(diffs)

    def diffOfTwoMatrix(self, y_pred, y_test):
        diffs = []
        for i in range(0, y_test.shape[0]):
            yPred = y_pred[i]
            yTest = y_test.iloc[i].values
            diff = [
                abs((test - pred) / test) if test != 0 else 0
                for pred, test in zip(yPred, yTest)
            ]
            diffs.append(sum(diff) / len(diff))
        print(sorted(diffs))
        return diffs

    def loadFromFile(self, model_file):
        self.model = pickle.load(open(model_file, 'rb'))
        self.TRAINED = True

    def predict(self, two_vecs):
        pred_vec = self.model.predict(two_vecs)
        return pred_vec

    def write_to_file(self, output):
        # save the model to disk
        pickle.dump(self.model, open(output, 'wb'))
        self.output_loc = output

    def dump_into_machine(self, machine):
        machine.model_params['MModel'] = dict()
        machine.model_params['MModel']["file"] = self.output_loc
        machine.model_params['MModel']["mse"] = self.mse
        machine.model_params['MModel']["mae"] = self.mae
        machine.model_params['MModel']["r2"] = self.r2
        machine.model_params['MModel']["diff"] = self.diff
