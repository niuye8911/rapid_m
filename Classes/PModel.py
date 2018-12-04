import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class PModel:
    def __init__(self, csv_file, model_file):
        self.csv_file = csv_file
        self.model_file = model_file
        self.model = None
        self.TRAINED = False

    def train(self):
        data = pd.read_csv(self.csv_file)
        x = data[
            ['ACYC', 'AFREQ', 'C0res%', 'C10res%', 'C1res%', 'C2res%', 'C3res%',
             'C4res%', 'C5res%', 'C6res%', 'C7res%', 'C8res%', 'C9res%', 'EXEC',
             'FREQ', 'INST', 'INSTnom', 'INSTnom%', 'IPC', 'L2HIT', 'L2MISS',
             'L2MPI', 'L3HIT', 'L3MISS', 'L3MPI', 'PhysIPC', 'PhysIPC%',
             'Proc Energy (Joules)', 'READ', 'TIME(ticks)', 'WRITE']]
        y = data['Slowdown']

        x_train, self.x_test, y_train, self.y_test = train_test_split(x, y,
                                                                      test_size=0.4,
                                                                      random_state=101)
        model = LinearRegression()
        model.fit(x_train, y_train)

        # save the model to disk
        pickle.dump(model, open(self.model_file, 'wb'))

        self.TRAINED = True

    def validate(self):
        # load the model from disk
        if not self.TRAINED:
            self.model = self.loadFromFile(self.model_file, 'rb')
        y_pred = self.model.predict(self.x_test)
        return np.sqrt(metrics.mean_squared_error(self.y_test, y_pred)), \
               r2_score(self.y_test, y_pred)

    def loadFromFile(self, model_file):
        self.model = pickle.load(open(model_file, 'rb'))
        self.TRAINED = True

    def predict(self, system_profile):
        if not self.TRAINED:
            self.model = self.loadFromFile(self.model_file, 'rb')
        pred_slowdown = self.model.predict(system_profile)[0]
        return pred_slowdown
