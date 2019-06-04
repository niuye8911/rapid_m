# FC 2Layer Neural Network

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from models.RapidModel import *
from keras.models import load_model
from sklearn.externals import joblib
import pickle
import numpy as np
import time


class RapidNN(RapidModel):

    REGRESSER = 'regressor'
    SCALER = 'scaler'

    def __init__(self, file_path=''):
        self.input_dim = -1
        RapidModel.__init__(self, 'NN', file_path)
        if file_path == '':
            self.model = None

    def fromFile(self, file_path):
        full_path_scaler = file_path + '_scaler.pkl'
        full_path_regressor = file_path + '_regresser.h5'
        estimator = joblib.load(full_path_scaler)
        estimator.named_steps[RapidNN.REGRESSER].model = load_model(
            full_path_regressor)
        self.model = estimator

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.input_dim = X.shape[1]
            print(self.input_dim)
            time1 = time.time()
            self.model = self.nnTrain(X, Y)
            time2 = time.time()
            return time2 - time1
        return -1
        # note: self.model is now a pipe line

    def predict(self, x):
        ''' predict the result '''
        if self.model is not None:
            return self.model.predict(x)


    def save(self, file_path):
        # save the model
        regresser = self.model.named_steps[RapidNN.REGRESSER]
        regresser.model.save(file_path + '_regresser.h5')
        # save the scaler
        # remove the regresser
        self.model.named_steps[RapidNN.REGRESSER].model = None
        joblib.dump(self.model, file_path + '_scaler.pkl')

    def initNNModel(self):
        model = Sequential()
        # first layer
        model.add(
            Dense(self.input_dim * self.input_dim,
                  input_dim=self.input_dim,
                  kernel_initializer='normal',
                  activation='relu'))
        # second layer
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def nnTrain(self, X, Y):
        # use neural net for training
        seed = 7
        np.random.seed(seed)
        estimators = []
        estimators.append((RapidNN.SCALER, StandardScaler()))
        estimators.append((RapidNN.REGRESSER,
                           KerasRegressor(build_fn=self.initNNModel,
                                          epochs=30,
                                          batch_size=10,
                                          verbose=2)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=10, random_state=seed)
        pipeline.fit(X, Y)
        #scores = pipeline.evaluate(X, Y)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        return pipeline
