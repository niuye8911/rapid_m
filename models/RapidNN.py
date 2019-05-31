# FC 2Layer Neural Network

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from models.RapidModel import *
import pickle
import numpy as np


class RapidNN(RapidModel):

    REGRESSER = 'regressor'
    SCALER = 'scaler'

    def __init__(self, name='NN', file_path=''):
        self.input_dim = -1
        RapidModel.__init__(self, name, file_path)
        if file_path == '':
            self.model = None

    def fromFile(self, file_path):
        pass

    def fit(self, X, Y):
        ''' train the model '''
        if self.model is None:
            self.input_dim = X.shape[1]
            self.model = self.nnTrain(X, Y)
        # note: self.model is now a pipe line

    def predict(self, x):
        ''' predict the result '''
        if self.model is not None:
            return self.model.predict(x)

    def validate(self, X, Y):
        ''' validate the trained model '''
        if self.model is None:
            return -1
        pred = self.model.predict(X)
        r2 = r2_score(Y, pred)
        return r2

    def save(self, file_path):
        # save the model
        regresser = self.model.named_steps[RapidNN.REGRESSER]
        regresser.model.save(file_path + '_regresser.h5')
        # save the scaler
        # remove the regresser
        self.model.named_steps[RapidNN.REGRESSER] = None
        pickle.dump(self.model, open(file_path + '_scaler.pkl', 'wb'))

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
                                          epochs=100,
                                          batch_size=5,
                                          verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=10, random_state=seed)
        pipeline.fit(X, Y)
        #scores = pipeline.evaluate(X, Y)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        return pipeline
