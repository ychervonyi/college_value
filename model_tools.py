from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.models import model_from_json
from sklearn import linear_model
from sklearn.externals import joblib

from data_tools import get_data
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)


def compute_college_scores(model_type, x, y):
    # Load student model
    print("Loading model...")
    student_model = Model(model_type=model_type, model_name='student')
    student_model.load()

    assert student_model is not None, "Can't load a model"

    # Actual earnings
    y_raw = y

    # Predicted earnings
    y_predicted = student_model.predict(x)

    # College score
    college_score = np.divide(y_raw, y_predicted)
    return college_score


def train_model(x, y, feature_names, model_name, batch=16, n_epochs=300,
                learning_rate=0.1, model_type='sklearn', save=False):
    # Make test and train set
    train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=0)

    params = {'batch': batch, 'epochs': n_epochs, 'learning_rate': learning_rate}
    print("Training %s model..." % model_name)
    model = Model(model_type=model_type, model_name=model_name, input_shape=train_X.shape[1], params=params)
    model.create()
    model.train(train_X, train_y, test_X, test_y)
    if save:
        model.save()

    predictions = model.predict(test_X)
    r_squared = r2_score(test_y, predictions)

    model.print()

    print("R squared: %.4f" % r_squared)
    print("Features: %s" % feature_names)
    print("Number of examples: %d" % len(y))


class Model(object):
    def __init__(self, model_type, model_name, input_shape=None, params=None):
        assert model_type in ('sklearn', 'keras')
        self._type = model_type
        self._input_shape = input_shape
        self._model_name = model_name
        self._model = None

        self._params = params

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def type(self):
        return self._type

    def create(self):
        if self._type == 'keras':
            model = Sequential()
            model.add(Dense(1, input_shape=(self._input_shape,),
                            kernel_initializer='uniform',
                            activation='linear',
                            # kernel_regularizer='l2',
                            use_bias=False))
            model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=self._params['learning_rate']))
            self._model = model
        elif self._type == 'sklearn':
            model = linear_model.Ridge()
            self._model = model

    def load(self):
        if self._type == 'keras':
            json_file = open('%s.json' % self._model_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self._model = model_from_json(loaded_model_json)
            # load weights into new model
            self._model.load_weights("%s.h5" % self._model_name)
            # need to compile, otherwise it returns nonsense (such as negative values)
            self._model.compile(loss='mse', optimizer='adam')
            self._input_shape = self.get_n_coefs()
        elif self._type == 'sklearn':
            self._model = joblib.load('%s.pkl' % self._model_name)
            self._input_shape = self.get_n_coefs()
        print("Loaded %s model from disk" % self._model_name)

    def train(self, train_X, train_y, test_X, test_y):
        if self._type == 'keras':
            history = self._model.fit(train_X, train_y,
                                      batch_size=self._params['batch'],
                                      epochs=self._params['epochs'],
                                      verbose=1,
                                      validation_data=(test_X, test_y)
                                      )

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.draw()

            # Evaluate and print MSE
            score = self._model.evaluate(test_X, test_y, verbose=0)
            print("Test loss: %.4f" % score)
        elif self._type == 'sklearn':
            self._model.fit(train_X, train_y)
        print("%s model trained" % self._model_name)

    def get_n_coefs(self):
        n_coefs = None
        if self._type == 'keras':
            n_coefs = self._model.get_weights()
        elif self._type == 'sklearn':
            n_coefs = self._model.coef_.shape[1]
        return n_coefs

    def save(self):
        if self._type == 'keras':
            # serialize model to JSON
            model_json = self._model.to_json()
            with open("%s.json" % self._model_name, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self._model.save_weights("%s.h5" % self._model_name)
        elif self._type == 'sklearn':
            joblib.dump(self._model, '%s.pkl' % self._model_name)
        print("Model saved to %s" % self._model_name)

    def predict(self, x):
        return self._model.predict(x)

    def print(self):
        print("============================================")
        if self._type == 'keras':
            print("Model weights: %s" % self._model.get_weights())
            print(self._model.summary())
            print("Params: %s" % self._params)
        elif self._type == 'sklearn':
            print('Coefficients: \n', self._model.coef_)
        print("============================================")


