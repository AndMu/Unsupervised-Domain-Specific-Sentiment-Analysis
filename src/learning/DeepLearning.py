import abc
import os
from os import path, makedirs

import gc
from keras import backend as k
from keras import callbacks
from keras.layers import Embedding, LSTM, Dropout, Dense, np, Flatten, Activation, \
    Reshape, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.regularizers import l2
from pathlib2 import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

import utilities.Constants as Constants
from learning import logger
from utilities.Utilities import Utilities

seed = 7
np.random.seed(seed)


class BaseDeepStrategy(object):
    __metaclass__ = abc.ABCMeta

    counter = 0

    def __init__(self, project_name, sub_project, time_steps):
        self.counter = BaseDeepStrategy.counter
        BaseDeepStrategy.counter += 1
        self.time_steps = time_steps
        self.project_name = project_name + "_" + sub_project
        logger.info('%s with timestamp %i', project_name, time_steps)
        self.project_path = path.join(Constants.TEMP, 'Deep', project_name)
        self.epochs_number = 20
        self.model = None

    def add_embeddings(self, model):
        vectors = self.loader.parser.word2vec.embedding_matrix
        model.add(Embedding(input_dim=vectors.shape[0],
                          output_dim=vectors.shape[1],
                          input_length=self.time_steps,
                          weights=[vectors],
                          trainable=False))

    def add_output(self, model):
        model.add(Dropout(0.5))
        total_classes = self.loader.convertor.total_classes()
        if self.loader.convertor.is_binary():
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
        else:
            model.add(Dense(total_classes))
            model.add(Activation('softmax'))

    def init_mode(self):
        if not hasattr(self, 'model') or self.model is None:
            self.model = self.construct_model()
            self.model.summary()
            if self.loader.convertor.is_binary():
                self.model.compile(loss="binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
            else:
                self.model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

    def get_file_name(self, name):
        file_path = path.join(self.project_path, 'weights', name, str(self.time_steps) + '_keras-lstm.h5')
        weights = path.dirname(file_path)
        if not Path(weights).exists():
            makedirs(weights)
        return file_path

    def load(self, name):
        self.init_mode()
        cache_file_name = self.get_file_name(name)
        if path.exists(cache_file_name):
            logger.info('Loading weights [%s]...', cache_file_name)
            self.model.load_weights(cache_file_name)
        else:
            logger.info('Weights file not found - [%s]...', cache_file_name)

    def save(self, name):
        cache_file_name = self.get_file_name(name)
        logger.info('Saving weights [%s]...', cache_file_name)
        self.model.save_weights(cache_file_name)

    def delete_weights(self, name):
        cache_file_name = self.get_file_name(name)
        if Path(cache_file_name).exists():
            logger.info('Deleting weight %s', cache_file_name)
            os.remove(cache_file_name)

    def test(self, test_x, test_y):
        logger.info('Testing with %i records', len(test_x))
        self.init_mode()
        loss, acc = self.model.evaluate(test_x, test_y, Constants.TESTING_BATCH)
        logger.info('Test loss / Test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    def predict_proba(self, test_x):
        logger.info('Predict with %i records', len(test_x))
        self.init_mode()
        y = self.model.predict_proba(test_x, Constants.TESTING_BATCH)
        return y

    def predict(self, test_x):
        logger.info('Predict with %i records', len(test_x))
        self.init_mode()
        y = self.model.predict(test_x, Constants.TESTING_BATCH)
        return y

    def test_predict(self, test_x, test_y):
        logger.info('Test predict_proba with %i records', len(test_x))
        self.init_mode()
        result_y_prob = self.model.predict_proba(test_x, batch_size=Constants.TESTING_BATCH, verbose=1)
        result_y = Utilities.make_single_dimension(result_y_prob)
        result_y_prob_single = self.loader.convertor.make_single(result_y_prob)

        Utilities.measure_performance(test_y, result_y)
        Utilities.measure_performance_auc(test_y, result_y, result_y_prob)

        return result_y, result_y_prob_single

    def get_classes(self, y):
        if self.loader.convertor.is_binary():
            return np.unique(check_array(y, ensure_2d=False, allow_nd=True))
        else:
            single = Utilities.make_single_dimension(y)
            return np.unique(check_array(single, ensure_2d=False, allow_nd=True))

    def fit(self, train_x, train_y):
        logger.info('Training with %i records', len(train_x))
        self.init_mode()
        Utilities.count_occurences(train_y)
        cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=2)]
        self.model.fit(train_x, train_y, batch_size=Constants.TRAINING_BATCH, callbacks=cbks, epochs=self.epochs_number,
                   validation_split=0.25, shuffle=True)

    def construct_model(self):
        model = Sequential()
        return model

    @abc.abstractmethod
    def copy(self):
        return None


class ConvNetAlgorithm(BaseDeepStrategy):

    def __init__(self, loader, project_name, time_steps, top_words=10000):
        self.time_steps = time_steps
        self.top_words = top_words
        self.loader = loader
        self.word_vector_size = self.loader.parser.word2vec.vector_size
        super(ConvNetAlgorithm, self).__init__(project_name, self.get_name(), time_steps)

    def get_name(self):
        return self.loader.parser.word2vec.name + '_OneDimensional'

    def construct_model(self):

        if not self.loader.convertor.is_binary():
            raise StandardError("This model support dual data only")

        k.set_image_data_format('channels_first')
        # Number of feature maps (outputs of convolutional layer)
        n_fm = 500
        # kernel size of convolutional layer
        kernel_size = 8
        conv_input_height = self.time_steps
        conv_input_width = self.word_vector_size

        logger.info('conv_input_height: %s conv_input_width: %s', conv_input_height, conv_input_width)

        model = Sequential()
        # Embedding layer (lookup table of trainable word vectors)
        self.add_embeddings(model)
        # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
        model.add(Reshape((1, conv_input_height, conv_input_width)))

        # first convolutional layer
        model.add(Convolution2D(n_fm,
                                kernel_size,
                                conv_input_width,
                                border_mode='valid',
                                W_regularizer=l2(0.0001)))
        # ReLU activation
        model.add(Activation('relu'))

        # aggregate data in every feature map to scalar using MAX operation
        model.add(MaxPooling2D(pool_size=(conv_input_height - kernel_size + 1, 1)))

        model.add(Flatten())
        # Inner Product layer (as in regular neural network, but without non-linear activation function)
        self.add_output(model)

        return model

    def compile(self):
        if self.loader.convertor.is_binary():
            self.model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=["accuracy"])
        else:
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def copy(self):
        copy_instance = ConvNetAlgorithm(self.loader, self.project_name, self.time_steps, self.top_words)
        return copy_instance

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()


class BaselineLSTM(ConvNetAlgorithm):

    def get_name(self):
        return self.loader.parser.word2vec.name + '_LTSM_' + str(self.time_steps)

    @abc.abstractmethod
    def construct_ltsm_model(self, model):
        return None

    def construct_model(self):
        model = Sequential()
        self.construct_ltsm_model(model)
        self.add_output(model)
        return model


class WeightsLSTM(BaselineLSTM):

    def __init__(self, loader, project_name, time_steps, top_words=10000, lstm=150):
        self.lstm = lstm
        super(WeightsLSTM, self).__init__(loader, project_name, time_steps, top_words)

    def get_name(self):
        vector_size = self.loader.parser.word2vec.embedding_matrix.shape[1]
        return self.loader.parser.word2vec.name + '_LTSM_Weights_' + str(self.time_steps) + '_' + str(vector_size)

    def construct_ltsm_model(self, model):
        self.add_embeddings(model)
        model.add(LSTM(self.lstm, return_sequences=False))

    def copy(self):
        copy_instance = WeightsLSTM(self.loader, self.project_name, self.time_steps, self.top_words, self.lstm)
        return copy_instance

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()


class DeepSklearnWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y):
        y = self.classifier.loader.convertor.create_vector(y)
        result = self.classifier.fit(X, y)
        self.classifier.save('Multi_' + str(self.classifier.counter))
        self.classes_ = self.classifier.get_classes(y)
        del self.classifier.model
        gc.collect()
        return result

    def predict_proba(self, X):
        self.classifier.load('Multi_' + str(self.classifier.counter))
        y = self.classifier.predict_proba(X)
        if len(self.classes_) == 2:
            y = Utilities.make_binary_prob(y)
        del self.classifier.model
        gc.collect()
        return y