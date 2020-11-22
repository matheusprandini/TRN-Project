import keras as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers.recurrent import LSTM


class TRN():

    def __init__(self):
        self.model = None

    def compile_model(self, learningRate):
        self.model.compile(loss="categorical_crossentropy", optimizer=K.optimizers.Adam(
            learning_rate=learningRate), metrics=["accuracy"])

    def train_model(self, trainGenerator, numEpochs):
        self.model.fit(trainGenerator, epochs=numEpochs)
