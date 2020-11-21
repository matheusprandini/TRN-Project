import keras as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers.recurrent import LSTM


class TRN():

    def build_model(self, numFeatures, numClasses):
        self.model = K.models.Sequential()
        self.model.add(Input(shape=(numFeatures, 1)))
        self.model.add(LSTM(numFeatures, dropout=0.1))
        self.model.add(Dense(numClasses))
