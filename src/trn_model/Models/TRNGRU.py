import keras as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers.recurrent import GRU
from Entity.TRN import TRN

class TRNGRU(TRN):

    def __init__(self, featureExtractorName):
        super().__init__(featureExtractorName)

    def build_model(self, numTimesteps, numFeatures, numClasses):
        self.model = K.models.Sequential()
        self.model.add(GRU(numFeatures, input_shape=(numTimesteps, numFeatures), dropout=0.1))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(numClasses, activation="softmax"))
        print(self.model.summary())
