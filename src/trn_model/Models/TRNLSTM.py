import keras as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers.recurrent import LSTM
from Entity.TRN import TRN

class TRNLSTM(TRN):

    def __init__(self):
        super().__init__()

    def build_model(self, numFeatures, numClasses):
        self.model = K.models.Sequential()
        self.model.add(Input(shape=(numFeatures, 1)))
        self.model.add(LSTM(numFeatures, dropout=0.1))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(numClasses, activation="softmax"))
        print(self.model.summary())
