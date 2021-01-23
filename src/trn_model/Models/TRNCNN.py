import keras as K
import time
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, Conv2D, MaxPooling2D
from Entity.TRN import TRN

class TRNCNN(TRN):

    def __init__(self, featureExtractorName):
        super().__init__(featureExtractorName)

    def build_model(self, numTimesteps, numFeatures, numClasses):
        self.model = K.models.Sequential()
        self.model.add(Conv2D(64, kernel_size=(5,5), input_shape=(numTimesteps, numFeatures, 1)))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(numClasses, activation="softmax"))
        print(self.model.summary())

    def predict(self, chunk, generateFeatures=False):
        chunkShape = (1,chunk.shape[1],chunk.shape[2],1)
        return super().predict(chunk, chunkShape, generateFeatures)
