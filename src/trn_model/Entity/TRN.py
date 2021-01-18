import keras as K
import time
from feature_extractor.Factory.FeatureExtractorFactory import FeatureExtractorFactory


class TRN():

    def __init__(self, featureExtractorName):
        self.model = None
        self.featureExtractor = FeatureExtractorFactory.get_model(featureExtractorName)

    def load_model(self, path):
        self.model = K.models.load_model(path)
        print(self.model.summary())

    def compile_model(self, learningRate):
        self.model.compile(loss="categorical_crossentropy", optimizer=K.optimizers.Adam(
            learning_rate=learningRate), metrics=["accuracy"])

    def train_model(self, trainGenerator, numEpochs):
        self.model.fit(trainGenerator, epochs=numEpochs)

    def predict(self, chunk):
        initialTime = time.time()
        features = self.featureExtractor.predict(chunk)
        prediction = self.model.predict(features.reshape(1, features.shape[0], features.shape[1], 1))
        totalTime = time.time() - initialTime
        return prediction, totalTime
