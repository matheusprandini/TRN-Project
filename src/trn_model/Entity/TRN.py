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
    
    def predict(self, chunk, chunkShape, generateFeatures=False):
        initialTime = time.time()
        if generateFeatures:
            chunk = self.featureExtractor.predict(chunk)
        prediction = self.model.predict(chunk.reshape(chunkShape))
        totalTime = time.time() - initialTime
        return prediction, totalTime
