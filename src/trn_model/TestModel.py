import sys
import glob
import numpy as np
from os import path
from Data.Labels import Labels
from Factory.TRNFactory import TRNFactory

sys.path.append(path.join(path.dirname(__file__), '..'))

from utils.JsonHandler import JsonHandler

def get_test_results(datasetPath, model):
    rightPredictions = 0
    wrongPredictions = 0
    classes = list(Labels.get_classes().keys())
    for className in classes:
        print("Testing " + className + "...")

        videosPath = datasetPath + className + "/"
        videos = sorted(glob.glob(videosPath + "*"))

        actionLabel = Labels.get_classes()[className]

        for video in videos:
            chunks = glob.glob(video + "/*")
            chunks.sort()

            probabilities = []
            for chunk in chunks:
                sample = np.load(chunk)
                sample = sample.reshape(-1, sample.shape[0], sample.shape[1])
                predictions = model.predict(sample)
                probabilities.append(predictions)

            probabilitiesMean = np.mean(probabilities, axis=0) 
            action = np.argmax(probabilitiesMean)
            if action == actionLabel:
                rightPredictions += 1
            else:
                wrongPredictions += 1

    print(rightPredictions, wrongPredictions)

if __name__ == '__main__':

    # Read config file
    configFile = JsonHandler.read_json("../../conf/test-trn-model-config.json")

    testDataset = configFile["datasetInfo"]["test"]

    modelPath = configFile["modelInfo"]["path"]
    trnType = configFile["modelInfo"]["trnType"]
    featureExtractorName = configFile["modelInfo"]["featureExtractor"]

    trn = TRNFactory.get_model(trnType, featureExtractorName)
    trn.load_model(modelPath)

    get_test_results(testDataset, trn)
