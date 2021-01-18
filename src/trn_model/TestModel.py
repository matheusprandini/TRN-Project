import sys
import glob
import time
import numpy as np
from os import path
from Data.Labels import Labels
from Entity.TRN import TRN

sys.path.append(path.join(path.dirname(__file__), '..'))

from utils.JsonHandler import JsonHandler

def get_test_results(datasetPath, model):
    predictionsTime = []
    totalCorrectPredictions = 0
    totalWrongPredictions = 0
    classes = list(Labels.get_classes().keys())
    for className in classes:

        videosPath = datasetPath + className + "/"
        videos = sorted(glob.glob(videosPath + "*"))

        actionLabel = Labels.get_classes()[className]

        localCorrectPredictions = 0
        localWrongPredictions = 0

        for video in videos:

            chunks = glob.glob(video + "/*")
            chunks.sort()

            probabilities = []
            for chunk in chunks:
                sample = np.load(chunk)
                sample = sample.reshape(-1, sample.shape[0], sample.shape[1])
                initialTime = time.time()
                predictions = model.predict(sample)
                predictionTime = time.time() - initialTime
                predictionsTime.append(predictionTime)
                probabilities.append(predictions)

            probabilitiesMean = np.mean(probabilities, axis=0) 
            action = np.argmax(probabilitiesMean)
            if action == actionLabel:
                localCorrectPredictions += 1
            else:
                localWrongPredictions += 1
        
        totalCorrectPredictions += localCorrectPredictions
        totalWrongPredictions += localWrongPredictions

        localAccuracy = localCorrectPredictions / (localCorrectPredictions + localWrongPredictions)
        print("Accuracy for ", className, ":", localAccuracy )

    totalAccuracy = totalCorrectPredictions / (totalCorrectPredictions + totalWrongPredictions)
    print("Global Accuracy:", totalAccuracy)
    print("Average Prediction Time:", np.mean(predictionsTime))

if __name__ == '__main__':

    # Read config file
    configFile = JsonHandler.read_json("../../conf/test-trn-model-config.json")

    testDataset = configFile["datasetInfo"]["test"]
    modelPath = configFile["modelInfo"]["path"]

    trn = TRN("")
    trn.load_model(modelPath)

    get_test_results(testDataset, trn)
