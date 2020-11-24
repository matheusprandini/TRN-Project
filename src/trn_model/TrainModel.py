import sys
import os
import keras as K
import numpy as np
from os import path
from Factory.TRNFactory import TRNFactory
from Data.DataGenerator import DataGenerator
from Data.DataPartitions import DataPartitions

sys.path.append(path.join(path.dirname(__file__), '..'))

from utils.JsonHandler import JsonHandler

if __name__ == '__main__':

    # Read config file
    configFile = JsonHandler.read_json("../../conf/train-trn-model-config.json")

    trainDataset = configFile["datasetInfo"]["train"]
    validationDataset = configFile["datasetInfo"]["validation"]
    
    numTimesteps = int(configFile["modelInfo"]["numTimesteps"])
    hiddenLayerSize = int(configFile["modelInfo"]["hiddenLayerSize"])
    numClasses = int(configFile["modelInfo"]["numClasses"])
    trnType = configFile["modelInfo"]["trnType"]
    featureExtractorName = configFile["modelInfo"]["featureExtractor"]
    
    epochs = int(configFile["hyperparameters"]["epochs"])
    learningRate = int(configFile["hyperparameters"]["learningRate"])
    
    finalModelFilename = configFile["resultInfo"]["finalModelFilename"]

    partitions = {}
    labels = {}

    partitions["train"], labels["train"] = DataPartitions.create_partitions_and_labels(trainDataset)
    partitions["validation"], labels["validation"] = DataPartitions.create_partitions_and_labels(validationDataset)
    
    trainGenerator = DataGenerator(partitions["train"], labels["train"])
    validationGenerator = DataGenerator(partitions["validation"], labels["validation"])

    trn = TRNFactory.get_model(trnType, featureExtractorName)
    trn.build_model(numTimesteps, hiddenLayerSize, numClasses)
    trn.compile_model(learningRate)
    trn.train_model(trainGenerator, epochs)

