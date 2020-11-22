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

    inputDataset = configFile["inputDataset"]
    hiddenLayerSize = int(configFile["modelInfo"]["hiddenLayerSize"])
    numClasses = int(configFile["modelInfo"]["numClasses"])
    trnType = configFile["modelInfo"]["trnType"]
    epochs = int(configFile["hyperparameters"]["epochs"])
    learningRate = int(configFile["hyperparameters"]["learningRate"])
    finalModelFilename = configFile["resultInfo"]["finalModelFilename"]

    partition, labels = DataPartitions.create_partitions_and_labels(inputDataset)
    trainGenerator = DataGenerator(partition, labels)

    trn = TRNFactory.get_model(trnType)
    trn.build_model(hiddenLayerSize, numClasses)
    trn.compile_model(learningRate)
    trn.train_model(trainGenerator, epochs)

