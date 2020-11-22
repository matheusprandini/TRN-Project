from utils.JsonHandler import JsonHandler
from DatasetHandler import DatasetHandler

if __name__ == '__main__':
    
    # Read config file
    configFile = JsonHandler.read_json("../conf/create-data-config.json")

    inputDataset = configFile["datasetInfo"]["inputDataset"]
    outputDataset = configFile["datasetInfo"]["outputDataset"]
    classesList = configFile["datasetInfo"]["classes"]
    chunkSize = int(configFile["datasetInfo"]["chunkSize"])
    imageSize = tuple((configFile["datasetInfo"]["imageSize"], configFile["datasetInfo"]["imageSize"]))
    featureExtractorName = configFile["featureExtractor"]["name"]

    dataset = DatasetHandler(inputDataset, outputDataset, classesList, chunkSize, imageSize, featureExtractorName)
    dataset.create_data()