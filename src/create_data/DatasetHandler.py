import os
import glob
import sys
import numpy as np
from os import path
from VideoFramesHandler import VideoFramesHandler
from feature_extractor.Factory.FeatureExtractorFactory import FeatureExtractorFactory

class DatasetHandler():

    def __init__(self, inputDataset, outputDataset, classes, chunkSize, imageSize, featureExtractorName):
        self.inputDataset = inputDataset # Input directory
        self.outputDataset = outputDataset # Output directory
        self.classes = classes # Classes to extract information
        self.chunkSize = chunkSize # Chunk size
        self.imageSize = imageSize # Image size
        self.featureExtractorModel = FeatureExtractorFactory.get_model(featureExtractorName) # Feature extractor model

    def create_data(self):
        for className in self.classes:
            print("Processing " + className + " data...")

            loadPath = self.inputDataset + className + "/"
            savePath = self.outputDataset + className + "/"

            # Get all files in the "loadPath" directory
            files = glob.glob(loadPath + "*")

            # Start counter
            videoCount = 0

            for filepath in files:

                # Create "saveVideoPath" directory
                saveVideoPath = savePath + "Video" + str(videoCount) + "/"
                os.makedirs(saveVideoPath, exist_ok=True)
                
                # Extract all frames
                videoFrames = VideoFramesHandler.extract_frames(filepath, self.imageSize)
                
                # Save frames in chunks
                self.process_chunk_data(saveVideoPath, videoFrames)
                
                # Update counter
                videoCount += 1

    def process_chunk_data(self, path, frames):
        chunkCount = 0
        for i in range(0, len(frames), self.chunkSize):

            # Filename
            if chunkCount < 10:
                chunkName = "chunk_" + "0" + str(chunkCount)
            else:
                chunkName = "chunk_" + str(chunkCount)

            # Get data (frames)
            data = frames[i:i+self.chunkSize]

            # Checks if the number of frames is less than the chunk size. 
            # If so, normalizes chunk with the last frame.
            if len(data) < self.chunkSize:
                missingFrames = self.chunkSize - len(data)
                for i in range(missingFrames):
                    data.append(data[-1])

            # Save preprocessed data
            data = np.array(data)
            preprocessedData = self.featureExtractorModel.predict(data)
            np.save(path + chunkName, preprocessedData)

            # Update Chunk count
            chunkCount += 1
