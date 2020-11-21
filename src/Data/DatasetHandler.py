import os
import cv2
import glob
import sys
import numpy as np
from os import path
from Model.FeatureExtractorFactory import FeatureExtractorFactory

class DatasetHandler():

    def __init__(self, inputDataset, outputDataset, classes, chunkSize, imageSize, featureExtractorName):
        self.inputDataset = inputDataset # Input directory
        self.outputDataset = outputDataset # Output directory
        self.classes = classes # Classes to extract information
        self.chunkSize = chunkSize # Chunk size
        self.imageSize = imageSize # Image size
        self.featureExtractorModel = FeatureExtractorFactory().get_model(featureExtractorName) # Feature extractor model

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
                videoFrames = self.extract_frames(filepath)
                
                # Save frames in chunks
                self.process_chunk_data(saveVideoPath, videoFrames)
                
                # Update counter
                videoCount += 1

    def process_chunk_data(self, path, frames):
        chunkCount = 0
        for i in range(0, len(frames), self.chunkSize):

            # Filename
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
            preprocessedData = self.featureExtractorModel.predict(data).flatten()
            np.save(path + chunkName, preprocessedData.reshape(preprocessedData.shape[0],1))

            # Update Chunk count
            chunkCount += 1


    ## Read video file and return all frames
    def extract_frames(self, filepath):

        # Retrieves video information
        video = cv2.VideoCapture(filepath)
        frames = []

        try:
            while True:
                # Get current state
                ret, frame = video.read()
                
                if not ret:
                    break

                # Resized frame
                frame = cv2.resize(frame, self.imageSize)

                # Normalized frame
                frame = frame / 255.0
                
                # Append frame
                frames.append(frame)
        finally:
            video.release()
            
        return frames
