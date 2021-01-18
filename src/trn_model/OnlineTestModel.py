import cv2
import sys
import glob
import time
import numpy as np
from os import path
from Data.Labels import Labels

sys.path.append(path.join(path.dirname(__file__), '..'))

from Entity.TRN import TRN
from utils.JsonHandler import JsonHandler

## Read video file and return all frames
def extract_frames(filepath, imageSize=(224,224)):

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
            frame = cv2.resize(frame, imageSize)

            # Normalized frame
            frame = frame / 255.0
            
            # Append frame
            frames.append(frame)
    finally:
        video.release()
        
    return frames

def get_test_results(datasetPath, model):
    preprocessTime = []
    classes = list(Labels.get_classes().keys())
    for className in classes[:1]:

        videos = sorted(glob.glob(datasetPath + "*"))

        for video in videos[:1]:
            print(video)

            frames = extract_frames(video)
            examples = []
            probabilities = []
            numChunks = 0

            for i in range (0,len(frames),2*8):
                numChunks += 2
                chunk1 = frames[i:i+8]

                if len(chunk1) < 8:
                    missingFrames = 8 - len(chunk1)
                    for i in range(missingFrames):
                        chunk1.append(chunk1[-1])
                sample1 = np.array(chunk1)
                predictions1, totalTime = model.predict(sample1)
                probabilities.append(predictions1)
                preprocessTime.append(totalTime)

                if i+8 <= len(frames):
                    chunk2 = frames[i+8:i+8+8]
                    if len(chunk2) < 8:
                        missingFrames = 8 - len(chunk2)
                        for i in range(missingFrames):
                            chunk2.append(chunk2[-1])
                    sample2 = np.array(chunk2)
                    predictions2, totalTime = model.predict(sample2)
                    probabilities.append(predictions2)
                    preprocessTime.append(totalTime)

                probabilitiesMean = np.mean(probabilities, axis=0)
                action = np.argmax(probabilitiesMean)
                probabilities = []
                print("Prediction " + str(np.round(numChunks * (8/30), 2)) + "s:", action, probabilitiesMean[0][action])

                for frame in chunk1:
                    examples.append([frame, action])
                for frame in chunk2:
                    examples.append([frame, action])

    print(len(examples))
    print("Avg prediction time:", np.mean(preprocessTime))
    #np.save("mario-test-1.npy", examples[:400])
    #np.save("mario-test-2.npy", examples[400:800])
    #np.save("mario-test-3.npy", examples[800:])
                    

if __name__ == '__main__':

    # Read config file
    configFile = JsonHandler.read_json("../../conf/test-trn-model-config.json")

    testDataset = configFile["datasetInfo"]["test"]
    modelPath = configFile["modelInfo"]["path"]

    trn = TRN("ResNet50V2")
    trn.load_model(modelPath)

    get_test_results(testDataset, trn)
