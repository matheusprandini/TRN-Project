import os
import cv2
import glob
import numpy as np
from Utils.JsonHandler import JsonHandler


def load_video(path, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frames.append(frame)

    finally:
        cap.release()
    return np.array(frames)


def create_data(path_to_load, path_to_save, className):
    count = 0

    files = glob.glob(path_to_load + className + "/*")
    os.makedirs(path_to_save + className + "/")

    for filepath in files:
        videoFrames = load_video(filepath)
        for videoFrame in videoFrames:
            name = className + "_" + str(count) + ".jpg"
            cv2.imwrite(path_to_save + className + "/" + name, videoFrame)
            count += 1


# Read config file
configFile = JsonHandler.read_json("../conf/create-data-config.json")

inputDataset = configFile["inputDataset"]
outputDataset = configFile["outputDataset"]
classesList = configFile["classes"]

# Create Data
for className in classesList:
    print("Processing " + className + " data...")
    create_data(inputDataset, outputDataset, className)
