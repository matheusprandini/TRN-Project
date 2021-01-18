import os
from Labels import Labels


class DataPartitions():

    @staticmethod
    def create_partitions_and_labels(dataPath):
        partition = []
        labels = {}

        allFiles = os.listdir(dataPath)

        for classFolder in allFiles:
            allVideosPath = dataPath + classFolder + "/"
            allVideos = os.listdir(allVideosPath)
            for video in allVideos:
                allChunksPath = allVideosPath + video + "/"
                allChunks = os.listdir(allChunksPath)
                for chunk in allChunks:
                    chunkPath = allChunksPath + chunk
                    partition.append(chunkPath)
                    labels[chunkPath] = Labels.get_classes()[classFolder]

        return partition, labels
