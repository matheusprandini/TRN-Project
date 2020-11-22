import os

class DataPartitions():

    classLabels = {"HighJump": 0, "LongJump": 1, "BasketballDunk": 2}

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
                    labels[chunkPath] = DataPartitions.classLabels[classFolder]

        return partition, labels