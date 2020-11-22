import os

class DataPartitions():

    classLabels = {"HighJump": 0, "LongJump": 1, "BasketballDunk": 2}

    def create_partitions_and_labels(self, dataPath):
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
                    labels[chunkPath] = self.classLabels[classFolder]

        return partition, labels