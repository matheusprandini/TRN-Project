import keras as K
import numpy as np

class DataGenerator(K.utils.Sequence):

    def __init__(self, listIDs, labels, dim=(6,1280), batchSize=8, numClasses=20, shuffle=True):
        self.listIDs = listIDs
        self.labels = labels
        self.batchSize = batchSize
        self.dim = dim
        self.numClasses = numClasses
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.listIDs) / self.batchSize))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        listIDsBatch = [self.listIDs[k] for k in indexes]
        X, y = self.__data_generation(listIDsBatch)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.listIDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, listIDsBatch):
        X = np.empty((self.batchSize, *self.dim))
        y = np.empty((self.batchSize), dtype=int)

        for i, ID in enumerate(listIDsBatch):
            X[i,] = np.load(ID)
            y[i] = self.labels[ID]

        return X, K.utils.to_categorical(y, num_classes=self.numClasses)