from keras.preprocessing.image import ImageDataGenerator
from TRNEntity.TRN import TRN
import sys
import os
import keras as K
import numpy as np
from os import path

sys.path.append(path.join(path.dirname(__file__), '..'))

class DataGenerator(K.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(7680,1), n_classes=3, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(ID)

            # Store class
            y[i] = self.labels[ID]

        return X, K.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == '__main__':
    trn = TRN()
    trn.build_model(512, 3)
    print(trn.model.summary())

    all_files_loc = "/home/matheus_prandini/Doutorado/ThumosChunkData/train/"
    all_files = os.listdir(all_files_loc)
    print(all_files)

    partition = []
    classLabels = {"HighJump": 0, "LongJump": 1, "BasketballDunk": 2}
    labels = {}

    for classFolder in all_files:
        all_videos_loc = all_files_loc + classFolder + "/"
        all_videos = os.listdir(all_videos_loc)
        for video in all_videos:
            all_chunks_loc = all_videos_loc + video + "/"
            all_chunks = os.listdir(all_chunks_loc)
            for chunk in all_chunks:
                chunk_path = all_chunks_loc + chunk
                partition.append(chunk_path)
                labels[chunk_path] = classLabels[classFolder]

    train_generator = DataGenerator(partition, labels)

    # Training model

    checkpointer = K.callbacks.ModelCheckpoint(
        filepath="est.hdf5",
        verbose=1,
        save_best_only=True)

    earlyStopper = K.callbacks.EarlyStopping(patience=5)

    batch_size = 8

    trn.model.compile(loss="categorical_crossentropy",
                      optimizer=K.optimizers.Adam(learning_rate=0.0005), metrics=["accuracy"])

    trn.model.fit(
        train_generator,
        epochs=10,
        callbacks=[checkpointer, earlyStopper])

    '''train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        '/home/matheus_prandini/Doutorado/ThumosImageData/train/',
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42)

    trn.model.compile(loss="categorical_crossentropy",
                      optimizer=K.optimizers.Adam(learning_rate=0.0005), metrics=["accuracy"])

    trn.model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        callbacks=[checkpointer, earlyStopper])'''
