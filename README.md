# TRN-Project

Modified version of the following work: [Temporal Recurrent Networks for Online Action Detection](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Temporal_Recurrent_Networks_for_Online_Action_Detection_ICCV_2019_paper.pdf).

## Dependencies

The following dependencies must be installed to run this project:

- Python3: sudo apt install python3-pip
- Numpy: pip3 install numpy
- Tensorflow: pip3 install tensorflow
- Keras: pip3 install keras
- OpenCv: pip3 install opencv-python (needs to upgrade pip: pip3 install --upgrade pip)

## Execution

### Create Data

Extract frames from videos and save on an organized directory.

The `create-data-config.json` configuration file (under conf/ directory) has the following structure:

```
{
    "inputDataset": "/path/to/ThumosDataset/",
    "outputDataset": "/path/to/ThumosImageData/",
    "classes": ["class1", "class2", "class3", ...]
}
```

- inputDataset: directory to load video data.
- outputDataset: directory to save image data.
- classes: classes to extract frames.

Steps to execute this code:

- Download and extract ([ThumosDataset](https://drive.google.com/drive/folders/1VWx35zK6tUbTS-lzE03M4Bc74rd_XIcG)).
- Create 'create-data-config.json'.
- Execute the following command under **src/create_data** directory: ``` python3 CreateDataset.py ```

### Train Model

It is possible to train models locally or in the cloud using ([Google Colab](https://colab.research.google.com/ogle.com/drive/folders/1VWx35zK6tUbTS-lzE03M4Bc74rd_XIcG)).

#### Locally

The `train-trn-model-config.json` configuration file (under conf/ directory) has the following structure:

```
{
    "datasetInfo": {
        "train": "/path/to/train/dir/",
        "validation": "/path/to/validation/dir/"
    },
    "modelInfo": {
        "numTimesteps": number of time steps used by a recurrent model,
        "hiddenLayerSize": number of neurons of the hidden layers, 
        "numClasses": number of classes,
        "trnType": type of TRN model ("cnn", "convlstm", "gru" or "lstm"),
        "featureExtractor": type of feature extractor model ("MobileNetV2", "ResNet50V2", "VGG16")
    },
    "hyperparameters": {
        "epochs": number of epochs to train the model,
        "learningRate": learning rate used to train the model
    },
    "resultInfo": {
        "finalModelFilename": name of the final model
    }
}
```

Besides this, it is necessary to define the classes and their labels in the `labels.json` configuration file (under conf/ directory) has the following structure:

```
{
    "classes": {"class0": 0, "class1": 1,  "class2": 1, ...}
}
```

Execute the following command under **src/trn_model** directory:

```
python3 TrainModel.py
```

#### Google Colab

Execute the steps provided in this ([Jupyter Notebook](https://colab.research.google.com/drive/1uD0qbmM-MimaDDhcDrNCxkorujY3Lf6f)).

### Test Model

It is possible to test the models trained using two strategies: **offline** and **online**.

The `test-trn-model-config.json` configuration file (under conf/ directory) has the following structure:

```
{
    "datasetInfo": {
        "test": "/path/to/test/dir/"
    },
    "modelInfo": {
        "path": "/path/to/model/model_name.h5",
        "trnType": type of TRN model ("cnn", "convlstm", "gru" or "lstm"),
        "featureExtractor": type of feature extractor model ("MobileNetV2", "ResNet50V2", "VGG16")
    }
}
```

Besides this, it is necessary to define the classes and their labels in the `labels.json` configuration file (under conf/ directory) has the following structure:

```
{
    "classes": {"class0": 0, "class1": 1,  "class2": 1, ...}
}
```

#### Offline

Execute the following command under **src/trn_model** directory:

```
python3 OfflineTestModel.py
```

#### Online

Execute the following command under **src/trn_model** directory:

```
python3 OnlineTestModel.py
```
