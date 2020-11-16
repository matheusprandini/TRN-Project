# TRN-Project

Modified version of the following work: [Temporal Recurrent Networks for Online Action Detection](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Temporal_Recurrent_Networks_for_Online_Action_Detection_ICCV_2019_paper.pdf).

## Dependencies

The following dependencies must be installed to run this project:

- Python3: pip3 install python3-pip
- Numpy: pip3 install numpy
- Tensorflow: pip3 install tensorflow
- Keras: pip3 install keras
- OpenCv: pip3 install opencv-python

## Execution

### Create Data

Extract frames from videos and save on an organized directory.

The `create-data-config.json` configuration file has the following structure:

```
{
    "inputDataset": "/path/to/ThumosDataset/",
    "outputDataset": "/path/to/ThumosImageData/",
    "classes": ["class1", "class2", "class3", ...]
}
```

- inputDataset: directory to load video data (Download [ThumosDataset](https://drive.google.com/drive/folders/1VWx35zK6tUbTS-lzE03M4Bc74rd_XIcG)).
- outputDataset: directory to save image data.
- classes: classes to extract frames.

Execute the following command under src directory:

```
python3 CreateData.py
```

### Get Feature Extractor Model

Build the feature extractor model based on VGG16.

Execute the following command under src directory:

```
python3 FeatureExtractorModel.py
```
