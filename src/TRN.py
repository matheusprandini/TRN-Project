import keras as K
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

'''vgg16_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

cnn_model = K.models.Sequential()
cnn_model.add(vgg16_model)
cnn_model.add(K.layers.Flatten())
cnn_model.add(K.layers.Dense(4096, activation="relu"))
cnn_model.add(K.layers.Dense(4096, activation="relu"))
cnn_model.add(K.layers.LSTM(units=10))'''

'''for layer in cnn_model.layers:
    layer.trainable = False'''

'''lstm_model = K.models.Sequential()
lstm_model.add(K.layers.LSTM(10, return_sequences=False, return_state=True, input_shape=(4096,1)))

classifier = K.models.Sequential()
classifier.add(K.layers.Dense(3))

model = TimeDistributed(cnn_model)
model = rnn(lstm_model)
model = dense(classifier)'''

'''model_input = K.layers.Input(shape=(1, 224, 224, 3))
features = cnn_model(model_input)
output, state_h, state_c = K.layers.LSTMCell(10)(features)
classifier = K.layers.Dense(3)(output)'''

'''model = K.models.Sequential()
model.add(K.layers.LSTM(10, input_shape=(4096,1)))
print(model.summary())'''

x = K.layers.Input(shape=(4096,1))
print(x.shape)
output, state_h, state_c = K.layers.LSTM(units=10)(x)
