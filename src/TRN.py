import keras as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras.applications.vgg16 import VGG16

vgg16_model = K.models.Sequential()
vgg16_model.add(VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))
vgg16_model.add(Flatten())
vgg16_model.add(Dense(4096, activation="relu"))
vgg16_model.add(Dense(4096, activation="relu"))

lstm_model = K.models.Sequential()
lstm_model.add(Reshape((4096,1)))
lstm_model.add(Dropout(.1))
lstm_model.add(LSTM(4096))
lstm_model.add(Reshape((4096,1)))
lstm_model.add(Dropout(.1))
lstm_model.add(LSTM(4096))
lstm_model.add(Dense(3))

input_shape = (224, 224, 3)
visible = Input(shape=input_shape)
features = vgg16_model(visible)
reshaped_features = Reshape((4096,1))(features)
classifier = lstm_model(reshaped_features)

model = Model(inputs=visible, outputs=classifier)
print(model.summary())
