import keras as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator

## Building model

vgg16_model = K.models.Sequential()
vgg16_model.add(ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))
vgg16_model.add(Flatten())
vgg16_model.add(Dense(512, activation="relu"))
vgg16_model.add(Dense(128, activation="relu"))

for layer in vgg16_model.layers:
    layer.trainable = False

lstm_model = K.models.Sequential()
lstm_model.add(Reshape((128,1)))
lstm_model.add(Dropout(.1))
lstm_model.add(LSTM(128))
lstm_model.add(Reshape((128,1)))
lstm_model.add(Dropout(.1))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(3))

input_shape = (224, 224, 3)
visible = Input(shape=input_shape)
features = vgg16_model(visible)
classifier = lstm_model(features)

model = Model(inputs=visible, outputs=classifier)
print(model.summary())


## Training model

batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    '/home/matheus_prandini/Doutorado/ThumosImageData/',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42,
    subset="training")

validation_generator = train_datagen.flow_from_directory(
    '/home/matheus_prandini/Doutorado/ThumosImageData/',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42,
    subset="validation")

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=1)
