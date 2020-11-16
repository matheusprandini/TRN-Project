import keras as K
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

class VGG16Model():

    def __init__(self, transfer_learning=False):
        self.model = self.build_model(transfer_learning)

    def build_model(self, transfer_learning):
        model = K.models.Sequential()
        model.add(VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(4096, activation="relu"))
        model.add(K.layers.Dense(4096, activation="relu"))

        if not transfer_learning:
            for layer in model.layers:
                layer.trainable = False
        
        return model

vgg16_model = VGG16Model().model
print(vgg16_model.summary())

# Utilizando o modelo para extrair features de uma imagem x:
# x = x.reshape((-1,224,224,3))
# features = vgg16_model.predict(x)



# Código caso for necessário efetuar transfer learning para o dataset do THUMOS

'''

# Fine-tuning model for THUMOS'14

model = K.models.Sequential()
model.add(res_model)
model.add(K.layers.Dense(3, activation='softmax'))
model.layers[0].trainable = False
print(model.summary())

batch_size = 8

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'Data',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    subset="training")

validation_generator = train_datagen.flow_from_directory(
    'Data',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    subset="validation")

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=1)'''
