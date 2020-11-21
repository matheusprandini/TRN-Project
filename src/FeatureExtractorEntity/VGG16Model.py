import keras as K
from keras.applications.vgg16 import VGG16

class VGG16Model():

    def build_model(self, weights, includeTop, inputShape):
        baseModel = VGG16(weights=weights, include_top=includeTop, input_shape=inputShape)

        model = K.models.Model(
            inputs=baseModel.input,
            outputs=baseModel.get_layer('fc2').output
        )

        for layer in model.layers:
            layer.trainable = False
        
        return model

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