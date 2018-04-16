# load the data
from dataParser import get_data
from keras.models import load_model
import keras.optimizers as optimizers
from dataGenerator import DataGenerator

model = load_model('./cnn_model.h5')

# load pandas df
data = get_data()
print("Got data!")

generator = DataGenerator(pandasFile=data, batch_size=32)
model.compile(optimizer=optimizers.adam(clipvalue=1.),
              loss='categorical_crossentropy',
              metrics=['acc'])
model.fit_generator(generator=generator.gen_data(), use_multiprocessing=True, workers=4, epochs=5, steps_per_epoch=generator.__len__(), class_weight={0 : 0.0001,  1: 0.0004})

model.save('./model.h5')
