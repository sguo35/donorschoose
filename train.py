# load the data
from dataParser import get_data
from keras.models import load_model
from dataGenerator import DataGenerator

model = load_model('./cnn_model.h5')

# load pandas df
data = get_data()
print("Got data!")

generator = DataGenerator(pandasFile=data, batch_size=64)

model.fit_generator(generator=generator.gen_data(), use_multiprocessing=False, workers=1, epochs=5, steps_per_epoch=generator.__len__())

model.save('./model.h5')
