# load the data
from load_data import load_data

grades, subject_cat, subject_subcat, titles, essays_1, essays_2, resources, num_projs, proj_price, y = load_data("./train.csv")

from keras.models import load_model

model = load_model('./model.h5')

from keras.callbacks import *

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

history = model.fit([grades, subject_cat, subject_subcat, titles, essays_1, essays_2, resources, proj_price], y, validation_split=0.1,
                    verbose=1,
          epochs=1, batch_size=32, callbacks=[reduce_lr])

model.save('./model.h5')