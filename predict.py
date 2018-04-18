# load the data
from dataParser import get_data
from keras.models import load_model
import keras.optimizers as optimizers
from dataPredictGenerator import DataGenerator

import tensorflow as tf
def auc_roc(y_true, y_pred):
	print(y_pred)
	# any tensorflow metric
	value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

	# find all variables created for this metric
	metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

	# Add metric variables to GLOBAL_VARIABLES collection.
	# They will be initialized for new session.
	for v in metric_vars:
		tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

	# force to update metric values
	with tf.control_dependencies([update_op]):
		value = tf.identity(value)
		return value

import keras.backend as K
# experimental ranking loss
def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = -K.sigmoid(pos-neg) # use loss = K.maximum(1.0 + neg - pos, 0.0) if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true
model = load_model('./weights.h5', custom_objects={'ranking_loss': ranking_loss, 'auc_roc': auc_roc})

# load pandas df
data = get_data()
print("Got data!")
train_data = data[:1500]
valid_data = data[-100:]

generator = DataGenerator(pandasFile=data, batch_size=32)
results = model.predict_generator(generator=generator.gen_data(), use_multiprocessing=False, workers=1, steps=generator.__len__(), verbose=1)
import numpy
a = numpy.asarray(results)
numpy.savetxt("./results.csv", results, delimiter=",")
print(results)