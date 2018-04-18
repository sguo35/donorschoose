# load the data
from dataParser import get_data
from keras.models import load_model
import keras.optimizers as optimizers
from dataGenerator import DataGenerator

model = load_model('./cnn_model.h5')

# load pandas df
data = get_data()
print("Got data!")
train_data = data[:-1000]
valid_data = data[-1000:]

import tensorflow as tf
def auc_roc(y_true, y_pred):
	#print(y_pred)
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
from cyclicLR import CyclicLR
clr = CyclicLR(base_lr=0.01, max_lr=0.05,
						step_size=200., mode='triangular2')

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='./weights.h5', verbose=1, save_best_only=True)

import keras.backend as K
# experimental ranking loss
def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = -K.sigmoid(pos-neg) # use loss = K.maximum(1.0 + neg - pos, 0.0) if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true


generator = DataGenerator(pandasFile=train_data, batch_size=32)
valid_gen = DataGenerator(pandasFile=valid_data, batch_size=32)
model.compile(optimizer=optimizers.sgd(nesterov=True, lr=0.01),
			  loss=ranking_loss,
			  metrics=[auc_roc, 'acc'])
model.fit_generator(generator=generator.gen_data(), use_multiprocessing=True, workers=4, epochs=5, steps_per_epoch=generator.__len__(), validation_data=valid_gen.gen_data(), validation_steps=valid_gen.__len__(), callbacks=[clr, checkpointer])

model.save('./model.h5')
