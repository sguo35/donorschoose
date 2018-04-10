# Stacked LSTM with residual connections in depth direction.
#
# Naturally LSTM has something like residual connections in time.
# Here we add residual connection in depth.
#
# Inspired by Google's Neural Machine Translation System (https://arxiv.org/abs/1609.08144).
# They observed that residual connections allow them to use much deeper stacked RNNs.
# Without residual connections they were limited to around 4 layers of depth.
#
# It uses Keras 2 API.

from keras.layers import CuDNNLSTM, Lambda
from keras.layers.merge import add

def make_residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        x_rnn = LSTM(rnn_width, 
        recurrent_dropout=rnn_dropout, 
        dropout=rnn_dropout, 
        return_sequences=return_sequences, 
        implementation=2, 
        kernel_regularizer=regularizers.l2(0.01))(x)

        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn])
    return x


# Example usage
from keras import regularizers
from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, \
GlobalMaxPool1D,SpatialDropout1D,CuDNNGRU,Bidirectional,PReLU,GRU, BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

# changeable parameters
MULTIPLIER = int(1)
L2_REGULARIZATION = 0.0

def get_model():
    # Grade category is shape (4)
    input_grade_category = Input((4))

    # Grade Category network - 1x32 ReLU layer
    grade_category_network = Dense(32 * MULTIPLIER, activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION))(input_grade_category)

    # Subject category is 30x72 time series
    input_subject_category = Input((30,72))

    # Subject Category network
    # 2x64 rLSTM
    subject_category_network = make_residual_lstm_layers(input=input_subject_category, rnn_width=64 * MULTIPLIER, rnn_depth=2 * MULTIPLIER)

    # subject subcategory is 30x72 time series
    input_subject_subcategory = Input((30, 72))

    # project title is 100x72 time series
    input_project_title = Input((100, 72))

    # essay 1 is 1500x72 time series
    input_essay_1 = Input((1500,72))

    # essay 2 is 1500x72 time series
    input_essay_2 = Input((1500, 72))

    # resource summary is 200x72 time series
    input_resource_summary = Input((200,72))

    # number of projects approved is just one input
    input_num_proj_approved = Input((1))
    
    predictions = Dense(2, activation="softmax")(x)
    model = Model(inputs=[input_grade_category, 
    input_subject_category, input_subject_subcategory,
    input_project_title, input_essay_1, input_essay_2,
    input_resource_summary, input_num_proj_approved], outputs=predictions)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

model = get_model()
from keras.callbacks import *

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

history = model.fit([X_train_cat, X_train_num, X_train_words], X_train_target, validation_split=0.1,
                    verbose=1,
          epochs=100, batch_size=128, callbacks=[reduce_lr])
model.save('./model.h5')