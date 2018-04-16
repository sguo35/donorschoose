# Example usage
from keras import regularizers
from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Conv1D, GlobalMaxPool1D,SpatialDropout1D,CuDNNGRU,Bidirectional,PReLU,GRU, BatchNormalization
from keras.layers import GlobalAveragePooling1D
from resnet import residual_network
from keras.models import Model
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

# changeable parameters
MULTIPLIER = int(1)
L2_regularizer = 0.01

def get_model():
    # Grade category is shape (4)
    input_grade_category = Input(shape=(4,))

    # Grade Category network - 1x64 ReLU layer
    grade_category_network = Dense(8 * MULTIPLIER, activation='relu', kernel_regularizer=regularizers.l2(L2_regularizer))(input_grade_category)

    # Subject category is 30x72 time series
    input_subject_category = Input((30,72))

    # Subject Category network
    # 2x64 rLSTM
    subject_category_network = residual_network(input_subject_category)
    subject_category_network = Dense(8, activation='relu')(subject_category_network)

    # subject subcategory is 30x72 time series
    input_subject_subcategory = Input((30, 72))

    # Subject subcategory network
    # 2x64 rLSTM
    subject_subcategory_network = residual_network(input_subject_subcategory)
    subject_subcategory_network = Dense(8, activation='relu')(subject_subcategory_network)
    # project title is 100x72 time series
    input_project_title = Input((100, 72))

    # Project title network - 2x64 rLSTM
    project_title_network = residual_network(input_project_title)
    project_title_network = Dense(8, activation='relu')(project_title_network)
    # essay 1 is 1500x72 time series
    input_essay_1 = Input((1500,72))

    # Essay 1 network - 4x128 rLSTM
    essay_1_network = residual_network(input_essay_1)
    # downscale the network to 64 units
    essay_1_network = Dense(8 * MULTIPLIER, activation='relu', kernel_regularizer=regularizers.l2(L2_regularizer))(essay_1_network)

    # essay 2 is 1500x72 time series
    input_essay_2 = Input((1500, 72))

    # essay 2 network - 4x128 rLSTM
    essay_2_network = residual_network(input_essay_2)
    # downscale the network to 64 units
    essay_2_network = Dense(8 * MULTIPLIER, activation='relu', kernel_regularizer=regularizers.l2(L2_regularizer))(essay_2_network)

    # resource summary is 200x72 time series
    input_resource_summary = Input((200,72))

    # resource summary network - 2x128 rLSTM
    resource_summary_network = residual_network(input_resource_summary)
    # downscale the network to 64 units
    resource_summary_network = Dense(8 * MULTIPLIER, activation='relu', kernel_regularizer=regularizers.l2(L2_regularizer))(resource_summary_network)

    # number of projects approved is just one input
    input_num_proj_approved = Input((1,))

    # Project price scaled
    input_proj_price = Input((1,))
    proj_price_network = Dense(8 * MULTIPLIER, activation='relu', kernel_regularizer=regularizers.l2(L2_regularizer))(input_proj_price)

    # not using this for now
    network = concatenate([grade_category_network, subject_category_network, subject_subcategory_network, project_title_network, essay_1_network,
    essay_2_network, resource_summary_network, proj_price_network])
    network = BatchNormalization()(network)
    
    # Fully connected layer
    for i in range(1 * MULTIPLIER):
        network = Dense(8 * MULTIPLIER, activation='relu', kernel_regularizer=regularizers.l2(L2_regularizer))(network)

    predictions = Dense(2, activation="softmax")(network)
    model = Model(inputs=[input_grade_category, 
    input_subject_category, input_subject_subcategory,
    input_project_title, input_essay_1, input_essay_2,
    input_resource_summary, input_proj_price], outputs=predictions)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

    return model

model = get_model()

print(model.summary())

model.save('./cnn_model.h5')