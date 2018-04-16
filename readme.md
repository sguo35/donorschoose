# DonorsChoose model
- `Dataparser.py` holds data loading function
- `dataGenerator.py` holds data generator code
- `model.py` holds the model definition
- `model_cnn.py` holds the CNN based model (ResNet based)
- `train.py` holds training code
- You will need to manually download `test.csv` and `resources.csv`

# Training
- Run `python model_cnn.py` to generate the model first.
- Run `python train.py` to train. You will need at least 2GB VRAM
- ~~Change `LSTM` to `CuDNNLSTM` in `model.py`'s `make_residual_lstm` function if you have a GPU with sufficient memory~~

# Libraries
- Keras
- Numpy
- sklearn
- Pandas
- Tensorflow backend
# TODO
- ~~Write parser / data loader for `resources.csv`~~
- ~~Use LSTM for resources?~~
- ~~Write the actual model itself~~
- Find hyperparameters
- Dropout vs. L2?
# Model
- Residual LSTMs
- Each sequence data will have 4-8x128 rLSTM (with batch norm? or Bidirectional?)
- One hot data is 2x64 ReLU + BatchNorm
- Concatenate all the outputs together and stack on another 2-4x128 ReLU + BatchNorm
- Final output is 2 unit softmax
- Not sure if we want to use hinge loss as I've read it optimizes for AUC (which is what we're judged on) or just standard `categorical_crossentropy`
- We can try Adam and SGD w/momentum

# CNN Model
- ResNet-18 but with 1D convolutions for sequential data
- Rest is same as LSTM model, but with only 1x8 ReLU for onehot data
- TODO: Try models with bigger ResNets (34, 50, etc)
- TODO: get data parser for 4 essay format
# Changes
- I removed all the old format 4 essay ones (~8k) from training, I'm pretty sure we need to create a separate model for those but for now `train.csv.tar.lzma` has only the 2 essay entries
