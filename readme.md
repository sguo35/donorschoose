# DonorsChoose model
- `load_data.py` holds data loading / preprocessing functions
- `model.py` holds the model definition and training code
- You will need to manually download `test.csv` and `resources.csv`

# Training
- Run `keras model.py` to train. You will need at least 2GB VRAM
- Change `LSTM` to `CuDNNLSTM` in `model.py`'s `make_residual_lstm` function if you have a GPU with sufficient memory
- Because loading the entire dataset into memory is impossible unless you have 64GB+ RAM, I currently load 3k entries at a time, randomly selected. You will have to restart training periodically with existing saved model to sample the entire dataset (TODO: fix this)

# Libraries
- Keras
- Numpy
- sklearn
- Pandas
- Tensorflow backend
# TODO
- ~~Write parser / data loader for `resources.csv`~~
- Use LSTM for resources?
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
# Changes
- I removed all the old format 4 essay ones (~8k) from training, I'm pretty sure we need to create a separate model for those but for now `train.csv.tar.lzma` has only the 2 essay entries
