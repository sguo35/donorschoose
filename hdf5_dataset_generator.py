import h5py

"""
*****DO NOT USE ---- UNFINISHED CODE****
"""


dictionary = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890,.-&! ()'

def initialize_datasets(location, image_dimensions):
    f = h5py.File(location+"/train-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'w')
    dsetx = f.create_dataset("X", (1, image_dimensions[0], image_dimensions[1], 1),
                             maxshape=(None, image_dimensions[0], image_dimensions[1], 1))
    dsety = f.create_dataset("Y", (1, 2), maxshape=(None, 2))
    f.close()
    f = h5py.File(location+"/val-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'w')
    dsetx = f.create_dataset("X", (1, image_dimensions[0], image_dimensions[1], 1),
                             maxshape=(None, image_dimensions[0], image_dimensions[1], 1))
    dsety = f.create_dataset("Y", (1, 2), maxshape=(None, 2))
    f.close()
    f = h5py.File(location+"/test-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'w')
    dsetx = f.create_dataset("X", (1, image_dimensions[0], image_dimensions[1], 1),
                             maxshape=(None, image_dimensions[0], image_dimensions[1], 1))
    dsety = f.create_dataset("Y", (1, 2), maxshape=(None, 2))
    f.close()

def write_to_hdf5(data, dataset_file):
    f = h5py.File(dataset_file, 'a')
    dsetx = f['X']
    dsety = f['Y']
    num_examples = len(data)
    old_len = len(dsety)
    dsetx.resize(old_len+num_examples, axis=0)
    dsety.resize(old_len+num_examples, axis=0)
    for i in range(num_examples):
        dsetx[old_len+i] = data[i][0]
        dsety[old_len+i] = data[i][1]
    f.close()

def one_hot_string(string, length):
    one_hot = []
    for i in range(length):
        # one extra for unknown
        char_one_hot = np.zeros(shape=(72), dtype='float32')
        # diction lookup
        if i < len(string):
            char_index = dictionary.find(string[i])
        else:
            char_index = -2
        # if we can't find it, last one is unknowns
        if char_index == -2:
            char_one_hot[char_index] = 1.
        if char_index == -1:
            char_one_hot[71] = 1.
        # only mark unknown if we're not past the string length
        # append arr to time series
        one_hot.append(char_one_hot)
    return one_hot


def main(data_directory):
    initialize_datasets(data_directory)
    data = get_data()


if __name__ == "__main__":
    data_directory = "./"
    main(data_directory)
