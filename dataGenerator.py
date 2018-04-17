import numpy as np
import keras
import math
import pandas as pd


# 71 characters in our diction
dictionary = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890,.-&! ()'

class DataGenerator():
    'Generates data for Keras'
    def __init__(self, pandasFile, batch_size=32, 
                shuffle=True):
        self.list_IDs = pandasFile.index.values
        self.pandasFile = pandasFile
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return abs(int(np.floor(len(self.list_IDs) / self.batch_size)))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return list_IDs_temp

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def gen_data(self):
        i = 0
        while True:
            for j in range(self.__len__()):
                ids = self.__getitem__(j)
                yield self.__data_generation(list_IDs_temp=ids)
            self.on_epoch_end()

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        import math
        #print(self.pandasFile)
        # get raw arrays from pandas
        train_raw = []
        for i in list_IDs_temp:
            filtered = self.pandasFile[self.pandasFile['id'].str.contains(i, na=False)].values
            
            if len(filtered) > 0 and type(filtered[0][5]) != float:
                # add to raw data
                train_raw.append(filtered[0])
        # raw data arrays
        grade_category = []
        subject_category = []
        subject_subcategory = []
        project_title = []
        essay_1 = []
        essay_2 = []
        resource = []
        num_proj = []
        proj_price = []
        y = []
        #ct = 0
        for row in train_raw:
            #print(row)
            #ct += 1
            #print(ct, end='\r')
            try:
                #print(row)
                if type(row[5]) != float:
                    # row[5] is grade category - we're using 1hot encoding here w/4 categories
                    grade_one_hot = np.zeros(shape=(4), dtype='float32')
                    if row[5] == "Grades PreK-2":
                        grade_one_hot[0] = 1.
                    if row[5] == "Grades 3-5":
                        grade_one_hot[1] = 1.
                    if row[5] == "Grades 6-8":
                        grade_one_hot[2] = 1.
                    if row[5] == "Grades 9-12":
                        grade_one_hot[3] = 1.

                    # row[6] - subject categories - char LSTM 1hot since there's like 30 categories        
                    subject_category_one_hot = one_hot_string(row[6], 30)
                    # row[7] - subject subcategories - char LSTM 1hot since there's like 30 categories
                    subject_subcategory_one_hot = one_hot_string(row[7], 30)

                    # row[8] - project titles - char LSTM 1hot
                    project_title_one_hot = one_hot_string(row[8], 100)

                    # row[9] - Essay 1 - char LSTM 1 hot
                    # TODO: perhaps don't use char for such long paragraphs?
                    # Although attention can resolve that
                    essay_1_one_hot = one_hot_string(row[9], 1500)

                    # row[10] - Essay 2 - char LSTM 1 hot
                    essay_2_one_hot = one_hot_string(row[10], 1500)
                    # row[13] - Resource Summary - char LSTM 1 hot
                    resource_one_hot = one_hot_string(row[13], 200)
                    # row[14] - Number of projects approved
                    num_proj_ex = int(row[14])
                    # row[15] - If projected is approved
                    temp_app = int(row[15])
                    # one hot arr for approval
                    y_one_hot = np.zeros(shape=(2), dtype='float32')
                    if temp_app == 1:
                        y_one_hot[0] = 1.
                    else:
                        y_one_hot[1] = 1.

                    # row[16] is cost - rescale by dividing by 1500
                    cost = row[24] / 1500. 
                    proj_price.append(cost)

                    grade_category.append(grade_one_hot)
                    subject_category.append(subject_category_one_hot)
                    subject_subcategory.append(subject_subcategory_one_hot)
                    project_title.append(project_title_one_hot)
                    essay_1.append(essay_1_one_hot)
                    essay_2.append(essay_2_one_hot)
                    resource.append(resource_one_hot)
                    num_proj.append(num_proj_ex)
                    y.append(y_one_hot)

            except:
                continue
        #print([np.array(grade_category), np.array(subject_category), np.array(subject_subcategory), np.array(project_title), np.array(essay_1), np.array(essay_2), np.array(resource), np.array(proj_price)], np.array(y))
        return [np.array(grade_category), np.array(subject_category), np.array(subject_subcategory), np.array(project_title), np.array(essay_1), np.array(essay_2), np.array(resource), np.array(proj_price)], np.array(y)




# converts a string to onehot encoded time series
# length is the length of the time series
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

