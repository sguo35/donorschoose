import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# 71 characters in our diction
dictionary = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890,.-&! ()'


# Loads data into the form we're using
# `file` is a string with the path to the file to load
# Returns arrays as shown below
def load_data(file):
    # get the raw numpy array
    import total_price_data from Dataparser
    raw_data = total_price_data()

    # raw data arrays
    grade_category = []
    subject_category = []
    subject_subcategory = []
    project_title = []
    essay_1 = []
    essay_2 = []
    resource = []
    num_proj = []
    y = []

    for row in train_raw:
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

        grade_category.append(grade_one_hot)
        subject_category.append(subject_category_one_hot)
        subject_subcategory.append(subject_subcategory_one_hot)
        project_title.append(project_title_one_hot)
        essay_1.append(essay_1_one_hot)
        essay_2.append(essay_2_one_hot)
        resource.append(resource_one_hot)
        num_proj.append(num_proj_ex)
        y.append(y_one_hot)
    return grade_category, subject_category, 
    subject_subcategory, project_title, 
    essay_1, essay_2, resource, num_proj, y



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
        if char_index != -1:
            char_one_hot[char_index] = 1.
        else if char_index != -2:
            char_one_hot[71] = 1.
        # only mark unknown if we're not past the string length
        # append arr to time series
        one_hot.append(char_one_hot)
    return one_hot


            

    