import os
from sklearn.model_selection import train_test_split

"""
Method for splitting the original dataset into corresponding train, validation and test files
The ration of training, validation and testing is determined by method parameters
"""
def make_split(input_filepath, output_filepath, test_percent, valid_percent, encoding="utf8", languages=None):
    # Reads files from original dataset
    x_train = open(input_filepath + "/x_train.txt", encoding=encoding)
    y_train = open(input_filepath + '/y_train.txt', encoding=encoding)
    x_test = open(input_filepath + "/x_test.txt", encoding=encoding)
    y_test = open(input_filepath + "/y_test.txt", encoding=encoding)

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    # Creates files for split dataset
    x_train_split = open(output_filepath + '/x_train_split.txt', 'w', encoding=encoding)
    y_train_split = open(output_filepath + '/y_train_split.txt', 'w', encoding=encoding)
    x_valid_split = open(output_filepath + '/x_val_split.txt', 'w', encoding=encoding)
    y_valid_split = open(output_filepath + '/y_val_split.txt', 'w', encoding=encoding)
    x_test_split = open(output_filepath + '/x_test_split.txt', 'w', encoding=encoding)
    y_test_split = open(output_filepath + '/y_test_split.txt', 'w', encoding=encoding)

    # Splits the data and reads into new variables
    x_train_data = x_train.read().splitlines()
    x_test_data = x_test.read().splitlines()
    y_train_data = y_train.read().splitlines()
    y_test_data = y_test.read().splitlines()

    # Combines data
    x_train_data += x_test_data
    y_train_data += y_test_data

    # If the language parameter is not empy, reads only the given languages
    # Used for when you only want to differantiate specific languages
    if languages is not None:
        filter_x = []
        filter_y = []
        for i in range(len(x_train_data)):
            if y_train_data[i] in languages:
                filter_x.append(x_train_data[i])
                filter_y.append(y_train_data[i])
        x_train_data = filter_x
        y_train_data = filter_y

    # Defines the datio based on the percentage parameters and splits dataset accordingly
    ratio = valid_percent / (test_percent + valid_percent)
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_train_data, y_train_data,
                                                                            test_size=(test_percent + valid_percent),
                                                                            random_state=42)
    x_test_data, x_val_data, y_test_data, y_val_data = train_test_split(x_test_data, y_test_data, test_size=ratio,
                                                                        random_state=42)
    # Writes the data into its corresponding file
    for i in range(len(x_train_data)):
        x_train_split.write(x_train_data[i]+"\n")
        y_train_split.write(y_train_data[i]+"\n")

    for i in range(len(x_val_data)):
        x_valid_split.write(x_val_data[i]+"\n")
        y_valid_split.write(y_val_data[i]+"\n")

    for i in range(len(x_test_data)):
        x_test_split.write(x_test_data[i]+"\n")
        y_test_split.write(y_test_data[i]+"\n")

    x_train.close()
    y_train.close()
    x_test.close()
    y_test.close()
    x_train_split.close()
    y_train_split.close()
    x_valid_split.close()
    y_valid_split.close()
    x_test_split.close()
    y_test_split.close()


input_filepath = "../input"
output_filepath = "../input/dataset"
validation_percent = 0.1
test_percent = 0.1

make_split(input_filepath, output_filepath, validation_percent, test_percent)
