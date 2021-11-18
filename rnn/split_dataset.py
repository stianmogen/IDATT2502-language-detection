import os
from sklearn.model_selection import train_test_split


def make_split(input_filepath, output_filepath, test_percent, valid_percent, encoding="utf8"):
    x_train = open(input_filepath + "/x_train.txt", encoding=encoding)
    y_train = open(input_filepath + '/y_train.txt', encoding=encoding)
    x_test = open(input_filepath + "/x_test.txt", encoding=encoding)
    y_test = open(input_filepath + "/y_test.txt", encoding=encoding)

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    x_train_split = open(output_filepath + '/x_train_split.txt', 'w', encoding=encoding)
    y_train_split = open(output_filepath + '/y_train_split.txt', 'w', encoding=encoding)
    x_valid_split = open(output_filepath + '/x_val_split.txt', 'w', encoding=encoding)
    y_valid_split = open(output_filepath + '/y_val_split.txt', 'w', encoding=encoding)
    x_test_split = open(output_filepath + '/x_test_split.txt', 'w', encoding=encoding)
    y_test_split = open(output_filepath + '/y_test_split.txt', 'w', encoding=encoding)

    x_train_data = x_train.readlines()
    x_test_data = x_test.readlines()
    y_train_data = y_train.readlines()
    y_test_data = y_test.readlines()

    x_train_data += x_test_data
    y_train_data += y_test_data

    ratio = valid_percent / (test_percent + valid_percent)
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_train_data, y_train_data,
                                                                            test_size=(test_percent + valid_percent),
                                                                            random_state=42)
    x_test_data, x_val_data, y_test_data, y_val_data = train_test_split(x_test_data, y_test_data, test_size=ratio,
                                                                        random_state=42)

    for i in range(len(x_train_data)):
        if (y_train_data[i] == y_train_data[0] or
                y_train_data[i] == y_train_data[1] or
                y_train_data[i] == y_train_data[2] or
                y_train_data[i] == y_train_data[3] or
                y_train_data[i] == y_train_data[4] or
                y_train_data[i] == y_train_data[5] or
                y_train_data[i] == y_train_data[6] or
                y_train_data[i] == y_train_data[7] or
                y_train_data[i] == y_train_data[8] or
                y_train_data[i] == y_train_data[9] or
                y_train_data[i] == y_train_data[10] or
                y_train_data[i] == y_train_data[11] or
                y_train_data[i] == y_train_data[12] or
                y_train_data[i] == y_train_data[13] or
                y_train_data[i] == y_train_data[14] or
                y_train_data[i] == y_train_data[15] or
                y_train_data[i] == y_train_data[16] or
                y_train_data[i] == y_train_data[17] or
                y_train_data[i] == y_train_data[18] or
                y_train_data[i] == y_train_data[19] or
                y_train_data[i] == y_train_data[20] or
                y_train_data[i] == y_train_data[21] or
                #y_train_data[i] == y_train_data[22] or
                y_train_data[i] == y_train_data[23]):
            print(y_train_data[i], i)
            x_train_split.write(x_train_data[i])
            y_train_split.write(y_train_data[i])

    for i in range(len(x_val_data)):
        if  (y_val_data[i] == y_train_data[0] or
                y_val_data[i] == y_train_data[1] or
                y_val_data[i] == y_train_data[2] or
                y_val_data[i] == y_train_data[3] or
                y_val_data[i] == y_train_data[4] or
                y_val_data[i] == y_train_data[5] or
                y_val_data[i] == y_train_data[6] or
                y_val_data[i] == y_train_data[7] or
                y_val_data[i] == y_train_data[8] or
                y_val_data[i] == y_train_data[9] or
                y_val_data[i] == y_train_data[10] or
                y_val_data[i] == y_train_data[11] or
                y_val_data[i] == y_train_data[12] or
                y_val_data[i] == y_train_data[13] or
                y_val_data[i] == y_train_data[14] or
                y_val_data[i] == y_train_data[15] or
                y_val_data[i] == y_train_data[16] or
                y_val_data[i] == y_train_data[17] or
                y_val_data[i] == y_train_data[18] or
                y_val_data[i] == y_train_data[19] or
                y_val_data[i] == y_train_data[20] or
                y_val_data[i] == y_train_data[21] or
                #y_val_data[i] == y_train_data[22] or
                y_val_data[i] == y_train_data[23]):
            x_valid_split.write(x_val_data[i])
            y_valid_split.write(y_val_data[i])

    for i in range(len(x_test_data)):
        if (y_test_data[i] == y_train_data[0] or
                y_test_data[i] == y_train_data[1] or
                y_test_data[i] == y_train_data[2] or
                y_test_data[i] == y_train_data[3] or
                y_test_data[i] == y_train_data[4] or
                y_test_data[i] == y_train_data[5] or
                y_test_data[i] == y_train_data[6] or
                y_test_data[i] == y_train_data[7] or
                y_test_data[i] == y_train_data[8] or
                y_test_data[i] == y_train_data[9] or
                y_test_data[i] == y_train_data[10] or
                y_test_data[i] == y_train_data[11] or
                y_test_data[i] == y_train_data[12] or
                y_test_data[i] == y_train_data[13] or
                y_test_data[i] == y_train_data[14] or
                y_test_data[i] == y_train_data[15] or
                y_test_data[i] == y_train_data[16] or
                y_test_data[i] == y_train_data[17] or
                y_test_data[i] == y_train_data[18] or
                y_test_data[i] == y_train_data[19] or
                y_test_data[i] == y_train_data[20] or
                y_test_data[i] == y_train_data[21] or
                #y_test_data[i] == y_train_data[22] or
                y_test_data[i] == y_train_data[23]):
            x_test_split.write(x_test_data[i])
            y_test_split.write(y_test_data[i])

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
train_percent = 1 - validation_percent - test_percent
