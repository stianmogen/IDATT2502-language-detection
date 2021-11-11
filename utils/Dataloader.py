import os

import pandas as pd


class Dataloader:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def read_file(self, file, encoding="utf8"):
        with open(os.path.join(self.input_dir, file), encoding=encoding) as f:
            data = f.read()
            data = data.split("\n")
            data.pop(-1)
            return data

    def get_dataframes(self):
        x_train = pd.DataFrame(self.read_file("x_train_split.txt"), columns=["sentence"])
        y_train = pd.DataFrame(self.read_file("y_train_split.txt"), columns=["language"])

        x_val = pd.DataFrame(self.read_file("x_val_split.txt"), columns=["sentence"])
        y_val = pd.DataFrame(self.read_file("y_val_split.txt"), columns=["language"])

        x_test = pd.DataFrame(self.read_file("x_test_split.txt"), columns=["sentence"])
        y_test = pd.DataFrame(self.read_file("y_test_split.txt"), columns=["language"])

        return x_train, y_train, x_val, y_val, x_test, y_test

