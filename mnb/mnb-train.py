import os
import pickle
import random
import re

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

import pandas as pd

from utils.Dataloader import Dataloader

seed = 1111
random.seed(seed)
np.random.RandomState(seed)

device = ""
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0")
else:
    torch.manual_seed(seed)
    device = torch.device("cpu")

print(device)

# Constants

INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
x_train, y_train, x_val, y_val, x_test, y_test = dataloader.get_dataframes()

print("x_train, x_test shape:")
print(x_train.shape)
print(x_test.shape)


def remove_special(sentence):
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|()-_)(}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', ' ', sentence.lower().strip())


x_train = x_train['sentence'].apply(remove_special)
y_train = y_train['language']

x_test = x_test['sentence'].apply(remove_special)
y_test = y_test['language']


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

min_values = [1, 2, 3]
max_values = [1, 2, 3]


def predict(text, cv, model):
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang = le.inverse_transform(lang)  # finding the language corresponding the the predicted value
    print("The langauge is in", lang[0])


def run():
    root_out_path = "out/"
    if not os.path.exists(root_out_path):
        os.makedirs(root_out_path)
    for min_value in min_values:
        for max_value in max_values:
            print("\n", max_value, min_value)
            if max_value >= min_value:

                cv = CountVectorizer(analyzer='char', ngram_range=(min_value, max_value))

                x_train_t = cv.fit_transform(x_train)
                x_test_t = cv.transform(x_test)

                print("Vectorized x_train, x_test: ")
                print(x_train_t.shape)
                print(x_test_t.shape)

                model = MultinomialNB()
                model.fit(x_train_t, y_train)

                y_pred = model.predict(x_test_t)

                ac = accuracy_score(y_test, y_pred) * 100

                print(f"Accuracy is: {ac:.1f}%")

                file = open(os.path.join(root_out_path, f"model{min_value}{max_value}acc{int(ac)}.sav"), 'wb')
                pickle.dump(model, file)


run()






