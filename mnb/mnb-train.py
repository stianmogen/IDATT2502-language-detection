import os
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

cv = CountVectorizer(analyzer='char', ngram_range=(1, 4))

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

print("\nVectorized x_train, x_test: ")
print(x_train.shape)
print(x_test.shape)

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

ac = accuracy_score(y_test, y_pred)

print("Accuracy is :", ac)


def predict(text):
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang = le.inverse_transform(lang)  # finding the language corresponding the the predicted value
    print("The langauge is in", lang[0])


predict("my name is Simon")

predict("mi nombre es Simon")

predict("Hei jeg heter Simon")

predict("Her vil vi i dagane framover dele dikt av Olav H. Hauge. Vi vil la Hauges poesi gje våre eigne tankar både pause og perspektiv. Gjer som Hauge, og lat diktet vere ei lauvhytte eller eit snøhus som du kan krype inn i, og vere i, i stille kveldar.")

predict("내 이름은 zisun")

predict("меня зовут цисун")
