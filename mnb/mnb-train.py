import os
import pickle
import random
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

from utils.Dataloader import Dataloader

seed = 1111
random.seed(seed)
np.random.RandomState(seed)


INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
x_train, y_train, x_val, y_val, _, _ = dataloader.get_dataframes()

print("x_train, x_val shape:")
print(x_train.shape)
print(x_val.shape)


def remove_special(sentence):
    return re.sub(r'[\\/:*«`\'?¿";!<>,.|()-_)}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', '', sentence.lower().strip())


x_train = x_train['sentence'].apply(remove_special)
y_train = y_train['language']

x_val = x_val['sentence'].apply(remove_special)
y_val = y_val['language']


le = LabelEncoder()
y_train = le.fit_transform(y_train)

y_val = le.transform(y_val)

with open("out/labelencoder", 'wb') as fout:
    pickle.dump(le, fout)

fout.close()

min_values = [1, 2, 3]
max_values = [1, 2, 3]
analyzers = ["word", "char"]

with open("labelencoder", 'wb') as fout:
    pickle.dump(le, fout)

fout.close()


analyzers = {"char": [1, 2, 3], "word": [1]}


def predict(text, cv, model):
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang = le.inverse_transform(lang)  # finding the language corresponding the the predicted value
    print("The langauge is in", lang[0])


def run():
    root_out_path = "out/"
    if not os.path.exists(root_out_path):
        os.makedirs(root_out_path)

    for analyzer in analyzers:
        for max_value in analyzers[analyzer]:
            for min_value in range(1, max_value + 1):
                type_out_path = f"{max_value}{min_value}/"

                if not os.path.exists(root_out_path + type_out_path):
                    os.makedirs(root_out_path + type_out_path)

                print("\n", max_value, min_value)

                cv = CountVectorizer(analyzer=analyzer, ngram_range=(min_value, max_value))

                x_train_t = cv.fit_transform(x_train)
                x_val_t = cv.transform(x_val)

                file = open(os.path.join(root_out_path, type_out_path, "vectorizer"), 'wb')
                pickle.dump(cv, file)
                file.close()

                print("Vectorized x_train, x_val: ")
                print(x_train_t.shape)
                print(x_val_t.shape)

                model = MultinomialNB()
                model.fit(x_train_t, y_train)

                y_pred = model.predict(x_val_t)

                ac = accuracy_score(y_val, y_pred) * 100

                print(f"Accuracy is: {ac:.1f}%")

                file = open(os.path.join(root_out_path + type_out_path, "model.sav"), 'wb')
                pickle.dump(model, file)
                file.close()


run()






