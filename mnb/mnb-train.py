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

# Uses dataloader to read the training and validation data
INPUT_DIR = "../input/dataset"
dataloader = Dataloader(INPUT_DIR)
# Uses the get_dataframes() method in dataloader to fetch the relevant dataframes for training
x_train, y_train, x_val, y_val, _, _ = dataloader.get_dataframes()

# Prints the shape of the dataframes
print("x_train, x_val shape:")
print(x_train.shape)
print(x_val.shape)

# Special characters are removed and not used in training
def remove_special(sentence):
    return re.sub(r'[\\/:*«`\'?¿";!<>,.|()-_)}{#$%@^&~+-=–—‘’“”„†•…′ⁿ№−、《》「」『』（），－：；]', '', sentence.lower().strip())

# Defines training and validation by their respective sentence and language data
x_train = x_train['sentence'].apply(remove_special)
y_train = y_train['language']

x_val = x_val['sentence'].apply(remove_special)
y_val = y_val['language']

# Label encoder giving indexes to data for efficient lookup
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

with open("out/labelencoder", 'wb') as fout:
    pickle.dump(le, fout)

fout.close()

min_values = [1, 2, 3]
max_values = [1, 2, 3]

fout.close()

analyzers = {"char": [1, 2, 3], "word": [1]}


def run():
    # Creates directory if does not exist
    root_out_path = "out/"
    if not os.path.exists(root_out_path):
        os.makedirs(root_out_path)

    # Creates path for the specific analysers defined
    for analyzer in analyzers:
        analyzer_out_path = f"{root_out_path}{analyzer}/"

        if not os.path.exists(analyzer_out_path):
            os.makedirs(analyzer_out_path)

        # Loops for the analysing options defined by the user
        for max_value in analyzers[analyzer]:
            for min_value in range(1, max_value + 1):
                size_out_path = f"{analyzer_out_path}{max_value}{min_value}/"

                if not os.path.exists(size_out_path):
                    os.makedirs(size_out_path)

                print("\n", max_value, min_value)

                # Vecotrizes data depending on the given analyzer and n-gram models
                cv = CountVectorizer(analyzer=analyzer, ngram_range=(min_value, max_value))
                # Transforms data with vectorizer
                x_train_t = cv.fit_transform(x_train)
                x_val_t = cv.transform(x_val)

                file = open(os.path.join(size_out_path, "vectorizer"), 'wb')
                pickle.dump(cv, file)
                file.close()

                print("Vectorized x_train, x_val: ")
                print(x_train_t.shape)
                print(x_val_t.shape)

                # Creates an instance of the sklearn Multinomial Naive Bayes model and fits data
                model = MultinomialNB()
                model.fit(x_train_t, y_train)

                # Predicts the language based on the validation data
                y_pred = model.predict(x_val_t)
                # Gives a percentage score based on the amount of correct guesses
                ac = accuracy_score(y_val, y_pred) * 100

                print(f"Accuracy is: {ac:.1f}%")
                # Saves the model in the corresponding directory
                file = open(os.path.join(size_out_path, "model.sav"), 'wb')
                pickle.dump(model, file)
                file.close()


run()






